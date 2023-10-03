"""
Backbone model - UNet architecture, with attention

This file implements a UNet design for the imaging backbone.
The input to the model is [B, T, C_in, H, W]. The output of the model is [B, T, C, H, W].
For every resolution level, the image size will be reduced by x2, with the number of channels increasing by x2.

Please ref to the project page for the network design.
"""

import os
import sys
import logging
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision

from pathlib import Path
from argparse import Namespace

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Current_DIR))

Model_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Model_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from imaging_attention import *
from cells import *
from blocks import *
from setup.setup_utils import get_device
from utils.status import model_info

from backbone_base import STCNNT_Base_Runtime, set_window_patch_sizes_keep_num_window, set_window_patch_sizes_keep_window_size, DownSample, UpSample

__all__ = ['STCNNT_Unet_model']

#-------------------------------------------------------------------------------------
def STCNNT_Unet_model(config, pre_feature_channels):
    """
    Simple function to return STCCNT Unet model.
    Additionally, function computes feature_channels, a list of ints containing the number of channels in each feature returned by the model.
    """
    config.no_in_channel = pre_feature_channels[-1]
    model = STCNNT_Unet(config=config)
    feature_channels = [int(config.backbone_unet.C)]
    return model, feature_channels


# -------------------------------------------------------------------------------------------------
class _unet_attention(nn.Module):
    """
    Unet attention scheme

    The query q is from the lower resolution level [B, T, C_q, H, W];
    The value x is from the higher resolution level [B, T, C, H, W]

    Output is a gated value tensor [B, T, C, H, W]
    """

    def __init__(self, C_q=32, C=16) -> None:
        super().__init__()

        self.C_q = C_q
        self.C = C

        self.conv_query = Conv2DExt(in_channels=self.C_q, out_channels=self.C, kernel_size=[1,1], stride=[1,1], padding=[0,0])
        self.conv_x = Conv2DExt(in_channels=self.C, out_channels=self.C, kernel_size=[1,1], stride=[1,1], padding=[0,0])

        self.conv_gate = Conv2DExt(in_channels=self.C, out_channels=1, kernel_size=[1,1], stride=[1,1], padding=[0,0])

    def forward(self, q:Tensor, x:Tensor) -> Tensor:
        B, T, C_q, H, W = q.shape
        B, T, C, H, W = x.shape

        v = F.relu(self.conv_query(q) + self.conv_x(x), inplace=False)
        g = torch.sigmoid(self.conv_gate(v)) # [B, T, 1, H, W]

        y = x * g

        return y

# -------------------------------------------------------------------------------------------------
# stcnnt hrnet

class STCNNT_Unet(STCNNT_Base_Runtime):
    """
    This class implemented the stcnnt version of Unet with maximal 5 down/upsample levels.
    
    The attention window_size and patch_size are in the unit of pixels and set for the top level resolution. For every downsample level,
    they are reduced by x2 to keep the number of windows roughly the same.
    
    The minimal window size is 16 and minimal patch size is 4.
    """

    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
            - load (bool): whether to try loading from config.load_path or not

        @args (from config):

            ---------------------------------------------------------------
            model specific arguments
            ---------------------------------------------------------------

            - C (int): number of channels, when resolution is reduced by x2, number of channels will increase by x2
            - num_resolution_levels (int): number of resolution levels; each deeper level will reduce spatial size by x2

            - block_str (str | list of strings): order of attention types and mixer
                format is list of XYXYXYXY...
                - X is "L", "G" or "T" or "V" for attention type
                - Y is "0" or "1" for with or without mixer
                - requires len(att_types[i]) to be even

                This string is the "Block string" to define the attention layers in a block.
                If a list of string is given, each string defines the attention structure for a resolution level.
                The last string is the bridge structure.

            - use_unet_attention (bool): whether to use unet attention from lower resolution to higher resolution
            - use_interpolation (bool): whether to use interpolation in downsample layer; if False, use stride convolution
            - with_conv (bool): whether to add conv in down/upsample layers; if False, only interpolation is performed

            ---------------------------------------------------------------
            Shared arguments used in this model
            ---------------------------------------------------------------
            - C_in (int): number of input channels

            - height (int list): expected heights of the input
            - width (int list): expected widths of the input

            - a_type ("conv", "lin"): type of attention in spatial heads
            - cell_type ("sequential", "parallel"): type of attention cell
            - window_size (int): size of window for local and global att
            - patch_size (int): size of patch for local and global att
            - window_sizing_method (str): "mixed", "keep_window_size", "keep_num_window"
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - stride_t (int): special stride for temporal attention k,q matrices
            - normalize_Q_K (bool): whether to use layernorm to normalize Q and K, as in 22B ViT paper
            - att_dropout_p (float): probability of dropout for attention coefficients
            - dropout (float): probability of dropout
            - att_with_output_proj (bool): whether to add output projection in the attention layer
            - scale_ratio_in_mixer (float): channel scaling ratio in the mixer
            - norm_mode ("layer", "batch", "instance"):
                layer - norm along C, H, W; batch - norm along B*T; or instance norm along H, W for C
            - interp_align_c (bool):
                whether to align corner or not when interpolating
            - residual (bool):
                whether to add long skip residual connection or not

            - optim ("adamw", "sgd", "nadam"): choices for optimizer
            - scheduler ("ReduceOnPlateau", "StepLR", "OneCycleLR"):
                choices for learning rate schedulers
            - global_lr (float): global learning rate
            - beta1, beta2 (float): parameters for adam
            - weight_decay (float): parameter for regularization
            - all_w_decay (bool): whether to separate model params for regularization

            - load_path (str): path to load the weights from
        """
        super().__init__(config)

        self.check_class_specific_parameters(config)

        C = config.backbone_unet.C
        num_resolution_levels = config.backbone_unet.num_resolution_levels
        block_str = config.backbone_unet.block_str
        use_unet_attention = config.backbone_unet.use_unet_attention
        use_interpolation = config.backbone_unet.use_interpolation
        with_conv = config.backbone_unet.with_conv

        assert C >= config.no_in_channel, "Number of channels should be larger than C_in"
        assert num_resolution_levels <= 5 and num_resolution_levels>=1, "Maximal number of resolution levels is 5"

        self.C = C
        self.num_resolution_levels = num_resolution_levels
                
        if isinstance(block_str, list):
            self.block_str = block_str if len(block_str)>=self.num_resolution_levels+1 else [block_str[0] for n in range(self.num_resolution_levels+1)] # with bridge
        else:
            self.block_str = [block_str for n in range(self.num_resolution_levels+1)]
        
        self.use_unet_attention = use_unet_attention
        self.use_interpolation = use_interpolation
        self.with_conv = with_conv

        c = copy.deepcopy(config)
        
        # compute number of windows and patches
        self.num_wind = [c.height//c.window_size[0], c.width//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]

        kwargs = {
            "C_in":c.no_in_channel,
            "C_out":self.C,
            "H":c.height,
            "W":c.width,
            "a_type":c.a_type,
            "window_size": c.window_size,
            "patch_size": c.patch_size,
            "is_causal":c.is_causal,
            "dropout_p":c.dropout_p,
            "n_head":c.n_head,

            "kernel_size":(c.kernel_size, c.kernel_size),
            "stride":(c.stride, c.stride),
            "padding":(c.padding, c.padding),

            "mixer_kernel_size":(c.mixer_kernel_size, c.mixer_kernel_size),
            "mixer_stride":(c.mixer_stride, c.mixer_stride),
            "mixer_padding":(c.mixer_padding, c.mixer_padding),

            "stride_s": (c.stride_s, c.stride_s),
            "stride_t":(c.stride_t, c.stride_t),

            "separable_conv": c.separable_conv,

            "norm_mode":c.norm_mode,
            "interpolate":"none",
            "interp_align_c":c.interp_align_c,

            "cell_type": c.cell_type,
            "normalize_Q_K": c.normalize_Q_K, 
            "att_dropout_p": c.att_dropout_p,
            "att_with_output_proj": c.att_with_output_proj, 
            "scale_ratio_in_mixer": c.scale_ratio_in_mixer,

            "cosine_att": c.cosine_att,
            "att_with_relative_postion_bias": c.att_with_relative_postion_bias,
            "block_dense_connection": c.block_dense_connection,

            "num_wind": self.num_wind,
            "num_patch": self.num_patch,

            "mixer_type": c.mixer_type,
            "shuffle_in_window": c.shuffle_in_window,
            
            "use_einsum": c.use_einsum,
            "temporal_flash_attention": c.temporal_flash_attention,

            "activation_func": c.activation_func
        }

        window_sizes = []
        patch_sizes = []

        if num_resolution_levels >= 1:
            # define D0
            kwargs["C_in"] = c.no_in_channel
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            
            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D0")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="D0")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D0")
                
            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])
            
            kwargs["att_types"] = self.block_str[0]
            self.D0 = STCNNT_Block(**kwargs)

            self.down_0 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)

        if num_resolution_levels >= 2:
            # define D1
            kwargs["C_in"] = self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2
            
            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D1")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="D1")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]] , window_sizes[0], patch_sizes[0], module_name="D1")
                
            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])
            
            kwargs["att_types"] = self.block_str[1]
            self.D1 = STCNNT_Block(**kwargs)

            self.down_1 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)

        if num_resolution_levels >= 3:
            # define D2
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height // 4
            kwargs["W"] = c.width // 4
            
            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D2")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[1], patch_sizes[1], module_name="D2")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , [v//2 for v in self.num_wind], self.num_patch, module_name="D2")
                
            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])
            
            kwargs["att_types"] = self.block_str[2]
            self.D2 = STCNNT_Block(**kwargs)

            self.down_2 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)

        if num_resolution_levels >= 4:
            # define D3
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height // 8
            kwargs["W"] = c.width // 8
            
            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D3")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="D3")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="D3")
                
            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])
            
            kwargs["att_types"] = self.block_str[3]
            self.D3 = STCNNT_Block(**kwargs)

            self.down_3 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)

        if num_resolution_levels >= 5:
            # define D4
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 16*self.C
            kwargs["H"] = c.height // 16
            kwargs["W"] = c.width // 16
            
            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D4")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[3], patch_sizes[3], module_name="D4")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , [v//4 for v in self.num_wind], self.num_patch, module_name="D4")
                
            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])
            
            kwargs["att_types"] = self.block_str[4]
            self.D4 = STCNNT_Block(**kwargs)

            self.down_4 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)

        # define the bridge
        kwargs["C_in"] = kwargs["C_out"]
        kwargs["att_types"] = self.block_str[-1]
        kwargs["H"] //= 2
        kwargs["W"] //= 2
        kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[-1], patch_sizes[-1], module_name="bridge")
        self.bridge = STCNNT_Block(**kwargs)

        if num_resolution_levels >= 5:
            self.up_4 = UpSample(N=1, C_in=16*self.C, C_out=16*self.C, method=c.upsample_method, with_conv=self.with_conv)
            if self.use_unet_attention:
                self.attention_4 = _unet_attention(C_q=16*self.C, C=16*self.C)

            kwargs["C_in"] = 32*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height // 16
            kwargs["W"] = c.width // 16
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[4], patch_sizes[4], module_name="U4")
            kwargs["att_types"] = self.block_str[4]
            self.U4 = STCNNT_Block(**kwargs)

        if num_resolution_levels >= 4:
            self.up_3 = UpSample(N=1, C_in=8*self.C, C_out=8*self.C, method=c.upsample_method, with_conv=self.with_conv)
            if self.use_unet_attention:
                self.attention_3 = _unet_attention(C_q=8*self.C, C=8*self.C)

            kwargs["C_in"] = 16*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height // 8
            kwargs["W"] = c.width // 8
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[3], patch_sizes[3], module_name="U3")
            kwargs["att_types"] = self.block_str[3]
            self.U3 = STCNNT_Block(**kwargs)

        if num_resolution_levels >= 3:
            self.up_2 = UpSample(N=1, C_in=4*self.C, C_out=4*self.C, method=c.upsample_method, with_conv=self.with_conv)
            if self.use_unet_attention:
                self.attention_2 = _unet_attention(C_q=4*self.C, C=4*self.C)

            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height // 4
            kwargs["W"] = c.width // 4
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="U2")
            kwargs["att_types"] = self.block_str[2]
            self.U2 = STCNNT_Block(**kwargs)

        if num_resolution_levels >= 2:
            self.up_1 = UpSample(N=1, C_in=2*self.C, C_out=2*self.C, method=c.upsample_method, with_conv=self.with_conv)
            if self.use_unet_attention:
                self.attention_1 = _unet_attention(C_q=2*self.C, C=2*self.C)

            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[1], patch_sizes[1], module_name="U1")
            kwargs["att_types"] = self.block_str[1]
            self.U1 = STCNNT_Block(**kwargs)

        if num_resolution_levels >= 1:
            self.up_0 = UpSample(N=1, C_in=self.C, C_out=self.C, method=c.upsample_method, with_conv=self.with_conv)
            if self.use_unet_attention:
                self.attention_0 = _unet_attention(C_q=self.C, C=self.C)

            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="U0")
            kwargs["att_types"] = self.block_str[0]
            self.U0 = STCNNT_Block(**kwargs)

        self.permute = torchvision.ops.misc.Permute([0,2,1,3,4])


    def check_class_specific_parameters(self, config):
        if not "backbone_unet" in config:
            raise "backbone_unet namespace should exist in config"
               
        err_str = lambda x : f"{x} should exist in config.backbone_unet"        

        para_list = ["C", "num_resolution_levels", "block_str", "use_unet_attention", "use_interpolation", "with_conv"]        
        for arg_name in para_list:            
            if not arg_name in config.backbone_unet:
                raise ValueError(err_str(arg_name))
            
            
    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): the input image, [B, T, Cin, H, W]

        @rets:
            - y_hat (5D torch.Tensor): output tensor, [B, T, Cout, H, W]
        """

        x = self.permute(x)

        B, T, Cin, H, W = x.shape

        # first we go down the resolution ... 
        if self.num_resolution_levels >= 1:
            x_0, _ = self.D0(x)
            x_d_0 = self.down_0(x_0)

        if self.num_resolution_levels >= 2:
            x_1, _ = self.D1(x_d_0)
            x_d_1 = self.down_1(x_1)

        if self.num_resolution_levels >= 3:
            x_2, _ = self.D2(x_d_1)
            x_d_2 = self.down_2(x_2)

        if self.num_resolution_levels >= 4:
            x_3, _ = self.D3(x_d_2)
            x_d_3 = self.down_3(x_3)

        if self.num_resolution_levels >= 5:
            x_4, _ = self.D4(x_d_3)
            x_d_4 = self.down_4(x_4)

        # now we go up the resolution ...
        if self.num_resolution_levels == 1:
            y_d_0, _ = self.bridge(x_d_0)
            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        if self.num_resolution_levels == 2:
            y_d_1, _ = self.bridge(x_d_1)
            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1) if self.use_unet_attention else x_1
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        if self.num_resolution_levels == 3:
            y_d_2, _ = self.bridge(x_d_2)
            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2) if self.use_unet_attention else x_2
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1) if self.use_unet_attention else x_1
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        if self.num_resolution_levels == 4:
            y_d_3, _ = self.bridge(x_d_3)
            y_3 = self.up_3(y_d_3)
            x_gated_3 = self.attention_3(q=y_3, x=x_3) if self.use_unet_attention else x_3
            y_d_2, _ = self.U3(torch.cat((x_gated_3, y_3), dim=2))

            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2) if self.use_unet_attention else x_2
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1) if self.use_unet_attention else x_1
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        if self.num_resolution_levels == 5:
            y_d_4, _ = self.bridge(x_d_4)
            y_4 = self.up_4(y_d_4)
            x_gated_4 = self.attention_4(q=y_4, x=x_4) if self.use_unet_attention else x_4
            y_d_3, _ = self.U4(torch.cat((x_gated_4, y_4), dim=2))

            y_3 = self.up_3(y_d_3)
            x_gated_3 = self.attention_3(q=y_3, x=x_3) if self.use_unet_attention else x_3
            y_d_2, _ = self.U3(torch.cat((x_gated_3, y_3), dim=2))

            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2) if self.use_unet_attention else x_2
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1) if self.use_unet_attention else x_1
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0) if self.use_unet_attention else x_0
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        y_hat = self.permute(y_hat)
        
        return [y_hat]

    def __str__(self):
        return create_generic_class_str(obj=self, exclusion_list=[nn.Module, OrderedDict, STCNNT_Block, DownSample, UpSample, _unet_attention])

# -------------------------------------------------------------------------------------------------

def tests():

    from utils.benchmark import benchmark_all, benchmark_memory, pytorch_profiler
    from utils.setup_training import set_seed
    from colorama import Fore, Style

    device = get_device()
        
    B,T,C,H,W = 1, 12, 1, 256, 256
    test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)
    
    parser = add_backbone_STCNNT_args()
    ns = Nestedspace()
    config = parser.parse_args(namespace=ns)
    
    config.no_in_channel = C
    config.C_out = C
    config.height = [H]
    config.width = [W]
    config.batch_size = B
    config.time = T
    config.norm_mode = "instance3d"
    config.a_type = "lin"

    config.window_size = [H//8, W//8]
    config.patch_size = [H//32, W//32]
    
    config.num_wind =[8, 8]
    config.num_patch =[4, 4]
    
    config.window_sizing_method = "mixed"
    
    # losses
    config.losses = ["mse"]
    config.loss_weights = [1.0]
    config.load_path = None

    # to be tested
    config.residual = True
    config.device = device
    config.channels = [16,32,64]
    config.all_w_decay = True
    config.optim = "adamw"
    config.scheduler = "StepLR"

    config.complex_i = False

    # ---------------------------------------------------------------------

    config.backbone_unet.block_str = ["C2C2C2",
                        "C3C3C3",
                        "C2C2C2",
                        "C3C3C3",
                        "C2C2C2"]

    model = STCNNT_Unet(config=config)
    model.to(device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(test_in)

    config.with_timer = False
    print(f"{Fore.GREEN}-------------> CONV_Unet---einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in, grad=None, min_run_time=5, desc='CONV_Unet', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in, desc='CONV_Unet', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # ---------------------------------------------------------------------

    config.backbone_unet.block_str = ["T1L1G1",
                        "T1L1G1",
                        "T1L1G1",
                        "T1L1G1",
                        "T1L1G1"]

    model = STCNNT_Unet(config=config)
    model.to(device=device)


    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(test_in)

    config.with_timer = False
    print(f"{Fore.GREEN}-------------> STCNNT_Unet---einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in, grad=None, min_run_time=5, desc='STCNNT_Unet', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in, desc='STCNNT_Unet', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # ---------------------------------------------------------------------

    config.stride_s = 1
    config.separable_conv = False
    config.use_einsum = False
    
    model = STCNNT_Unet(config=config)
    model.to(device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(test_in)
            
    print(f"{Fore.GREEN}-------------> STCNNT_Unet---einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in.to(device=device), grad=None, min_run_time=5, desc='STCNNT_Unet-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in.to(device=device), desc='STCNNT_Unet-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)
    
    # ---------------------------------------------------------------------

    model = STCNNT_Unet(config=config)
    model.to(device=device)
    with torch.no_grad():
        model_summary = model_info(model, config)
    print(f"Configuration for this run:\n{config}")
    print(f"Model Summary:\n{str(model_summary)}")

    print("Passed all tests")

if __name__=="__main__":
    tests()
