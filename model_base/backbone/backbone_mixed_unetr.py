"""
Backbone model - UNet architecture, with attention

This file implements a UNetr design for the imaging backbone.
The input to the model is [B, T, C_in, H, W]. The output of the model is [B, T, C, H, W].
For every resolution level, the image size will be reduced by x2, with the number of channels increasing by x2.

If the T//2 for current level is less than min_T, the scaling down will only be applied to H and W.

The transformer_for_up_branch decides whether conv or transformers are used for upsampling branch.

n_heads is a list to decide number of heads for each level.

Please ref to the project page for the network design.
"""

import os
import sys
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from pathlib import Path
from argparse import Namespace

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from losses import *
from imaging_attention import *
from cells import *
from blocks import *
from utils import get_device, model_info, add_backbone_STCNNT_args, Nestedspace

from backbone_base import STCNNT_Base_Runtime, set_window_patch_sizes_keep_num_window, set_window_patch_sizes_keep_window_size, DownSample, UpSample, WindowPartition3D, WindowPartition2D

__all__ = ['STCNNT_Mixed_Unetr']

# -------------------------------------------------------------------------------------------------

class _unet_attention(nn.Module):
    """
    Unet attention scheme

    The query q is from the lower resolution level [B, T, C_q, H, W];
    The value x is from the higher resolution level [B, T, C, H, W]

    Output is a gated value tensor [B, T, C, H, W]
    """

    def __init__(self, C_q=32, C=16, use_conv_3d=False) -> None:
        super().__init__()

        self.C_q = C_q
        self.C = C
        self.use_conv_3d = use_conv_3d

        if use_conv_3d:
            self.conv_query = Conv3DExt(in_channels=self.C_q, out_channels=self.C, kernel_size=[1,1,1], stride=[1,1,1], padding=[0,0,0])
            self.conv_x = Conv3DExt(in_channels=self.C, out_channels=self.C, kernel_size=[1,1,1], stride=[1,1,1], padding=[0,0,0])
            self.conv_gate = Conv3DExt(in_channels=self.C, out_channels=1, kernel_size=[1,1,1], stride=[1,1,1], padding=[0,0,0])
        else:
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

def create_conv(in_channels, out_channels, kernel_size=[3,3,3], stride=[1,1,1], padding=[1,1,1], bias=False, separable_conv=False, use_conv_3d=True):
    if use_conv_3d:
        conv = Conv3DExt(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, separable_conv=separable_conv)
    else:
        conv = Conv2DExt(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, separable_conv=separable_conv)

    return conv

# -------------------------------------------------------------------------------------------------

class _encoder_on_skip_connection(nn.Module):
    
    def __init__(self, H=32, W=32, C_in=32, C_out=32, norm_mode="instance2d", activation="prelu", bias=False, separable_conv=False, use_conv_3d=True) -> None:
            super().__init__()

            self.C_in = C_in
            self.C_out = C_out

            self.conv1 = create_conv(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[3,3,3], stride=[1,1,1], padding=[1,1,1], bias=bias, separable_conv=separable_conv, use_conv_3d=use_conv_3d)
            self.norm1 = create_norm(norm_mode=norm_mode, C=self.C_out, H=H, W=W)
            self.nl1 = create_activation_func(name=activation)
            
            self.conv2 = create_conv(in_channels=self.C_out, out_channels=self.C_out, kernel_size=[3,3,3], stride=[1,1,1], padding=[1,1,1], bias=bias, separable_conv=separable_conv, use_conv_3d=use_conv_3d)
            self.norm2 = create_norm(norm_mode=norm_mode, C=self.C_out, H=H, W=W)
            self.nl2 = create_activation_func(name=activation)
            
    def forward(self, x):
        
        residual = x
        res = self.conv1(x)
        res = self.norm1(res)
        res = self.nl1(res)
        res = self.conv2(res)
        res = self.norm2(res)
        res += residual
        res = self.nl2(res)
        
        return res

class _decoder_conv(_encoder_on_skip_connection):

    def __init__(self, H=32, W=32, C_in=32, C_out=32, norm_mode="instance2d", activation="prelu", bias=False, separable_conv=False, use_conv_3d=True) -> None:
            super().__init__(H=H, W=W, C_in=C_in, C_out=C_out, norm_mode=norm_mode, activation=activation, bias=bias, separable_conv=separable_conv, use_conv_3d=use_conv_3d)


# -------------------------------------------------------------------------------------------------

# stcnnt hrnet

class STCNNT_mixed_Unetr(STCNNT_Base_Runtime):
    """
    This class implemented the stcnnt version of Unetr with maximal 5 down/upsample levels.

    The attention window_size and patch_size are in the unit of pixels and set for the top level resolution. For every downsample level,
    they are reduced by x2 to keep the number of windows roughly the same.

    min_T is set to control whether downsampling is performed along T/D dimension.

    encoder_on_skip_connection controls whether conv layers are added to skip connections.

    transformer_for_upsampling controls whether to use imaging transformers for the upsampling branch.

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
            - min_T (int): minimal T/D dimension to allow downsampling along T or D
            - encoder_on_skip_connection (bool): if true, add conv layers on the skip connection
            - transformer_for_upsampling (bool): if true, use transformer for the upsampling; otherwise, use conv layers
            - n_heads (list): number of heads for each resolution level
        """
        super().__init__(config)

        self.check_class_specific_parameters(config)

        C = config.backbone_mixed_unetr.C
        num_resolution_levels = config.backbone_mixed_unetr.num_resolution_levels
        block_str = config.backbone_mixed_unetr.block_str
        use_unet_attention = config.backbone_mixed_unetr.use_unet_attention
        use_interpolation = config.backbone_mixed_unetr.use_interpolation
        with_conv = config.backbone_mixed_unetr.with_conv
        min_T = config.backbone_mixed_unetr.min_T
        encoder_on_skip_connection = config.backbone_mixed_unetr.encoder_on_skip_connection
        transformer_for_upsampling = config.backbone_mixed_unetr.transformer_for_upsampling
        n_heads = config.backbone_mixed_unetr.n_heads
        use_conv_3d = config.backbone_mixed_unetr.use_conv_3d

        assert C >= config.C_in, "Number of channels should be larger than C_in"
        assert num_resolution_levels <= 5 and num_resolution_levels>=1, "Maximal number of resolution levels is 5"

        self.C = C
        self.num_resolution_levels = num_resolution_levels

        if isinstance(block_str, list):
            self.block_str = block_str if len(block_str)>=self.num_resolution_levels+1 else [block_str[0] for n in range(self.num_resolution_levels+1)] # with bridge
        else:
            self.block_str = [block_str for n in range(self.num_resolution_levels+1)]

        if isinstance(n_heads, list):
            self.n_heads = n_heads if len(n_heads)>=self.num_resolution_levels+1 else [n_heads[0] for n in range(self.num_resolution_levels+1)] # with bridge
        else:
            self.n_heads = [int(n_heads) for n in range(self.num_resolution_levels+1)]

        self.use_unet_attention = use_unet_attention
        self.use_interpolation = use_interpolation
        self.with_conv = with_conv
        self.min_T = min_T
        self.encoder_on_skip_connection = encoder_on_skip_connection
        self.transformer_for_upsampling = transformer_for_upsampling
        self.n_heads = n_heads
        self.use_conv_3d = use_conv_3d

        c = config

        # window partition layer
        H = c.height[0]
        W = c.width[0]
        T = c.time

        if T//2 > min_T:
            self.window_partition = WindowPartition3D()
            C_in_wp = 8 * c.C_in
            T = T//2
            is_3D_window_partition = True
        else:
            self.window_partition = WindowPartition2D()
            C_in_wp = 4 * c.C_in
            is_3D_window_partition = False

        self.conv_window_partition = create_conv(in_channels=C_in_wp, out_channels=self.C, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)

        if encoder_on_skip_connection:
            self.E = _encoder_on_skip_connection(H=H, W=W, C_in=c.C_in, C_out=self.C, norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)
            self.EW = _encoder_on_skip_connection(H=H//2, W=W//2, C_in=self.C, C_out=self.C, norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)
        else:
            self.E = nn.Identity()

        # compute number of windows and patches
        c.height[0] = H//2
        c.width[0] = W//2

        self.num_wind = [c.height[0]//c.window_size[0], c.width[0]//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]

        kwargs = {
            "C_in":self.C,
            "C_out":self.C,
            "H":c.height[0],
            "W":c.width[0],
            "a_type":c.a_type,
            "window_size": c.window_size,
            "patch_size": c.patch_size,
            "is_causal":c.is_causal,
            "dropout_p":c.dropout_p,
            #"n_head":c.n_head,

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
            "temporal_flash_attention": c.temporal_flash_attention
        }

        window_sizes = []
        patch_sizes = []

        if num_resolution_levels >= 1:
            # define D0
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D0")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="D0")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D0")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[0]
            kwargs["n_head"] = n_heads[0]
            self.D0 = STCNNT_Block(**kwargs)

            if encoder_on_skip_connection:
                self.E0 = _encoder_on_skip_connection(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_out"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)
            else:
                self.E0 = nn.Identity()

            self.down_0 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv, is_3D=(T//2 >= min_T))

        if num_resolution_levels >= 2:
            # define D1
            kwargs["C_in"] = self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D1")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="D1")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]] , window_sizes[0], patch_sizes[0], module_name="D1")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[1]
            kwargs["n_head"] = n_heads[1]
            self.D1 = STCNNT_Block(**kwargs)

            if encoder_on_skip_connection:
                self.E1 = _encoder_on_skip_connection(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_out"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)
            else:
                self.E1 = nn.Identity()

            self.down_1 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv, is_3D=(T//4 >= min_T))

        if num_resolution_levels >= 3:
            # define D2
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D2")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[1], patch_sizes[1], module_name="D2")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , [v//2 for v in self.num_wind], self.num_patch, module_name="D2")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[2]
            kwargs["n_head"] = n_heads[2]
            self.D2 = STCNNT_Block(**kwargs)

            if encoder_on_skip_connection:
                self.E2 = _encoder_on_skip_connection(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_out"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)
            else:
                self.E2 = nn.Identity()

            self.down_2 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv, is_3D=(T//8 >= min_T))

        if num_resolution_levels >= 4:
            # define D3
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0] // 8
            kwargs["W"] = c.width[0] // 8

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D3")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="D3")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="D3")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[3]
            kwargs["n_head"] = n_heads[3]
            self.D3 = STCNNT_Block(**kwargs)

            if encoder_on_skip_connection:
                self.E3 = _encoder_on_skip_connection(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_out"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)
            else:
                self.E3 = nn.Identity()

            self.down_3 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv, is_3D=(T//16 >= min_T))

        if num_resolution_levels >= 5:
            # define D4
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 16*self.C
            kwargs["H"] = c.height[0] // 16
            kwargs["W"] = c.width[0] // 16

            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D4")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[3], patch_sizes[3], module_name="D4")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , [v//4 for v in self.num_wind], self.num_patch, module_name="D4")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[4]
            kwargs["n_head"] = n_heads[4]
            self.D4 = STCNNT_Block(**kwargs)

            if encoder_on_skip_connection:
                self.E4 = _encoder_on_skip_connection(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_out"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)
            else:
                self.E4 = nn.Identity()

            self.down_4 = DownSample(N=1, C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv, is_3D=(T//32 >= min_T))

        # -----------------------------------------------------
        # define the bridge
        # -----------------------------------------------------
        kwargs["C_in"] = kwargs["C_out"]
        kwargs["att_types"] = self.block_str[-1]
        kwargs["H"] //= 2
        kwargs["W"] //= 2
        kwargs["n_head"] = n_heads[-1]
        kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[-1], patch_sizes[-1], module_name="bridge")
        self.bridge = STCNNT_Block(**kwargs)

        # -----------------------------------------------------
        if num_resolution_levels >= 5:
            self.up_4 = UpSample(N=1, C_in=16*self.C, C_out=16*self.C, with_conv=self.with_conv, is_3D=self.down_4.is_3D)
            if self.use_unet_attention:
                self.attention_4 = _unet_attention(C_q=16*self.C, C=16*self.C, use_conv_3d=use_conv_3d)

            kwargs["C_in"] = 32*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0] // 16
            kwargs["W"] = c.width[0] // 16
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[4], patch_sizes[4], module_name="U4")
            kwargs["att_types"] = self.block_str[4]
            kwargs["n_head"] = n_heads[4]
            if transformer_for_upsampling:
                self.U4 = STCNNT_Block(**kwargs)
            else:
                self.U4 = _decoder_conv(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_in"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)

        if num_resolution_levels >= 4:
            self.up_3 = UpSample(C_in=8*self.C, C_out=8*self.C, with_conv=self.with_conv, is_3D=self.down_3.is_3D)
            if self.use_unet_attention:
                self.attention_3 = _unet_attention(C_q=8*self.C, C=8*self.C, use_conv_3d=use_conv_3d)

            kwargs["C_in"] = 16*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0] // 8
            kwargs["W"] = c.width[0] // 8
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[3], patch_sizes[3], module_name="U3")
            kwargs["att_types"] = self.block_str[3]
            kwargs["n_head"] = n_heads[3]
            if transformer_for_upsampling:
                self.U3 = STCNNT_Block(**kwargs)
            else:
                self.U3 = _decoder_conv(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_in"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)

        if num_resolution_levels >= 3:
            self.up_2 = UpSample(C_in=4*self.C, C_out=4*self.C, with_conv=self.with_conv, is_3D=self.down_2.is_3D)
            if self.use_unet_attention:
                self.attention_2 = _unet_attention(C_q=4*self.C, C=4*self.C, use_conv_3d=use_conv_3d)

            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[2], patch_sizes[2], module_name="U2")
            kwargs["att_types"] = self.block_str[2]
            kwargs["n_head"] = n_heads[2]
            if transformer_for_upsampling:
                self.U2 = STCNNT_Block(**kwargs)
            else:
                self.U2 = _decoder_conv(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_in"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)

        if num_resolution_levels >= 2:
            self.up_1 = UpSample(C_in=2*self.C, C_out=2*self.C, with_conv=self.with_conv, is_3D=self.down_1.is_3D)
            if self.use_unet_attention:
                self.attention_1 = _unet_attention(C_q=2*self.C, C=2*self.C, use_conv_3d=use_conv_3d)

            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[1], patch_sizes[1], module_name="U1")
            kwargs["att_types"] = self.block_str[1]
            kwargs["n_head"] = n_heads[1]
            if transformer_for_upsampling:
                self.U1 = STCNNT_Block(**kwargs)
            else:
                self.U1 = _decoder_conv(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_in"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)

        if num_resolution_levels >= 1:
            self.up_0 = UpSample(C_in=self.C, C_out=self.C, with_conv=self.with_conv, is_3D=self.down_0.is_3D)
            if self.use_unet_attention:
                self.attention_0 = _unet_attention(C_q=self.C, C=self.C, use_conv_3d=use_conv_3d)

            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="U0")
            kwargs["att_types"] = self.block_str[0]
            kwargs["n_head"] = n_heads[0]
            if transformer_for_upsampling:
                self.U0 = STCNNT_Block(**kwargs)
            else:
                self.U0 = _decoder_conv(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_in"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)

        # -----------------------------------------------------
        if self.use_unet_attention:
            self.attention_w = _unet_attention(C_q=self.C, C=self.C, use_conv_3d=use_conv_3d)

        # -----------------------------------------------------
        kwargs["C_in"] = 2*self.C
        kwargs["C_out"] = 2*self.C
        kwargs["H"] = c.height[0]
        kwargs["W"] = c.width[0]
        kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="UW")
        kwargs["att_types"] = self.block_str[0]
        kwargs["n_head"] = n_heads[0]
        if transformer_for_upsampling:
            self.UW = STCNNT_Block(**kwargs)
        else:
            self.UW = _decoder_conv(H=kwargs["H"], W=kwargs["W"], C_in=kwargs["C_in"], C_out=kwargs["C_out"], norm_mode=c.norm_mode, activation=c.activation_func, bias=False, separable_conv=c.separable_conv, use_conv_3d=use_conv_3d)

        # -----------------------------------------------------

        self.up_w = UpSample(C_in=2*self.C, C_out=2*self.C, with_conv=self.with_conv, is_3D=is_3D_window_partition)

    # -------------------------------------------------------------------------------------------

    def check_class_specific_parameters(self, config):
        if not "backbone_mixed_unetr" in config:
            raise "backbone_mixed_unetr namespace should exist in config"

        err_str = lambda x : f"{x} should exist in config.backbone_mixed_unetr"

        para_list = ["C", "num_resolution_levels", "block_str", "use_unet_attention", "use_interpolation", "with_conv", "min_T", "encoder_on_skip_connection", "transformer_for_upsampling", "n_heads" , "use_conv_3d"]
        for arg_name in para_list:
            if not arg_name in config.backbone_mixed_unetr:
                raise ValueError(err_str(arg_name))

    def _get_gated_output(self, x, x_E, y):
        if self.encoder_on_skip_connection and self.use_unet_attention:
            return self.attention_0(q=y, x=x_E)
        elif not self.encoder_on_skip_connection and self.use_unet_attention:
            return self.attention_0(q=y, x=x)
        elif self.encoder_on_skip_connection and not self.use_unet_attention:
            return x_E
        else: # not self.encoder_on_skip_connection and not self.use_unet_attention:
            return x


    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): the input image, [B, T, Cin, H, W]

        @rets:
            - y_hat (5D torch.Tensor): output tensor, [B, T, Cout, H, W]
        """

        B, T, Cin, H, W = x.shape

        # apply the window partition
        x_w = self.window_partition(x)
        x_w = self.conv_window_partition(x_w)

        if self.encoder_on_skip_connection:
            x_E = self.E(x)
            x_EW = self.EW(x_w)

        # first we go down the resolution 
        if self.num_resolution_levels >= 1:
            x_0, _ = self.D0(x_w)
            if self.encoder_on_skip_connection: x_E0 = self.E0(x_0)
            x_d_0 = self.down_0(x_0)

        if self.num_resolution_levels >= 2:
            x_1, _ = self.D1(x_d_0)
            if self.encoder_on_skip_connection: x_E1 = self.E1(x_1)
            x_d_1 = self.down_1(x_1)

        if self.num_resolution_levels >= 3:
            x_2, _ = self.D2(x_d_1)
            if self.encoder_on_skip_connection: x_E2 = self.E2(x_2)
            x_d_2 = self.down_2(x_2)

        if self.num_resolution_levels >= 4:
            x_3, _ = self.D3(x_d_2)
            if self.encoder_on_skip_connection: x_E3 = self.E3(x_3)
            x_d_3 = self.down_3(x_3)

        if self.num_resolution_levels >= 5:
            x_4, _ = self.D4(x_d_3)
            if self.encoder_on_skip_connection: x_E4 = self.E3(x_4)
            x_d_4 = self.down_4(x_4)

        # now we go up the resolution ...
        if self.num_resolution_levels == 1:
            y_d_0, _ = self.bridge(x_d_0)
            y_0 = self.up_0(y_d_0)
            x_gated_0 = self._get_gated_output(x_0, x_E0, y_0)
            y_w, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        if self.num_resolution_levels == 2:
            y_d_1, _ = self.bridge(x_d_1)
            y_1 = self.up_1(y_d_1)
            x_gated_1 = self._get_gated_output(x_1, x_E1, y_1)
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self._get_gated_output(x_0, x_E0, y_0)
            y_w, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        if self.num_resolution_levels == 3:
            y_d_2, _ = self.bridge(x_d_2)
            y_2 = self.up_2(y_d_2)
            x_gated_2 = self._get_gated_output(x_2, x_E2, y_2)
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self._get_gated_output(x_1, x_E1, y_1)
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self._get_gated_output(x_0, x_E0, y_0)
            y_w, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        if self.num_resolution_levels == 4:
            y_d_3, _ = self.bridge(x_d_3)
            y_3 = self.up_3(y_d_3)
            x_gated_3 = self._get_gated_output(x_3, x_E3, y_3)
            y_d_2, _ = self.U3(torch.cat((x_gated_3, y_3), dim=2))

            y_2 = self.up_2(y_d_2)
            x_gated_2 = self._get_gated_output(x_2, x_E2, y_2)
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self._get_gated_output(x_1, x_E1, y_1)
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self._get_gated_output(x_0, x_E0, y_0)
            y_w, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        if self.num_resolution_levels == 5:
            y_d_4, _ = self.bridge(x_d_4)
            y_4 = self.up_4(y_d_4)
            x_gated_4 = self._get_gated_output(x_4, x_E4, y_4)
            y_d_3, _ = self.U4(torch.cat((x_gated_4, y_4), dim=2))

            y_3 = self.up_3(y_d_3)
            x_gated_3 = self._get_gated_output(x_3, x_E3, y_3)
            y_d_2, _ = self.U3(torch.cat((x_gated_3, y_3), dim=2))

            y_2 = self.up_2(y_d_2)
            x_gated_2 = self._get_gated_output(x_2, x_E2, y_2)
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))

            y_1 = self.up_1(y_d_1)
            x_gated_1 = self._get_gated_output(x_1, x_E1, y_1)
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))

            y_0 = self.up_0(y_d_0)
            x_gated_0 = self._get_gated_output(x_0, x_E0, y_0)
            y_w, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))

        y_gated_w = self._get_gated_output(x_w, x_EW, y_w)
        y_hat = self.UW(y_gated_w)
        y_hat = self.up_w(y_hat)

        if self.encoder_on_skip_connection:
            y_hat = torch.cat((y_hat, x_E), dim=2)

        return y_hat

    def __str__(self):
        return create_generic_class_str(obj=self, exclusion_list=[nn.Module, OrderedDict, STCNNT_Block, DownSample, UpSample, _unet_attention])

# -------------------------------------------------------------------------------------------------

def tests():

    from utils.benchmark import benchmark_all, benchmark_memory, pytorch_profiler
    from utils.setup_training import set_seed
    from colorama import Fore, Style

    device = get_device()

    B,T,C,H,W = 1, 256, 1, 256, 256
    test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)

    parser = add_backbone_STCNNT_args()
    ns = Nestedspace()
    config = parser.parse_args(namespace=ns)

    config.C_in = C
    config.C_out = C
    config.height = [H]
    config.width = [W]
    config.batch_size = B
    config.time = T
    config.norm_mode = "instance2d"
    config.a_type = "conv"

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
    config.device = None
    config.channels = [16,32,64]
    config.all_w_decay = True
    config.optim = "adamw"
    config.scheduler = "StepLR"

    config.complex_i = False

    config.backbone_mixed_unetr.block_str = ["T1L1G1",
                        "T1L1G1",
                        "T1L1G1",
                        "T1L1G1",
                        "T1L1G1"]

    config.backbone_mixed_unetr.C = 32
    config.backbone_mixed_unetr.num_resolution_levels = 4
    config.backbone_mixed_unetr.use_unet_attention = 1
    config.backbone_mixed_unetr.use_interpolation = 1
    config.backbone_mixed_unetr.with_conv = 1
    config.backbone_mixed_unetr.min_T = 16
    config.backbone_mixed_unetr.encoder_on_skip_connection = 1
    config.backbone_mixed_unetr.transformer_for_upsampling = 1
    config.backbone_mixed_unetr.n_heads = [4, 8, 16, 32, 32]
    config.backbone_mixed_unetr.use_conv_3d = 1

    model = STCNNT_mixed_Unetr(config=config)
    model.to(device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(test_in)

    config.with_timer = False
    print(f"{Fore.GREEN}-------------> STCNNT_Unet---einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in, grad=None, min_run_time=5, desc='STCNNT_Unet', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in, desc='STCNNT_Unet', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    config.stride_s = 1
    config.separable_conv = False
    config.use_einsum = False
    
    model = STCNNT_mixed_Unetr(config=config)
    model.to(device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            y = model(test_in)
            
    print(f"{Fore.GREEN}-------------> STCNNT_Unet---einsum-{config.use_einsum}-stride_s-{config.stride_s}-separable_conv-{config.separable_conv} <----------------------{Style.RESET_ALL}")
    benchmark_all(model, test_in.to(device=device), grad=None, min_run_time=5, desc='STCNNT_Unet-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    benchmark_memory(model, test_in.to(device=device), desc='STCNNT_Unet-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)
    
    model = STCNNT_mixed_Unetr(config=config)
    model.to(device=device)
    with torch.no_grad():
        model_summary = model_info(model, config)
    print(f"Configuration for this run:\n{config}")
    print(f"Model Summary:\n{str(model_summary)}")

    print("Passed all tests")

if __name__=="__main__":
    tests()
