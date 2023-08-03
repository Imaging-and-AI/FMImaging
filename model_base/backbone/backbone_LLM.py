"""
Backbone model - LLMs architecture, stack of transformers

This file implements a stack of transformer design for the imaging backbone.
The input to the model is [B, T, C_in, H, W]. The output of the model is [B, T, N*C, H, W].

The model includes a number of stages. Each stage is a block.
C is the model base number of channels. Every stage will increase the number of feature maps.

Please ref to the project page for the network design.
"""

import os
import sys
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from pathlib import Path
from argparse import Namespace

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from losses import *
from imaging_attention import *
from cells import *
from blocks import *
from utils import get_device, model_info, add_backbone_STCNNT_args, Nestedspace

from backbone_base import STCNNT_Base_Runtime, set_window_patch_sizes_keep_num_window, set_window_patch_sizes_keep_window_size

__all__ = ['STCNNT_LLMnet']

# -------------------------------------------------------------------------------------------------
# stcnnt LLMnet

class STCNNT_LLMnet(STCNNT_Base_Runtime):
    """
    This class implemented the stcnnt version of stack of transformers with maximal 5 levels.
    """

    def __init__(self, config, total_steps=1, load=False) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
            - load (bool): whether to try loading from config.load_path or not

        @args (from config):

            ---------------------------------------------------------------
            model specific arguments
            ---------------------------------------------------------------

            - C (int): number of channels, model base number of feature maps
            - num_stages (int): number of stages; each stage is a block

            - block_str (str | list of strings): order of attention types and mixer
                format is list of XYXYXYXY...
                - X is "L", "G" or "T" or "V" for attention type
                - Y is "0" or "1" for with or without mixer
                - requires len(att_types[i]) to be even

                This string is the "Block string" to define the attention layers in a block.
                If a list of string is given, each string defines the attention structure for a stage.

            - add_skip_connections (bool): whether to add skip connections between stages; if True, densenet type connections are added; if False, LLM type network is created.

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

            - losses (list of "ssim", "ssim3D", "l1", "mse"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether we are dealing with complex images or not

            - load_path (str): path to load the weights from
        """
        super().__init__(config)

        self.check_class_specific_parameters(config)
        
        C = config.backbone_LLM.C
        num_stages = config.backbone_LLM.num_stages
        block_str = config.backbone_LLM.block_str
        add_skip_connections = config.backbone_LLM.add_skip_connections

        assert C >= config.C_in, "Number of channels should be larger than C_in"
        assert num_stages <= 5 and num_stages>=2, "Maximal number of stages is 5"

        self.C = C
        self.num_stages = num_stages
        
        if isinstance(block_str, list):
            self.block_str = block_str if len(block_str)>=self.num_stages else [block_str[0] for n in range(self.num_stages)] # with bridge
        else:
            self.block_str = [block_str for n in range(self.num_stages)]
            
        self.add_skip_connections = add_skip_connections

        c = config

        # compute number of windows and patches
        self.num_wind = [c.height[0]//c.window_size[0], c.width[0]//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]

        kwargs = {
            "C_in":c.C_in,
            "C_out":C,
            "H":c.height[0],
            "W":c.width[0],
            "a_type":c.a_type,            
            "window_size": c.window_size,
            "patch_size": c.patch_size,
            "is_causal":c.is_causal,
            "dropout_p":c.dropout_p,
            "n_head":c.n_head,
            
            "kernel_size":(c.kernel_size, c.kernel_size),
            "stride":(c.stride, c.stride),
            "padding":(c.padding, c.padding),

            "stride_s": (c.stride_s, c.stride_s),
            "stride_t":(c.stride_t, c.stride_t),

            "separable_conv": c.separable_conv,

            "mixer_kernel_size":(c.mixer_kernel_size, c.mixer_kernel_size),
            "mixer_stride":(c.mixer_stride, c.mixer_stride),
            "mixer_padding":(c.mixer_padding, c.mixer_padding),
            
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

        if num_stages >= 1:
            # define B0
            kwargs["C_in"] = c.C_in
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="B0")
            kwargs["att_types"] = self.block_str[0]
            self.B0 = STCNNT_Block(**kwargs)

        if num_stages >= 2:
            # define B1
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="B1")
            kwargs["att_types"] = self.block_str[1]
            self.B1 = STCNNT_Block(**kwargs)

        if num_stages >= 3:
            # define B2
            kwargs["C_in"] = 2*self.C if add_skip_connections else self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="B2")
            kwargs["att_types"] = self.block_str[2]
            self.B2 = STCNNT_Block(**kwargs)

        if num_stages >= 4:
            # define B3
            kwargs["C_in"] = 4*self.C if add_skip_connections else 2*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="B3")
            kwargs["att_types"] = self.block_str[3]
            self.B3 = STCNNT_Block(**kwargs)

        if num_stages >= 5:
            # define B4
            kwargs["C_in"] = 8*self.C if add_skip_connections else 4*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="B4")
            kwargs["att_types"] = self.block_str[4]
            self.B4 = STCNNT_Block(**kwargs)
   
    
    def check_class_specific_parameters(self, config):
        if not "backbone_LLM" in config:
            raise "backbone_LLM namespace should exist in config"
               
        err_str = lambda x : f"{x} should exist in config.backbone_LLM"        

        para_list = ["C", "num_stages", "block_str", "add_skip_connections"]        
        for arg_name in para_list:            
            if not arg_name in config.backbone_LLM:
                raise ValueError(err_str(arg_name))
                
    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): the input image, [B, T, Cin, H, W]

        @rets:
            - y_hat (5D torch.Tensor): output feature maps
        """

        B, T, Cin, H, W = x.shape

        if self.add_skip_connections:
            x0, _ = self.B0(x)
            y_hat = x0

            if self.num_stages >= 2:
                x1, _ = self.B1(x0)
                y_hat = x1

            if self.num_stages >= 3:
                x2, _ = self.B2(torch.cat((x0, x1), dim=2))
                y_hat = x2

            if self.num_stages >= 4:
                x3, _ = self.B3(torch.cat((x0, x1, x2), dim=2))
                y_hat = x3

            if self.num_stages >= 5:
                x4, _ = self.B4(torch.cat((x0, x1, x2, x3), dim=2))
                y_hat = x4
        else:
            y_hat, _ = self.B0(x)
            if self.num_stages >= 2:
                y_hat, _ = self.B1(y_hat)
            if self.num_stages >= 3:
                y_hat, _ = self.B2(y_hat)
            if self.num_stages >= 4:
                y_hat, _ = self.B3(y_hat)
            if self.num_stages >= 5:
                y_hat, _ = self.B4(y_hat)

        return y_hat

    def __str__(self):
        return create_generic_class_str(obj=self, exclusion_list=[nn.Module, OrderedDict, STCNNT_Block])

# -------------------------------------------------------------------------------------------------

def tests():

    B,T,C,H,W = 2,4,1,256,256
    test_in = torch.rand(B,T,C,H,W)

    parser = add_backbone_STCNNT_args()
    ns = Nestedspace()
    config = parser.parse_args(namespace=ns)
    
    config.backbone_LLM.C = 16
    config.backbone_LLM.num_stages = 4
    config.backbone_LLM.block_str = 'T1L1G1T1L1G1T1L1G1'
    config.backbone_LLM.add_skip_connections = True

    # attention modules
    config.kernel_size = 3
    config.stride = 1
    config.padding = 1
    config.stride_t = 2
    config.dropout_p = 0.1
    config.C_in = C
    config.C_out = C
    config.height = [H]
    config.width = [W]
    config.batch_size = B
    config.time = T
    config.norm_mode = "instance2d"
    config.a_type = "conv"
    config.is_causal = False
    config.n_head = 8
    config.interp_align_c = True
    
    config.window_size = [H//8, W//8]
    config.patch_size = [H//32, W//32]
    
    config.num_wind =[8, 8]
    config.num_patch =[4, 4]
    
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

    config.cell_type = "sequential"
    config.normalize_Q_K = True 
    config.att_dropout_p = 0.0
    config.att_with_output_proj = True 
    config.scale_ratio_in_mixer  = 1.0
    
    config.cosine_att = True
    config.att_with_relative_postion_bias = True
    
    config.block_dense_connection = True
    
    config.summary_depth = 4

    config.complex_i = False

    device = get_device()

    config.optim = "adamw"
    config.scheduler = "ReduceLROnPlateau"
    config.all_w_decay = True

    model = STCNNT_LLMnet(config=config)
    model.to(device=device)

    print(model)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    test_out = model(test_in.to(device=device))

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    print(f"forward time: {elapsed_time_ms:.3f}ms")
    print(get_gpu_ram_usage(device=device))

    B,T,C,H,W = 2,8,1,128,128
    test_in2 = torch.rand(B,T,C,H,W)
    test_out2 = model(test_in2.to(device=device))

    del model, test_out
    torch.cuda.empty_cache()

    print(get_gpu_ram_usage(device=device))

    print("Passed optimizers and schedulers")

    model = STCNNT_LLMnet(config=config)
    model_summary = model_info(model, config)
    print(f"Configuration for this run:\n{config}")
    print(f"Model Summary:\n{str(model_summary)}")

    print("Passed all tests")

if __name__=="__main__":
    tests()
