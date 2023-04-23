"""
Backbone model - HRNet architecture

This file implements a HRNet design for the imaging backbone. The input to the model is [B, T, C_in, H, W]. The output of the model is [B, T, N*C, H, W].
N is the number of resolution levels. C is the number of channels at the original resolution. For every resolution level, the image size will be reduced by x2, with the number of channels increasing by x2.

Besides the aggregated output tensor, this backbone model also outputs the per-resolution-level feature maps as a list.

Please ref to the project page for the network design.

"""

import os
import sys
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from argparse import Namespace

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from losses import *
from attention_modules import *
from utils.utils import get_device, model_info

from base_models import STCNNT_Base_Runtime

# -------------------------------------------------------------------------------------------------
# building blocks

class _D2(torch.Module):
    """
    Downsample by 2 layer

    This module takes in a [B, T, C, H, W] tensor and downsample it to [B, T, C, H//2, W//2]
    
    By default, the operation is performed with a bilinear interpolation. If with_conv is True, a 1x1 convolution is added after interpolation.
    If with_interpolation is False, the stride convolution is used.
    """
    
    def __init__(self, C_in=16, C_out=-1, use_interpolation=True, with_conv=True) -> None:
        super().__init__()
        
        self.C_in = C_in
        self.C_out = C_out if C_out>0 else C_in
        
        self.use_interpolation = use_interpolation
        self.with_conv = with_conv

        self.stride_conv = None
        self.conv = None
        
        if not self.use_interpolation:
            self.stride_conv = Conv2DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        elif self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv2DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[1,1], stride=[1,1], padding=[0,0])
            
        
    def forward(self, x:Tensor) -> Tensor:        
        B, T, C, H, W = x.shape
        if self.use_interpolation:
            y = F.interpolate(x, scale_factor=(0.5, 0.5), mode="bilinear", align_corners=False, recompute_scale_factor=False)
            if self.conv:
                y = self.conv(y)
        else:
            y = self.stride_conv(x)
        
        return y
    

class _DownSample(torch.Module):
    """
    Downsample by x4, by using two D2 layers
    """
    
    def __init__(self, N=2, C_in=16, C_out=-1, use_interpolation=True, with_conv=True) -> None:
        super().__init__()
        
        C_out = C_out if C_out>0 else C_in
        
        layers = [('D2_0', _D2(C_in=C_in, C_out=C_out, use_interpolation=use_interpolation, with_conv=with_conv))]
        for n in range(1, N):
            layers.append( (f'D2_{n}', _D2(C_in=C_out, C_out=C_out, use_interpolation=use_interpolation, with_conv=with_conv)) )
                    
        self.block = nn.Sequential(layers)
        
    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)
       
# -------------------------------------------------------------------------------------------------
   
class _U2(torch.Module):
    """
    Upsample by 2

    This module takes in a [B, T, Cin, H, W] tensor and upsample it to [B, T, Cout, 2*H, 2*W]
    
    By default, the operation is performed with a bilinear interpolation. If with_conv is True, a 1x1 convolution is added after interpolation.
    """
    
    def __init__(self, C_in=16, C_out=-1, with_conv=True) -> None:
        super().__init__()
        
        self.C_in = C_in
        self.C_out = C_out if C_out>0 else C_in
        
        self.with_conv = with_conv

        self.conv = None        
        if self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv2DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[1,1], stride=[1,1], padding=[0,0])
            
        
    def forward(self, x:Tensor) -> Tensor:        
        B, T, C, H, W = x.shape
        y = F.interpolate(x, size=(2*H, 2*W), mode="bilinear", align_corners=False, recompute_scale_factor=False)
        if self.with_conv:
            y = self.conv(y)
        
        return y
    

class _UpSample(torch.Module):
    """
    Upsample by N times
    """
    
    def __init__(self, N=2, C_in=16, C_out=-1, with_conv=True) -> None:
        super().__init__()
        
        C_out = C_out if C_out>0 else C_in
        
        layers = [('U2_0', _U2(C_in=C_in, C_out=C_out, with_conv=with_conv))]
        for n in range(1, N):
            layers.append( (f'U2_{n}', _U2(C_in=C_out, C_out=C_out, with_conv=with_conv)) )
                    
        self.block = nn.Sequential(layers)
        
    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)
    
# -------------------------------------------------------------------------------------------------
# stcnnt hrnet

class STCNNT_HRnet(STCNNT_Base_Runtime):
    """
    This class implemented the stcnnt version of HRnet with maximal 5 levels.
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
            
            - C (int): number of channels, when resolution is reduced by x2, number of channels will increase by x2
            - num_resolution_levels (int): number of resolution levels; each deeper level will reduce spatial size by x2
            
            - block_str (a str or a list of strings): order of attention types and mixer
                format is list of XYXYXYXY...
                - X is "L", "G" or "T" for attention type
                - Y is "0" or "1" for with or without mixer
                - only first one is used for this model to create consistent blocks
                - requires len(att_types[0]) to be even
                
                This string is the "Block string" to define the attention layers in a block. If a list of string is given,  each string defines the attention structure for a resolution level.
                      
            - use_interpolation (bool): whether to use interpolation in downsample layer; if False, use stride convolution
                                       
            ---------------------------------------------------------------    
            Shared arguments used in this model
            ---------------------------------------------------------------
            - C_in (int): number of input channels
                
            - height (int list): expected heights of the input
            - width (int list): expected widths of the input                    
            
            - a_type ("conv", "lin"):
                type of attention in spatial heads
            - window_size (int): size of window for local and global att
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - dropout (float): probability of dropout
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

        C = config.C
        num_resolution_levels = config.num_resolution_levels
        block_str = config.block_str
        use_interpolation = config.use_interpolation
        
        assert(C >= config.C_in, "Number of channels should be larger than C_in")
        assert(num_resolution_levels <= 5 and num_resolution_levels>=2, "Maximal number of resolution levels is 5")

        self.C = C
        self.num_resolution_levels = num_resolution_levels
        self.block_str = block_str if isinstance(block_str, list) else [block_str for n in range(self.num_resolution_levels)]
        
        c = config
        kwargs = {            
            "att_types":c.att_types[0],             
            "C_in":c.C_in, 
            "C_out":c.channels[0],\
            "H":c.height[0], 
            "W":c.width[0], 
            "a_type":c.a_type,\
            "is_causal":c.is_causal, 
            "dropout_p":c.dropout_p,\
            "n_head":c.n_head, 
            "kernel_size":(c.kernel_size, c.kernel_size),\
            "stride":(c.stride, c.stride), 
            "padding":(c.padding, c.padding),\
            "norm_mode":c.norm_mode,\
            "interpolate":"none", 
            "interp_align_c":c.interp_align_c
        }

        if num_resolution_levels >= 1:
            # define B00
            kwargs["C_in"] = c.C_in
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["a_type"] = self.block_str[0]
            self.B00 = STCNNT_Block(**kwargs)
            
            # output stage 0
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["a_type"] = self.block_str[0]
            self.output_B0 = STCNNT_Block(**kwargs)
            
        if num_resolution_levels >= 2:
            # define B01
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["a_type"] = self.block_str[0]
            self.B01 = STCNNT_Block(**kwargs)
            
            # define B11
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs["a_type"] = self.block_str[1]
            self.B11 = STCNNT_Block(**kwargs)
        
            # define down sample
            self.down_00_11 = _DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
            
            # define output B1
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs["a_type"] = self.block_str[1]
            self.output_B1 = STCNNT_Block(**kwargs)
            
        if num_resolution_levels >= 3:
            # define B02
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["a_type"] = self.block_str[0]
            self.B02 = STCNNT_Block(**kwargs)
            
            # define B12
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs["a_type"] = self.block_str[1]
            self.B12 = STCNNT_Block(**kwargs)
        
            # define B22
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4
            kwargs["a_type"] = self.block_str[2]
            self.B22 = STCNNT_Block(**kwargs)
            
            # define down sample
            self.down_01_12 = _DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_01_22 = _DownSample(N=2, C_in=self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_11_22 = _DownSample(N=1, C_in=2*self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
        
            # define output B2
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4
            kwargs["a_type"] = self.block_str[2]
            self.output_B2 = STCNNT_Block(**kwargs)
        
        if num_resolution_levels >= 4:
            # define B03
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["a_type"] = self.block_str[0]
            self.B02 = STCNNT_Block(**kwargs)
            
            # define B13
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs["a_type"] = self.block_str[1]
            self.B12 = STCNNT_Block(**kwargs)
        
            # define B23
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4
            kwargs["a_type"] = self.block_str[2]
            self.B22 = STCNNT_Block(**kwargs)
            
            # define B33
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0] // 8
            kwargs["W"] = c.width[0] // 8
            kwargs["a_type"] = self.block_str[3]
            self.B33 = STCNNT_Block(**kwargs)
            
            # define down sample
            self.down_02_13 = _DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_02_23 = _DownSample(N=2, C_in=self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_02_33 = _DownSample(N=3, C_in=self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_12_23 = _DownSample(N=1, C_in=2*self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_12_33 = _DownSample(N=2, C_in=2*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_22_33 = _DownSample(N=1, C_in=4*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            
            # define output B3
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0] // 8
            kwargs["W"] = c.width[0] // 8
            kwargs["a_type"] = self.block_str[3]
            self.output_B3 = STCNNT_Block(**kwargs)
            
        if num_resolution_levels >= 5:
            # define B04
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["a_type"] = self.block_str[0]
            self.B04 = STCNNT_Block(**kwargs)
            
            # define B14
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs["a_type"] = self.block_str[1]
            self.B14 = STCNNT_Block(**kwargs)
        
            # define B24
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4
            kwargs["a_type"] = self.block_str[2]
            self.B24 = STCNNT_Block(**kwargs)
            
            # define B34
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0] // 8
            kwargs["W"] = c.width[0] // 8
            kwargs["a_type"] = self.block_str[3]
            self.B34 = STCNNT_Block(**kwargs)
            
            # define B44
            kwargs["C_in"] = 16*self.C
            kwargs["C_out"] = 16*self.C
            kwargs["H"] = c.height[0] // 16
            kwargs["W"] = c.width[0] // 16
            kwargs["a_type"] = self.block_str[4]
            self.B44 = STCNNT_Block(**kwargs)
            
            # define down sample
            self.down_03_14 = _DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_03_24 = _DownSample(N=2, C_in=self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_03_34 = _DownSample(N=3, C_in=self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_03_44 = _DownSample(N=4, C_in=self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)
            
            self.down_13_24 = _DownSample(N=1, C_in=2*self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_13_34 = _DownSample(N=2, C_in=2*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_13_44 = _DownSample(N=3, C_in=2*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)
            
            self.down_23_34 = _DownSample(N=1, C_in=4*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
            self.down_23_44 = _DownSample(N=2, C_in=4*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)
            
            self.down_33_44 = _DownSample(N=1, C_in=8*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)
    
            # define output B4
            kwargs["C_in"] = 16*self.C
            kwargs["C_out"] = 16*self.C
            kwargs["H"] = c.height[0] // 16
            kwargs["W"] = c.width[0] // 16
            kwargs["a_type"] = self.block_str[4]
            self.output_B4 = STCNNT_Block(**kwargs)
            
        # fusion stage
        self.down_0_1 = _DownSample(N=1, C_in=self.C, C_out=2*self.C, use_interpolation=use_interpolation, with_conv=True)
        self.down_0_2 = _DownSample(N=2, C_in=self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
        self.down_0_3 = _DownSample(N=3, C_in=self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
        self.down_0_4 = _DownSample(N=4, C_in=self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)

        self.down_1_2 = _DownSample(N=1, C_in=2*self.C, C_out=4*self.C, use_interpolation=use_interpolation, with_conv=True)
        self.down_1_3 = _DownSample(N=2, C_in=2*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
        self.down_1_4 = _DownSample(N=3, C_in=2*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)
        
        self.down_2_3 = _DownSample(N=1, C_in=4*self.C, C_out=8*self.C, use_interpolation=use_interpolation, with_conv=True)
        self.down_2_4 = _DownSample(N=2, C_in=4*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)
        
        self.down_3_4 = _DownSample(N=1, C_in=8*self.C, C_out=16*self.C, use_interpolation=use_interpolation, with_conv=True)
        

        self.up_1_0 = _UpSample(N=1, C_in=2*self.C, C_out=self.C, with_conv=True)
        self.up_2_0 = _UpSample(N=2, C_in=4*self.C, C_out=self.C, with_conv=True)
        self.up_3_0 = _UpSample(N=3, C_in=8*self.C, C_out=self.C, with_conv=True)
        self.up_4_0 = _UpSample(N=4, C_in=16*self.C, C_out=self.C, with_conv=True)
        
        self.up_2_1 = _UpSample(N=1, C_in=4*self.C, C_out=2*self.C, with_conv=True)
        self.up_3_1 = _UpSample(N=2, C_in=8*self.C, C_out=2*self.C, with_conv=True)
        self.up_4_1 = _UpSample(N=3, C_in=16*self.C, C_out=2*self.C, with_conv=True)
        
        self.up_3_2 = _UpSample(N=1, C_in=8*self.C, C_out=4*self.C, with_conv=True)
        self.up_4_2 = _UpSample(N=2, C_in=16*self.C, C_out=4*self.C, with_conv=True)
        
        self.up_4_3 = _UpSample(N=1, C_in=16*self.C, C_out=8*self.C, with_conv=True)
                
        self.up_1 = _UpSample(N=1, C_in=2*self.C, C_out=self.C, with_conv=True)
        self.up_2 = _UpSample(N=2, C_in=4*self.C, C_out=self.C, with_conv=True)
        self.up_3 = _UpSample(N=3, C_in=8*self.C, C_out=self.C, with_conv=True)
        self.up_4 = _UpSample(N=4, C_in=16*self.C, C_out=self.C, with_conv=True)
        
        # set up remaining stuff
        device = get_device(device=c.device)
        self.set_up_loss(device=device)
        self.set_up_optim_and_scheduling(total_steps=total_steps)

        if load and c.load_path is not None:
            self.load(device=device)
    
    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): the input image, [B, T, Cin, H, W]
            
        @@rets:       
            - y_hat (5D torch.Tensor): aggregated output tensor
            - y_level_outputs (Tuple): tuple of tensor for every resolution level
        """

        B, T, Cin, H, W = x.shape
        
        y_hat = None
        y_level_outputs = None
        
        # compute the block outputs
        if self.num_resolution_levels >= 1:
            x_00 = self.B00(x)
            
        if self.num_resolution_levels >= 2:
            x_01 = self.B01(x_00)        
            x_11 = self.B11(self.down_00_11(x_01))
                        
        if self.num_resolution_levels >= 3:
            x_02 = self.B02(x_01)                       
            x_12 = self.B12(x_11 + self.down_01_12(x_01))
            x_22 = self.B22(self.down_11_22(x_11) + self.down_01_22(x_01))
                        
        if self.num_resolution_levels >= 4:
            x_03 = self.B03(x_02)            
            x_13 = self.B13(x_12 + self.down_02_13(x_02))            
            x_23 = self.B23(x_22 + self.down_12_23(x_12) + self.down_02_23(x_02))
            x_33 = self.B33(self.down_22_33(x_22) 
                            + self.down_12_33(x_12) 
                            + self.down_02_33(x_02)
                            )
            
        if self.num_resolution_levels >= 5:
            x_04 = self.B04(x_03)
            x_14 = self.B14(x_13 + self.down_03_14(x_03))            
            x_24 = self.B24(x_23 + self.down_13_24(x_13) + self.down_03_24(x_03))
            
            x_34 = self.B34(x_33 
                            + self.down_23_34(x_23) 
                            + self.down_13_34(x_13) 
                            + self.down_03_34(x_03)
                            )
            
            x_44 = self.B44(self.down_33_44(x_33) 
                            + self.down_23_44(x_23) 
                            + self.down_13_44(x_13) 
                            + self.down_03_44(x_03)
                            )
        
        if self.num_resolution_levels == 1:
            y_hat_0 = self.output_B0(x_00)
            y_hat = y_hat_0
            y_level_outputs = (y_hat_0, )
            
        if self.num_resolution_levels == 2:           
            y_hat_0 = x_01 + self.up_1_0(x_11)
            y_hat_1 = x_11 + self.down_0_1(x_01)
            
            y_hat_0 = self.output_B0(y_hat_0)
            y_hat_1 = self.output_B1(y_hat_1)
            
            y_hat = torch.cat((y_hat_0, self.up_1(y_hat_1)), dim=2)
            
            y_level_outputs = (y_hat_0, y_hat_1)
            
        if self.num_resolution_levels == 3:           
            y_hat_0 = x_02 + self.up_1_0(x_12) + self.up_2_0(x_22)
            y_hat_1 = self.down_0_1(x_02) + x_12 + self.up_2_1(x_22)
            y_hat_2 = self.down_0_2(x_02) + self.down_1_2(x_12) + x_22
            
            y_hat_0 = self.output_B0(y_hat_0)
            y_hat_1 = self.output_B1(y_hat_1)
            y_hat_2 = self.output_B2(y_hat_2)
            
            y_hat = torch.cat((y_hat_0, self.up_1(y_hat_1), self.up_2(y_hat_2)), dim=2)
            y_level_outputs = (y_hat_0, y_hat_1, y_hat_2)
            
        if self.num_resolution_levels == 4:            
            y_hat_0 = x_03 + self.up_1_0(x_13) + self.up_2_0(x_23) + self.up_3_0(x_33)
            y_hat_1 = self.down_0_1(x_03) + x_13 + self.up_2_1(x_23) + self.up_3_1(x_33)
            y_hat_2 = self.down_0_2(x_03) + self.down_1_2(x_13) + x_23 + self.up_3_2(x_33)
            y_hat_3 = self.down_0_3(x_03) + self.down_1_3(x_13) + self.down_2_3(x_23) + x_33
            
            y_hat_0 = self.output_B0(y_hat_0)
            y_hat_1 = self.output_B1(y_hat_1)
            y_hat_2 = self.output_B2(y_hat_2)
            y_hat_3 = self.output_B3(y_hat_3)
            
            y_hat = torch.cat(
                (
                    y_hat_0, 
                    self.up_1(y_hat_1), 
                    self.up_2(y_hat_2), 
                    self.up_3(y_hat_3)
                 ), dim=2)
            
            y_level_outputs = (y_hat_0, y_hat_1, y_hat_2, y_hat_3)
                  
        if self.num_resolution_levels == 5:            
            y_hat_0 =               x_04    + self.up_1_0(x_14)         + self.up_2_0(x_24)         + self.up_3_0(x_34)         + self.up_4_0(x_44)
            y_hat_1 = self.down_0_1(x_04)   +               x_14        + self.up_2_1(x_24)         + self.up_3_1(x_34)         + self.up_4_1(x_44)
            y_hat_2 = self.down_0_2(x_04)   + self.down_1_2(x_14)       +               x_24        + self.up_3_2(x_34)         + self.up_4_2(x_44)
            y_hat_3 = self.down_0_3(x_04)   + self.down_1_3(x_14)       + self.down_2_3(x_24)       +             x_34          + self.up_4_3(x_44) 
            y_hat_4 = self.down_0_4(x_04)   + self.down_1_4(x_14)       + self.down_2_4(x_24)       + self.down_3_4(x_34)       +             x_44
                        
            y_hat_0 = self.output_B0(y_hat_0)
            y_hat_1 = self.output_B1(y_hat_1)
            y_hat_2 = self.output_B2(y_hat_2)
            y_hat_3 = self.output_B3(y_hat_3)
            y_hat_4 = self.output_B4(y_hat_4)
            
            y_hat = torch.cat(
                (
                    y_hat_0, 
                    self.up_1(y_hat_1), 
                    self.up_2(y_hat_2), 
                    self.up_3(y_hat_3),
                    self.up_4(y_hat_4)
                 ), dim=2)
            
            y_level_outputs = (y_hat_0, y_hat_1, y_hat_2, y_hat_3, y_hat_4)
                  
        return y_hat, y_level_outputs

# -------------------------------------------------------------------------------------------------

def tests():

    B,T,C,H,W = 32, 12, 1, 512, 512
    test_in = torch.rand(B,T,C,H,W)

    config = Namespace()
    # optimizer and scheduler
    config.weight_decay = 0.1
    config.global_lr = 0.001
    config.beta1 = 0.9
    config.beta2 = 0.99
    # attention modules
    config.kernel_size = 3
    config.stride = 1
    config.padding = 1
    config.dropout_p = 0.1
    config.C_in = C
    config.C_out = C
    config.height = [H]
    config.width = [W]
    config.batch_size = B
    config.time = T
    config.norm_mode = "instance3d"
    config.a_type = "conv"
    config.is_causal = False
    config.n_head = 8
    config.interp_align_c = True
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

    config.C = 16
    config.num_resolution_levels = 5
    config.block_str = ['', '', '', '', '']
    config.use_interpolation = True

    optims = ["adamw", "sgd", "nadam"]
    schedulers = ["StepLR", "OneCycleLR", "ReduceLROnPlateau"]
    all_w_decays = [True, False]

    for optim in optims:
        for scheduler in schedulers:
            for all_w_decay in all_w_decays:
                config.optim = optim    
                config.scheduler = scheduler
                config.all_w_decay = all_w_decay

                model = STCNNT_HRnet(config=config)
                test_out = model(test_in)
                loss = model.computer_loss(test_out, test_in)

                print(loss)

    print("Passed optimizers and schedulers")

    heads_and_channelss = [(8,[16,32,64]),(5,[5,50,15]),(13,[13,13,13])]
    residuals = [True, False]

    for n_head, channels in heads_and_channelss:
        for residual in residuals:
            config.n_head = n_head
            config.channels = channels
            config. residual = residual

            model = STCNNT_HRnet(config=config)
            test_out = model(test_in)
            loss = model.computer_loss(test_out, test_in)

            print(loss)

    print("Passed channels and residual")

    devices = ["cuda", "cpu", "cuda:0"]

    for device in devices:
        config.device = device

        model = STCNNT_HRnet(config=config)
        test_out = model(test_in)
        loss = model.computer_loss(test_out, test_in)

        print(loss)

    print("Passed devices")

    model_summary = model_info(model, config)
    print(f"Configuration for this run:\n{config}")
    print(f"Model Summary:\n{str(model_summary)}")
    
    print("Passed all tests")


if __name__=="__main__":
    tests()
