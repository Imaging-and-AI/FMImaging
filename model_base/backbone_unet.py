"""
Backbone model - UNet architecture, with attention

This file implements a UNet design for the imaging backbone. The input to the model is [B, T, C_in, H, W]. The output of the model is [B, T, C, H, W].
For every resolution level, the image size will be reduced by x2, with the number of channels increasing by x2.

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

from losses import *
from attention_modules import *
from utils.utils import get_device, model_info

from base_models import STCNNT_Base_Runtime

# -------------------------------------------------------------------------------------------------
# building blocks

class _D2(nn.Module):
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
            y = F.interpolate(x.view((B*T, C, H, W)), scale_factor=(0.5, 0.5), mode="bilinear", align_corners=False, recompute_scale_factor=False)
            y = torch.reshape(y, (B, T, C, H//2, W//2))
            if self.conv:
                y = self.conv(y)
        else:
            y = self.stride_conv(x)
        
        return y
          
# -------------------------------------------------------------------------------------------------
   
class _U2(nn.Module):
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
        y = F.interpolate(x.view((B*T, C, H, W)), size=(2*H, 2*W), mode="bilinear", align_corners=False, recompute_scale_factor=False)
        y = torch.reshape(y, (B, T, C, 2*H, 2*W))
        if self.with_conv:
            y = self.conv(y)
        
        return y

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
                
                This string is the "Block string" to define the attention layers in a block. 
                If a list of string is given, each string defines the attention structure for a resolution level. The last string is the bridge structure.
                      
            - use_interpolation (bool): whether to use interpolation in downsample layer; if False, use stride convolution
            - with_conv (bool): whether to add conv in down/upsample layers; if False, only interpolation is performed
                                       
            ---------------------------------------------------------------    
            Shared arguments used in this model
            ---------------------------------------------------------------
            - C_in (int): number of input channels
            - C_out (int): number of output channels
                
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
        with_conv = config.with_conv
        
        assert C >= config.C_in, "Number of channels should be larger than C_in"
        assert num_resolution_levels <= 5 and num_resolution_levels>=1, "Maximal number of resolution levels is 5"

        self.C = C
        self.num_resolution_levels = num_resolution_levels
        self.block_str = block_str if isinstance(block_str, list) else [block_str for n in range(self.num_resolution_levels+1)] # with bridge
        self.use_interpolation = use_interpolation
        self.with_conv = with_conv
        
        c = config
        kwargs = {            
            "C_in":c.C_in, 
            "C_out":c.C,
            "H":c.height[0], 
            "W":c.width[0], 
            "a_type":c.a_type,
            "window_size": c.window_size, 
            "is_causal":c.is_causal, 
            "dropout_p":c.dropout_p,
            "n_head":c.n_head, 
            "kernel_size":(c.kernel_size, c.kernel_size),
            "stride":(c.stride, c.stride), 
            "padding":(c.padding, c.padding),
            "norm_mode":c.norm_mode,
            "interpolate":"none", 
            "interp_align_c":c.interp_align_c
        }

        if num_resolution_levels >= 1:
            # define D0
            kwargs["C_in"] = c.C_in
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["att_types"] = self.block_str[0]
            self.D0 = STCNNT_Block(**kwargs)
            
            self.down_0 = _D2(C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)
                
        if num_resolution_levels >= 2:
            # define D1
            kwargs["C_in"] = self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs["att_types"] = self.block_str[1]
            self.D1 = STCNNT_Block(**kwargs)
            
            self.down_1 = _D2(C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)
            
        if num_resolution_levels >= 3:
            # define D2
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4
            kwargs["att_types"] = self.block_str[2]
            self.D2 = STCNNT_Block(**kwargs)
            
            self.down_2 = _D2(C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)
            
        if num_resolution_levels >= 4:
            # define D3
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0] // 8
            kwargs["W"] = c.width[0] // 8
            kwargs["att_types"] = self.block_str[3]
            self.D3 = STCNNT_Block(**kwargs)
            
            self.down_3 = _D2(C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)
            
        if num_resolution_levels >= 5:
            # define D4
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 16*self.C
            kwargs["H"] = c.height[0] // 16
            kwargs["W"] = c.width[0] // 16
            kwargs["att_types"] = self.block_str[4]
            self.D4 = STCNNT_Block(**kwargs)
            
            self.down_4 = _D2(C_in=kwargs["C_out"], C_out=kwargs["C_out"], use_interpolation=self.use_interpolation, with_conv=self.with_conv)
            
        # define the bridge
        kwargs["C_in"] = kwargs["C_out"]
        kwargs["att_types"] = self.block_str[-1]
        self.bridge = STCNNT_Block(**kwargs)
        
        if num_resolution_levels >= 5:
            self.up_4 = _U2(C_in=16*self.C, C_out=16*self.C, with_conv=self.with_conv)
            self.attention_4 = _unet_attention(C_q=16*self.C, C=16*self.C)
            
            kwargs["C_in"] = 32*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0] // 16
            kwargs["W"] = c.width[0] // 16
            kwargs["att_types"] = self.block_str[4]
            self.U4 = STCNNT_Block(**kwargs)
            
        if num_resolution_levels >= 4:
            self.up_3 = _U2(C_in=8*self.C, C_out=8*self.C, with_conv=self.with_conv)
            self.attention_3 = _unet_attention(C_q=8*self.C, C=8*self.C)
            
            kwargs["C_in"] = 16*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0] // 8
            kwargs["W"] = c.width[0] // 8
            kwargs["att_types"] = self.block_str[3]
            self.U3 = STCNNT_Block(**kwargs)    
                
        if num_resolution_levels >= 3:
            self.up_2 = _U2(C_in=4*self.C, C_out=4*self.C, with_conv=self.with_conv)
            self.attention_2 = _unet_attention(C_q=4*self.C, C=4*self.C)
            
            kwargs["C_in"] = 8*self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4
            kwargs["att_types"] = self.block_str[2]
            self.U2 = STCNNT_Block(**kwargs) 
                     
        if num_resolution_levels >= 2:
            self.up_1 = _U2(C_in=2*self.C, C_out=2*self.C, with_conv=self.with_conv)
            self.attention_1 = _unet_attention(C_q=2*self.C, C=2*self.C)
            
            kwargs["C_in"] = 4*self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs["att_types"] = self.block_str[1]
            self.U1 = STCNNT_Block(**kwargs) 
            
        if num_resolution_levels >= 1:
            self.up_0 = _U2(C_in=self.C, C_out=self.C, with_conv=self.with_conv)
            self.attention_0 = _unet_attention(C_q=self.C, C=self.C)
            
            kwargs["C_in"] = 2*self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["att_types"] = self.block_str[0]
            self.U0 = STCNNT_Block(**kwargs) 
                                                    
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
            - y_hat (5D torch.Tensor): output tensor, [B, T, Cout, H, W]
        """

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
            x_gated_0 = self.attention_0(q=y_0, x=x_0)
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))
            
        if self.num_resolution_levels == 2:
            y_d_1, _ = self.bridge(x_d_1)
            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1)
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))
                             
            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0)
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))
            
        if self.num_resolution_levels == 3:
            y_d_2, _ = self.bridge(x_d_2)
            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2)
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))
            
            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1)
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))
                             
            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0)
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))
            
        if self.num_resolution_levels == 4:
            y_d_3, _ = self.bridge(x_d_3)
            y_3 = self.up_3(y_d_3)
            x_gated_3 = self.attention_3(q=y_3, x=x_3)
            y_d_2, _ = self.U3(torch.cat((x_gated_3, y_3), dim=2))
            
            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2)
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))
            
            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1)
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))
                             
            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0)
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))
            
        if self.num_resolution_levels == 5:
            y_d_4, _ = self.bridge(x_d_4)
            y_4 = self.up_4(y_d_4)
            x_gated_4 = self.attention_4(q=y_4, x=x_4)
            y_d_3, _ = self.U4(torch.cat((x_gated_4, y_4), dim=2))
            
            y_3 = self.up_3(y_d_3)
            x_gated_3 = self.attention_3(q=y_3, x=x_3)
            y_d_2, _ = self.U3(torch.cat((x_gated_3, y_3), dim=2))
            
            y_2 = self.up_2(y_d_2)
            x_gated_2 = self.attention_2(q=y_2, x=x_2)
            y_d_1, _ = self.U2(torch.cat((x_gated_2, y_2), dim=2))
            
            y_1 = self.up_1(y_d_1)
            x_gated_1 = self.attention_1(q=y_1, x=x_1)
            y_d_0, _ = self.U1(torch.cat((x_gated_1, y_1), dim=2))
                             
            y_0 = self.up_0(y_d_0)
            x_gated_0 = self.attention_0(q=y_0, x=x_0)
            y_hat, _ = self.U0(torch.cat((x_gated_0, y_0), dim=2))
            
        return y_hat

    def __str__(self):
        res = create_generic_class_str(obj=self, exclusion_list=[nn.Module, OrderedDict, STCNNT_Block, _D2, _U2, _unet_attention])
        return res
    
# -------------------------------------------------------------------------------------------------

def tests():

    B,T,C,H,W = 8, 8, 1, 256, 256
    test_in = torch.rand(B,T,C,H,W, dtype=torch.float32)

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
    config.window_size = 16
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

    config.summary_depth = 4

    config.C = 16
    config.num_resolution_levels = 4
    config.block_str = ["T1L1G1", 
                        "T1L1G1T1L1G1", 
                        "T1L1G1T1L1G1T1L1G1", 
                        "T1L1G1T1L1G1T1L1G1T1L1G1", 
                        "T1L1G1T1L1G1T1L1G1T1L1G1"]
    
    config.use_interpolation = True
    config.with_conv = True

    config.optim = "adamw"    
    config.scheduler = "ReduceLROnPlateau"
    config.all_w_decay = True
    device = get_device()
    
    model = STCNNT_Unet(config=config)
    model.to(device=device)
            
    print(model)
                    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    test_out = model(test_in.to(device=device))
    loss = model.loss_f(test_out, 2*test_out)
    
    loss.backward()
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    
    print(f"forward time: {elapsed_time_ms:.3f}ms")
    get_gpu_ram_usage(device=device)

    del model, test_out
    torch.cuda.empty_cache()
    
    get_gpu_ram_usage(device=device)

    model = STCNNT_Unet(config=config)
    model.to(device=device)
    with torch.no_grad():
        model_summary = model_info(model, config)
    print(f"Configuration for this run:\n{config}")
    print(f"Model Summary:\n{str(model_summary)}")
    
    print("Passed all tests")


if __name__=="__main__":
    tests()
