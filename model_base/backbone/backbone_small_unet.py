"""
Main base models of STCNNT
Provides implementation for the following:
    - CNNT_Unet:
        - the original CNNT_Unet
        - 2 down, 2 up and then a final layer
        - original was made of 4 temporal cells per block
        - now can use spatial cells as well
"""

import os
import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim

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
from backbone import *
from utils.utils import get_device, create_generic_class_str

__all__ = ['CNNT_Unet']

# -------------------------------------------------------------------------------------------------
# CNNT unet

class CNNT_Unet(STCNNT_Base_Runtime):
    """
    CNNT Unet implementation with 2 downsample and 2 upsample layers
    Concatenates the outputs with skip connections
    Final layer does not interpolate
    Instead uses output projection to get desired output channels
    """
    def __init__(self, config, total_steps=1, load=False) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
            - load (bool): whether to try loading from config.load_path or not
        @args (from config):
            - channels (int list): number of channels in each of the 3 layers
            - att_types (str list): order of attention types and their following mlps
                format is list of XYXYXYXY...
                - X is "L", "G" or "T" for attention type
                - Y is "0" or "1" for with or without mixer
                - only first one is used for this model to create consistent blocks
                - requires len(att_types[0]) to be even
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - height (int list): expected heights of the input
            - width (int list): expected widths of the input
            - a_type ("conv", "lin"):
                type of attention in spatial heads
            - window_size (int): size of window for local and global att
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int): convolution parameters
            - stride_t (int): special stride for temporal attention k,q matrices
            - dropout (float): probability of dropout
            - norm_mode ("layer", "batch", "instance"):
                layer - norm along C, H, W; batch - norm along B*T; or instance
            - interp_align_c (bool):
                whether to align corner or not when interpolating
            - residual (bool):
                whether to add long skip residual connection or not
            - losses (list of "ssim", "ssim3D", "l1", "mse"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether we are dealing with complex images or not
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

        c = config # shortening due to numerous uses

        for h in c.height: assert not h % 8, f"height {h} should be divisible by 8"
        for w in c.width: assert not w % 8, f"width {w} should be divisible by 8"
        assert len(c.channels) == 3, f"Requires exactly 3 channel numbers"

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
            "stride_t":(c.stride_t, c.stride_t), 
            "norm_mode":c.norm_mode,\
            "interpolate":"down", 
            "interp_align_c":c.interp_align_c,
            "cell_type": c.cell_type,
            "normalize_Q_K": c.normalize_Q_K, 
            "att_dropout_p": c.att_dropout_p,
            "att_with_output_proj": c.att_with_output_proj, 
            "scale_ratio_in_mixer": c.scale_ratio_in_mixer,
            "window_size": c.window_size,
            "patch_size": c.patch_size
        }

        window_sizes = []
        patch_sizes = []
        
        self.num_windows_h = c.height[0]//c.window_size
        self.num_windows_w = c.width[0]//c.window_size
        self.num_patch = c.window_size//c.patch_size
        
        kwargs = self.set_window_patch_sizes_keep_num_window(kwargs, kwargs["H"], self.num_windows_h, self.num_patch, module_name="D1")
        window_sizes.append(kwargs["window_size"])
        patch_sizes.append(kwargs["patch_size"])
        
        self.down1 = STCNNT_Block(**kwargs)

        kwargs["C_in"] = c.channels[0]
        kwargs["C_out"] = c.channels[1]
        kwargs["H"] = c.height[0]//2
        kwargs["W"] = c.width[0]//2
        kwargs = self.set_window_patch_sizes_keep_window_size(kwargs, kwargs["H"] , window_sizes[0], patch_sizes[0], module_name="D2")
        window_sizes.append(kwargs["window_size"])
        patch_sizes.append(kwargs["patch_size"])
        
        self.down2 = STCNNT_Block(**kwargs)
        
        kwargs["C_in"] = c.channels[1]
        kwargs["C_out"] = c.channels[2]
        kwargs["H"] = c.height[0]//4
        kwargs["W"] = c.width[0]//4
        kwargs["interpolate"] = "up"
        
        kwargs = self.set_window_patch_sizes_keep_num_window(kwargs, kwargs["H"], self.num_windows_h//2, self.num_patch, module_name="U1")
        window_sizes.append(kwargs["window_size"])
        patch_sizes.append(kwargs["patch_size"])
        
        self.up1 = STCNNT_Block(**kwargs)

        kwargs["C_in"] = c.channels[1]+c.channels[2]
        kwargs["C_out"] = c.channels[2]
        kwargs["H"] = c.height[0]//2
        kwargs["W"] = c.width[0]//2
        
        kwargs = self.set_window_patch_sizes_keep_window_size(kwargs, kwargs["H"] , window_sizes[1], patch_sizes[1], module_name="U2")
        
        self.up2 = STCNNT_Block(**kwargs)

        kwargs["C_in"] = c.channels[0]+c.channels[2]
        kwargs["C_out"] = c.channels[1]
        kwargs["H"] = c.height[0]
        kwargs["W"] = c.width[0]
        kwargs["interpolate"] = "none"
        
        kwargs = self.set_window_patch_sizes_keep_num_window(kwargs, kwargs["H"], self.num_windows_h, self.num_patch, module_name="final")
        
        self.final = STCNNT_Block(**kwargs)

        self.output_proj = Conv2DExt(c.channels[1], c.C_out, kernel_size=kwargs["kernel_size"],\
                                        stride=kwargs["stride"], padding=kwargs["padding"])
           
    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): the input image
        """

        _,_,_,H,W = x.shape
        assert not(H % 8 or W % 8),\
            f"Require H and W dimension sizes to be divisible by 8"
                                                # x :[B, T,  C, 64, 64]
        x1, x1_interp = self.down1(x)           # x1:[B, T, 16, 64, 64], x1_interp:[B, T, 16, 32, 32]
        x2, x2_interp = self.down2(x1_interp)   # x2:[B, T, 32, 32, 32], x2_interp:[B, T, 32, 16, 16]

        y1, y1_interp = self.up1(x2_interp)     # y1:[B, T, 64, 16, 16], y1_interp:[B, T, 64, 32, 32]
        c1 = torch.cat((y1_interp, x2), dim=2)  # c1:[B, T, 96, 32, 32]
        y2, y2_interp = self.up2(c1)            # y2:[B, T, 64, 32, 32], y2_interp:[B, T, 64, 64, 64] 
        c2 = torch.cat((y2_interp, x1), dim=2)  # c2:[B, T, 80, 64, 64] 

        z1, z1_interp = self.final(c2)          # z1:[B, T, 32, 64, 64], z1_interp:[B, T, 32, 64, 64]

        output = self.output_proj(z1)

        return output

# -------------------------------------------------------------------------------------------------

def tests():

    B,T,C,H,W = 2,4,1,64,64
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
    config.stride_t = 2
    config.dropout_p = 0.1
    config.C_in = C
    config.C_out = C
    config.height = [H]
    config.width = [W]
    config.norm_mode = "instance3d"
    config.att_types = ["T0T1T0T1"]
    config.a_type = "conv"
    config.is_causal = False
    config.n_head = 8
    config.interp_align_c = True
    config.cell_type = 'sequential'
    config.normalize_Q_K = True 
    config.att_dropout_p = 0.0
    config.att_with_output_proj = True 
    config.scale_ratio_in_mixer = 4.0
            
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

    optims = ["adamw", "sgd", "nadam"]
    schedulers = ["StepLR", "OneCycleLR", "ReduceLROnPlateau"]
    all_w_decays = [True, False]

    loss = nn.MSELoss()

    for optim in optims:
        for scheduler in schedulers:
            for all_w_decay in all_w_decays:
                config.optim = optim    
                config.scheduler = scheduler
                config.all_w_decay = all_w_decay

                cnnt_unet = CNNT_Unet(config=config)
                test_out = cnnt_unet(test_in)
                res = loss(test_out, test_in)

                print(res)

    print("Passed optimizers and schedulers")

    heads_and_channelss = [(8,[16,32,64]),(5,[5,50,15]),(13,[13,13,13])]
    residuals = [True, False]

    for n_head, channels in heads_and_channelss:
        for residual in residuals:
            config.n_head = n_head
            config.channels = channels
            config. residual = residual

            cnnt_unet = CNNT_Unet(config=config)
            test_out = cnnt_unet(test_in)
            res = loss(test_out, test_in)

            print(res)

    print("Passed channels and residual")

    devices = ["cuda", "cpu", "cuda:0"]

    for device in devices:
        config.device = device

        cnnt_unet = CNNT_Unet(config=config)
        test_out = cnnt_unet(test_in)
        res = loss(test_out, test_in)

        print(res)

    print("Passed devices")

    print("Passed all tests")

if __name__=="__main__":
    tests()
