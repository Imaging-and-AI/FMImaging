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
from backbone_base import STCNNT_Base_Runtime, set_window_patch_sizes_keep_num_window, set_window_patch_sizes_keep_window_size
from utils import get_device, create_generic_class_str, add_backbone_STCNNT_args, Nestedspace

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
    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
            - load (bool): whether to try loading from config.load_path or not
        @args (from config):
        
            Model specific arguments
            
            - channels (int list): number of channels in each of the 3 layers
            - block_str (str list): order of attention types and their following mlps
                format is list of XYXYXYXY...
                - X is "L", "G" or "T" for attention type
                - Y is "0" or "1" for with or without mixer
                - only first one is used for this model to create consistent blocks
                - requires len(att_types[0]) to be even
                
            Common shared arguments
            
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

        block_str = c.backbone_small_unet.block_str
        
        if isinstance(block_str, list):
            block_str = block_str if len(block_str)>=3 else [block_str[0] for n in range(3)] # with bridge
        else:
            block_str = [block_str for n in range(3)]

        channels = c.backbone_small_unet.channels

        for h in c.height: assert not h % 8, f"height {h} should be divisible by 8"
        for w in c.width: assert not w % 8, f"width {w} should be divisible by 8"
        assert len(channels) == 3, f"Requires exactly 3 channel numbers"
        
        self.num_wind = [c.height[0]//c.window_size[0], c.width[0]//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]
        
        kwargs = {
            "att_types":block_str[0], 
            "C_in":c.C_in, 
            "C_out":channels[0],
            "H":c.height[0], 
            "W":c.width[0], 
            "a_type":c.a_type,
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
            "interpolate":"down", 
            "interp_align_c":c.interp_align_c,
            "cell_type": c.cell_type,
            "normalize_Q_K": c.normalize_Q_K, 
            "att_dropout_p": c.att_dropout_p,
            "att_with_output_proj": c.att_with_output_proj, 
            "scale_ratio_in_mixer": c.scale_ratio_in_mixer,
            "window_size": c.window_size,
            "patch_size": c.patch_size,
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
               
        kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="D1")
        window_sizes.append(kwargs["window_size"])
        patch_sizes.append(kwargs["patch_size"])
        
        kwargs["att_types"] = block_str[0]
        self.down1 = STCNNT_Block(**kwargs)

        kwargs["C_in"] = channels[0]
        kwargs["C_out"] = channels[1]
        kwargs["H"] = c.height[0]//2
        kwargs["W"] = c.width[0]//2
        kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], window_sizes[0], patch_sizes[0], module_name="D2")
        window_sizes.append(kwargs["window_size"])
        patch_sizes.append(kwargs["patch_size"])
        kwargs["att_types"] = block_str[1]
        self.down2 = STCNNT_Block(**kwargs)
        
        kwargs["C_in"] = channels[1]
        kwargs["C_out"] = channels[2]
        kwargs["H"] = c.height[0]//4
        kwargs["W"] = c.width[0]//4
        kwargs["interpolate"] = "up"
        
        kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], [v//2 for v in self.num_wind], self.num_patch, module_name="U1")
        window_sizes.append(kwargs["window_size"])
        patch_sizes.append(kwargs["patch_size"])
        kwargs["att_types"] = block_str[-1]
        self.up1 = STCNNT_Block(**kwargs)

        kwargs["C_in"] = channels[1]+channels[2]
        kwargs["C_out"] = channels[2]
        kwargs["H"] = c.height[0]//2
        kwargs["W"] = c.width[0]//2
        
        kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]] , window_sizes[1], patch_sizes[1], module_name="U2")
        kwargs["att_types"] = block_str[1]
        self.up2 = STCNNT_Block(**kwargs)

        kwargs["C_in"] = channels[0]+channels[2]
        kwargs["C_out"] = channels[1]
        kwargs["H"] = c.height[0]
        kwargs["W"] = c.width[0]
        kwargs["interpolate"] = "none"
        
        kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="final")
        kwargs["att_types"] = block_str[0]
        self.final = STCNNT_Block(**kwargs)

        self.output_proj = Conv2DExt(channels[1], c.C_out, kernel_size=kwargs["kernel_size"],\
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

    parser = add_backbone_STCNNT_args()
    ns = Nestedspace()
    config = parser.parse_args(namespace=ns)
    
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
    config.a_type = "conv"
    config.is_causal = False
    config.n_head = 8
    config.interp_align_c = True
    config.cell_type = 'sequential'
    config.normalize_Q_K = True 
    config.att_dropout_p = 0.0
    config.att_with_output_proj = True 
    config.scale_ratio_in_mixer = 4.0
               
    config.window_size = [H//8, W//8]
    config.patch_size = [H//16, W//16]
    
    config.num_wind =[8, 8]
    config.num_patch =[2, 2]
    
    config.cosine_att = True
    config.att_with_relative_postion_bias = True
            
    # losses
    config.losses = ["mse"]
    config.loss_weights = [1.0]
    config.load_path = None
    # to be tested
    config.device = None
    config.channels = [16,32,64]
    config.all_w_decay = True
    config.optim = "adamw"
    config.scheduler = "StepLR"

    config.backbone_small_unet = Namespace()
    config.backbone_small_unet.block_str = ["T1L1G1","T1L1G1","T1L1G1"]
    config.backbone_small_unet.channels = [16,32,64]

    config.block_dense_connection = True
    
    config.complex_i = False

    optims = ["adamw", "sgd", "nadam"]
    schedulers = ["StepLR", "OneCycleLR", "ReduceLROnPlateau"]
    all_w_decays = [True, False]

    loss = nn.MSELoss()

    for optim in optims:
        for scheduler in schedulers:
            for all_w_decay in all_w_decays:
                
                print(optim, scheduler, all_w_decay)
                
                config.optim = optim    
                config.scheduler = scheduler
                config.all_w_decay = all_w_decay

                cnnt_unet = CNNT_Unet(config=config)
                test_out = cnnt_unet(test_in)
                res = loss(test_out, test_in)

                print(res)

    print("Passed optimizers and schedulers")

    B,T,C,H,W = 2,4,1,128,128
    test_in2 = torch.rand(B,T,C,H,W)
    test_out2 = cnnt_unet(test_in2)
    
    print("Passed different image size")
    
    heads_and_channelss = [(8,[16,32,64]),(5,[5,50,15]),(13,[13,13,13])]

    for n_head, channels in heads_and_channelss:
        config.n_head = n_head
        config.backbone_small_unet.channels = channels
        
        cnnt_unet = CNNT_Unet(config=config)
        test_out = cnnt_unet(test_in)
        res = loss(test_out, test_in)

        print(res)

    print("Passed channels")

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
