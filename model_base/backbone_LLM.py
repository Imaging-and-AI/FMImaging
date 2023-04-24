"""
Backbone model - LLMs architecture, stack of transformers

This file implements a stack of transformer design for the imaging backbone. The input to the model is [B, T, C_in, H, W]. The output of the model is [B, T, N*C, H, W].

The model includes a number of stages. Each stage is a block. C is the model base number of channels. Every stage will increase the number of feature maps.

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
            
            - block_str (a str or a list of strings): order of attention types and mixer
                format is list of XYXYXYXY...
                - X is "L", "G" or "T" for attention type
                - Y is "0" or "1" for with or without mixer
                - only first one is used for this model to create consistent blocks
                - requires len(att_types[0]) to be even
                
                This string is the "Block string" to define the attention layers in a block. If a list of string is given,  each string defines the attention structure for a stage.
                           
            - add_skip_connections (bool): whether to add skip connections between stages; if True, densenet type connections are added; if False, LLM type network is created.
            
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
        num_stages = config.num_stages
        block_str = config.block_str
        add_skip_connections = config.add_skip_connections
        
        assert C >= config.C_in, "Number of channels should be larger than C_in"
        assert num_stages <= 5 and num_stages>=2, "Maximal number of stages is 5"

        self.C = C
        self.num_stages = num_stages
        self.block_str = block_str if isinstance(block_str, list) else [block_str for n in range(self.num_stages)]
        self.add_skip_connections = add_skip_connections
        
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

        if num_stages >= 1:
            # define B0
            kwargs["C_in"] = c.C_in
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["att_types"] = self.block_str[0]
            self.B0 = STCNNT_Block(**kwargs)
                        
        if num_stages >= 2:
            # define B1
            kwargs["C_in"] = self.C
            kwargs["C_out"] = self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["att_types"] = self.block_str[1]
            self.B1 = STCNNT_Block(**kwargs)
        
        if num_stages >= 3:
            # define B2
            kwargs["C_in"] = 2*self.C if add_skip_connections else self.C
            kwargs["C_out"] = 2*self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["att_types"] = self.block_str[2]
            self.B2 = STCNNT_Block(**kwargs)
            
        if num_stages >= 4:
            # define B3
            kwargs["C_in"] = 4*self.C if add_skip_connections else 2*self.C
            kwargs["C_out"] = 4*self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["att_types"] = self.block_str[3]
            self.B3 = STCNNT_Block(**kwargs)
            
        if num_stages >= 5:
            # define B4
            kwargs["C_in"] = 8*self.C if add_skip_connections else 4*self.C
            kwargs["C_out"] = 8*self.C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs["att_types"] = self.block_str[4]
            self.B4 = STCNNT_Block(**kwargs)
            
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
            - y_hat (5D torch.Tensor): output feature maps
        """

        B, T, Cin, H, W = x.shape
               
        if self.add_skip_connections:
            x0 = self.B0(x)
            y_hat = x0
            
            if self.num_stages >= 2:
                x1 = self.B1(x0)
                y_hat = x1
                
            if self.num_stages >= 3:
                x2 = self.B2(torch.cat((x0, x1), dim=2))
                y_hat = x2
                
            if self.num_stages >= 4:
                x3 = self.B3(torch.cat((x0, x1, x2), dim=2))
                y_hat = x3
                
            if self.num_stages >= 5:
                x4 = self.B4(torch.cat((x0, x1, x2, x3), dim=2))
                y_hat = x4
        else:
            y_hat = self.B0(x)
            if self.num_stages >= 2:
                y_hat = self.B1(y_hat)
            if self.num_stages >= 3:
                y_hat = self.B2(y_hat)
            if self.num_stages >= 4:
                y_hat = self.B3(y_hat)
            if self.num_stages >= 5:
                y_hat = self.B4(y_hat)
                  
        return y_hat

# -------------------------------------------------------------------------------------------------

def tests():

    B,T,C,H,W = 32,16,1,512,512
    test_in = torch.rand(B,T,C,H,W)

    config = Namespace()
    config.C = 16
    config.num_stages = 5
    config.block_str = 'T1L1G1T1L1G1'
    config.add_skip_connections = True
        
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
    config.att_types = ["T0T1T0T1"]
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

    optims = ["adamw", "sgd", "nadam"]
    schedulers = ["StepLR", "OneCycleLR", "ReduceLROnPlateau"]
    all_w_decays = [True, False]

    for optim in optims:
        for scheduler in schedulers:
            for all_w_decay in all_w_decays:
                config.optim = optim    
                config.scheduler = scheduler
                config.all_w_decay = all_w_decay

                model = STCNNT_LLMnet(config=config)
                test_out = model(test_in)

    print("Passed optimizers and schedulers")

    heads_and_channelss = [(8,[16,32,64]),(5,[5,50,15]),(13,[13,13,13])]
    residuals = [True, False]

    for n_head, channels in heads_and_channelss:
        for residual in residuals:
            config.n_head = n_head
            config.channels = channels
            config.residual = residual

            model = STCNNT_LLMnet(config=config)
            test_out = model(test_in)
            loss = model.computer_loss(test_out, test_in)

            print(loss)

    print("Passed channels and residual")

    devices = ["cuda", "cpu", "cuda:0"]

    for device in devices:
        config.device = device

        model = STCNNT_LLMnet(config=config)
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
