"""
Main base model for backbone
Provides implementation for the following:
    - STCNNT_Base_Runtime_Model:
        - the base class that setups the optimizer scheduler and loss
        - also provides ability to save and load checkpoints
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
from blocks import *
from utils.utils import get_device, create_generic_class_str

__all__ = ['STCNNT_Base_Runtime']

# -------------------------------------------------------------------------------------------------
# Base model for rest to inherit

class STCNNT_Base_Runtime(nn.Module):
    """
    Base Runtime model of STCNNT
    Sets up the optimizer, scheduler and loss
    Provides generic save and load functionality
    """
    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
        """
        super().__init__()
        self.config = config

    @property
    def device(self):
        return next(self.parameters()).device

    def configure_optim_groups(self):
        """
        Copied (and modified) from mingpt: https://github.com/karpathy/minGPT
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the optimizer groups.

        @args (from config):
            - weight_decay (float): weight decay coefficient for regularization
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv3d)
        blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def set_up_optim_and_scheduling(self, total_steps=1):
        """
        Sets up the optimizer and the learning rate scheduler using the config
        @args:
            - total_steps (int): total training steps. used for OneCycleLR
        @args (from config):
            - optim ("adamw", "sgd", "nadam"): choices for optimizer
            - scheduler ("ReduceOnPlateau", "StepLR", "OneCycleLR"):
                choices for learning rate schedulers
            - global_lr (float): global learning rate
            - beta1, beta2 (float): parameters for adam
            - weight_decay (float): parameter for regularization
            - all_w_decay (bool): whether to separate model params for regularization
                if False then norms and embeddings do not experience weight decay
        """
        c = self.config # short config name because of multiple uses
        self.optim = None
        self.sched = None
        self.stype = None
        if c.optim is None:
            return

        optim_groups = self.configure_optim_groups() if not c.all_w_decay else self.parameters()

        if c.optim == "adamw":
            self.optim = optim.AdamW(optim_groups, lr=c.global_lr, betas=(c.beta1, c.beta2), weight_decay=c.weight_decay)
        elif c.optim == "sgd":
            self.optim = optim.SGD(optim_groups, lr=c.global_lr, momentum=0.9, weight_decay=c.weight_decay)
        elif c.optim == "nadam":
            self.optim = optim.NAdam(optim_groups, lr=c.global_lr, betas=(c.beta1, c.beta2), weight_decay=c.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer not implemented: {c.optim}")

        if c.scheduler == "ReduceLROnPlateau":
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode="min", factor=0.75,
                                                                    patience=5, cooldown=3, min_lr=1e-8,
                                                                    verbose=True)
            self.stype = "ReduceLROnPlateau"
        elif c.scheduler == "StepLR":
            self.sched = optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.8, last_epoch=-1,
                                                        verbose=True)
            self.stype = "StepLR"
        elif c.scheduler == "OneCycleLR":
            self.sched = optim.lr_scheduler.OneCycleLR(self.optim, max_lr=c.global_lr*4, total_steps=total_steps,
                                                            pct_start=0.3, anneal_strategy="cos", verbose=True)
            self.stype = "OneCycleLR"
        else:
            raise NotImplementedError(f"Scheduler not implemented: {c.scheduler}")

    def set_up_loss(self, device="cpu"):
        """
        Sets up the combined loss
        @args:
            - device (torch.device): device to setup the loss on
        @args (from config):
            - losses (list of "ssim", "ssim3D", "l1", "mse"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether we are dealing with complex images or not
        """
        self.loss_f = Combined_Loss(self.config.losses, self.config.loss_weights,
                                    complex_i=self.config.complex_i, device=device)

    def set_window_patch_sizes_keep_num_window(self, kwargs, H, num_windows_h, num_patch, module_name=None):
        
        num_windows_h = 2 if num_windows_h<2 else num_windows_h
        num_patch = 2 if num_patch<2 else num_patch
        
        kwargs["window_size"] = H // num_windows_h
        kwargs["patch_size"] = kwargs["window_size"] // num_patch
        
        kwargs["window_size"] = 1 if kwargs["window_size"]<1 else kwargs["window_size"]
        kwargs["patch_size"] = 1 if kwargs["patch_size"]<1 else kwargs["patch_size"]
        
        info_str = f" --> image size {H} - windows size {kwargs['''window_size''']} - patch size {kwargs['''patch_size''']}"
        if module_name is not None:
            info_str = module_name + info_str
            
        print(info_str)
        
        return kwargs
    
    def set_window_patch_sizes_keep_window_size(self, kwargs, H, window_size, patch_size, module_name=None):        
        
        if H//window_size < 2:
            window_size = H//2
            
        if window_size//patch_size < 2:
            patch_size = window_size//2
        
        kwargs["window_size"] = window_size
        kwargs["patch_size"] = patch_size
        
        kwargs["window_size"] = 1 if kwargs["window_size"]<1 else kwargs["window_size"]
        kwargs["patch_size"] = 1 if kwargs["patch_size"]<1 else kwargs["patch_size"]
        
        info_str = f" --> image size {H} - windows size {kwargs['''window_size''']} - patch size {kwargs['''patch_size''']}"
        if module_name is not None:
            info_str = module_name + info_str
            
        print(info_str)
        
        return kwargs
    
    def save(self, epoch):
        """
        Save model checkpoints
        @args:
            - epoch (int): current epoch of the training cycle
        @args (from config):
            - date (datetime str): runtime date
            - checkpath (str): directory to save checkpoint in
        """
        save_file_name = f"{self.config.date}_epoch-{epoch}.pth"
        save_path = os.path.join(self.config.check_path, save_file_name)
        logging.info(f"Saving weights at {save_path}")
        torch.save(self.state_dict(), save_path)

    def load(self, device=None):
        """
        Load a checkpoint from the load path in config
        @args:
            - device (torch.device): device to setup the model on
        @args (from config):
            - load_path (str): path to load the weights from
        """
        logging.info(f"Loading model from {self.config.load_path}")
        device = get_device(device=device)
        self.load_state_dict(torch.load(self.config.load_path, map_location=device))

# -------------------------------------------------------------------------------------------------

def tests():
    print("Passed all tests")

if __name__=="__main__":
    tests()
