"""
A base model for specific task
"""

import os
import sys
import logging
import abc 
from abc import ABC
from colorama import Fore, Style

import torch
import torch.nn as nn
import torch.optim as optim
from Sophia import SophiaG 

from pathlib import Path
from argparse import Namespace

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from imaging_attention import *
from backbone.blocks import *
from utils import get_device, create_generic_class_str, optimizer_to

__all__ = ['STCNNT_Task_Base']

# -------------------------------------------------------------------------------------------------
# Base model for rest to inherit

class STCNNT_Task_Base(nn.Module, ABC):
    """
    Base Runtime model for a specific task
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
        blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.parameter.Parameter)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(p, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif (pn.endswith('weight') or pn.endswith('relative_position_bias_table')) and isinstance(p, blacklist_weight_modules):
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
        @outputs:
            - self.optim: optimizer
            - self.sched: scheduler
            - self.stype: scheduler type name
        """
        c = self.config # short config name because of multiple uses
        self.optim = None
        self.sched = None
        self.stype = None
        self.curr_epoch = 0
        if c.optim is None:
            return

        optim_groups = self.configure_optim_groups() if not c.all_w_decay else self.parameters()

        if c.optim == "adamw":
            self.optim = optim.AdamW(optim_groups, lr=c.global_lr, betas=(c.beta1, c.beta2), weight_decay=c.weight_decay)
        elif c.optim == "sgd":
            self.optim = optim.SGD(optim_groups, lr=c.global_lr, momentum=0.9, weight_decay=c.weight_decay)
        elif c.optim == "nadam":
            self.optim = optim.NAdam(optim_groups, lr=c.global_lr, betas=(c.beta1, c.beta2), weight_decay=c.weight_decay)
        elif c.optim == "sophia":
            self.optim = SophiaG(optim_groups, lr=c.global_lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=c.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer not implemented: {c.optim}")

        if c.scheduler_type == "ReduceLROnPlateau":
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode="min", factor=c.scheduler.ReduceLROnPlateau.factor,
                                                                    patience=c.scheduler.ReduceLROnPlateau.patience, 
                                                                    cooldown=c.scheduler.ReduceLROnPlateau.cooldown, 
                                                                    min_lr=c.scheduler.ReduceLROnPlateau.min_lr,
                                                                    verbose=True)
            self.stype = "ReduceLROnPlateau"
        elif c.scheduler_type == "StepLR":
            self.sched = optim.lr_scheduler.StepLR(self.optim, 
                                                   step_size=c.scheduler.StepLR.step_size, 
                                                   gamma=c.scheduler.StepLR.gamma, 
                                                   last_epoch=-1,
                                                   verbose=True)
            self.stype = "StepLR"
        elif c.scheduler_type == "OneCycleLR":
            self.sched = optim.lr_scheduler.OneCycleLR(self.optim, max_lr=c.global_lr, total_steps=total_steps,
                                                            pct_start=0.2, anneal_strategy="cos", verbose=False)
            self.stype = "OneCycleLR"
        else:
            raise NotImplementedError(f"Scheduler not implemented: {c.scheduler_type}")

    @abc.abstractmethod
    def set_up_loss(self, device="cpu"):
        """
        Sets up the loss
        @args:
            - device (torch.device): device to setup the loss on
            
        @output:
            self.loss_f should be set
        """
        pass
    
    def save(self, epoch):
        """
        Save model checkpoints
        @args:
            - epoch (int): current epoch of the training cycle
        @args (from config):
            - date (datetime str): runtime date
            - checkpath (str): directory to save checkpoint in
        """
        run_name = self.config.run_name.replace(" ", "_")
        save_file_name = f"{run_name}_epoch-{epoch}.pth"
        save_path = os.path.join(self.config.check_path, save_file_name)
        logging.info(f"{Fore.YELLOW}Saving model status at {save_path}{Style.RESET_ALL}")
        torch.save({
            "epoch":epoch,
            "model_state": self.state_dict(), 
            "optimizer_state": self.optim.state_dict(), 
            "scheduler_state": self.sched.state_dict(),
            "config": self.config,
            "scheduler_type":self.stype
        }, save_path)

    def load(self, device=None):
        """
        Load a checkpoint from the load path in config
        @args:
            - device (torch.device): device to setup the model on
        @args (from config):
            - load_path (str): path to load the weights from
        """
        logging.info(f"{Fore.YELLOW}Loading model from {self.config.load_path}{Style.RESET_ALL}")

        device = get_device(device=device)

        status = torch.load(self.config.load_path, map_location=self.config.device)
        self.config = status['config']

        if 'model_state' in status:
            self.load_state_dict(status['model_state'])
        else:
            self.load_state_dict(status['model'])

        if 'optimizer_state' in status:
            self.optim.load_state_dict(status['optimizer_state'])
            optimizer_to(self.optim, device=self.config.device)

        if 'scheduler_state' in status:
            self.sched.load_state_dict(status['scheduler_state'])

        if 'scheduler_type' in status:
            self.stype = status['scheduler_type']

        if 'epoch' in status:
            self.curr_epoch = status['epoch']

# -------------------------------------------------------------------------------------------------

def tests():
    
    config = Namespace()
    
    # should raise an exception
    base_model = STCNNT_Task_Base(config=config)    
    
    print("Passed all tests")

if __name__=="__main__":
    tests()
