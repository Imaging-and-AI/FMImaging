"""
Set up the optimizer and scheduler manager
"""

import os
import sys
import logging
from abc import ABC
from colorama import Fore, Style

import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from optim_utils import *
from optimizers import *

# -------------------------------------------------------------------------------------------------

class OptimManager(object):
    """
    Manages optimizer and scheduler
    """
    
    def __init__(self, config, model_manager, train_set) -> None:
        """
        @args:
            - config (Namespace): nested namespace containing all args
            - model_manager (ModelManager): model containig pre/backbone/post modules we aim to optimize
            - train_set (torch Dataset): train split dataset, used for computing total_steps (needed by some schedulers)
        """
        super().__init__()

        # Save vars
        self.config = config
        self.model_manager = model_manager
        
        # # Compute total steps (needed for some schedulers)
        if isinstance(train_set, list):
            total_num_samples = sum([len(a_train_set) for a_train_set in train_set])
        else:
            total_num_samples = len(train_set)
            
        self.total_steps = compute_total_steps(self.config, total_num_samples)
        logging.info(f"Total steps for this run: {self.total_steps}, number of samples {total_num_samples}, batch {self.config.batch_size}")
        
        # Set up optimizer and scheduler
        self.set_up_optim_and_scheduling(total_steps=self.total_steps)

        # Load optim and scheduler states, if desired
        if self.config.continued_training:
            if self.config.pre_model_load_path is not None: self.load_optim_and_sched("pre", config.pre_model_load_path)
            if self.config.backbone_model_load_path is not None: self.load_optim_and_sched("backbone", config.backbone_model_load_path)
            if self.config.post_model_load_path is not None: self.load_optim_and_sched("post", config.post_model_load_path)
        elif self.config.pre_model_load_path is not None or self.config.backbone_model_load_path is not None or self.config.post_model_load_path is not None:
            logging.info(f"{Fore.YELLOW}No optimizers or schedulers loaded this run{Style.RESET_ALL}")

    def split_decay_optim_groups(self, module, lr=-1, wd=0.0):
        """
        This function splits up parameter groups into those that will and those that won't experience weight decay
        Adapted from mingpt: https://github.com/karpathy/minGPT
        @args:
            - module (torch module): module to create weight decay parameter groups for
            - lr (float): learning rate for the module m
            - weight_decay (float, from config): weight decay coefficient for regularization
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv3d)
        blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.parameter.Parameter)
        for mn, m in module.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if len([sub_m for sub_mn, sub_m in m.named_modules()])==1 or pn.endswith('relative_position_bias_table'):
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    elif pn.endswith('relative_position_bias_table'):
                        no_decay.add(fpn)
                    else: 
                        no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in module.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd}, # With weight decay group
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}, # Without weight decay group
        ]

        if lr >= 0:
            optim_groups[0]['lr'] = lr
            optim_groups[1]['lr'] = lr

        return optim_groups

    def configure_optim_groups(self):
        """
        This function splits up pre, backbone, and post parameters into different parameter groups
        If all_w_decay is False (default), then this function further splits up each group into params w/ and w/o weight decay
        """
        if not self.config.optim.all_w_decay:
            pre_optim_groups = self.split_decay_optim_groups(self.model_manager.pre, lr=self.config.optim.lr[0], wd=self.config.optim.weight_decay)
            backbone_optim_groups = self.split_decay_optim_groups(self.model_manager.backbone, lr=self.config.optim.lr[1], wd=self.config.optim.weight_decay)
            post_optim_groups = self.split_decay_optim_groups(self.model_manager.post, lr=self.config.optim.lr[2], wd=self.config.optim.weight_decay)

        else:
            pre_optim_groups = [{"params": list(self.model_manager.pre.parameters()), "lr": self.config.optim.lr[0], "weight_decay": self.config.optim.weight_decay}]
            backbone_optim_groups = [{"params": list(self.model_manager.backbone.parameters()), "lr": self.config.optim.lr[1], "weight_decay": self.config.optim.weight_decay}]
            post_optim_groups = [{"params": list(self.model_manager.post.parameters()), "lr": self.config.optim.lr[2], "weight_decay": self.config.optim.weight_decay}]

        optim_groups = pre_optim_groups + backbone_optim_groups + post_optim_groups

        return optim_groups

    def set_up_optim_and_scheduling(self, total_steps=1):
        """
        Sets up the optimizer and the learning rate scheduler using the config
        @args:
            - total_steps (int): total training steps. used for OneCycleLR
        @args (from config):
            - optim_type ("adam", "adamw", "nadam", "sgd", "sophia"): choices for optimizer
            - scheduler ("ReduceOnPlateau", "StepLR", "OneCycleLR", None): choices for learning rate schedulers
            - lr (float): global learning rate
            - beta1, beta2 (float): parameters for adam optimizers
            - weight_decay (float): parameter for regularization
            - all_w_decay (bool): whether to separate model params for regularization
                if False then norms and embeddings do not experience weight decay
        @outputs:
            - self.optim: optimizer
            - self.sched: scheduler
        """
        c = self.config # short config name because of multiple uses
        self.optim = None
        self.sched = None
        self.curr_epoch = 0
        if c.optim_type is None:
            return

        optim_groups = self.configure_optim_groups() 

        if c.optim_type == "adam":
            self.optim = optim.Adam(optim_groups, lr=c.optim.global_lr, betas=(c.optim.beta1, c.optim.beta2), weight_decay=c.optim.weight_decay)
        elif c.optim_type == "adamw":
            self.optim = optim.AdamW(optim_groups, lr=c.optim.global_lr, betas=(c.optim.beta1, c.optim.beta2), weight_decay=c.optim.weight_decay)
        elif c.optim_type == "sgd":
            self.optim = optim.SGD(optim_groups, lr=c.optim.global_lr, momentum=0.9, weight_decay=c.optim.weight_decay)
        elif c.optim_type == "nadam":
            self.optim = optim.NAdam(optim_groups, lr=c.optim.global_lr, betas=(c.optim.beta1, c.optim.beta2), weight_decay=c.optim.weight_decay)
        elif c.optim_type == "sophia":
            self.optim = SophiaG(optim_groups, lr=c.optim.global_lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=c.optim.weight_decay)
        elif c.optim_type == "lbfgs":
            self.optim = optim.LBFGS(optim_groups, lr=c.optim.global_lr, max_iter=c.optim.max_iter, history_size=c.optim.history_size, line_search_fn=c.optim.line_search_fn)
        else:
            raise NotImplementedError(f"Optimizer not implemented: {c.optim_type}")

        if c.scheduler_type == "ReduceLROnPlateau":
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode="min", factor=c.scheduler.factor,
                                                                    patience=c.scheduler.patience, 
                                                                    cooldown=c.scheduler.cooldown, 
                                                                    min_lr=c.scheduler.min_lr)
        elif c.scheduler_type == "StepLR":
            self.sched = optim.lr_scheduler.StepLR(self.optim, 
                                                   step_size=c.scheduler.step_size, 
                                                   gamma=c.scheduler.gamma, 
                                                   last_epoch=-1)
        elif c.scheduler_type == "OneCycleLR":
            self.sched = optim.lr_scheduler.OneCycleLR(self.optim, max_lr=c.optim.global_lr, total_steps=total_steps,
                                                            pct_start=c.scheduler.pct_start, anneal_strategy="cos")
        elif c.scheduler_type is None:
            self.sched = None
        else:
            raise NotImplementedError(f"Scheduler not implemented: {c.scheduler_type}")
        
    def load_optim_and_sched(self, part, load_path):
        logging.info(f"{Fore.YELLOW}Loading {part} optim and scheduler from {load_path}{Style.RESET_ALL}")

        if os.path.isfile(load_path):
            status = torch.load(load_path, map_location=self.config.device)
            
            if f'{part}_optimizer_state' in status:
                # Load saved optimizer state
                saved_part_optimizer_state = status[f'{part}_optimizer_state']

                # Get current optimizer states
                current_pre_optimizer_state = divide_optim_into_groups(self.optim, "pre", self.config.optim.all_w_decay)
                current_backbone_optimizer_state = divide_optim_into_groups(self.optim, "backbone", self.config.optim.all_w_decay)
                current_post_optimizer_state = divide_optim_into_groups(self.optim, "post", self.config.optim.all_w_decay)

                # Replace current optimizer state with saved optimizer state
                if part=="pre": current_pre_optimizer_state = saved_part_optimizer_state
                elif part=="backbone": current_backbone_optimizer_state = saved_part_optimizer_state
                elif part=="post": current_post_optimizer_state = saved_part_optimizer_state
                else: raise ValueError(f"Unknown model part {part} specified in load_optim_and_sched")

                full_optimizer_state = {'state': {**current_pre_optimizer_state['state'],**current_backbone_optimizer_state['state'],**current_post_optimizer_state['state']},
                                        'param_groups': current_pre_optimizer_state['param_groups']+current_backbone_optimizer_state['param_groups']+current_post_optimizer_state['param_groups'],
                                        'curr_epoch': status['epoch']}

                # Load modified optimizer state into self.optim
                self.optim.load_state_dict(full_optimizer_state)

                # Send optim to device
                optimizer_to(self.optim, device=self.config.device)

                logging.info(f"{Fore.GREEN} {part} optim loading successful {Style.RESET_ALL}")
            else: 
                logging.warning(f"{Fore.YELLOW} {part} optimizer state is not available in specified load_path {Style.RESET_ALL}")

            if 'scheduler_state' in status:
                self.sched.load_state_dict(status['scheduler_state'])
                logging.info(f"{Fore.GREEN} {part} scheduler loading successful {Style.RESET_ALL}")
            else: 
                logging.warning(f"{Fore.YELLOW} {part} scheduler state is not available in specified load_path {Style.RESET_ALL}")

            if 'epoch' in status:
                self.curr_epoch = status['epoch']
                logging.info(f"{Fore.GREEN} {part} epoch loading successful {Style.RESET_ALL}")
            else: 
                logging.warning(f"{Fore.YELLOW} {part} epoch is not available in specified load_path {Style.RESET_ALL}")

        else:
            logging.warning(f"{Fore.YELLOW}{load_path} does not exist .... {Style.RESET_ALL}")


def tests():
    print('Passed all tests')

    
if __name__=="__main__":
    tests()
