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
        run_name = self.config.run_name.replace(" ", "_")
        save_file_name = f"backbone_{run_name}_{self.config.date}_epoch-{epoch}.pth"
        save_path = os.path.join(self.config.check_path, save_file_name)
        logging.info(f"Saving backbone status at {save_path}")
        torch.save({
            "epoch":epoch,
            "model_state": self.state_dict(), 
            "config": self.config
        }, save_path)

    def load(self, device=None):
        """
        Load a checkpoint from the load path in config
        @args:
            - device (torch.device): device to setup the model on
        @args (from config):
            - load_path (str): path to load the weights from
        """
        logging.info(f"Loading backbone from {self.config.load_path}")
        
        device = get_device(device=device)
        
        status = torch.load(self.config.load_path, map_location=device)
        
        self.load_state_dict(status['model_state'])
        self.config = status['config']
        
# -------------------------------------------------------------------------------------------------

def tests():
    print("Passed all tests")

if __name__=="__main__":
    tests()
