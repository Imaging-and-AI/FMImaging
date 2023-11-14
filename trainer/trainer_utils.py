"""
Helper functions for train manager
"""

import os
import torch

# -------------------------------------------------------------------------------------------------         
def clean_after_training():
    """Clean after the training
    """
    #os.system("kill -9 $(ps aux | grep torchrun | grep -v grep | awk '{print $2}') ")
    #os.system("kill -9 $(ps aux | grep wandb | grep -v grep | awk '{print $2}') ")
    #os.system("kill -9 $(ps aux | grep mri | grep -v grep | awk '{print $2}') ")
    pass

# -------------------------------------------------------------------------------------------------
def get_bar_format():
    """Get the default bar format
    """
    return '{desc}{percentage:3.0f}%|{bar:10}{r_bar}'