"""
Requirements to use a custom loss with the general codebase: 
- Define a loss class with a __call__ method that adheres to the following:
    @args: 
        outputs (torch tensor): model outputs
        targets (torch tensor): model targets 
    @rets:
        loss (float): loss value which model will aim to minimize
    Note that the custom loss __init___ args can be customized via the custom run.py file
"""

import torch.nn as nn

class custom_loss(object):
    """
    Example custom loss function
    """
    def __init__(self, config):

        self.config = config

    def __call__(self, outputs, targets):

        loss_value = nn.functional.cross_entropy(outputs, targets)
        
        return loss_value