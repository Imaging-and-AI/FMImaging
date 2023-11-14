"""
Defines the loss function specified in the config
"""

import sys
from torch import nn
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from loss_functions import *

# -------------------------------------------------------------------------------------------------
def get_loss_func(config):
    """
    Sets up the loss
    @args:
        - config (namespace): contains args for defining loss
    @output:
        - loss_f: loss function
    """
    if config.loss_type=='CrossEntropy':
        loss_f = nn.CrossEntropyLoss()
    elif config.loss_type=='MSE':
        loss_f = nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss function not implemented: {config.loss_type}")
    return loss_f
        
# -------------------------------------------------------------------------------------------------
def tests():
    pass

if __name__=="__main__":
    tests()
