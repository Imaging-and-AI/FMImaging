"""
Support functions for models
"""

import torch
from collections import OrderedDict
import numpy as np
import torch.nn as nn

# -------------------------------------------------------------------------------------------------    
def create_generic_class_str(obj : object, exclusion_list=[torch.nn.Module, OrderedDict]) -> str:
    """
    Create a generic name of a class
    @args:
        - obj (object): the class to make string of
        - exclusion_list (object list): the objects to exclude from the class string
    @rets:
        - class_str (str): the generic class string
    """
    name = type(obj).__name__

    vars_list = []
    for key, value in vars(obj).items():
        valid = True
        for type_e in exclusion_list:
            if isinstance(value, type_e) or key.startswith('_'):
                valid = False
                break
        
        if valid:
            vars_list.append(f'{key}={value!r}')
            
    vars_str = ',\n'.join(vars_list)
    return f'{name}({vars_str})'


# -------------------------------------------------------------------------------------------------    

class IdentityModel(nn.Module):
    """
    Simple class to implement identity model in format requierd by codebase
    """
    def __init__(self):
        super().__init__()
        self.identity_layer = nn.Identity()

    def forward(self, x):
        return [self.identity_layer(x)]

def identity_model(config, input_feature_channels):
    """
    Simple function to return identity model and feature channels in format requierd by codebase
    """
    model = IdentityModel()
    output_feature_channels = input_feature_channels
    return model, output_feature_channels

                        
