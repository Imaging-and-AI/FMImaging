"""
Loss for qperf
"""

import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

import torch
import torch.nn as nn

class qperf_loss:

    def __init__(self, config):

        self.config = config

        losses = [self.str_to_loss(loss) for loss in config.losses]
        self.losses = list(zip(losses, config.loss_weights))

    def str_to_loss(self, loss_name):

        if loss_name=="mse":
            loss_f = nn.MSELoss(reduction='mean')
        elif loss_name=="l1":
            loss_f = nn.L1Loss(reduction='mean')
        else:
            raise NotImplementedError(f"Loss type not implemented: {loss_name}")

        return loss_f
    
    def __call__(self, outputs, targets):

        y_hat, params_estimated = outputs
        y, params, aif_p = targets

        valid_N = aif_p[-1]

        combined_loss = 0
        for loss_f, weight in self.losses:
            v = weight*loss_f(y_hat[:valid_N, :], y[:valid_N, :])
            if not torch.isnan(v):
                combined_loss += v

        combined_loss += torch.sum(self.config.loss_weights_params * torch.abs(params_estimated-params))

        return combined_loss