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

    def __init__(self, losses, loss_weights, device="cpu"):
        assert len(losses)>0, f"At least one loss is required to setup"
        assert len(losses)<=len(loss_weights), f"Each loss should have its weight"

        self.device = device

        losses = [self.str_to_loss(loss) for loss in losses]
        self.losses = list(zip(losses, loss_weights))

    def str_to_loss(self, loss_name):

        if loss_name=="mse":
            loss_f = nn.MSELoss(reduction='mean')
        elif loss_name=="l1":
            loss_f = nn.L1Loss(reduction='mean')
        else:
            raise NotImplementedError(f"Loss type not implemented: {loss_name}")

        return loss_f
    
    def __call__(self, outputs, targets, weights=[2.0, 1.0, 1.0, 1.0, 1.0, 2.0]):

        y_hat, params_estimated = outputs
        y, params = targets

        combined_loss = 0
        for loss_f, weight in self.losses:
            v = weight*loss_f(y_hat, y)
            if not torch.isnan(v):
                combined_loss += v

        combined_loss += torch.sum(weights*torch.abs(params_estimated-params))

        return combined_loss