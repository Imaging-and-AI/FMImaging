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

class qperf_mse(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, y, y_hat, mask, N):
        v = torch.sum(mask* ( (y_hat-y)**2 ))/N
        return v

class qperf_l1(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, y, y_hat, mask, N):
        v = torch.sum(mask* torch.abs(y_hat-y))/N
        return v

class qperf_loss:

    def __init__(self, config):

        self.config = config

        losses = [self.str_to_loss(loss) for loss in config.losses]
        self.losses = list(zip(losses, config.loss_weights))

    def str_to_loss(self, loss_name):

        if loss_name=="mse":
            loss_f = qperf_mse(self.config)
        elif loss_name=="l1":
            loss_f = qperf_l1(self.config)
        else:
            raise NotImplementedError(f"Loss type not implemented: {loss_name}")

        return loss_f
    
    def __call__(self, outputs, targets):

        y_hat, params_estimated = outputs
        y, params = targets

        valid_N = params[:, -1]

        B, T = y.shape

        y_hat = y_hat.to(dtype=y.dtype).squeeze()

        mask = torch.ones((B, T), dtype=y.dtype).to(device=y.device)
        for b in range(B):
            mask[b, int(valid_N[b]):T] = 0

        N = torch.sum(valid_N)

        combined_loss = 0
        for loss_f, weight in self.losses:
            v = weight*loss_f(y, y_hat, mask, N)
            if not torch.isnan(v):
                combined_loss += v

        num_params = params_estimated.shape[1]
        for n in range(num_params):
            combined_loss += self.config.loss_weights_params[n] * torch.sum(torch.abs(params_estimated[:, n]-params[:, n]))/B

        return combined_loss