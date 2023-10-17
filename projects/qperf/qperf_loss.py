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

import numpy as np

import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

from loss.loss_functions.gaussian import create_window_1d

# --------------------------------------------------------
class qperf_mse(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, y, y_hat, mask, N):
        v = torch.sum(mask* ( (y_hat-y)**2 ))/N
        if(torch.any(torch.isnan(v))):
            raise NotImplementedError(f"nan in qperf_mse")
            v = torch.mean(0.0 * y_hat)
        return v

# --------------------------------------------------------
class qperf_max_absolute_error(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, y, y_hat, mask=None, N=None):
        B = y.shape[0]
        if mask is not None:
            v, _ = torch.max(mask * torch.abs(y-y_hat), dim=1)
        else:
            v, _ = torch.max(torch.abs(y-y_hat), dim=1)
        v = torch.sum(v) / B
        if(torch.any(torch.isnan(v))):
            raise NotImplementedError(f"nan in qperf_max_absolute_error")
            v = torch.mean(0.0 * y_hat)
        return v
    
# --------------------------------------------------------
class qperf_l1(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, y, y_hat, mask, N):
        v = torch.sum(mask* torch.abs(y_hat-y))/N
        if(torch.any(torch.isnan(v))):
            raise NotImplementedError(f"nan in qperf_l1")
            v = torch.mean(0.0 * y_hat)
        return v

# --------------------------------------------------------

class qperf_gaussian(object):
    def __init__(self, config):
        self.config = config
        self.sigmas = [0.25, 0.5, 1.0]

        # compute kernels
        self.kernels = []
        for sigma in self.sigmas:
            k_1d = create_window_1d(sigma=sigma, halfwidth=7, voxelsize=1.0, order=1)
            kx, = k_1d.shape
            k_1d = torch.from_numpy(np.reshape(k_1d, (1, 1, kx))).to(torch.float32)
            self.kernels.append(k_1d.to(device=config.device))

    def __call__(self, y, y_hat, mask=None, N=None):
        B, T = y.shape

        loss = 0
        for k_1d in self.kernels:
            grad_y = F.conv1d(y[:,None,:].to(dtype=k_1d.dtype), k_1d, bias=None, stride=1, padding='same')
            grad_y = grad_y.squeeze()

            grad_y_hat = F.conv1d(y_hat[:,None,:].to(dtype=k_1d.dtype), k_1d, bias=None, stride=1, padding='same')
            grad_y_hat = grad_y_hat.squeeze()

            if mask is not None:
                v = torch.sum(mask* torch.abs(grad_y_hat-grad_y))/N
            else:
                v = torch.mean(torch.abs(grad_y_hat-grad_y))
            loss += v

        loss /= len(self.kernels)

        if(torch.any(torch.isnan(loss))):
            raise NotImplementedError(f"nan in qperf_gaussian")
            loss = torch.mean(0.0 * y_hat)
        return loss

# --------------------------------------------------------
class qperf_loss:

    def __init__(self, config):

        self.config = config

        losses = [self.str_to_loss(loss) for loss in config.losses]
        self.losses = list(zip(losses, config.loss_weights))
        self.msle = torchmetrics.MeanSquaredLogError()

    def str_to_loss(self, loss_name):

        if loss_name=="mse":
            loss_f = qperf_mse(self.config)
        elif loss_name=="max_ae":
            loss_f = qperf_max_absolute_error(self.config)
        elif loss_name=="l1":
            loss_f = qperf_l1(self.config)
        elif loss_name=="gauss":
            loss_f = qperf_gaussian(self.config)
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

        self.msle.to(device=params.device)

        num_params = params_estimated.shape[1]
        for n in range(num_params):
            #v = torch.abs(params_estimated[:, n]-params[:, n])
            #combined_loss += self.config.loss_weights_params[n] * torch.sum(v)/B

            v1 = torch.mean(torch.abs(params_estimated[:, n] - params[:, n]))
            if(torch.any(torch.isnan(v1))):
                raise NotImplementedError(f"nan in v1 = torch.mean(torch.abs(params_estimated[:, n] - params[:, n]))")
                v1 = torch.mean(0.0 * params_estimated[:, n])

            # v2 = torch.mean( ( torch.log(1 + torch.abs(params_estimated[:, n])) - torch.log(1 + torch.abs(params[:, n])) ) ** 2)
            # if(torch.any(torch.isnan(v2))):
            #     raise NotImplementedError(f"nan in v2 = torch.mean( ( torch.log(1+params_estimated[:, n]) - torch.log(1+params[:, n]) ) ** 2)")
            #     v2 = torch.mean(0.0 * params_estimated[:, n])

            # v3 = torch.mean( (params_estimated[:, n] - params[:, n]) ** 2)
            # if(torch.any(torch.isnan(v3))):
            #     raise NotImplementedError(f"nan in v3 = torch.mean( (params_estimated[:, n] - params[:, n]) ** 2)")
            #     v3 = torch.mean(0.0 * params_estimated[:, n])

            combined_loss += self.config.loss_weights_params[n] * (v1)

        return combined_loss
# --------------------------------------------------------
