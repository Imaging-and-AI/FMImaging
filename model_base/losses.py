"""
Losses and Metrics

Provides implmentation of the following types of losses:
    - SSIM: Structural Similarity Index Measure for 2D
    - SSIM3D: Structural Similarity Index Measure for 3D
    - L1: Mean Absolute Error
    - MSE: Mean Squared Error
    - FSIM: Feature Similarity Index Measure
    - MSSSIM: multi-scale SSIM
    - Perpendicular: perp loss for complex images
    - Combined: Any weighed combination of the above

Allows custom weights for each indvidual loss calculation as well
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.setup_training import get_device
from utils.pytorch_ssim import SSIM, SSIM3D
from utils.msssim import MS_SSIM, ms_ssim
from utils.gaussian import create_window_2d, create_window_3d
import piq

# -------------------------------------------------------------------------------------------------
# Feature Similarity Index Measure (FSIM) loss

class FSIM_Loss:
    """
    Weighted FSIM loss
    """
    def __init__(self, chromatic=False, data_range=None, complex_i=False, device='cpu'):
        """
        @args:
            - chromatic (bool) : flag to compute FSIMc, which also takes into account chromatic components
            - data_range (float): max data value in the training; if none, determine from data
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        self.complex_i = complex_i
        self.chromatic = chromatic
        self.data_range = data_range


    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:,:,:1]*outputs[:,:,:1] + outputs[:,:,1:]*outputs[:,:,1:])
            targets_im = torch.sqrt(targets[:,:,:1]*targets[:,:,:1] + targets[:,:,1:]*targets[:,:,1:])
        else:
            outputs_im = outputs
            targets_im = targets

        B, T, C, H, W = targets_im.shape
        outputs_im = torch.reshape(outputs_im, (B*T, C, H, W))
        targets_im = torch.reshape(targets_im, (B*T, C, H, W))

        data_range = self.data_range
        if self.data_range is None:
            data_range = torch.max(torch.cat((targets_im, outputs_im), dim=0))

        loss = piq.fsim(outputs_im, targets_im, reduction='none', data_range=data_range, chromatic=self.chromatic)

        if weights is not None:

            if weights.ndim==1:
                weights_used = weights.expand(T,B).permute(1,0).reshape(B*T)
            elif weights.ndim==2:
                weights_used = weights.reshape(B*T)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for FSIM_Loss")
            
            v = torch.sum(weights_used*loss) / (torch.sum(weights_used) + torch.finfo(torch.float16).eps)
        else:
            v = torch.mean(loss)

        if(torch.any(torch.isnan(v))):
            v = torch.tensor(1.0, requires_grad=True)

        return (1.0-v)
    
# -------------------------------------------------------------------------------------------------
# SSIM loss

class SSIM_Loss:
    """
    Weighted SSIM loss
    """
    def __init__(self, window_size=11, complex_i=False, device='cpu'):
        """
        @args:
            - window_size (int): size of the window to use for loss computation
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        self.complex_i = complex_i
        self.ssim_loss = SSIM(window_size=window_size, size_average=False, device=device)

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:,:,:1]*outputs[:,:,:1] + outputs[:,:,1:]*outputs[:,:,1:])
            targets_im = torch.sqrt(targets[:,:,:1]*targets[:,:,:1] + targets[:,:,1:]*targets[:,:,1:])
        else:
            outputs_im = outputs
            targets_im = targets

        B, T, C, H, W = targets_im.shape
        outputs_im = torch.reshape(outputs_im, (B*T, C, H, W))
        targets_im = torch.reshape(targets_im, (B*T, C, H, W))

        if weights is not None:

            if weights.ndim==1:
                weights_used = weights.expand(T,B).permute(1,0).reshape(B*T)
            elif weights.ndim==2:
                weights_used = weights.reshape(B*T)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for SSIM_Loss")

            v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / (torch.sum(weights_used) + torch.finfo(torch.float16).eps)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = torch.tensor(1.0, requires_grad=True)

        return (1.0-v_ssim)

# -------------------------------------------------------------------------------------------------
# SSIM3D loss

class SSIM3D_Loss:
    """
    Weighted SSIM3D loss
    """
    def __init__(self, window_size=11, complex_i=False, device='cpu'):
        """
        @args:
            - window_size (int): size of the window to use for loss computation
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        self.complex_i = complex_i
        self.ssim_loss = SSIM3D(window_size=window_size, size_average=False, device=device)

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:,:,:1]*outputs[:,:,:1] + outputs[:,:,1:]*outputs[:,:,1:])
            targets_im = torch.sqrt(targets[:,:,:1]*targets[:,:,:1] + targets[:,:,1:]*targets[:,:,1:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.permute(outputs_im, (0, 2, 1, 3, 4))
        targets_im = torch.permute(targets_im, (0, 2, 1, 3, 4))

        if weights is not None:

            if not weights.ndim==1:
                raise NotImplementedError(f"Only support 1D(Batch) weights for SSIM3D_Loss")
            v_ssim = torch.sum(weights*self.ssim_loss(outputs_im, targets_im)) / (torch.sum(weights) + torch.finfo(torch.float16).eps)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = torch.tensor(1.0, requires_grad=True)

        return (1.0-v_ssim)

# -------------------------------------------------------------------------------------------------
# MSSSIM loss

class MSSSIM_Loss:
    """
    Weighted MSSSIM loss
    """
    def __init__(self, window_size=5, complex_i=False, data_range=256.0, device='cpu'):
        """
        @args:
            - window_size (int): size of the window to use for loss computation
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        self.complex_i = complex_i
        self.data_range = data_range
        self.msssim_loss = MS_SSIM(data_range=data_range, size_average=False, win_size=window_size, channel=1, spatial_dims=2)


    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:,:,:1]*outputs[:,:,:1] + outputs[:,:,1:]*outputs[:,:,1:])
            targets_im = torch.sqrt(targets[:,:,:1]*targets[:,:,:1] + targets[:,:,1:]*targets[:,:,1:])
        else:
            outputs_im = outputs
            targets_im = targets

        B, T, C, H, W = targets_im.shape
        outputs_im = torch.reshape(outputs_im, (B*T, C, H, W))
        targets_im = torch.reshape(targets_im, (B*T, C, H, W))

        # make it B, C, T, H, W, so calling 3D msssim
        #outputs_im = torch.permute(outputs_im, (0, 2, 1, 3, 4))
        #targets_im = torch.permute(targets_im, (0, 2, 1, 3, 4))

        v = self.msssim_loss(outputs_im, targets_im)
        v = v.squeeze()

        if weights is not None:
            if weights.ndim==1:
                weights_used = weights.expand(T,B).permute(1,0).reshape(B*T)
            elif weights.ndim==2:
                weights_used = weights.reshape(B*T)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for SSIM_Loss")

            v_ssim = torch.sum(weights_used*v) / (torch.sum(weights_used) + torch.finfo(torch.float16).eps)
        else:
            v_ssim = torch.mean(v)

        v_ssim = torch.clamp(v_ssim, 0.0, 1.0)

        return (1.0-v_ssim)

# -------------------------------------------------------------------------------------------------
# L1/mae loss

class L1_Loss:
    """
    Weighted L1 loss
    """
    def __init__(self, complex_i=False):
        """
        @args:
            - complex_i (bool): whether images are 2 channelled for complex data
        """
        self.complex_i = complex_i

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            diff_L1 = torch.abs(outputs[:,:,0]-targets[:,:,0]) + torch.abs(outputs[:,:,1]-targets[:,:,1])
        else:
            diff_L1 = torch.abs(outputs-targets)

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for L1_Loss")

            v_l1 = torch.sum(weights*diff_L1) / (torch.sum(weights) + torch.finfo(torch.float16).eps)
        else:
            v_l1 = torch.sum(diff_L1)

        if(torch.any(torch.isnan(v_l1))):
            raise NotImplementedError(f"nan in L1_Loss")
            v_l1 = torch.mean(0.0 * outputs)

        return v_l1 / diff_L1.numel()

# -------------------------------------------------------------------------------------------------
# MSE loss

class MSE_Loss:
    """
    Weighted MSE loss
    """
    def __init__(self, complex_i=False):
        """
        @args:
            - complex_i (bool): whether images are 2 channelled for complex data
        """
        self.complex_i = complex_i

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            diff_mag_square = torch.square(outputs[:,:,0]-targets[:,:,0]) + torch.square(outputs[:,:,1]-targets[:,:,1])
        else:
            diff_mag_square = torch.square(outputs-targets)

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for MSE_Loss")

            v_l2 = torch.sum(weights*diff_mag_square) / (torch.sum(weights) + torch.finfo(torch.float16).eps)
        else:
            v_l2 = torch.sum(diff_mag_square)

        if(torch.any(torch.isnan(v_l2))):
            raise NotImplementedError(f"nan in MSE_Loss")
            v_l2 = torch.mean(0.0 * outputs)

        return v_l2 / diff_mag_square.numel()
    
# -------------------------------------------------------------------------------------------------
# Charbonnier Loss

class Charbonnier_Loss:
    """
    Charbonnier Loss (L1)
    """
    def __init__(self, complex_i=False, eps=1e-3):
        """
        @args:
            - complex_i (bool): whether images are 2 channelled for complex data
            - eps (float): epsilon, different values can be tried here
        """
        self.complex_i = complex_i
        self.eps = eps

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            diff_L1 = torch.abs(outputs[:,:,0]-targets[:,:,0]) + torch.abs(outputs[:,:,1]-targets[:,:,1])
        else:
            diff_L1 = torch.abs(outputs-targets)

        loss = torch.sqrt(diff_L1 * diff_L1 + self.eps * self.eps)

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for L1_Loss")

            v_l1 = torch.sum(weights*loss) / torch.sum(weights)
        else:
            v_l1 = torch.sum(loss)

        return v_l1 / loss.numel()

# -------------------------------------------------------------------------------------------------
# Perceptual Loss

class VGGPerceptualLoss(torch.nn.Module):
    """
    Perceptual Loss (VGG Loss)
    """
    def __init__(self, complex_i=False, resize=False, interpolate_mode='bilinear'):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.interpolate_mode = interpolate_mode
        self.complex_i = complex_i
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # can also try feature_layers=[2], style_layers=[0, 1, 2, 3]
    def __call__(self, outputs, targets, feature_layers=[0, 1, 2, 3], style_layers=[], weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:,:,:1]*outputs[:,:,:1] + outputs[:,:,1:]*outputs[:,:,1:])
            targets_im = torch.sqrt(targets[:,:,:1]*targets[:,:,:1] + targets[:,:,1:]*targets[:,:,1:])
        else:
            outputs_im = outputs
            targets_im = targets

        B, T, C, H, W = targets_im.shape
        outputs_im = torch.reshape(outputs_im, (B*T, C, H, W))
        targets_im = torch.reshape(targets_im, (B*T, C, H, W))

        if outputs_im.shape[1] != 3:
            outputs_im = outputs_im.repeat(1, 3, 1, 1)
            targets_im = targets_im.repeat(1, 3, 1, 1)
        outputs_im = (outputs_im-self.mean) / self.std
        targets_im = (targets_im-self.mean) / self.std
        
        if self.resize:
            outputs_im = self.transform(outputs_im, mode=self.interpolate_mode, size=(224, 224), align_corners=False)
            targets_im = self.transform(targets_im, mode=self.interpolate_mode, size=(224, 224), align_corners=False)
        
        loss = 0.0
        x = outputs_im
        y = targets_im
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        if weights is not None:
            if weights.ndim==1:
                weights_used = weights.expand(T,B).permute(1,0).reshape(B*T)
            elif weights.ndim==2:
                weights_used = weights.reshape(B*T)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights")
            v_vgg = torch.sum(weights_used*loss) / (torch.sum(weights_used) + torch.finfo(torch.float16).eps)
        else:
            v_vgg = torch.mean(loss)

        return v_vgg
    
# -------------------------------------------------------------------------------------------------
# PSNR

class PSNR:
    """
    PSNR as a comparison metric
    """
    def __init__(self, range=1.0):
        """
        @args:
            - range (float): max range of the values in the images
        """
        self.range=range

    def __call__(self, outputs, targets):

        num = self.range * self.range
        den = torch.mean(torch.square(targets - outputs)) + torch.finfo(torch.float16).eps

        return 10 * torch.log10(num/den)

class PSNR_Loss:
    """
    PSNR as a comparison metric
    """
    def __init__(self, range=1.0):
        """
        @args:
            - range (float): max range of the values in the images
        """
        self.range=range

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape

        num = self.range * self.range
        den = torch.square(targets - outputs) + torch.finfo(torch.float16).eps

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for PSNR_Loss")

            v_l2 = torch.sum(weights*torch.log10(num/den)) / (torch.sum(weights) + torch.finfo(torch.float16).eps)
        else:
            v_l2 = torch.sum(torch.log10(num/den))

        if(torch.any(torch.isnan(v_l2))):
            raise NotImplementedError(f"nan in PSNR_Loss")
            v_l2 = torch.mean(0.0 * outputs)

        return 10 - v_l2 / den.numel()

# -------------------------------------------------------------------------------------------------

def perpendicular_loss_complex(X, Y):
    """perpendicular loss for complex MR images
    
    from https://gitlab.com/computational-imaging-lab/perp_loss/-/blob/main/PerpLoss_-_Image_reconstruction.ipynb

    Args:
        X (complex images): torch complex images
        Y (complex images): torch complex images
        
    Outputs:
        final_term: the loss in the same size as X and Y
    """
    assert X.is_complex()
    assert Y.is_complex()

    mag_input = torch.abs(X)
    mag_target = torch.abs(Y)
    cross = torch.abs(X.real * Y.imag - X.imag * Y.real)

    angle = torch.atan2(X.imag, X.real) - torch.atan2(Y.imag, Y.real)
    ploss = torch.abs(cross) / (mag_input + 1e-8)

    aligned_mask = (torch.cos(angle) < 0).bool()

    final_term = torch.zeros_like(ploss)
    final_term[aligned_mask] = mag_target[aligned_mask] + (mag_target[aligned_mask] - ploss[aligned_mask])
    final_term[~aligned_mask] = ploss[~aligned_mask]
    
    return final_term

class Perpendicular_Loss:
    """
    Perpendicular loss
    """
    def __init__(self):
        """
        @args:            
        """

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape

        loss = perpendicular_loss_complex(outputs[:,:,0,:,:]+1j*outputs[:,:,1,:,:], targets[:,:,0,:,:]+1j*targets[:,:,1,:,:])

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for Perpendicular_Loss")

            v = torch.sum(weights*loss) / (torch.sum(weights) + torch.finfo(torch.float16).eps)
        else:
            v = torch.sum(loss)

        if(torch.any(torch.isnan(v))):
            raise NotImplementedError(f"nan in Perpendicular_Loss")
            v = torch.mean(0.0 * outputs)

        return v / targets.numel()

# -------------------------------------------------------------------------------------------------

class GaussianDeriv_Loss:
    """
    Weighted gaussian derivative loss for 2D
    For every sigma, the gaussian derivatives are computed for outputs and targets along the magnitude of H and W
    The l1 loss are computed to measure the agreement of gaussian derivatives
    
    If sigmas have more than one value, every sigma in sigmas are used to compute a guassian derivative tensor
    The mean l1 is returned
    """
    def __init__(self, sigmas=[0.5, 1.0, 1.25], complex_i=False, device='cpu'):
        """
        @args:
            - sigmas (list): sigma for every scale
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        self.complex_i = complex_i
        self.sigmas = sigmas

        # compute kernels
        self.kernels = []
        for sigma in sigmas:
            k_2d = create_window_2d(sigma=(sigma, sigma), halfwidth=(3, 3), voxelsize=(1.0, 1.0), order=(1,1))
            kx, ky = k_2d.shape
            k_2d = torch.from_numpy(np.reshape(k_2d, (1, 1, kx, ky))).to(torch.float32)
            self.kernels.append(k_2d.to(device=device))

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:,:,:1]*outputs[:,:,:1] + outputs[:,:,1:]*outputs[:,:,1:])
            targets_im = torch.sqrt(targets[:,:,:1]*targets[:,:,:1] + targets[:,:,1:]*targets[:,:,1:])
        else:
            outputs_im = outputs
            targets_im = targets

        B, T, C, H, W = targets_im.shape
        outputs_im = torch.reshape(outputs_im, (B*T, C, H, W))
        targets_im = torch.reshape(targets_im, (B*T, C, H, W))

        loss = 0
        for k_2d in self.kernels:
            grad_outputs_im = F.conv2d(outputs_im, k_2d, bias=None, stride=1, padding='same', groups=C)
            grad_targets_im = F.conv2d(targets_im, k_2d, bias=None, stride=1, padding='same', groups=C)
            loss += torch.mean(torch.abs(grad_outputs_im-grad_targets_im), dim=(1, 2, 3), keepdim=True)

        loss /= len(self.kernels)

        if weights is not None:

            if weights.ndim==1:
                weights_used = weights.expand(T,B).permute(1,0).reshape(B*T)
            elif weights.ndim==2:
                weights_used = weights.reshape(B*T)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for GaussianDeriv_Loss")

            v = torch.sum(weights_used*loss) / (torch.sum(weights_used) + torch.finfo(torch.float16).eps)
        else:
            v = torch.mean(loss)

        if(torch.any(torch.isnan(v))):
            raise NotImplementedError(f"nan in GaussianDeriv_Loss")
            v = torch.mean(0.0 * outputs)

        return v

# -------------------------------------------------------------------------------------------------

class GaussianDeriv3D_Loss:
    """
    Weighted gaussian derivative loss for 3D
    For every sigma, the gaussian derivatives are computed for outputs and targets along the magnitude of T, H, W
    The l1 loss are computed to measure the agreement of gaussian derivatives
    
    If sigmas have more than one value, every sigma in sigmas are used to compute a guassian derivative tensor
    The mean l1 is returned
    """
    def __init__(self, sigmas=[0.5, 1.0, 1.25], sigmas_T=[0.5, 1.0, 1.25], complex_i=False, device='cpu'):
        """
        @args:
            - sigmas (list): sigma for every scale along H and W
            - sigmas_T (list): sigma for every scale along T
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        self.complex_i = complex_i
        self.sigmas = sigmas
        self.sigmas_T = sigmas_T

        assert len(self.sigmas_T) == len(self.sigmas)

        # compute kernels
        self.kernels = []
        for sigma, sigma_T in zip(sigmas, sigmas_T):
            k_3d = create_window_3d(sigma=(sigma, sigma, sigma_T), halfwidth=(3, 3, 3), voxelsize=(1.0, 1.0, 1.0), order=(1,1,1))
            kx, ky, kz = k_3d.shape
            k_3d = torch.from_numpy(np.reshape(k_3d, (1, 1, kx, ky, kz))).to(torch.float32)
            k_3d = torch.permute(k_3d, [0, 1, 4, 2, 3])
            self.kernels.append(k_3d.to(device=device))

    def __call__(self, outputs, targets, weights=None):

        B, T, C, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:,:,:1]*outputs[:,:,:1] + outputs[:,:,1:]*outputs[:,:,1:])
            targets_im = torch.sqrt(targets[:,:,:1]*targets[:,:,:1] + targets[:,:,1:]*targets[:,:,1:])
        else:
            outputs_im = outputs
            targets_im = targets

        B, T, C, H, W = targets_im.shape
        outputs_im = torch.permute(outputs_im, (0, 2, 1, 3, 4))
        targets_im = torch.permute(targets_im, (0, 2, 1, 3, 4))

        loss = 0
        for k_3d in self.kernels:
            grad_outputs_im = F.conv3d(outputs_im, k_3d, bias=None, stride=1, padding='same', groups=C)
            grad_targets_im = F.conv3d(targets_im, k_3d, bias=None, stride=1, padding='same', groups=C)
            loss += torch.mean(torch.abs(grad_outputs_im-grad_targets_im), dim=(1, 2, 3, 4), keepdim=True)

        loss /= len(self.kernels)

        if weights is not None:
            if not weights.ndim==1:
                raise NotImplementedError(f"Only support 1D(Batch) weights for GaussianDeriv3D_Loss")
            v = torch.sum(weights*loss) / (torch.sum(weights) + torch.finfo(torch.float16).eps)
        else:
            v = torch.mean(loss)

        if(torch.any(torch.isnan(v))):
            raise NotImplementedError(f"nan in GaussianDeriv3D_Loss")
            v = torch.mean(0.0 * outputs)

        return v

# -------------------------------------------------------------------------------------------------
# Combined loss class

class Combined_Loss:
    """
    Combined loss for image enhancement
    Sums multiple loss with their respective weights
    """
    def __init__(self, losses, loss_weights, complex_i=False, device="cpu") -> None:
        """
        @args:
            - losses (list of "ssim", "ssim3D", "l1", "mse"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        assert len(losses)>0, f"At least one loss is required to setup"
        assert len(losses)<=len(loss_weights), f"Each loss should have its weight"

        self.complex_i = complex_i
        self.device = device

        losses = [self.str_to_loss(loss) for loss in losses]
        self.losses = list(zip(losses, loss_weights))

    def str_to_loss(self, loss_name):

        if loss_name=="mse":
            loss_f = MSE_Loss(complex_i=self.complex_i)
        elif loss_name=="l1":
            loss_f = L1_Loss(complex_i=self.complex_i)
        elif loss_name=="charbonnier":
            loss_f = Charbonnier_Loss(complex_i=self.complex_i)
        elif loss_name=="perceptual":
            loss_f = VGGPerceptualLoss(complex_i=self.complex_i)
            loss_f.to(self.device)
        elif loss_name=="ssim":
            loss_f = SSIM_Loss(window_size=5, complex_i=self.complex_i, device=self.device)
        elif loss_name=="ssim3D":
            loss_f = SSIM3D_Loss(window_size=5, complex_i=self.complex_i, device=self.device)
        elif loss_name=="psnr":
            loss_f = PSNR_Loss(range=2048.0)
        elif loss_name=="perpendicular":
            loss_f = Perpendicular_Loss()
        elif loss_name=="msssim":
            loss_f = MSSSIM_Loss(window_size=3, complex_i=self.complex_i, data_range=256, device=self.device)
        elif loss_name=="gaussian":
            loss_f = GaussianDeriv_Loss(sigmas=[0.25, 0.5, 1.0, 1.5], complex_i=self.complex_i, device=self.device)
        elif loss_name=="gaussian3D":
            loss_f = GaussianDeriv3D_Loss(sigmas=[0.25, 0.5, 1.0], sigmas_T=[0.25, 0.5, 0.5], complex_i=self.complex_i, device=self.device)
        else:
            raise NotImplementedError(f"Loss type not implemented: {loss_name}")

        return loss_f
    
    def __call__(self, outputs, targets, weights=None):

        #combined_loss = torch.mean(0.0 * outputs)
        combined_loss = 0
        for loss_f, weight in self.losses:
            v = weight*loss_f(outputs=outputs, targets=targets, weights=weights)
            if not torch.isnan(v):
                combined_loss += v

        # combined_loss = sum([weight*loss_f(outputs=outputs, targets=targets, weights=weights) \
        #                         for loss_f, weight in self.losses])
        
        return combined_loss

# -------------------------------------------------------------------------------------------------
 
def tests():

    device = get_device()

    import numpy as np

    Project_DIR = Path(__file__).parents[1].resolve()

    clean_a = np.load(os.path.join(Project_DIR, 'data/microscopy/clean1.npy'))
    clean_b = np.load(os.path.join(Project_DIR, 'data/microscopy/clean2.npy'))
    noisy_a = np.load(os.path.join(Project_DIR, 'data/microscopy/noisy.npy'))

    H, W = clean_a.shape
    clean_a = torch.from_numpy(clean_a.reshape((1, 1, 1, H ,W)))
    clean_b = torch.from_numpy(clean_b.reshape((1, 1, 1, H ,W)))
    noisy_a = torch.from_numpy(noisy_a.reshape((1, 1, 1, H ,W)))

    B,T,C,H,W = 4,8,1,64,64

    im_1 = torch.rand(B,T,C,H,W)
    im_2 = torch.rand(B,T,C,H,W)
    im_3 = im_2.clone()
    im_3[0,0,0,0,0]+=0.1
    wt_1 = torch.rand(B)
    wt_2 = torch.rand(B,T)

    fsim_loss_f = FSIM_Loss()
    fsim_1 = fsim_loss_f(im_1, im_1)
    assert fsim_1==0

    fsim_2 = fsim_loss_f(im_2, im_2)
    assert fsim_2==0

    fsim_3 = fsim_loss_f(im_1, im_2)
    assert 0<=fsim_3<=1

    fsim_4 = fsim_loss_f(im_2, im_1)
    assert 0<=fsim_4<=1

    assert fsim_3==fsim_4

    f1 = fsim_loss_f(clean_a, clean_b)
    assert 0<=f1<=1

    f2 = fsim_loss_f(clean_a, noisy_a)
    assert 0<=f2<=1

    assert f2>f1

    print("Passed fsim")    

    ssim_loss_f = SSIM_Loss()

    ssim_1 = ssim_loss_f(im_1, im_1)
    assert ssim_1==0

    ssim_2 = ssim_loss_f(im_2, im_2)
    assert ssim_2==0

    ssim_3 = ssim_loss_f(im_1, im_2)
    assert 0<=ssim_3<=1

    ssim_4 = ssim_loss_f(im_2, im_1)
    assert 0<=ssim_4<=1

    assert ssim_3==ssim_4

    print("Passed ssim2D")

    ssim_1 = ssim_loss_f(im_1, im_1, weights=wt_1)
    assert ssim_1==0

    ssim_2 = ssim_loss_f(im_2, im_2, weights=wt_1)
    assert ssim_2==0

    ssim_3 = ssim_loss_f(im_1, im_2, weights=wt_2)
    assert 0<=ssim_3<=1

    ssim_4 = ssim_loss_f(im_2, im_1, weights=wt_2)
    assert 0<=ssim_4<=1

    assert ssim_3==ssim_4

    print("Passed ssim2D weighted")

    ssim3d_loss_f = SSIM3D_Loss()

    ssim_5 = ssim3d_loss_f(im_1, im_1)
    assert ssim_5==0

    ssim_6 = ssim3d_loss_f(im_2, im_2)
    assert ssim_6==0

    ssim_7 = ssim3d_loss_f(im_1, im_2)
    assert 0<=ssim_7<=1

    ssim_8 = ssim3d_loss_f(im_2, im_1)
    assert 0<=ssim_8<=1

    assert ssim_7==ssim_8

    print("Passed ssim3D")

    ssim_5 = ssim3d_loss_f(im_1, im_1, weights=wt_1)
    assert ssim_5==0

    ssim_6 = ssim3d_loss_f(im_2, im_2, weights=wt_1)
    assert ssim_6==0

    ssim_7 = ssim3d_loss_f(im_1, im_2, weights=wt_1)
    assert 0<=ssim_7<=1

    ssim_8 = ssim3d_loss_f(im_2, im_1, weights=wt_1)
    assert 0<=ssim_8<=1

    assert ssim_7==ssim_8

    print("Passed ssim3D weighted")

    l1_loss_f = L1_Loss()
    l1_loss_f_t = torch.nn.L1Loss()

    l1_1 = l1_loss_f(im_1, im_1)
    assert l1_1==0

    l1_2 = l1_loss_f(im_2, im_2)
    assert l1_2==0

    l1_3 = l1_loss_f(im_1, im_2)
    assert l1_3>=0

    l1_4 = l1_loss_f(im_2, im_3)
    l1_5 = l1_loss_f_t(im_2, im_3)
    assert l1_4==l1_5

    print("Passed L1")

    l1_loss_f = L1_Loss()

    l1_1 = l1_loss_f(im_1, im_1, weights=wt_1)
    assert l1_1==0

    l1_2 = l1_loss_f(im_2, im_2, weights=wt_2)
    assert l1_2==0

    l1_3 = l1_loss_f(im_1, im_2, weights=wt_2)
    assert l1_3>=0

    print("Passed L1 Weighted")

    mse_loss_f = MSE_Loss()
    mse_loss_f_t = torch.nn.MSELoss()

    mse_1 = mse_loss_f(im_1, im_1)
    assert mse_1==0

    mse_2 = mse_loss_f(im_2, im_2)
    assert mse_2==0

    mse_3 = mse_loss_f(im_1, im_2)
    assert mse_3>=0

    mse_4 = mse_loss_f(im_2, im_3)
    mse_5 = mse_loss_f_t(im_2, im_3)
    assert mse_4==mse_5

    print("Passed MSE")

    mse_1 = mse_loss_f(im_1, im_1, weights=wt_1)
    assert mse_1==0

    mse_2 = mse_loss_f(im_2, im_2, weights=wt_2)
    assert mse_2==0

    mse_3 = mse_loss_f(im_1, im_2, weights=wt_2)
    assert mse_3>=0

    print("Passed MSE weighted")

    psnr_f = PSNR(range=1.0)

    psnr_1 = psnr_f(im_1, im_1)
    assert psnr_1==torch.inf

    psnr_2 = psnr_f(im_2, im_3)
    assert psnr_2>=50

    psnr_3 = psnr_f(im_2, im_3)
    assert psnr_3>=50

    assert psnr_2==psnr_3

    print("Passed PSNR")

    combined_l_f = Combined_Loss(["mse", "l1", "ssim", "ssim3D", "msssim"], [1.0, 1.0, 1.0, 1.0, 1.0], complex_i=False)
    
    c_loss_1 = combined_l_f(im_2, im_3)
    assert c_loss_1>0

    print("Passed Combined Loss")


    noisy = np.load(str(Project_DIR) + '/data/loss/noisy_real.npy') + 1j * np.load(str(Project_DIR) + '/data/loss/noisy_imag.npy')
    print(noisy.shape)

    clean = np.load(str(Project_DIR) + '/data/loss/clean_real.npy') + 1j * np.load(str(Project_DIR) + '/data/loss/clean_imag.npy')
    print(clean.shape)

    pred = np.load(str(Project_DIR) + '/data/loss/pred_real.npy') + 1j * np.load(str(Project_DIR) + '/data/loss/pred_imag.npy')
    print(pred.shape)

    RO, E1, PHS, N = noisy.shape

    for k in range(N):
        perp_loss = Perpendicular_Loss()

        x = np.zeros((1, PHS, 2, RO, E1))
        y = np.zeros((1, PHS, 2, RO, E1))

        x[:,:,0,:,:] = np.transpose(np.real(noisy[:,:,:,k]), (2, 0, 1))
        x[:,:,1,:,:] = np.transpose(np.imag(noisy[:,:,:,k]), (2, 0, 1))
        y[:,:,0,:,:] = np.transpose(np.real(clean[:,:,:,k]), (2, 0, 1))
        y[:,:,1,:,:] = np.transpose(np.imag(clean[:,:,:,k]), (2, 0, 1))

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        v = perp_loss(x, y)

        print(f"sigma {k+1} - perp - {v}")
       
    # -----------------------------------------------------------------
    
    # further test ssim, msssim and perp loss
    noisy = np.load(str(Project_DIR) + '/data/loss/noisy.npy')
    print(noisy.shape)

    clean = np.load(str(Project_DIR) + '/data/loss/clean.npy')
    print(clean.shape)

    pred = np.load(str(Project_DIR) + '/data/loss/pred.npy')
    print(pred.shape)

    RO, E1, PHS, N = noisy.shape

    print("-----------------------")

    msssim_loss = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(kernel_size=5, reduction=None, data_range=256)
    msssim_loss.to(device=device)

    # 2D ssim
    x = torch.from_numpy(noisy[:,:,0,:]).to(device=device)
    y = torch.from_numpy(clean[:,:,0,:]).to(device=device)

    x = torch.permute(x, (2, 0, 1)).reshape([N, 1, RO, E1])
    y = torch.permute(y, (2, 0, 1)).reshape([N, 1, RO, E1])
    v = msssim_loss(x, y)
    print(f"sigma 1 to 10 - mssim - {v}")
        
    # 3D ssim
    x = torch.permute(torch.from_numpy(noisy), (3, 2, 0, 1))
    y = torch.permute(torch.from_numpy(clean), (3, 2, 0, 1))
    v = msssim_loss(x, y)
    print(f"sigma 1 to 10 - mssim - {v}")

    print("-----------------------")

    # loss code
    x = torch.permute(torch.from_numpy(noisy), (3, 2, 0, 1)).reshape((N, PHS, 1, RO, E1)).to(device=device)
    y = torch.permute(torch.from_numpy(clean), (3, 2, 0, 1)).reshape((N, PHS, 1, RO, E1)).to(device=device)
   
    msssim_loss = MSSSIM_Loss(window_size=3, data_range=128, device=device, complex_i=False)
    
    for k in range(N):
        v = msssim_loss(torch.unsqueeze(x[k], dim=0), torch.unsqueeze(y[k], dim=0), weights=torch.ones(1, device=device))
        print(f"msssim loss - {v}")

    print("-----------------------")

    # loss code
    x = torch.permute(torch.from_numpy(noisy), (3, 2, 0, 1)).reshape((N, PHS, 1, RO, E1)).to(device=device)
    y = torch.permute(torch.from_numpy(clean), (3, 2, 0, 1)).reshape((N, PHS, 1, RO, E1)).to(device=device)

    gauss_loss = GaussianDeriv_Loss(sigmas=[0.25, 0.5, 1.0, 1.5], device=device, complex_i=False)

    for k in range(N):
        v = gauss_loss(torch.unsqueeze(x[k], dim=0), torch.unsqueeze(y[k], dim=0), weights=torch.ones(1, device=device))
        print(f"gauss loss - {v}")

    print("-----------------------")

    # loss code
    x = torch.permute(torch.from_numpy(noisy), (3, 2, 0, 1)).reshape((N, PHS, 1, RO, E1)).to(device=device)
    y = torch.permute(torch.from_numpy(clean), (3, 2, 0, 1)).reshape((N, PHS, 1, RO, E1)).to(device=device)

    gauss_loss = GaussianDeriv3D_Loss(sigmas=[0.25, 0.5, 1.0, 1.25], sigmas_T=[0.25, 0.5, 0.5, 0.5], device=device, complex_i=False)

    for k in range(N):
        v = gauss_loss(torch.unsqueeze(x[k], dim=0), torch.unsqueeze(y[k], dim=0), weights=torch.ones(1, device=device))
        print(f"gauss 3D loss - {v}")

    print("Passed all tests")

if __name__=="__main__":
    tests()
