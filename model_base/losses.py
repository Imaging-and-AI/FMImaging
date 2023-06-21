"""
Losses and Metrics

Provides implmentation of the following types of losses:
    - SSIM: Structural Similarity Index Measure for 2D
    - SSIM3D: Structural Similarity Index Measure for 3D
    - L1: Mean Absolute Error
    - MSE: Mean Squared Error
    - Combined: Any weighed combination of the above

Allows custom weights for each indvidual loss calculation as well
"""

import os
import sys
import torch
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.pytorch_ssim import SSIM, SSIM3D
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
            
            v = torch.sum(weights_used*loss) / torch.sum(weights_used)
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

            v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights_used)
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
            v_ssim = torch.sum(weights*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = torch.tensor(1.0, requires_grad=True)

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

            v_l1 = torch.sum(weights*diff_L1) / torch.sum(weights)
        else:
            v_l1 = torch.sum(diff_L1)

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

            v_l2 = torch.sum(weights*diff_mag_square) / torch.sum(weights)
        else:
            v_l2 = torch.sum(diff_mag_square)

        return v_l2 / diff_mag_square.numel()

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
        den = torch.mean(torch.square(targets - outputs))

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
        den = torch.square(targets - outputs) + 1e-8

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for PSNR_Loss")

            v_l2 = torch.sum(weights*torch.log10(num/den)) / torch.sum(weights)
        else:
            v_l2 = torch.sum(torch.log10(num/den))

        return 10 - v_l2 / den.numel()

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
        elif loss_name=="ssim":
            loss_f = SSIM_Loss(window_size=11, complex_i=self.complex_i, device=self.device)
        elif loss_name=="ssim3D":
            loss_f = SSIM3D_Loss(window_size=11, complex_i=self.complex_i, device=self.device)
        elif loss_name=="psnr":
            loss_f = PSNR_Loss(range=2048.0)
        else:
            raise NotImplementedError(f"Loss type not implemented: {loss_name}")

        return loss_f
    
    def __call__(self, outputs, targets, weights=None):

        combined_loss = sum([weight*loss_f(outputs=outputs, targets=targets, weights=weights) \
                                for loss_f, weight in self.losses])
        
        return combined_loss

# -------------------------------------------------------------------------------------------------
 
def tests():

    import numpy as np

    Project_DIR = Path(__file__).parents[1].resolve()
    
    clean_a = np.load(os.path.join(Project_DIR, 'data/microscopy/clean1.npy'))
    clean_b = np.load(os.path.join(Project_DIR, 'data/microscopy/clean2.npy'))
    noisy_a = np.load(os.path.join(Project_DIR, 'data/microscopy/noisy.npy'))

    H, W = clean_a.shape
    clean_a = torch.from_numpy(clean_a.reshape((1, 1, 1, H ,W)))
    clean_b = torch.from_numpy(clean_b.reshape((1, 1, 1, H ,W)))
    noisy_a = torch.from_numpy(noisy_a.reshape((1, 1, 1, H ,W)))

    B,T,C,H,W = 4,8,3,32,32

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

    combined_l_f = Combined_Loss(["mse", "l1", "ssim", "ssim3D"], [1.0,1.0,1.0,1.0])
    
    c_loss_1 = combined_l_f(im_2, im_3)
    assert c_loss_1>0

    print("Passed Combined Loss")

    print("Passed all tests")

if __name__=="__main__":
    tests()
