"""
Implemntation of gaussian, gaussian derivatives etc.
"""

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ['create_window_2d', 'create_window_3d']

# -------------------------------------------------------------------------------------------------

def get_gaussionand_derivatives_1D(sigma, halfwidth, voxelsize):
    """compute guassian kernels

    Args:
        sigma (float): sigma in the unit of physical world
        halfwidth (float): sampled halfwidth
        voxelsize (float): voxel size, in the same unit of sigma

    Returns:
        kernelSamplePoints, Gx, Dx, Dxx: sampled locations, gaussian and its derivatives
    """
    s = np.arange( 2 * round(halfwidth*sigma/voxelsize) + 1)
    kernelSamplePoints=(s-round(halfwidth*sigma/voxelsize))*voxelsize

    Gx, Dx, Dxx = gaussian_fucntion(kernelSamplePoints, sigma)

    return kernelSamplePoints, Gx, Dx, Dxx

# -------------------------------------------------------------------------------------------------

def gaussian_fucntion(kernelSamplePoints, sigma):
    """compute gaussian and its derivatives

    Args:
        kernelSamplePoints (np array): sampled kernal points
        sigma (float): guassian sigma

    Returns:
        G, D, DD: guassian kernel, guassian 1st and 2nd order derivatives
    """

    N = 1/np.sqrt(2*np.pi*sigma*sigma)
    T = np.exp(-(kernelSamplePoints*kernelSamplePoints)/(2*sigma*sigma))

    G = N * T
    G = G / np.sum(G)

    D = N * (-kernelSamplePoints / (sigma*sigma)) * T
    D = D / np.sum(np.abs(D))

    DD = N * ((-1/(sigma*sigma)*T) + ((-kernelSamplePoints / (sigma*sigma)) * (-kernelSamplePoints / (sigma*sigma)) * T))
    DD = DD / np.sum(np.abs(DD))

    return G, D, DD

# -------------------------------------------------------------------------------------------------

def create_window_2d(sigma=(1.25, 1.25), halfwidth=(3, 3), voxelsize=(1.0, 1.0), order=(1,1)):
    """
    Creates a 2D gauss kernel
    """
    k_0 = get_gaussionand_derivatives_1D(sigma[0], halfwidth[0], voxelsize[0])
    k_1 = get_gaussionand_derivatives_1D(sigma[1], halfwidth[1], voxelsize[1])
    window = k_0[order[0]+1][:, np.newaxis] * k_1[order[1]+1][:, np.newaxis].T

    return window

# -------------------------------------------------------------------------------------------------

def create_window_3d(sigma=(1.25, 1.25, 1.25), halfwidth=(3, 3, 3), voxelsize=(1.0, 1.0, 1.0), order=(1,1,1)):
    """
    Creates a 3D gauss kernel
    """
    k_0 = get_gaussionand_derivatives_1D(sigma[0], halfwidth[0], voxelsize[0])
    k_1 = get_gaussionand_derivatives_1D(sigma[1], halfwidth[1], voxelsize[1])
    k_2 = get_gaussionand_derivatives_1D(sigma[2], halfwidth[2], voxelsize[2])
    window = k_0[order[0]+1][:, np.newaxis] * k_1[order[1]+1][:, np.newaxis].T

    window = window[:, :, np.newaxis] * np.expand_dims(k_2[order[2]+1], axis=(0,1))

    return window

# -------------------------------------------------------------------------------------------------
# Testing

def tests():

    kernelSamplePoints, G, D, DD = get_gaussionand_derivatives_1D(sigma=1.25, halfwidth=3.0, voxelsize=1.0)

    print(kernelSamplePoints)
    print(G)
    print(D)
    print(DD)

    kernel_2d = create_window_2d(sigma=(1.25, 3), halfwidth=(3, 4), voxelsize=(1.0, 2.0), order=(1,1))
    print(f"2D kernel \n {kernel_2d}")

    kernel_3d = create_window_3d(sigma=(1.25, 2.25, 1.0), halfwidth=(3, 4, 3), voxelsize=(1.0, 2.0, 1.0), order=(1,1,1))
    print(f"3D kernel \n {kernel_3d}")

    import os
    import sys
    import nibabel as nib
    from pathlib import Path

    Project_DIR = Path(__file__).parents[1].resolve()
    sys.path.insert(1, str(Project_DIR))

    noisy = np.load(str(Project_DIR) + '/data/loss/noisy_real.npy') + 1j * np.load(str(Project_DIR) + '/data/loss/noisy_imag.npy')
    print(noisy.shape)

    clean = np.load(str(Project_DIR) + '/data/loss/clean_real.npy') + 1j * np.load(str(Project_DIR) + '/data/loss/clean_imag.npy')
    print(clean.shape)

    pred = np.load(str(Project_DIR) + '/data/loss/pred_real.npy') + 1j * np.load(str(Project_DIR) + '/data/loss/pred_imag.npy')
    print(pred.shape)

    # -------------------------------------------------

    im = np.abs(clean[:,:,:,0])
    im = torch.from_numpy(im)
    im = torch.permute(im, (2, 0, 1)).unsqueeze(dim=1)
    print(im.shape)

    kx, ky = kernel_2d.shape

    k_2d = torch.from_numpy(np.reshape(kernel_2d, (1, 1, kx, ky))).to(torch.float32)
    print(k_2d)

    grad_im = F.conv2d(im, k_2d, bias=None, stride=1, padding=(kx//2, ky//2), groups=1)
    print(grad_im.shape)

    grad_im = grad_im.numpy().squeeze().transpose((1, 2, 0))
    #res_dir = os.path.join(str(Project_DIR), 'data', 'loss', 'res')
    res_dir = os.path.join('/export/Lab-Xue/projects/mri', 'loss', 'res')
    os.makedirs(res_dir, exist_ok=True)

    np.save(os.path.join(res_dir, 'grad_im.npy'), grad_im)
    nib.save(nib.Nifti1Image(grad_im, affine=np.eye(4)), os.path.join(res_dir, 'grad_im.nii'))
    nib.save(nib.Nifti1Image(clean[:,:,:,0], affine=np.eye(4)), os.path.join(res_dir, 'clean.nii'))

    # -------------------------------------------------

    im = np.abs(clean[:,:,:,0])
    im = torch.from_numpy(im)
    H, W, T = im.shape
    im = torch.permute(im, (2, 0, 1)).view((1, 1, T, H, W))
    print(im.shape)

    kx, ky, kz = kernel_3d.shape
    print(kernel_3d.shape)

    k_3d = torch.from_numpy(np.reshape(kernel_3d, (1, 1, kx, ky, kz))).to(torch.float32)
    print(k_3d.shape)
    k_3d = torch.permute(k_3d, [0, 1, 4, 2, 3])

    grad_im = F.conv3d(im, k_3d, bias=None, stride=1, padding='same', groups=1)
    print(grad_im.shape)

    grad_im = grad_im.numpy().squeeze().transpose((1, 2, 0))
    #res_dir = os.path.join(str(Project_DIR), 'data', 'loss', 'res')
    res_dir = os.path.join('/export/Lab-Xue/projects/mri', 'loss', 'res')
    os.makedirs(res_dir, exist_ok=True)

    np.save(os.path.join(res_dir, 'grad_im_3d.npy'), grad_im)
    nib.save(nib.Nifti1Image(grad_im, affine=np.eye(4)), os.path.join(res_dir, 'grad_im_3d.nii'))

if __name__=="__main__":
    tests()
