"""
Utilities functions for tasks and projects
"""
import os
import cv2
import wandb
import torch
import logging
import argparse
import tifffile
import numpy as np

from collections import OrderedDict
from skimage.util import view_as_blocks

from datetime import datetime
from torchinfo import summary

import torch.distributed as dist
from colorama import Fore, Style
import nibabel as nib

# -------------------------------------------------------------------------------------------------
# 2D image patch and repatch

def cut_into_patches(image_list, cutout):
    """
    Cuts a 2D image into non-overlapping patches.
    Assembles patches in the time dimension
    Pads up to the required length by symmetric padding.
    @args:
        - image_list (5D torch.Tensor list): list of image to cut
        - cutout (2-tuple): the 2D cutout shape of each patch
    @reqs:
        - all images should have the same shape besides 3rd dimension (C)
        - 1st and 2nd dimension should be 1 (B,T)
    @rets:
        - final_image_list (5D torch.Tensor list): patch images
        - original_shape (5-tuple): the original shape of the image
        - patch_shape (10-tuple): the shape of patch array
    """
    original_shape = image_list[0].shape

    final_image_list = []

    for image in image_list:
        assert image.ndim==5, f"Image should have 5 dimensions"
        assert image.shape[0]==1, f"Batch size should be 1"
        assert image.shape[1]==1, f"Time size should be 1"
        assert image.shape[-2:]==original_shape[-2:], f"Image should have the same H,W"
        final_image_list.append(image.numpy(force=True))

    B,T,C,H,W = image_list[0].shape
    pad_H = (-1*H)%cutout[0]
    pad_W = (-1*W)%cutout[1]
    pad_shape = ((0,0),(0,0),(0,0),(0,pad_H),(0,pad_W))

    for i, image in enumerate(final_image_list):
        C = image.shape[2]
        image_i = np.pad(image, pad_shape, 'symmetric')
        image_i = view_as_blocks(image_i, (1,1,C,*cutout))
        patch_shape = image_i.shape
        image_i = image_i.reshape(1,-1,*image_i.shape[-3:])
        final_image_list[i] = torch.from_numpy(image_i)

    return final_image_list, original_shape, patch_shape

# -------------------------------------------------------------------------------------------------

def repatch(image_list, original_shape, patch_shape):
    """
    Reassembles the patched image into the complete image
    Assumes the patches are assembled in the time dimension
    @args:
        - image_list (5D torch.Tensor list): patched images
        - original_shape (5-tuple): the original shape of the images
        - patch_shape (10-tuple): the shape of image as patches
    @reqs:
        - 1st dimension to be 1 (B)
    @rets:
        - final_image_list (5D torch.Tensor list): repatched images
    """
    HO,WO = original_shape[-2:]
    H = patch_shape[3]*patch_shape[8]
    W = patch_shape[4]*patch_shape[9]

    final_image_list = []

    for image in image_list:
        C = image.shape[-3]
        patch_shape_x = (*patch_shape[:-3],C,*patch_shape[-2:])
        image = image.reshape(patch_shape_x)
        image = image.permute(0,5,1,6,2,7,3,8,4,9)
        image = image.reshape(1,1,C,H,W)[:,:,:,:HO,:WO]
        final_image_list.append(image)

    return final_image_list

# -------------------------------------------------------------------------------------------------

def save_image_local(path, complex_i, name, noisy, predi, clean):
    """
    Saves the image locally as a 4D tiff [T,C,H,W]
    3 channels: noisy, predicted, clean
    If complex image then save the magnitude using first 2 channels
    Else use just the first channel
    @args:
        - path (str): the directory to save the images in
        - complex_i (bool): complex images or not
        - i (int): index of the image
        - noisy (5D numpy array): the noisy image
        - predi (5D numpy array): the predicted image
        - clean (5D numpy array): the clean image
    """

    if complex_i:
        save_x = np.sqrt(np.square(noisy[0,:,0]) + np.square(noisy[0,:,1]))
        save_p = np.sqrt(np.square(predi[0,:,0]) + np.square(predi[0,:,1]))
        save_y = np.sqrt(np.square(clean[0,:,0]) + np.square(clean[0,:,1]))
    else:
        save_x = noisy[0,:,0]
        save_p = predi[0,:,0]
        save_y = clean[0,:,0]

    composed_channel_wise = np.transpose(np.array([save_x, save_p, save_y]), (1,0,2,3))

    tifffile.imwrite(os.path.join(path, f"{name}.tif"),\
                        composed_channel_wise, imagej=True)

# -------------------------------------------------------------------------------------------------

def save_image_wandb(title, complex_i, noisy, predi, clean):
    """
    Logs the image to wandb as a 4D gif [T,C,H,3*W]
    3 width: noisy, predicted, clean
    If complex image then save the magnitude using first 2 channels
    Else use just the first channel
    @args:
        - title (str): title to log image with
        - complex_i (bool): complex images or not
        - noisy (5D numpy array): the noisy image
        - predi (5D numpy array): the predicted image
        - clean (5D numpy array): the clean image
    """

    if complex_i:
        save_x = np.sqrt(np.square(noisy[0,:,0]) + np.square(noisy[0,:,1]))
        save_p = np.sqrt(np.square(predi[0,:,0]) + np.square(predi[0,:,1]))
        save_y = np.sqrt(np.square(clean[0,:,0]) + np.square(clean[0,:,1]))
    else:
        save_x = noisy[0,:,0]
        save_p = predi[0,:,0]
        save_y = clean[0,:,0]

    if save_x.ndim==2:
        save_x = np.expand_dims(save_x, axis=0)
        save_p = np.expand_dims(save_p, axis=0)
        save_y = np.expand_dims(save_y, axis=0)
        
    T, H, W = save_x.shape
    composed_res = np.zeros((T, H, 3*W))
    composed_res[:,:H,0*W:1*W] = save_x
    composed_res[:,:H,1*W:2*W] = save_p
    composed_res[:,:H,2*W:3*W] = save_y

    composed_res = np.clip(composed_res, a_min=0.5*np.mean(composed_res), a_max=0.85*np.mean(composed_res))

    temp = np.zeros_like(composed_res)
    composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

    wandbvid = wandb.Video(composed_res[:,np.newaxis,:,:].astype('uint8'), fps=8, format="gif")
    wandb.log({title: wandbvid})

# -------------------------------------------------------------------------------------------------

def save_image_batch(complex_i, noisy, predi, clean):
    """
    Logs the image to wandb as a 5D gif [B,T,C,H,W]
    If complex image then save the magnitude using first 2 channels
    Else use just the first channel
    @args:
        - complex_i (bool): complex images or not
        - noisy (5D numpy array): the noisy image [B, T, C+1, H, W]
        - predi (5D numpy array): the predicted image [B, T, C, H, W]
        - clean (5D numpy array): the clean image [B, T, C, H, W]
    """

    if noisy.ndim == 4:
        noisy = np.expand_dims(noisy, axis=0)
        predi = np.expand_dims(predi, axis=0)
        clean = np.expand_dims(clean, axis=0)

    if complex_i:
        save_x = np.sqrt(np.square(noisy[:,:,0,:,:]) + np.square(noisy[:,:,1,:,:]))
        save_p = np.sqrt(np.square(predi[:,:,0,:,:]) + np.square(predi[:,:,1,:,:]))
        save_y = np.sqrt(np.square(clean[:,:,0,:,:]) + np.square(clean[:,:,1,:,:]))
    else:
        save_x = noisy[:,:,0,:,:]
        save_p = predi[:,:,0,:,:]
        save_y = clean[:,:,0,:,:]
       
    B, T, H, W = save_x.shape

    max_col = 16
    if B>max_col:
        num_row = B//max_col
        if max_col*num_row < B: 
            num_row += 1
        composed_res = np.zeros((T, 3*H*num_row, max_col*W))
        for b in range(B):
            r = b//max_col
            c = b - r*max_col
            for t in range(T):
                composed_res[t, 3*r*H:(3*r+1)*H, c*W:(c+1)*W] = save_x[b,t,:,:].squeeze()
                composed_res[t, (3*r+1)*H:(3*r+2)*H, c*W:(c+1)*W] = save_p[b,t,:,:].squeeze()
                composed_res[t, (3*r+2)*H:(3*r+3)*H, c*W:(c+1)*W] = save_y[b,t,:,:].squeeze()
    elif B>2:
        composed_res = np.zeros((T, 3*H, B*W))
        for b in range(B):
            for t in range(T):
                composed_res[t, :H, b*W:(b+1)*W] = save_x[b,t,:,:].squeeze()
                composed_res[t, H:2*H, b*W:(b+1)*W] = save_p[b,t,:,:].squeeze()
                composed_res[t, 2*H:3*H, b*W:(b+1)*W] = save_y[b,t,:,:].squeeze()
    else:
        composed_res = np.zeros((T, B*H, 3*W))
        for b in range(B):
            for t in range(T):
                composed_res[t, b*H:(b+1)*H, :W] = save_x[b,t,:,:].squeeze()
                composed_res[t, b*H:(b+1)*H, W:2*W] = save_p[b,t,:,:].squeeze()
                composed_res[t, b*H:(b+1)*H, 2*W:3*W] = save_y[b,t,:,:].squeeze()

    composed_res = np.clip(composed_res, a_min=0.5*np.median(composed_res), a_max=np.percentile(composed_res, 90))

    temp = np.zeros_like(composed_res)
    composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

    return np.repeat(composed_res[:,np.newaxis,:,:].astype('uint8'), 3, axis=1)

def save_inference_results(input, output, gmap, output_dir, noisy_image=None):

    os.makedirs(output_dir, exist_ok=True)

    if input is not None:
        if np.any(np.iscomplex(input)):
            res_name = os.path.join(output_dir, 'input_real.npy')
            print(res_name)
            np.save(res_name, input.real)
            nib.save(nib.Nifti1Image(input.real, affine=np.eye(4)), os.path.join(output_dir, 'input_real.nii'))

            res_name = os.path.join(output_dir, 'input_imag.npy')
            print(res_name)
            np.save(res_name, input.imag)
            nib.save(nib.Nifti1Image(input.imag, affine=np.eye(4)), os.path.join(output_dir, 'input_imag.nii'))

            input = np.abs(input)

        res_name = os.path.join(output_dir, 'input.npy')
        print(res_name)
        np.save(res_name, input)
        nib.save(nib.Nifti1Image(input, affine=np.eye(4)), os.path.join(output_dir, 'input.nii'))

    if gmap is not None:
        res_name = os.path.join(output_dir, 'gfactor.npy')
        print(res_name)
        np.save(res_name, gmap)
        nib.save(nib.Nifti1Image(gmap, affine=np.eye(4)), os.path.join(output_dir, 'gfactor.nii'))

    if output is not None:
        if np.any(np.iscomplex(output)):
            res_name = os.path.join(output_dir, 'output_real.npy')
            print(res_name)
            np.save(res_name, output.real)
            nib.save(nib.Nifti1Image(output.real, affine=np.eye(4)), os.path.join(output_dir, 'output_real.nii'))

            res_name = os.path.join(output_dir, 'output_imag.npy')
            print(res_name)
            np.save(res_name, output.imag)
            nib.save(nib.Nifti1Image(output.imag, affine=np.eye(4)), os.path.join(output_dir, 'output_imag.nii'))

            output = np.abs(output)

        res_name = os.path.join(output_dir, 'output.npy')
        print(res_name)
        np.save(res_name, output)
        nib.save(nib.Nifti1Image(output, affine=np.eye(4)), os.path.join(output_dir, 'output.nii'))
        
    if noisy_image is not None:
        if np.any(np.iscomplex(noisy_image)):
            res_name = os.path.join(output_dir, 'noisy_image_real.npy')
            print(res_name)
            np.save(res_name, output.real)
            nib.save(nib.Nifti1Image(noisy_image.real, affine=np.eye(4)), os.path.join(output_dir, 'noisy_image_real.nii'))

            res_name = os.path.join(output_dir, 'noisy_image_imag.npy')
            print(res_name)
            np.save(res_name, noisy_image.imag)
            nib.save(nib.Nifti1Image(noisy_image.imag, affine=np.eye(4)), os.path.join(output_dir, 'noisy_image_imag.nii'))

            noisy_image = np.abs(noisy_image)

        res_name = os.path.join(output_dir, 'noisy_image.npy')
        print(res_name)
        np.save(res_name, noisy_image)
        nib.save(nib.Nifti1Image(noisy_image, affine=np.eye(4)), os.path.join(output_dir, 'noisy_image.nii'))

# -------------------------------------------------------------------------------------------------

def normalize_image(image, percentiles=None, values=None, clip=True, clip_vals=[0,1]):
    """
    Normalizes image locally.
    @args:
        - image (numpy.ndarray or torch.tensor): the image to normalize
        - percentiles (2-tuple int or float within [0,100]): pair of percentiles to normalize with
        - values (2-tuple int or float): pair of values normalize with
        - clip (bool): whether to clip the image or not
        - clip_vals (2-tuple int or float): values to clip with
    @reqs:
        - only one of percentiles and values is required
    @return:
        - n_img (numpy.ndarray or torch.tensor): the image normalized wrt given params
            same type as the input image
    """
    assert (percentiles==None and values!=None) or (percentiles!=None and values==None)

    if type(image)==torch.Tensor:
        image_c = image.cpu().detach().numpy()
    else:
        image_c = image

    if percentiles != None:
        i_min = np.percentile(image_c, percentiles[0])
        i_max = np.percentile(image_c, percentiles[1])
    if values != None:
        i_min = values[0]
        i_max = values[1]

    n_img = (image - i_min)/(i_max - i_min)

    return np.clip(n_img, clip_vals[0], clip_vals[1]) if clip else n_img

# -------------------------------------------------------------------------------------------------

if __name__=="__main__":
    pass
