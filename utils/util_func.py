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
