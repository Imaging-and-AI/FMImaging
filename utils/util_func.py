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
