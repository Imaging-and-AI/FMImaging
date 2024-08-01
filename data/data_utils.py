"""
Support functions for generic dataloader
"""

import torch
import random
import torchvision
import cv2
import numpy as np
import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from augmentation_functions import *

# ------------------------------------------------------------------------------------------------
def custom_numpy_to_tensor(image,height,width,time,no_channels,interp=cv2.INTER_LINEAR):
        """
        Function that adjusts stored numpy image of shape (H x W (x D x C)) to tensor of shape (C x T/D x H x W)
        """

        def _resize_xy(img,new_shape):
            if (img.shape[0],img.shape[1])==new_shape: 
                return img
            return cv2.resize(img, (new_shape[1],new_shape[0]), interpolation=interp)

        def _resize_depth(img,new_d):
            if img.shape[-1] < new_d:
                pad = new_d - img.shape[-1]
                img = np.pad(img,((0,0),(0,0),(pad//2,pad-pad//2)))
            elif img.shape[-1] > new_d:
                crop = img.shape[-1] - new_d
                img = img[:,:,crop//2:img.shape[-1]-(crop-crop//2)]
            return img

        if len(image.shape) not in [2,3,4]:
            raise ValueError(f"Image shape should be H x W (x D x C), consisting of 2, 3, or 4 dimensions. Got {len(image.shape)} dimensions.")
        
        if no_channels==1 and time==1: # 2d, single dimensional
            if len(image.shape)==2:
                image = _resize_xy(image,(height,width))
                image = np.expand_dims(image,(2,3))
            elif len(image.shape)==3:
                assert image.shape[-1]==1, f"Single channel and depth/time specified, but third dimension has size {image.shape[-1]}"
                image = _resize_xy(image,(height,width))
                image = np.expand_dims(image,(3))
            elif len(image.shape)==4:
                assert image.shape[-1]==1, f"Single channel specified, but fourth (C) dimension has size {image.shape[-1]}"
                assert image.shape[-2]==1, f"Single depth/time specified, but third (D/T) dimension has size {image.shape[-2]}"
                image = _resize_xy(image,(height,width))

        elif no_channels>1 and time==1: # 2d, multi dimensional
            if len(image.shape)==2:
                raise ValueError("More than one input channel specified, but stored image only has two dimensions.")
            elif len(image.shape)==3:
                assert image.shape[-1]==no_channels, f"Channel dimension in stored numpy ({image.shape[-1]}) does not match specified channel dimension ({no_channels})"
                image = _resize_xy(image,(height,width))
                image = np.expand_dims(image,(2))
            elif len(image.shape)==4:
                assert image.shape[-1]==no_channels, f"Channel dimension in stored numpy ({image.shape[-1]}) does not match specified channel dimension ({no_channels})"
                assert image.shape[-2]==1, f"Single depth/time specified, but third (D/T) dimension has size {image.shape[-2]}"
                image = _resize_xy(image,(height,width))

        elif no_channels==1 and time>1: # 3d, single dimensional
            if len(image.shape)==2:
                raise ValueError("More than one time/depth dimension specified, but stored image only has two dimensions.")
            elif len(image.shape)==3:
                image = _resize_xy(image,(height,width))
                image = _resize_depth(image,time)
                image = np.expand_dims(image,(3))
            elif len(image.shape)==4:
                assert image.shape[-1]==1, f"Single channel specified, but fourth (C) dimension has size {image.shape[-1]}"
                image = _resize_xy(image,(height,width))
                image = _resize_depth(image,time)

        elif no_channels>1 and time>1: # 3d, multi dimensional
            if len(image.shape)==2:
                raise ValueError("More than one time dimension specified, but stored image only has two dimensions.")
            elif len(image.shape)==3:
                raise ValueError("More than one time/depth dimension and channel specified, but stored image only has three dimensions.")
            elif len(image.shape)==4:
                assert image.shape[-1]==no_channels, f"Channel dimension in stored numpy ({image.shape[-1]}) does not match specified channel dimension ({no_channels})"
                image = _resize_xy(image,(height,width))
                image = _resize_depth(image,time)

        else:
            raise ValueError(f"Expected no_input_channel and time to be >=1, got {no_channels} and {time}")

        image = torch.from_numpy(image)
        image = torch.permute(image,(-1,-2,0,1)) # Permute to C, T/D, H, W

        return image

# ------------------------------------------------------------------------------------------------
def select_patch(image, patch_height, patch_width, patch_time, use_indices=False):
    """
    Function to select random patch from an image
    
    @args
        image (torch or numpy tensor): image to select patch from
        patch_height (int or tuple): if not use_indices, this is the height of the patch to select; if use_indices, this is a tuple of row indices that defines the patch
        patch_width, see above
        patch_time, see above
        use_indices (bool): whether to interpret previous args as indices or dimensions
    """
    if use_indices:
        return image[:,patch_time[0]:patch_time[1], patch_height[0]:patch_height[1], patch_width[0]:patch_width[1]], patch_height, patch_width, patch_time
    else:
        patch_start_time = np.random.randint(0, image.shape[1]-patch_time-1)
        patch_end_time = patch_start_time + patch_time

        patch_start_height = np.random.randint(0, image.shape[2]-patch_height-1)
        patch_end_height = patch_start_height + patch_height

        patch_start_width = np.random.randint(0, image.shape[3]-patch_width-1)
        patch_end_width = patch_start_width + patch_width

        return image[:,patch_start_time:patch_end_time, patch_start_height:patch_end_height, patch_start_width:patch_end_width], (patch_start_height,patch_end_height), (patch_start_width, patch_end_width), (patch_start_time,patch_end_time)

# ------------------------------------------------------------------------------------------------
def define_transforms(config, split):
    """
    Function to create transform sequences for a dataset

    @args: 
        config (nested namespace): config specifying which transforms to apply
        split (str): which split this is; only some transforms are applied during testing (e.g., normalization)

    @rets: 
        input_transforms (torch.transform): sequences of transforms to apply to the input images 
        output_transforms (torch.transform): sequences of transforms to apply to the outputs (e.g., seg masks should have affine transforms to match transformed inputs, but not color jitters)

    """

    # Initialize transform lists
    input_transforms = []
    output_transforms = []

    # Add affine transforms to training inputs and outputs, if desired
    if config.affine_aug and split=='train':
        input_transforms += [torchvision.transforms.RandomApply([torchvision.transforms.RandomAffine(10,(0.1,0.1),(0.95,1.05),10)],p=.9)]
        output_transforms += [torchvision.transforms.RandomApply([torchvision.transforms.RandomAffine(10,(0.1,0.1),(0.95,1.05),10)],p=.9)]

    # Add brightness transform to training inputs, if desired
    if config.brightness_aug and split=='train':
        input_transforms += [torchvision.transforms.RandomApply([RandomBrightnessContrast()],p=.9)]
        if config.task_type=='enhance': output_transforms += [torchvision.transforms.RandomApply([RandomBrightnessContrast()],p=.9)]

    # Add gaussian blur transform to training inputs, if desired
    if config.gaussian_blur_aug and split=='train':
        input_transforms += [torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 5))],p=.15)]

    # TODO: Add more augmentation transforms here, if desired
    # TODO: Add normalization transform to training and testing inputs, if desired
        
    # Compose transform lists; if none, specify identity transform
    if len(input_transforms)>0: input_transforms = torchvision.transforms.Compose(input_transforms)
    else: input_transforms = torch.nn.Identity()
    if len(output_transforms)>0: output_transforms = torchvision.transforms.Compose(output_transforms)
    else: output_transforms = torch.nn.Identity()

    return input_transforms, output_transforms



