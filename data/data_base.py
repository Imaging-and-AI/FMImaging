"""
Generic dataloader for numpy files
"""
import logging
import torch
import pandas as pd
import os, sys, glob
import numpy as np
import random
import cv2
from colorama import Fore, Style
from pathlib import Path
import torchvision

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from data_utils import define_transforms, custom_numpy_to_tensor, select_patch
from augmentation_functions import *

# ------------------------------------------------------------------------------------------------
class NumpyDataset(torch.utils.data.Dataset):
    """
    Generic dataset class for reading in numpy files. 
    Numpys should be stored as H x W x (D/T, optional) x (C, optional).

    @args:
        config (namespace): config contatining all args for this run
        task_name (str): name of the task associated with this dataset
        task_ind (int): index of the task associated with this dataset (defined by the order in which the tasks are listed in the config)
        split (str, {'train','val','test'}): indicates which data split this is
    @rets:
        image (torch.tensor): image input as a torch tensor of shape C x D/T x H x W
        label (torch.tensor): either seg mask or classification label
        id (str): ID identifying this sample

    """
    def __init__(self, config, task_name, task_ind, split):

        # Extract values from config used in dataloader
        self.config = config
        self.task_name = task_name
        self.task_ind = task_ind
        self.split = split 

        self.data_loc = self.config.data_dir[task_ind] #(str): directory where numpy files are stored
        self.height = self.config.height[task_ind] #(int): height (number of rows) of numpys; if stored data is not this height, images will be interpolated to this height
        self.width = self.config.width[task_ind] #(int): width (number of cols) of numpys; if stored data is not this width, images will be interpolated to this height
        self.time = self.config.time[task_ind] #(int): time or depth of numpys; if stored data does not have this time/depth, images will be cropped or padded. Should be >=1.
        self.no_in_channel = self.config.no_in_channel[task_ind] #(int): number of input channels. Should be >=1
        self.no_out_channel = self.config.no_out_channel[task_ind] #(int): number of output channels (if doing seg) or number of output classes (if doing class)
        self.split_csv = self.config.split_csv_path[task_ind] #(str): path to the csv specifying the dataset splits
        self.task_type = self.config.task_type[task_ind] #(str): the task type
        self.patch = self.config.use_patches[task_ind] # whether to train on patches

        assert self.time>=1, "Time arg should be greater than or equal to 1"
        assert self.no_in_channel>=1, "Number of input channels arg should be greater than or equal to 1"
        assert self.no_out_channel>=1, "Number of output channels arg should be greater than or equal to 1"
        
        # Create transforms
        self.input_transform, self.output_transform = define_transforms(config, split)
        
        # Define list of IDs included in this split
        if self.split_csv is not None:
            # If a csv is specified, that csv determines which subject ID maps to which split
            split_df = pd.read_csv(self.split_csv)
            split_df = split_df[split_df.Split.isin([self.split])]
            self.split_subject_ids = list(split_df.SubjectID)

        else:
            # Else all available subject IDs are split up
            all_subject_ids = [subject_path.split('/')[-2] for subject_path in glob.glob(os.path.join(self.data_loc, '*', '*_input.npy'))]
            total_subject_ids = len(all_subject_ids)
            if split=='train': self.split_subject_ids = all_subject_ids[:int(0.6*total_subject_ids)]
            elif split=='val': self.split_subject_ids = all_subject_ids[int(0.6*total_subject_ids):int(0.8*total_subject_ids)]
            elif split=='test': self.split_subject_ids = all_subject_ids[int(0.8*total_subject_ids):]
            else: raise ValueError(f"Unknown split {split} specified, should be train, val, or test")

        if self.patch:
            # We'll approximate the number of patches per image so we call on each image many times per epoch, extracting many patches per image per epoch
            no_patch_per_image = int(self.height/self.config.patch_height) * int(self.width/self.config.patch_width) * int(self.time/self.config.patch_time)
            self.split_subject_ids = list(self.split_subject_ids) * no_patch_per_image

        logging.info(f"{Fore.MAGENTA}Size of {split} dataset: {len(self.split_subject_ids)}{Style.RESET_ALL}")

        if self.task_type=='class':
            # Read in labels for classification tasks
            self.metadata = pd.read_csv(glob.glob(os.path.join(self.data_loc,'*_metadata.csv'))[0])
        if self.task_type=='ss_image_restoration':
            if self.config.ss_image_restoration.resolution_factor>1:
                self.ss_resolution = True
                self.ss_resolution_transform = torchvision.transforms.Compose(
                    [torchvision.transforms.Resize(size=(self.height//self.config.ss_image_restoration.resolution_factor, self.width//self.config.ss_image_restoration.resolution_factor), 
                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                    antialias = True),
                    torchvision.transforms.Resize(size=(self.height, self.width), 
                                                    interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
                                                    antialias = True)],

                )
            else:
                self.ss_resolution = False
            if self.config.ss_image_restoration.noise_std>0:
                self.ss_noise = True
                self.ss_noise_transform = torchvision.transforms.Compose(
                    [GaussianNoise(mean=0.0, std=self.config.ss_image_restoration.noise_std),]
                )
            else:
                self.ss_noise = False
            if self.config.ss_image_restoration.mask_percent>0:
                self.ss_mask = True
            else:
                self.ss_mask = False

    def _class_loader(self, index):
         
        # Load image and adjust it to a correctly-sized tensor
        image_path = os.path.join(self.data_loc,self.split_subject_ids[index],self.split_subject_ids[index]+'_input.npy')
        image = np.load(image_path).astype('float32') # Expect H, W, (optional T/D), (optional C)
        image = custom_numpy_to_tensor(image,self.height,self.width,self.time,self.no_in_channel) # Returns standardized C, T/D, H, W 

        # Transform
        seed = np.random.randint(2147483647)  
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.input_transform(image)
        if self.patch:
            image, patch_x, patch_y, patch_z = select_patch(image, self.config.patch_height, self.config.patch_width, self.config.patch_time)

        # Get label 
        label = float(self.metadata[self.metadata.SubjectID.isin([self.split_subject_ids[index]])].Label.iloc[0])

        return image, torch.tensor(label, dtype=torch.long), self.split_subject_ids[index]
    
    def _seg_loader(self, index):

        # Load image and adjust it to a correctly-sized tensor
        image_path = os.path.join(self.data_loc,self.split_subject_ids[index],self.split_subject_ids[index]+'_input.npy')
        image = np.load(image_path).astype('float32') # Expect H, W, (optional T/D), (optional C)
        image = custom_numpy_to_tensor(image,self.height,self.width,self.time,self.no_in_channel) # Returns standardized C, T/D, H, W 

        # Transform
        seed = np.random.randint(2147483647)  
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.input_transform(image)
        if self.patch:
            image, patch_x, patch_y, patch_z = select_patch(image, self.config.patch_height, self.config.patch_width, self.config.patch_time)

        # Load seg mask and adjust it to a correctly-sized tensor
        seg = np.load(image_path.replace('_input','_output')).astype('float32')
        seg = custom_numpy_to_tensor(seg,self.height,self.width,self.time,1,cv2.INTER_NEAREST) # CURRENTLY ASSUMING ALL CLASSES MARKED IN ONE-CHANNEL SEG MASK (i.e., not one-hot and not multiclass)

        # Transform
        random.seed(seed)
        torch.manual_seed(seed)
        seg = self.output_transform(seg)
        if self.patch:
            seg, _, _, _ = select_patch(seg, patch_x, patch_y, patch_z, use_indices=True)
        seg = seg[0] # Assuming all classes marked in one map (i.e., not one-hot or multiclass), get 0th (only) channel

        return image, seg.type(torch.LongTensor), self.split_subject_ids[index]
        
    def _enhance_loader(self, index):

         # Load image and adjust it to a correctly-sized tensor
        image_path = os.path.join(self.data_loc,self.split_subject_ids[index],self.split_subject_ids[index]+'_input.npy')
        image = np.load(image_path).astype('float32') # Expect H, W, (optional T/D), (optional C)
        image = custom_numpy_to_tensor(image,self.height,self.width,self.time,self.no_in_channel) # Returns standardized C, T/D, H, W 

        # Transform
        seed = np.random.randint(2147483647)  
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.input_transform(image)
        if self.patch:
            image, patch_x, patch_y, patch_z = select_patch(image, self.config.patch_height, self.config.patch_width, self.config.patch_time)

        # Load output image and adjust it to a correctly-sized tensor
        out = np.load(image_path.replace('_input','_output')).astype('float32')
        out = custom_numpy_to_tensor(out,self.height,self.width,self.time,self.no_out_channel) 

        # Transform
        random.seed(seed)
        torch.manual_seed(seed)
        out = self.output_transform(out)
        if self.patch:
            out, _, _, _ = select_patch(out, patch_x, patch_y, patch_z, use_indices=True)
        return image, out.type(torch.FloatTensor), self.split_subject_ids[index]        

    def _ss_image_restoration_loader(self, index):
        # Load image and adjust it to a correctly-sized tensor
        image_path = os.path.join(self.data_loc,self.split_subject_ids[index],self.split_subject_ids[index]+'_input.npy')
        image = np.load(image_path).astype('float32') # Expect H, W, (optional T/D), (optional C)
        image = custom_numpy_to_tensor(image,self.height,self.width,self.time,self.no_in_channel) # Returns standardized C, T/D, H, W 

        # Transform
        seed = np.random.randint(2147483647)  
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.input_transform(image)
        if self.patch:
            image, patch_x, patch_y, patch_z = select_patch(image, self.config.patch_height, self.config.patch_width, self.config.patch_time)

        # Apply downsampling, if desired 
        if self.ss_resolution:
            lowres_image = self.ss_resolution_transform(image)

        # Apply noise, if desired
        if self.ss_noise:
            noisy_image = self.ss_noise_transform(image)

        # Randomly mask out patches, if desired
        if self.ss_resolution and self.ss_noise:
            masked_image = lowres_image + noisy_image
        elif self.ss_resolution and not self.ss_noise:
            masked_image = lowres_image
        elif not self.ss_resolution and self.ss_noise:
            masked_image = noisy_image
        else:
            masked_image = image.clone()
        
        if self.ss_mask:
            mask_percent = self.config.ss_image_restoration.mask_percent
            mask_patch_size = self.config.ss_image_restoration.mask_patch_size 

            num_patches_in_t = int(np.ceil(image.shape[1]/mask_patch_size[0]))
            num_patches_in_x = int(np.ceil(image.shape[2]/mask_patch_size[1]))
            num_patches_in_y = int(np.ceil(image.shape[3]/mask_patch_size[2]))
            num_patches = int(num_patches_in_t*num_patches_in_x*num_patches_in_y)

            num_patches_to_mask = int(np.ceil(mask_percent*num_patches))

            patches_to_mask = np.random.choice(num_patches, num_patches_to_mask, replace=False)
            patch_inds = np.unravel_index(patches_to_mask, (num_patches_in_t, num_patches_in_x, num_patches_in_y))
            masked_image += 1e-5 # Adding small value to all of image for tracking masked patches later on
            image += 1e-5 # Adjusting original image with small value to match
            mask = torch.zeros_like(image)
            for t, x, y in zip(*patch_inds):
                masked_image[:,t*mask_patch_size[0]:(t+1)*mask_patch_size[0], x*mask_patch_size[1]:(x+1)*mask_patch_size[1], y*mask_patch_size[2]:(y+1)*mask_patch_size[2]] = 0
                mask[:,t*mask_patch_size[0]:(t+1)*mask_patch_size[0], x*mask_patch_size[1]:(x+1)*mask_patch_size[1], y*mask_patch_size[2]:(y+1)*mask_patch_size[2]] = 1
            
            if self.config.loss_func[self.task_ind] in ['SSImageRestoration']:
                image = torch.concat([image, mask], axis=0) 

        return masked_image, image, self.split_subject_ids[index]
    
    def __getitem__(self, index):

        if self.task_type=='class':
            network_input, network_output, sample_id = self._class_loader(index)
        elif self.task_type=='seg':
            network_input, network_output, sample_id = self._seg_loader(index)
        elif self.task_type=='enhance':
            network_input, network_output, sample_id = self._enhance_loader(index)
        elif self.task_type=='ss_image_restoration':
            network_input, network_output, sample_id = self._ss_image_restoration_loader(index)
        else:
            raise ValueError('Unknown task type.')
        
        return network_input, network_output, sample_id
        

    def __len__(self):
        return len(self.split_subject_ids)

