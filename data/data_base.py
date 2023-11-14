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

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from data_utils import define_transforms, custom_numpy_to_tensor, select_patch

# ------------------------------------------------------------------------------------------------
class NumpyDataset(torch.utils.data.Dataset):
    """
    Generic dataset class for reading in numpy files. 
    Numpys should be stored as H x W x (D/T, optional) x (C, optional).

    @args:
        config (namespace): config contatining all args for this run
        split (str, {'train','val','test'}): indicates which data split this is
    @rets:
        image (torch.tensor): image input as a torch tensor of shape C x D/T x H x W
        label (torch.tensor): either seg mask or classification label
        id (str): ID identifying this sample

    """
    def __init__(self, config, split):

        # Extract values from config used in dataloader
        self.config = config
        self.data_loc = self.config.data_dir #(str): directory where numpy files are stored
        self.height = self.config.height #(int): height (number of rows) of numpys; if stored data is not this height, images will be interpolated to this height
        self.width = self.config.width #(int): width (number of cols) of numpys; if stored data is not this width, images will be interpolated to this height
        self.time = self.config.time #(int): time or depth of numpys; if stored data does not have this time/depth, images will be cropped or padded. Should be >=1.
        self.no_in_channel = self.config.no_in_channel #(int): number of input channels. Should be >=1
        self.no_out_channel = self.config.no_out_channel #(int): number of output channels (if doing seg) or number of output classes (if doing class)
        self.split_csv = self.config.split_csv_path #(str): path to the csv specifying the dataset splits
        self.task = self.config.task_type #(str): the task type, currently 'seg' or 'class'
        self.patch = self.config.use_patches # whether to train on patches
        self.split = split #(str): indicates data split, train val test

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
            # We'll approximate the number of patches per image so we call on each image many times per epoch, extracting many patches per image per epcoh
            no_patch_per_image = int(self.height/self.config.patch_height) * int(self.width/self.config.patch_width) * int(self.time/self.config.patch_time)
            self.split_subject_ids = list(self.split_subject_ids) * no_patch_per_image

        logging.info(f"{Fore.MAGENTA}Size of {split} dataset: {len(self.split_subject_ids)}{Style.RESET_ALL}")

        if self.task=='class':
            # Read in labels for classification tasks
            self.metadata = pd.read_csv(glob.glob(os.path.join(self.data_loc,'*_metadata.csv'))[0])

    def __getitem__(self, index):
        
        # Load image and adjust it to a correctly-sized tensor
        image_path = os.path.join(self.data_loc,self.split_subject_ids[index],self.split_subject_ids[index]+'_input.npy')
        image = np.load(image_path).astype('float32') # Expect H, W, (optional T/D), (optional C)
        min_input = np.min(image)
        image -= min_input
        max_input = np.max(image)
        image = image/max_input
        image = custom_numpy_to_tensor(image,self.height,self.width,self.time,self.no_in_channel) # Returns standardized C, T/D, H, W     

        # Transform
        seed = np.random.randint(2147483647)  
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.input_transform(image)
        if self.patch:
            image, patch_x, patch_y, patch_z = select_patch(image, self.config.patch_height, self.config.patch_width, self.config.patch_time)

        if self.task=='seg':
            # Load seg mask and adjust it to a correctly-sized tensor
            seg = np.load(image_path.replace('_input','_output')).astype('float32')
            seg = custom_numpy_to_tensor(seg,self.height,self.width,self.time,1,cv2.INTER_NEAREST) # CURRENTLY ASSUMING ALL CLASSES MARKED IN ONE-CHANNEL SEG MASK (i.e., not one-hot and not multiclass)

            # Transform
            random.seed(seed)
            torch.manual_seed(seed)
            seg = self.output_transform(seg)
            if self.patch:
                seg, _, _, _ = select_patch(seg, patch_x, patch_y, patch_z, use_indices=True)
            seg = seg[0] # Assuming all classes marked in one map, get 0th (only) channel
            return image, seg.type(torch.LongTensor), self.split_subject_ids[index]
        
        elif self.task=='enhance':
            # Load output image and adjust it to a correctly-sized tensor
            out = np.load(image_path.replace('_input','_output')).astype('float32')
            out = custom_numpy_to_tensor(out,self.height,self.width,self.time,self.no_out_channel) 
            out -= min_input
            out = out/max_input

            # Transform
            random.seed(seed)
            torch.manual_seed(seed)
            out = self.output_transform(out)
            if self.patch:
                out, _, _, _ = select_patch(out, patch_x, patch_y, patch_z, use_indices=True)
            return image, out.type(torch.FloatTensor), self.split_subject_ids[index]
        
        elif self.task=='class':
            label = float(self.metadata[self.metadata.SubjectID.isin([self.split_subject_ids[index]])].Label)
            return image, torch.tensor(label, dtype=torch.long), self.split_subject_ids[index]
        
        else:
            raise ValueError('Unkown task type.')
        

    def __len__(self):
        return len(self.split_subject_ids)

