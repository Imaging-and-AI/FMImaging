"""
Requirements to use a custom dataset with the general codebase: 
- Define a torch Dataset class (or a list of torch Datset classes) that returns the following
    @rets:
        image (torch.tensor): image input as a torch tensor of shape C x D/T x H x W
        label (torch.tensor): label of appropriate shape and type for task
        id (str): ID identifying each sample
    Dataset class should have __init__, __getitem__, and __len__ methods
    Note that the custom class dataset __init___ args can be customized via the custom run.py file
"""

import torch
import pandas as pd
import os, glob
import numpy as np
import cv2

# ------------------------------------------------------------------------------------------------
class custom_dataset(torch.utils.data.Dataset):
    """
    Example custom dataset
    """
    def __init__(self, config, split):
        
        self.config = config
        self.data_loc = config.data_dir
        self.height = config.height
        self.width = config.width
        self.split_csv = config.split_csv_path

        # Define list of IDs included in this split
        all_subject_IDs = list(glob.glob(os.path.join(self.data_loc,'*','*_input.npy')))
        if split=='train': self.split_subject_ids = all_subject_IDs[:int(0.6*len(all_subject_IDs))]
        elif split=='val': self.split_subject_ids = all_subject_IDs[int(0.6*len(all_subject_IDs)):int(0.8*len(all_subject_IDs))]
        else: self.split_subject_ids = all_subject_IDs[int(0.8*len(all_subject_IDs)):]

        self.metadata = pd.read_csv(glob.glob(os.path.join(self.data_loc,'*_metadata.csv'))[0])

    def __getitem__(self, index):
        
        # Load image and adjust it to a correctly-sized tensor
        image_path = self.split_subject_ids[index]
        image = np.load(image_path).astype('float32') # Expect H, W, (optional T/D), (optional C)
        image = cv2.resize(image, (self.height,self.width))
        image = np.expand_dims(image,(2,3))
        image = torch.from_numpy(image)
        image = torch.permute(image,(-1,-2,0,1)) # Permute to C, T/D, H, W

        # Get label for this ID
        subject_id = self.split_subject_ids[index].split('/')[-2]
        label = float(self.metadata[self.metadata.SubjectID.isin([subject_id])].Label)

        # Return image, label, ID
        return image, torch.tensor(label, dtype=torch.long), self.split_subject_ids[index]
        

    def __len__(self):
        return len(self.split_subject_ids)

