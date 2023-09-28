"""
Requirements to use a custom dataset with the general codebase: 
- Define a torch Dataset class that returns the following
    @rets:
        image (torch.tensor): image input as a torch tensor of shape C x D/T x H x W
        label (torch.tensor): label of appropriate shape and type for task
        id (str): ID identifying each sample
    Dataset class should have __init__, __getitem__, and __len__ methods
    Note that the custom class dataset __init___ args can be customized via the custom run.py file
"""

import torch
import torchvision as tv
from torchvision import transforms
import pandas as pd
import os, glob
import numpy as np
import cv2
    
# ------------------------------------------------------------------------------------------------
def transform_f(x):
    """
    transform function for cifar images
    @args:
        - x (cifar dataset return object): the input image
    @rets:
        - x (torch.Tensor): 4D torch tensor [T,C,H,W], T=1
    """
    return x.unsqueeze(0)

class cifar_dataset(torch.utils.data.Dataset):
    """
    Cifar custom dataset
    """
    def __init__(self, config, split):
        
        self.config = config
        self.data_loc = config.data_dir
        self.height = config.height
        self.width = config.width
        self.split = split

        if self.split=='train':
            transform = transforms.Compose([transforms.Resize((self.height,self.width)),  #resises the image so it can be perfect for our model.
                                                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                                                transforms.RandomHorizontalFlip(), # Flips the image w.r.t horizontal axis
                                                transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                                transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #Normalize all the images
                                                transform_f
                                ])
            self.dataset = tv.datasets.CIFAR10(root=self.data_loc, train=True,
                                        download=True, transform=transform)
            
        elif self.split=='val':
            transform = transforms.Compose([transforms.Resize((self.height,self.width)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                transform_f
                                                ])
            self.dataset = tv.datasets.CIFAR10(root=self.data_loc, train=False,
                                            download=True, transform=transform)
        
        elif self.split=='test':
            transform = transforms.Compose([transforms.Resize((self.height,self.width)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                transform_f
                                                ])
            self.dataset = tv.datasets.CIFAR100(root=self.data_loc, train=False,
                                        download=True, transform=transform)
   
        
    def __getitem__(self, index):
        
        image, label = self.dataset[index]
        id = str(index)
        
        return image, label, id

    def __len__(self):
        return len(self.dataset)

