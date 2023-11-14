import torch
import random

class RandomBrightnessContrast(object):
    """
    Adapted from albumentations, randomly adjusts the brightness of a torch tensor
    """
    def __init__(
        self,
        brightness_limit=0.3,
        contrast_limit=0.3,
    ):
        
        super(RandomBrightnessContrast, self).__init__()
        
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit 
        
    def __call__(self, img):
        
        alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
        beta = 0.0 + random.uniform(-self.brightness_limit, self.brightness_limit)

        timg = img.clone()
        timg *= alpha
        timg += beta * torch.mean(timg)        
        
        return timg