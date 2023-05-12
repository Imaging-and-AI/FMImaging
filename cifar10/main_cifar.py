"""
Main file for STCNNT Cifar10
"""
import wandb
import logging
import argparse
import torchvision as tv
from torchvision import transforms
import torchvision.transforms as T
import torch.multiprocessing as mp

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from trainer_cifar import trainer
from model_cifar import STCNNT_Cifar

# True class names
classes = ['airplane', 'automobile', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# -------------------------------------------------------------------------------------------------
# Extra args on top of shared args

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - parser (ArgumentParser): the argparse for STCNNT Cifar10
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT Cifar10")
    parser.add_argument("--data_root", type=str, default=None, help='root folder for the data')
    parser.add_argument("--data_set", type=str, default="cifar10", help='choice of dataset: "cifar10", "cifar100", "imagenet"')
    parser.add_argument("--head_channels", nargs='+', type=int, default=[8,128,10], help='number of channels for cifar head')
    
    parser = add_backbone_STCNNT_args(parser=parser)

    ns = Nestedspace()
    args = parser.parse_args(namespace=ns)
    
    return args

def check_args(config):
    """
    checks the cmd args to make sure they are correct
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - config (Namespace): the checked and updated argparse for Cifar10
    """
    assert config.run_name is not None, f"Please provide a \"--run_name\" for wandb"
    assert config.data_root is not None, f"Please provide a \"--data_root\" to load the data"
    assert config.log_path is not None, f"Please provide a \"--log_path\" to save the logs in"
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.model_path is not None, f"Please provide a \"--model_path\" to save the final model in"
    assert config.check_path is not None, f"Please provide a \"--check_path\" to save the checkpoints in"

    config.time = 1
    config.height = [32]
    config.width = [32]

    if config.data_set == "imagenet":
        config.height = [256]
        config.width = [256]
    
    return config

# -------------------------------------------------------------------------------------------------
# create the train and val sets

def transform_f(x):
    """
    transform function for cifar images
    @args:
        - x (cifar dataset return object): the input image
    @rets:
        - x (torch.Tensor): 4D torch tensor [T,C,H,W], T=1
    """
    return x.unsqueeze(0)

def create_dataset(config):
    """
    create the train and val set using torchvision datasets
    @args:
        - config (Namespace): runtime namespace for setup
    @args (from config):
        - data_root (str): root directory for the dataset
        - time (int): for assertion (==1)
        - height, width (int list): for assertion (==32)
    @rets:
        - train_set (torch Dataset): the train set
        - val_set (torch Dataset): the val set (same as test set)
    """
    if config.data_set == "cifar10" or config.data_set == "cifar100":
        assert config.time==1 and config.height[0]==32 and config.width[0]==32, f"For Cifar10, time height width should 1 32 32"
        
    if config.data_set == "imagenet":
        assert config.time==1 and config.height[0]==256 and config.width[0]==256, f"For ImageNet, time height width should 1 256 256"
    
    transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                            transforms.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                                            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                            transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #Normalize all the images
                                            transform_f
                               ])
    
    transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               transform_f
                               ])
    
    if config.data_set == "cifar10":
        train_set = tv.datasets.CIFAR10(root=config.data_root, train=True,
                                        download=True, transform=transform_train)

        val_set = tv.datasets.CIFAR10(root=config.data_root, train=False,
                                        download=True, transform=transform)
    elif config.data_set == "cifar100":
        train_set = tv.datasets.CIFAR100(root=config.data_root, train=True,
                                        download=True, transform=transform_train)

        val_set = tv.datasets.CIFAR100(root=config.data_root, train=False,
                                        download=True, transform=transform)
        
    elif config.data_set == "imagenet":
        
        transform_train = transforms.Compose([transforms.Resize((256, 256)),  #resises the image so it can be perfect for our model.
                                        transforms.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                                        transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                        transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                        transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #Normalize all the images
                                        transform_f
                            ])
    
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transform_f
                                ])
    
        train_set = tv.datasets.ImageNet(root=config.data_root, split="train", transform=transform_train)

        val_set = tv.datasets.ImageNet(root=config.data_root, split="val", transform=transform)
    else:
        raise NotImplementedError(f"Data set not implemented:{config.data_set}")

    return train_set, val_set

# -------------------------------------------------------------------------------------------------
# main function. spawns threads if going for distributed data parallel

def main():
    
    config = check_args(arg_parser())
    setup_run(config)

    train_set, val_set = create_dataset(config=config)

    num_samples = len(train_set)
    if config.ddp: 
        num_samples /= torch.cuda.device_count()

    total_steps = int(np.ceil(num_samples/config.batch_size)*config.num_epochs)
    
    model = STCNNT_Cifar(config=config, total_steps=total_steps)

    # model summary
    model_summary = model_info(model, config)
    logging.info(f"Configuration for this run:\n{config}")
    logging.info(f"Model Summary:\n{str(model_summary)}")

    if not config.ddp: # run in main process
        trainer(rank=-1, model=model, config=config,
                train_set=train_set, val_set=val_set)
    else: # spawn a process for each gpu
        mp.spawn(trainer, args=(model, config, train_set, val_set),
                    nprocs=config.world_size)

if __name__=="__main__":
    main()
