"""
Main file for STCNNT Cifar10
"""
import wandb
import logging
import argparse
import torchvision as tv
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
    parser.add_argument("--head_channels", nargs='+', type=int, default=[8,128,10], help='number of channels for cifar head')
    parser = add_shared_args(parser=parser)

    return parser.parse_args()

def check_args(config):
    """
    checks the cmd args to make sure they are correct
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - config (Namespcae): the checked and updated argparse for Cifar10
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
    return tv.transforms.ToTensor()(x).unsqueeze(0)

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
    assert config.time==1 and config.height[0]==32 and config.width[0]==32,\
        f"For Cifar10, time height width should 1 32 32"
    
    train_set = tv.datasets.CIFAR10(root=config.data_root, train=True,
                                    download=True, transform=transform_f)

    val_set = tv.datasets.CIFAR10(root=config.data_root, train=False,
                                    download=True, transform=transform_f)

    return train_set, val_set

# -------------------------------------------------------------------------------------------------
# main function. spawns threads if going for distributed data parallel

def main():
    
    config = check_args(arg_parser())
    setup_run(config)

    train_set, val_set = create_dataset(config=config)
    model = STCNNT_Cifar(config=config, total_steps=len(train_set)//config.batch_size)

    if not config.ddp: # run in main process
        trainer(rank=-1, model=model, config=config,
                train_set=train_set, val_set=val_set)
    else: # spawn a process for each gpu
        mp.spawn(trainer, args=(model, config, train_set, val_set),
                    nprocs=config.world_size)

if __name__=="__main__":
    main()
