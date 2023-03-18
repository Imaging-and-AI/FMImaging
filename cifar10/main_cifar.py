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
    parser = add_shared_args(parser=parser)

    return parser.parse_args()

# -------------------------------------------------------------------------------------------------
# create the train and val sets

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

    def transform_f(x):
        return tv.transforms.ToTensor()(x).unsqueeze(0)
    
    train_set = tv.datasets.CIFAR10(root=config.data_root, train=True,
                                    download=True, transform=transform_f)

    val_set = tv.datasets.CIFAR10(root=config.data_root, train=False,
                                    download=True, transform=transform_f)

    return train_set, val_set

# -------------------------------------------------------------------------------------------------
# main function. spawns threads if going for distributed data parallel

def main():
    
    args = arg_parser()
    assert args.run_name is not None, f"Please provide a \"--run_name\" for wandb"

    run = wandb.init(project=args.project, entity=args.wandb_entity, config=args,
                        name=args.run_name, notes=args.run_notes)
    config = wandb.config
    setup_run(config)

    train_set, val_set = create_dataset(config=config)
    model = STCNNT_Cifar(config=config, total_steps=len(train_set)//config.batch_size)
    trainable_params, total_params = get_number_of_params(model)

    logging.info(f"Configuration for this run:\n{config}")
    logging.info(f"Trainable parameters: {trainable_params:,}, Total parameters: {total_params:,}")

    if not config.ddp: # run in main process
        trainer(rank=-1, model=model, config=config,
                train_set=train_set, val_set=val_set)
    else: # spawn a process for each gpu
        mp.spawn(trainer, args=(model, config, train_set, val_set),
                    nprocs=config.world_size)

if __name__=="__main__":
    main()
