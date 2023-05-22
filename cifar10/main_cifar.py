"""
Main file for STCNNT Cifar10
"""
import torchvision as tv
from torchvision import transforms
import torchvision.transforms as T
import torch.multiprocessing as mp
import torch.distributed as dist

import os
import sys
import argparse
from datetime import datetime, timedelta

from pathlib import Path
Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from trainer_cifar import *
from model_cifar import STCNNT_Cifar

from trainer_base import Trainer_Base

# # True class names
# classes = ['airplane', 'automobile', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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

# -------------------------------------------------------------------------------------------------

class CifarTrainer(Trainer_Base):
    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
        """
        super().__init__(config)
    

    def check_args(self):
        """
        checks the cmd args to make sure they are correct
        """
        
        super().check_args()
        
        self.config.time = 1

        if self.config.data_set == 'cifar10':        
            self.config.height = [32]
            self.config.width = [32]
            self.config.num_classes = 10
        
        if self.config.data_set == 'cifar10' and self.config.data_root is None:
            self.config.data_root = "/export/Lab-Xue/projects/cifar10/data"
            
        if self.config.data_set == 'cifar100' and self.config.data_root is None:
            self.config.data_root = "/export/Lab-Xue/projects/cifar100/data"

        if self.config.data_set == "imagenet":
            self.config.height = [256]
            self.config.width = [256]
            self.config.num_classes = 1000
            if self.config.data_root is None:
                self.config.data_root = "/export/Lab-Xue/projects/imagenet/data"
   
    def run_task_trainer(self, rank=-1, wandb_run=None):
        trainer(rank=rank, config=self.config, wandb_run=wandb_run)

# -------------------------------------------------------------------------------------------------
def main():
    
    config_default = arg_parser()
    
    trainer = CifarTrainer(config_default)
    trainer.train()
    
# -------------------------------------------------------------------------------------------------
if __name__=="__main__":    
    main()
