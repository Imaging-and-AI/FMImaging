"""
File for testing the cifar 10 models
Can be run from command line to load and test a model
Provides functionality to also be called during runtime while/after training:
    - eval_test
"""
import json
import wandb
import logging
import argparse

import torch
import torchvision as tv
import torchmetrics
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from model_cifar import STCNNT_Cifar

# -------------------------------------------------------------------------------------------------
# setup for testing from cmd

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - parser (ArgumentParser): the argparse for STCNNT Cifar10
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT Cifar10 test evaluation")

    parser.add_argument("--data_root", type=str, default=None, help='root folder for the data')
    parser.add_argument("--data_set", type=str, default="cifar10", help='choice of dataset: "cifar10", "cifar100')
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path endswith ".pt" or ".pts"')
    parser.add_argument("--saved_model_config", type=str, default=None, help='the config of the model. required when using ".pt"')

    parser = add_shared_STCNNT_args(parser=parser)

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
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.saved_model_path is not None, f"Please provide a \"--saved_model_path\" for loading a checkpoint"

    assert config.saved_model_path.endswith(".pt") or config.saved_model_path.endswith(".pts"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\""
    assert not(config.saved_model_path.endswith(".pt")) or \
            (config.saved_model_path.endswith(".pt") and config.saved_model_config.endswith(".json")),\
            f"If loading from \"*.pt\" need a \"*.json\" config file"

    config.time = 1
    config.height = [32]
    config.width = [32]
    config.load_path = config.saved_model_path
    if config.log_path is None: config.log_path = config.results_path

    return config

# -------------------------------------------------------------------------------------------------
# load dataset and model

def transform_f(x):
    """
    transform function for cifar images
    @args:
        - x (cifar dataset return object): the input image
    @rets:
        - x (torch.Tensor): 4D torch tensor [T,C,H,W], T=1
    """
    return x.unsqueeze(0)

def create_base_test_set(config):
    """
    create the test set using torchvision datasets
    @args:
        - config (Namespace): runtime namespace for setup
    @args (from config):
        - data_root (str): root directory for the dataset
        - time (int): for assertion (==1)
        - height, width (int list): for assertion (==32)
    @rets:
        - test_set (torch Dataset): the test set
    """
    assert config.time==1 and config.height[0]==32 and config.width[0]==32,\
        f"For Cifar10, time height width should 1 32 32"
    
    transform = transforms.Compose([transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            transform_f
                            ])
        
    if config.data_set == "cifar10":
        test_set = tv.datasets.CIFAR10(root=config.data_root, train=False,
                                    download=True, transform=transform)
    elif config.data_set == "cifar100":
        test_set = tv.datasets.CIFAR100(root=config.data_root, train=False,
                                    download=True, transform=transform)
        
    elif config.data_set == "imagenet":
           
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transform_f
                                ])
    
        test_set = tv.datasets.ImageNet(root=config.data_root, split="val", transform=transform)
        
    else:
        raise NotImplementedError(f"Data set not implemented:{config.data_set}")

    return test_set

def load_model(config):
    """
    load a ".pt" or ".pts" model
    ".pt" models require ".json" to create the model
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - model (torch model): the model ready for inference
    """
    if config.saved_model_path.endswith(".pt"):
        config.load_path = config.saved_model_path
        model = STCNNT_Cifar(config=config)
    else:
        model = torch.jit.load(config.saved_model_path)

    return model

# -------------------------------------------------------------------------------------------------
# save results

def save_results(config, loss, acc, id=""):
    """
    save the results
    @args:
        - config (Namespace): runtime namespace for setup
        - loss (float): test loss
        - acc (float): test accuracy [0,1]
        - id (str): unique id to save results with
    """
    file_name = f"{config.run_name}_{config.date}_{id}_results"
    results_file_name = os.path.join(config.results_path, file_name)

    result_dict = {
        "test_loss" : loss,
        "test_acc" : acc
    }

    with open(f"{results_file_name}.json", "w") as file:
        json.dump(result_dict, file)

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    config = check_args(arg_parser())
    setup_run(config, dirs=["log_path"])

    model = load_model(config)
    wandb.init(project=config.project, entity=config.wandb_entity, config=config,
                        name=config.run_name, notes=config.run_notes)

    eval_test(-1, model, config, test_set=None, device="cuda")

if __name__=="__main__":
    main()
