"""
File for testing the cifar 10 models
Can be run from command line to load and test a model
Provides functionality to also be called during runtime while/after training:
    - eval_test
"""
import wandb
import logging
import argparse

import torch
import torchvision as tv
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from model_cifar import STCNNT_Cifar

# -------------------------------------------------------------------------------------------------
# main test function

def eval_test(model, config, test_set=None, device="cpu"):
    """
    The test evaluation.
    @args:
        - model (torch model): model to be tested
        - config (Namespace): config of the run
        - test_set (torch Dataset): the data to test on
        - device (torch.device): the device to run the test on
    @rets:
        - test_loss_avg (float): the average test loss
    """
    c = config # shortening due to numerous uses

    # if no test_set given then load the base set
    if test_set is None: test_set = create_base_test_set(config)

    test_loader = DataLoader(dataset=test_set, batch_size=c.batch_size, shuffle=True, sampler=None,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0)
    
    loss_f = torch.nn.CrossEntropyLoss()
    test_loss = AverageMeter()

    model.eval()
    model.to(device)

    test_loader_iter = iter(test_loader)
    total_iters = len(test_loader)
    with tqdm(total=total_iters) as pbar:

        for idx in  np.arange(total_iters):

            inputs, labels = next(test_loader_iter)

            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = loss_f(output, labels)
            test_loss.update(loss.item(), n=c.batch_size)

            wandb.log({"running_test_loss": loss.item()})

            pbar.update(1)
            pbar.set_description(f"Test {inputs.shape}, {test_loss.avg:.4f}")

    pbar.set_postfix_str(f"Test results: {test_loss.avg:.4f}")
    logging.info(f"Test results: {test_loss.avg:.4f}")
    wandb.log({"test_loss_avg":test_loss.avg})

    return test_loss.avg

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

    parser.add_argument("--data_root", type=str, default="C:/Users/rehmana2/Documents/Current/Projects/cifar10", help='root folder for the data')
    parser.add_argument("--saved_model_path", type=str, default="C:/Users/rehmana2/Documents/Current/Projects/cifar10", help='root folder for the data')
    parser.add_argument("--saved_model_config", type=str, default="C:/Users/rehmana2/Documents/Current/Projects/cifar10", help='root folder for the data')

    parser = add_shared_args(parser=parser)

    return parser.parse_args()

def check_args(args):
    """
    checks the cmd args to make sure they are correct
    @args:
        - args (Namespace): the argparser
    @rets:
        - args (Namespcae): the checked and updated argparse for Cifar10
    """
    assert args.run_name is not None, f"Please provide a \"--run_name\" for wandb"
    assert args.saved_model_path is not None, f"Please provide a \"--saved_model_path\" for loading a checkpoint"

    assert args.saved_model_path.endswith(".pt") or args.saved_model_path.endswith(".pts"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\""
    assert not(args.saved_model_path.endswith(".pt")) or \
            (args.saved_model_path.endswith(".pt") and args.saved_model_config.endswith(".json")),\
            f"If loading from \"*.pt\" need a \"*.json\" config file"
    
    args.load_path = args.saved_model_path
    args.time = 1
    args.height = [32]
    args.width = [32]

    return args

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

    def transform_f(x):
        return tv.transforms.ToTensor()(x).unsqueeze(0)
    
    test_set = tv.datasets.CIFAR10(root=config.data_root, train=False,
                                    download=True, transform=transform_f)

    return test_set

def load_model(args):
    """
    load a ".pt" or ".pts" model
    ".pt" models require ".json" to create the model
    @args:
        - args (Namespace): the argpaser
    @rets:
        - model (torch model): the model ready for inference
    """
    if args.saved_model_path.endswith(".pt"):
        args.load_path = args.saved_model_path
        model = STCNNT_Cifar(config=args)
    else:
        model = torch.jit.load(args.saved_model_path)

    return model

# -------------------------------------------------------------------------------------------------
# the main function for setup and eval call

def main():

    args = check_args(arg_parser())

    model = load_model(args)
    run = wandb.init(project=args.project, entity=args.wandb_entity, config=args, 
                        name=args.run_name, notes=args.run_notes)
    config = wandb.config

    setup_run(config)
    eval_test(model, config, test_set=None, device="cuda")

if __name__=="__main__":
    main()
