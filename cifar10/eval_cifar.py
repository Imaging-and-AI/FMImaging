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

def eval_test(model, config, test_set=None, device="cpu", id=""):
    """
    The test evaluation.
    @args:
        - model (torch model): model to be tested
        - config (Namespace): config of the run
        - test_set (torch Dataset): the data to test on
        - device (torch.device): the device to run the test on
        - id (str): unique id to log and save results with
    @rets:
        - test_loss_avg (float): the average test loss
        - test_loss_acc (float): the average test acc [0,1]
    """
    c = config # shortening due to numerous uses

    # if no test_set given then load the base set
    if test_set is None: test_set = create_base_test_set(config)

    test_loader = DataLoader(dataset=test_set, batch_size=c.batch_size, shuffle=False, sampler=None,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0)

    loss_f = torch.nn.CrossEntropyLoss()
    test_loss = AverageMeter()
    test_acc = AverageMeter()

    model.eval()
    model.to(device)

    test_loader_iter = iter(test_loader)
    total_iters = len(test_loader) if not c.debug else 10
    with tqdm(total=total_iters) as pbar:

        for idx in  np.arange(total_iters):

            inputs, labels = next(test_loader_iter)

            inputs = inputs.to(device)
            labels = labels.to(device)
            total = labels.size(0)

            output = model(inputs)
            loss = loss_f(output, labels)
            test_loss.update(loss.item(), n=total)

            _, predicted = torch.max(output.data, 1)
            correct = (predicted == labels).sum().item()
            test_acc.update(correct/total, n=total)

            wandb.log({f"running_test_loss_{id}": loss.item(),
                        f"running_test_acc_{id}": correct/total})

            pbar.update(1)
            pbar.set_description(f"Test {inputs.shape}, {loss.item():.4f}, {correct/total:.4f}")

    pbar.set_postfix_str(f"Test results: {test_loss.avg:.4f}, {test_acc.avg:.4f}")
    logging.info(f"Test results: {test_loss.avg:.4f}, {test_acc.avg:.4f}")
    wandb.log({f"test_loss_avg_{id}":test_loss.avg,
                f"test_acc_avg_{id}":test_acc.avg})

    save_results(config, test_loss.avg, test_acc.avg, id)

    return test_loss.avg, test_acc.avg

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
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path endswith ".pt" or ".pts"')
    parser.add_argument("--saved_model_config", type=str, default=None, help='the config of the model. required when using ".pt"')

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
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.saved_model_path is not None, f"Please provide a \"--saved_model_path\" for loading a checkpoint"

    assert config.saved_model_path.endswith(".pt") or config.saved_model_path.endswith(".pts"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\""
    assert not(config.saved_model_path.endswith(".pt")) or \
            (config.saved_model_path.endswith(".pt") and config.saved_model_config.endswith(".json")),\
            f"If loading from \"*.pt\" need a \"*.json\" config file"

    config.load_path = config.saved_model_path
    config.time = 1
    config.height = [32]
    config.width = [32]

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
    return tv.transforms.ToTensor()(x).unsqueeze(0)

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

    test_set = tv.datasets.CIFAR10(root=config.data_root, train=False,
                                    download=True, transform=transform_f)

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

    eval_test(model, config, test_set=None, device="cuda")

if __name__=="__main__":
    main()
