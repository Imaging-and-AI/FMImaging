"""
File for testing the MRI models
Can be run from command line to load and test a model
Provides functionality to also be called during runtime while/after training:
    - eval_test
"""
import json
import wandb
import logging
import argparse

import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from utils.setup_training import get_bar_format
from model_base.losses import *
from model_mri import STCNNT_MRI
from data_mri import load_mri_test_data
from trainer_mri import eval_val
from utils.running_inference import running_inference

# -------------------------------------------------------------------------------------------------
# setup for testing from cmd

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT MRI test evaluation")

    parser.add_argument("--data_root", type=str, default="/export/Lab-Xue/projects/mri/data", help='root folder for the data')
    parser.add_argument("--results_path", type=str, default="/export/Lab-Xue/projects/mri/results", help='folder for results')
    parser.add_argument("--test_files", type=str, nargs='+', default=["train_3D_3T_retro_cine_2020_small_test.h5"], help='list of test h5files')
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    
    #parser = add_shared_STCNNT_args(parser=parser)

    return parser.parse_args()

def check_args(config):
    """
    checks the cmd args to make sure they are correct
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - config (Namespace): the checked and updated argparse for MRI
    """
    assert config.data_root is not None, f"Please provide a \"--data_root\" to load the data"
    assert config.test_files is not None, f"Please provide a \"--test_files\" to load the data"
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.saved_model_path is not None, f"Please provide a \"--saved_model_path\" for loading a checkpoint"

    assert config.saved_model_path.endswith(".pt") or config.saved_model_path.endswith(".pts"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\""

    # get the config path
    fname = os.path.splitext(config.saved_model_path)[0]
    config.saved_model_config  = fname + '.config'

    return config

# -------------------------------------------------------------------------------------------------
# load model

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
        status = torch.load(config.saved_model_path, map_location=get_device())
        model = STCNNT_MRI(config=config)
        model.load_state_dict(status['model'])
    else:
        model = torch.jit.load(config.saved_model_path)

    return model

# -------------------------------------------------------------------------------------------------
# save results

def save_results(config, losses, id=""):
    """
    save the results
    @args:
        - config (Namespace): runtime namespace for setup
        - losses (float list): test losses:
            - model loss
            - mse loss
            - l1 loss
            - ssim loss
            - ssim3D loss
            - psnr
        - id (str): unique id to save results with
    """
    file_name = f"{config.run_name}_{config.date}_{id}_results"
    results_file_name = os.path.join(config.results_path, file_name)

    result_dict = {f"test_loss_{id}": losses[0],
                    f"test_mse_loss_{id}": losses[1],
                    f"test_l1_loss_{id}": losses[2],
                    f"test_ssim_loss_{id}": losses[3],
                    f"test_ssim3D_loss_{id}": losses[4],
                    f"test_psnr_{id}": losses[5]}

    with open(f"{results_file_name}.json", "w") as file:
        json.dump(result_dict, file)

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    c = check_args(arg_parser())
    
    patch_size_inference = c.patch_size_inference
    
    config_file = c.saved_model_config 
    print(f"{Fore.YELLOW}Load in config file - {config_file}")
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    config.data_root = c.data_root
    config.results_path = c.results_path
    config.test_files = c.test_files
    config.saved_model_path = c.saved_model_path
    config.pad_time = c.pad_time
    config.ddp = False
    
    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference
    
    setup_run(config, dirs=["log_path"])

    print(f"{Fore.YELLOW}Load in model file - {config.saved_model_path}")
    model = load_model(config)
    run = wandb.init(project=config.project, entity=config.wandb_entity, config=config,
                        name=f"Test_{config.run_name}_inference_{config.height[-1]}", notes=config.run_notes)

    print(f"Wandb name:\n{run.name}")
    
    test_set = load_mri_test_data(config=config)
    losses = eval_val(rank=-1, model=model, config=config, val_set=test_set, epoch=-1, device=get_device(), wandb_run=run, id="test")

    save_results(config, losses, id="")
        
if __name__=="__main__":
    main()
