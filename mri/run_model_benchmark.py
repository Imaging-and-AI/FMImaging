"""
Run MRI inference data
"""
import json
import wandb
import logging
import argparse
import copy
from time import time

import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from model_base.losses import *
from trainer_mri import load_model

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
    parser.add_argument("--saved_model_path", type=str, default="/export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230716_190117_960318_C-32-1_amp-False_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-50.pth", help='model checkpoints. endswith ".pth"')
    parser.add_argument("--model_type", type=str, default="MRI_double_net", help="if set, overwrite the config setting, STCNNT_MRI or MRI_hrnet, MRI_double_net")

    return parser.parse_args()

def check_args(args):
    """
    checks the cmd args to make sure they are correct
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - args (Namespace): the checked and updated argparse for MRI
    """
    assert args.saved_model_path.endswith(".pt") or args.saved_model_path.endswith(".pts") or args.saved_model_path.endswith(".onnx") or args.saved_model_path.endswith(".pth"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\" or \"*.onnx\" or \"*.pth\""

    # get the args path
    fname = os.path.splitext(args.saved_model_path)[0]
    args.saved_model_config  = fname + '.config'

    return args

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    args = check_args(arg_parser())
    print(args)
    
    print(f"{Fore.YELLOW}Load in model file - {args.saved_model_path}")
    model, config = load_model(args.saved_model_path, args.saved_model_config, args.model_type)

    device = get_device()

    B, T, C, H, W = 8, 16, 3, 64, 64

    x = torch.rand((B, T, C, H, W))

    model.eval()
    model.to(device=device)

    x = x.to(device=device)
    with torch.inference_mode():
        y = model(x.to(torch.float32))

    benchmark_all(model, x, grad=None, repeats=10, desc='MRI_double_net', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    benchmark_memory(model, x, desc='MRI_double_net', amp=True, amp_dtype=torch.bfloat16, verbose=True)

if __name__=="__main__":
    main()
