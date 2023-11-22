"""
Save pt model as onnx and pts
"""
import json
import wandb
import logging
import argparse
import copy
from time import time
import pickle

import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

import sys
from pathlib import Path

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from mri_model import *
from inference import apply_model, load_model, apply_model_3D, compare_model, load_model_onnx

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

    parser.add_argument("--input", default=None, help="model to load")
    parser.add_argument("--output", default=None, help="model to save")
    
    return parser.parse_args()

def check_args(args):
    """
    checks the cmd args to make sure they are correct
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - args (Namespace): the checked and updated argparse for MRI
    """

    # get the args path    
    if args.output is None:
        fname = os.path.splitext(args.input)[0]
        args.output  = fname + '.pts'

    fname = os.path.splitext(args.output)[0]
    args.output_onnx  = fname + '.onnx'
    args.config = fname + '.config'

    return args

# -------------------------------------------------------------------------------------------------
# load model

def load_pth_model(args):
    """
    load a ".pth" model
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - model (torch model): the model ready for inference
    """
    status = torch.load(args.input, map_location=get_device())
    config = status['config']
    model = create_model(config, config.model_type)
    model.load_state_dict(status['model_state'])
    return model, config

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    args = check_args(arg_parser())
    print(args)
    
    print(f"{Fore.YELLOW}Load in model file - {args.input}")
    model, config = load_pth_model(args)

    output_dir = Path(args.output).parents[0].resolve()
    os.makedirs(str(output_dir), exist_ok=True)

    device = get_device()

    model_input = torch.randn(1, config.no_in_channel, config.time, config.mri_height[-1], config.mri_width[-1], requires_grad=False)
    model_input = model_input.to(device)
    model.to(device)
    model.eval()
    print(f"input size {model_input.shape}")

    # model_scripted = torch.jit.trace(model, model_input, strict=False)
    # model_scripted.save(args.output)

    torch.onnx.export(model, model_input, args.output_onnx, 
                    export_params=True, 
                    opset_version=16, 
                    training =torch.onnx.TrainingMode.TRAINING,
                    do_constant_folding=False,
                    input_names = ['input'], 
                    output_names = ['output'], 
                    dynamic_axes={'input' : {0:'batch_size', 2: 'time', 3: 'H', 4: 'W'}, 
                                    'output' : {0:'batch_size', 2: 'time', 3: 'H', 4: 'W'}
                                    }
                    )

    with open(args.config, 'wb') as fid:
        pickle.dump(config, fid)

    model_jit = torch.jit.load(args.output)
    model_onnx, _ = load_model_onnx(model_dir=None, model_file=args.output_onnx)

    print(f"device is {device}")
    compare_model(config, model, model_jit, model_onnx, device=torch.device('cpu'), x=None)

if __name__=="__main__":
    main()
