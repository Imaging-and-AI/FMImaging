"""
Save pt model as onnx and pts
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
from model_mri import STCNNT_MRI
from trainer_mri import apply_model, compare_model

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

def load_pt_model(args):
    """
    load a ".pt" model
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - model (torch model): the model ready for inference
    """
    
    config = []
    
    status = torch.load(args.input, map_location=get_device())
    config = status['config']        
    model = STCNNT_MRI(config=config)
    model.load_state_dict(status['model'])
    return model, config

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    args = check_args(arg_parser())
    print(args)
    
    print(f"{Fore.YELLOW}Load in model file - {args.input}")
    model, config = load_pt_model(args)

    output_dir = Path(args.output).parents[0].resolve()
    os.makedirs(str(output_dir), exist_ok=True)

    device = get_device()

    model_input = torch.randn(1, config.time, config.C_in, config.height[-1], config.width[-1], requires_grad=False)
    model_input = model_input.to(device)
    model.to(device)

    print(f"input size {model_input.shape}")

    model_scripted = torch.jit.trace(model, model_input, strict=False)
    model_scripted.save(args.output)

    torch.onnx.export(model, model_input, args.output_onnx, 
                    export_params=True, 
                    opset_version=16, 
                    training =torch.onnx.TrainingMode.TRAINING,
                    do_constant_folding=False,
                    input_names = ['input'], 
                    output_names = ['output'], 
                    dynamic_axes={'input' : {0:'batch_size', 1: 'time', 3: 'H', 4: 'W'}, 
                                    'input' : {0:'batch_size', 1: 'time', 3: 'H', 4: 'W'}
                                    }
                    )

    with open(args.config, 'wb') as fid:
        pickle.dump(config, fid)

    model_jit = load_model(model_dir=None, model_file=args.output, map_location=device)
    model_onnx, _ = load_model_onnx(model_dir=None, model_file=args.output_onnx)
            
    print(f"device is {device}")
    compare_model(config, model, model_jit, model_onnx, device=device, x=None)
    
        
if __name__=="__main__":
    main()
