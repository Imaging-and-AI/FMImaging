"""
Save pt model as onnx and pts
"""
import argparse
import pickle

import torch
from torch.utils.data.dataloader import DataLoader

import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from mri_model import *
from inference import apply_model, load_model, apply_model_3D, compare_model, load_model_onnx, load_model_pre_backbone_post

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
    parser.add_argument("--batch_size", default=8, type=int, help="batch size for onnx")
    parser.add_argument("--remake", action="store_true", help="ignore check and remake the model")
    parser.add_argument("--only_save", action="store_true", help="only save the full model")

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
    args.output_pth  = fname

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
    #model, config = load_pth_model(args)

    output_dir = Path(args.output).parents[0].resolve()
    os.makedirs(str(output_dir), exist_ok=True)

    if os.path.exists(args.input):
        model, config = load_model(args.input)
    else:
        model, config = load_model_pre_backbone_post(args.input)
        print(f"{Fore.RED}--> save entire model at {args.output_pth} ...{Style.RESET_ALL}")
        model.save_entire_model(config.num_epochs, save_file_name=args.output_pth)
        if args.only_save: return 0

    config.log_dir = str(output_dir)

    print(f"{Fore.RED}{'*' * 60}{Style.RESET_ALL}")
    device = torch.device('cpu')
    model_input = torch.randn(args.batch_size, config.no_in_channel, config.time, config.mri_height[-1], config.mri_width[-1], requires_grad=False, dtype=torch.float32)
    model_input = model_input.to(device)
    model.to(device)
    model.eval()
    print(f"input size {model_input.shape}, {model_input.dtype}, {device}")

    model_scripted = torch.jit.trace(model, model_input, strict=False)
    model_scripted.save(args.output)
    print(f"{Fore.YELLOW}Save model as torch jit format - {args.output}{Style.RESET_ALL}")

    if ((not os.path.exists(args.output_onnx)) or args.remake):
        print(f"{Fore.RED}{'*' * 60}{Style.RESET_ALL}")
        device = get_device()
        model_input = model_input.to(device)
        model.to(device)
        kwargs = dict()
        x = (model_input,)
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True, op_level_debug=True)
        onnx_program = torch.onnx.dynamo_export(model,
                                                *x,
                                                **kwargs,
                                                export_options=export_options)
        onnx_program.save(args.output_onnx)
        print(f"{Fore.YELLOW}Save model as onnx format - {args.output_onnx}{Style.RESET_ALL}")

    # torch.onnx.export(model, model_input, args.output_onnx, 
    #                 export_params=True, 
    #                 opset_version=17, 
    #                 training =torch.onnx.TrainingMode.TRAINING,
    #                 do_constant_folding=False,
    #                 input_names = ['input'], 
    #                 output_names = ['output'], 
    #                 dynamic_axes={'input' : {0:'batch_size', 2: 'time', 3: 'H', 4: 'W'}, 
    #                                 'output' : {0:'batch_size', 2: 'time', 3: 'H', 4: 'W'}
    #                                 }
    #                 )

    config.batch_size = model_input.shape[0]

    with open(args.config, 'wb') as fid:
        pickle.dump(config, fid)

    print(f"save config - {args.config}, batch_size {config.batch_size}")

    model_jit = torch.jit.load(args.output)
    model_onnx, config = load_model_onnx(model_dir=None, model_file=args.output_onnx, use_cpu=False)
    compare_model(config, model, model_jit, model_onnx, device = get_device(), x=None, batch_size=args.batch_size)
    print(f"{Fore.RED}{'*' * 60}{Style.RESET_ALL}")
    model_onnx, config = load_model_onnx(model_dir=None, model_file=args.output_onnx, use_cpu=True)
    compare_model(config, model, model_jit, model_onnx, device=torch.device('cpu'), x=None, batch_size=args.batch_size)

if __name__=="__main__":

    main()
