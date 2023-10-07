"""
Run MRI inference data
"""
import os
import argparse
import copy
from time import time

import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

MRI_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(MRI_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[3].resolve()
sys.path.append(str(REPO_DIR))

from utils import *
from inference import apply_model, load_model, apply_model_3D

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

    parser.add_argument("--input_dir", default=None, help="folder to load the data")
    parser.add_argument("--output_dir", default=None, help="folder to save the data")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="scaling factor to adjust model strength; higher scaling means lower strength")
    parser.add_argument("--im_scaling", type=float, default=1.0, help="extra scaling applied to image")
    parser.add_argument("--gmap_scaling", type=float, default=1.0, help="extra scaling applied to gmap")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    parser.add_argument("--overlap", nargs='+', type=int, default=None, help='overlap for (T, H, W), e.g. (2, 8, 8), (0, 0, 0) means no overlap')

    parser.add_argument("--input_fname", type=str, default="im", help='input file name')
    parser.add_argument("--gmap_fname", type=str, default="gfactor", help='gmap input file name')

    parser.add_argument("--model_type", type=str, default=None, help="if set, overwrite the config setting, STCNNT_MRI or MRI_hrnet, MRI_double_net")

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
    args.saved_model_config  = fname + '.yaml'

    return args

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    args = check_args(arg_parser())
    print(args)

    print(f"---> support bfloat16 is {support_bfloat16(device=get_device())}")

    print(f"{Fore.YELLOW}Load in model file - {args.saved_model_path}")
    model, config = load_model(args.saved_model_path)

    patch_size_inference = args.patch_size_inference
    config.pad_time = args.pad_time
    config.ddp = False

    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference

    #setup_run(config, dirs=["log_path"])

    # load the data
    image = np.load(os.path.join(args.input_dir, f"{args.input_fname}_real.npy")) + np.load(os.path.join(args.input_dir, f"{args.input_fname}_imag.npy")) * 1j
    image /= args.im_scaling
    image = np.squeeze(image)

    gmap_file = f"{args.input_dir}/{args.gmap_fname}.npy"
    if os.path.exists(gmap_file):
        gmap = np.load(f"{args.input_dir}/{args.gmap_fname}.npy")
        gmap /= args.gmap_scaling
    else:
        gmap = np.ones((image.shape[:2]))

    if len(image.shape) == 3 and gmap.ndim==3 and gmap.shape[2]==image.shape[2]:
        output = apply_model_3D(image, model, gmap, config=config, scaling_factor=args.scaling_factor, device=get_device(), verbose=True)
        print(f"3D mode, {args.input_dir}, images - {image.shape}, gmap - {gmap.shape}, median gmap {np.median(gmap)}")
    else:
        if len(image.shape) == 2:
            image = image[:,:,np.newaxis,np.newaxis]

        if len(image.shape) == 3:
            image = image[:,:,:,np.newaxis]

        if(image.shape[3]>20):
            image = np.transpose(image, (0, 1, 3, 2))

        RO, E1, frames, slices = image.shape
        print(f"2DT mode, {args.input_dir}, images - {image.shape}, gmap - {gmap.shape}, median gmap {np.median(gmap)}")

        if(gmap.ndim==2):
            gmap = np.expand_dims(gmap, axis=2)

        if gmap.shape[2] >= slices and gmap.shape[2] == frames:
            image = np.transpose(image, (0, 1, 3, 2))
            RO, E1, frames, slices = image.shape

        if args.overlap:
            overlap_used = tuple(args.overlap)
        else:
            overlap_used = None

        output = apply_model(image, model, gmap, config=config, scaling_factor=args.scaling_factor, device=get_device(), overlap=overlap_used, verbose=True)

        # input = np.flip(image, axis=0)
        # output2 = apply_model(input, model, np.flip(gmap, axis=0), config=config, scaling_factor=args.scaling_factor, device=get_device())
        # output2 = np.flip(output2, axis=0)

        # input = np.flip(image, axis=1)
        # output3 = apply_model(input, model, np.flip(gmap, axis=1), config=config, scaling_factor=args.scaling_factor, device=get_device())
        # output3 = np.flip(output3, axis=1)

        # input = np.transpose(image, axes=(1, 0, 2, 3))
        # output4 = apply_model(input, model, np.transpose(gmap, axes=(1, 0, 2)), config=config, scaling_factor=args.scaling_factor, device=get_device())
        # output4 = np.transpose(output4, axes=(1, 0, 2, 3))

        # res = output + output2 + output3 + output4
        # output = res / 4
    
    # -------------------------------------------    

    print(f"{args.output_dir}, images - {image.shape}, {output.shape}")

    output = np.squeeze(output)

    save_inference_results(image, output, gmap, args.output_dir)

if __name__=="__main__":
    main()
