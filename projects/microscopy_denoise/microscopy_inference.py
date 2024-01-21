"""
Run micrsocopy inference
"""
import os
import h5py
import torch
import argparse
import tifffile
import numpy as np

from time import time
from tqdm import tqdm

import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

MRI_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(MRI_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

from utils import *
from microscopy_utils import *
from temp_utils.inference_utils import apply_model, load_model

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

    parser.add_argument("--input_dir", type=str, default=None, help="folder to load the data")
    parser.add_argument("--input_file_s", nargs='+', type=str, default=None, help="specific file(s) to load. .tiff or .h5. If None load the entire input_dir")
    parser.add_argument("--output_dir", type=str, default=None, help="folder to save the data")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    parser.add_argument("--overlap", nargs='+', type=int, default=None, help='overlap for (T, H, W), e.g. (2, 8, 8), (0, 0, 0) means no overlap')
    parser.add_argument("--no_clip_data", action="store_true", help="whether to not clip the data to [0,1] after scaling. default: do clip")
    parser.add_argument("--image_order", type=str, default="THW", help='the order of axis in the input image: THW or WHT')
    parser.add_argument("--device", type=str, default="cuda", help='the device to run on')
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size for running inference')

    parser.add_argument("--scaling_vals", type=float, nargs='+', default=[0, 4096], help='min max values to scale with respect to the scaling type')

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
    fname = os.path.splitext(args.saved_model_path)[0]
    args.saved_model_config  = fname + '.yaml'

    os.makedirs(args.output_dir, exist_ok=True)

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
    config.device = args.device

    config.height = config.micro_height
    config.width = config.micro_width
    config.time = config.micro_time

    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference

    print("Start loading data")

    images = data_all(args, config)

    # Each image is 3 tuple:
    # noisy_im, clean_im (optional), im_name

    print("End loading data")

    print("Start inference and saving")

    for noisy_im, clean_im, f_name in tqdm(images):

        predi_im = apply_model(model, noisy_im, config, config.device, overlap=args.overlap, batch_size=args.batch_size)

        predi_im = predi_im.cpu().numpy()

        np.save(os.path.join(args.output_dir, f"{f_name}_x.npy"), np.transpose(np.squeeze(noisy_im), [1, 2, 0])*args.scaling_vals[1])
        np.save(os.path.join(args.output_dir, f"{f_name}_pred.npy"), np.transpose(np.squeeze(predi_im), [1, 2, 0])*args.scaling_vals[1])
        if clean_im:
            np.save(os.path.join(args.output_dir, f"{f_name}_y.npy"), np.transpose(np.squeeze(clean_im), [1, 2, 0])*args.scaling_vals[1])

    print("End inference and saving")

if __name__=="__main__":
    main()
