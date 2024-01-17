"""
Run MRI inference data
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

REPO_DIR = Path(__file__).parents[3].resolve()
sys.path.append(str(REPO_DIR))

from utils import *
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
    parser.add_argument("--batch_size", type=int, default=4, help='batch_size for running inference')

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
# load and normalize data

def load_tiff(input_dir, tifffile_name):

    return [tifffile.imread(os.path.join(input_dir, tifffile_name)), None, f"{tifffile_name}"]

def load_h5(input_dir, h5file_name):

    images = []

    h5file = h5py.File(os.path.join(input_dir, h5file_name), mode='r')
    keys = list(h5file.keys())

    for key in keys:
        noisy_im = np.array(h5file[key+"/noisy_im"])
        clean_im = np.array(h5file[key+"/clean_im"])

        images.append([noisy_im, clean_im, f"{h5file_name}_{key}"])

    return images

def load_all(input_dir, file_names):

    images = []

    for file_name in tqdm(file_names):

        if file_name.endswith(".tif") or file_name.endswith(".tiff") or file_name.endswith(".TIFF"):
            images.append(load_tiff(input_dir, file_name))
        elif file_name.endswith(".h5") or file_name.endswith(".h5py"):
            images.extend(load_h5(input_dir, file_name))
        else:
            raise NotImplementedError(f"Only supported formats: '.tif', '.tiff', '.TIFF', '.h5', '.h5py'. Given format:{file_name}")

    return images

def data_all(args, config):

    c = config

    input_dir = args.input_dir
    file_names = args.input_file_s if args.input_file_s is not None else os.list_dir(input_dir)

    images = load_all(input_dir, file_names)

    for image in tqdm(images):

        if args.image_order=="HWT":
            image[0] = np.transpose(image[0], (2,0,1))
            if image[1] is not None:
                image[1] = np.transpose(image[1], (2,0,1))

        if c.scaling_type=="per":
            image[0] = torch.from_numpy(normalize_image(image[0], percentiles=c.scaling_vals, clip=not args.no_clip_data).astype(np.float32))[np.newaxis,:,np.newaxis]
            if image[1] is not None:
                image[1] = torch.from_numpy(normalize_image(image[1], percentiles=c.scaling_vals).astype(np.float32))[np.newaxis,:,np.newaxis]
        else:
            image[0] = torch.from_numpy(normalize_image(image[0], values=c.scaling_vals, clip=not args.no_clip_data).astype(np.float32))[np.newaxis,:,np.newaxis]
            if image[1] is not None:
                image[1] = torch.from_numpy(normalize_image(image[1], values=c.scaling_vals).astype(np.float32))[np.newaxis,:,np.newaxis]

    return images

# -------------------------------------------------------------------------------------------------
# save images

def save_one(saved_path, fname, x, output, y=None):

        noisy_im = x.numpy(force=True)
        predi_im = output.numpy(force=True)
        if y is not None:
            clean_im = y.numpy(force=True)
        else:
            clean_im = None

        np.save(os.path.join(saved_path, f"{fname}_x.npy"), noisy_im)
        np.save(os.path.join(saved_path, f"{fname}_output.npy"), predi_im)
        if clean_im is not None: np.save(os.path.join(saved_path, f"{fname}_y.npy"), clean_im)

        save_x = noisy_im[0,0]
        save_p = predi_im[0,0]
        save_y = clean_im[0,0] if clean_im is not None else []

        if clean_im is None:
            composed_channel_wise = np.transpose(np.array([save_x, save_p]), (1,0,2,3))
        else:
            composed_channel_wise = np.transpose(np.array([save_x, save_p, save_y]), (1,0,2,3))

        tifffile.imwrite(os.path.join(saved_path, f"{fname}_combined.tiff"),\
                            composed_channel_wise, imagej=True)

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

        save_one(args.output_dir, f_name, noisy_im, clean_im, predi_im)

    print("End inference and saving")

if __name__=="__main__":
    main()
