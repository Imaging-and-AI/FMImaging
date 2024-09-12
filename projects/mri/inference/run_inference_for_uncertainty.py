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

from setup import *
from utils import *
from inference import apply_model, load_model, apply_model_3D, load_model_pre_backbone_post, compute_uncertainty
from mri_data import MRIDenoisingDatasetTrain, load_mri_data
from mri_parser import mri_parser

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
    #parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    parser.add_argument("--overlap", nargs='+', type=int, default=None, help='overlap for (T, H, W), e.g. (2, 8, 8), (0, 0, 0) means no overlap')

    parser.add_argument("--input_fname", type=str, default="im", help='input file name')
    parser.add_argument("--gmap_fname", type=str, default="gfactor", help='gmap input file name')

    #parser.add_argument("--model_type", type=str, default=None, help="if set, overwrite the config setting, STCNNT_MRI or MRI_hrnet, MRI_double_net")

    args, unknown_args = parser.parse_known_args(namespace=Nestedspace())

    return args, unknown_args

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

    return args

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    args, unknown_args = arg_parser()
    print(args)

    project_config, unknown_args_project = mri_parser().parser.parse_known_args(namespace=Nestedspace())

    full_config = Nestedspace(**vars(args),
                              **vars(project_config))

    print(f"---> support bfloat16 is {support_bfloat16(device=get_device())}")

    print(f"{Fore.YELLOW}Load in model file - {full_config.saved_model_path}{Style.RESET_ALL}")
    if os.path.exists(full_config.saved_model_path):
        model, config = load_model(full_config.saved_model_path)
    else:
        model, config = load_model_pre_backbone_post(full_config.saved_model_path)

    # train_loader, val_loader
    print(f"{Fore.YELLOW}Load in data - {full_config.train_files[0]}{Style.RESET_ALL}")

    full_config.data_dir = full_config.data_root
    full_config.time = 12
    train_set, val_set, test_set = load_mri_data(config=full_config)

    train_loader = DataLoader(dataset=train_set[0], batch_size=1, shuffle=True, sampler=None,
                                num_workers=8, prefetch_factor=4, drop_last=True,
                                persistent_workers=True, pin_memory=False)

    val_loader = DataLoader(dataset=val_set[0], batch_size=1, shuffle=False, sampler=None,
                                num_workers=8, prefetch_factor=4, drop_last=True,
                                persistent_workers=True, pin_memory=False)

    # -----------------------------------------------------------------------
    config.height = config.mri_height
    config.width = config.mri_width

    patch_size_inference = full_config.patch_size_inference
    config.pad_time = full_config.pad_time
    config.ddp = False

    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference

    #setup_run(config, dirs=["log_path"])

    # load the data
    image = np.load(os.path.join(full_config.input_dir, f"{full_config.input_fname}_real.npy")) + np.load(os.path.join(full_config.input_dir, f"{full_config.input_fname}_imag.npy")) * 1j
    image /= full_config.im_scaling
    image = np.squeeze(image)

    gmap_file = f"{full_config.input_dir}/{full_config.gmap_fname}.npy"
    if os.path.exists(gmap_file):
        gmap = np.load(f"{full_config.input_dir}/{full_config.gmap_fname}.npy")
        gmap /= full_config.gmap_scaling
    else:
        gmap = np.ones((image.shape[:2]))

    if len(image.shape) == 2:
        image = image[:,:,np.newaxis,np.newaxis]

    if len(image.shape) == 3:
        image = image[:,:,:,np.newaxis]

    if(image.shape[3]>20):
        image = np.transpose(image, (0, 1, 3, 2))

    RO, E1, frames, slices = image.shape
    print(f"2DT mode, {full_config.input_dir}, images - {image.shape}, gmap - {gmap.shape}, median gmap {np.median(gmap)}")

    if(gmap.ndim==2):
        gmap = np.expand_dims(gmap, axis=2)

    if gmap.shape[2] >= slices and gmap.shape[2] == frames:
        image = np.transpose(image, (0, 1, 3, 2))
        RO, E1, frames, slices = image.shape

    if full_config.overlap:
        overlap_used = tuple(full_config.overlap)
    else:
        overlap_used = None

    #output = apply_model(image, model, gmap, config=config, scaling_factor=full_config.scaling_factor, device=get_device(), overlap=overlap_used, verbose=True)
    output = image
    sd = compute_uncertainty(image, train_loader, val_loader, model, gmap, config, scaling_factor=full_config.scaling_factor, device=get_device(), overlap=overlap_used, verbose=True)

    # -------------------------------------------    

    print(f"{full_config.output_dir}, images - {image.shape}, {output.shape}")

    output = np.squeeze(output)

    save_inference_results(input=image, output=output, gmap=gmap, output_dir=full_config.output_dir, noisy_image=None, sd_image=sd)

if __name__=="__main__":
    main()
