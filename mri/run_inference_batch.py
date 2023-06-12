"""
Run MRI inference data in the batch mode
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

import nibabel as nib

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from model_base.losses import *
from model_mri import STCNNT_MRI
from trainer_mri import apply_model

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

    parser.add_argument("--input_dir", default=None, help="folder to load the batch data, go to all subfolders")
    parser.add_argument("--output_dir", default=None, help="folder to save the data; subfolders are created for each case")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="scaling factor to adjust model strength; higher scaling means lower strength")
    parser.add_argument("--im_scaling", type=float, default=1.0, help="extra scaling applied to image")
    parser.add_argument("--gmap_scaling", type=float, default=1.0, help="extra scaling applied to gmap")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    parser.add_argument("--batch_size", type=int, default=16, help='after loading a batch, start processing')
    
    parser.add_argument("--input_fname", type=str, default="im", help='input file name')
    
    return parser.parse_args()

def check_args(args):
    """
    checks the cmd args to make sure they are correct
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - args (Namespace): the checked and updated argparse for MRI
    """
    assert args.saved_model_path.endswith(".pt") or args.saved_model_path.endswith(".pts") or args.saved_model_path.endswith(".onnx"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\" or \"*.onnx\""

    # get the args path
    fname = os.path.splitext(args.saved_model_path)[0]
    args.saved_model_config  = fname + '.config'

    return args

# -------------------------------------------------------------------------------------------------
# load model

def load_model(args):
    """
    load a ".pt" or ".pts" model
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - model (torch model): the model ready for inference
    """
    
    config = []
    
    config_file = args.saved_model_config
    if os.path.isfile(config_file):
        print(f"{Fore.YELLOW}Load in config file - {config_file}")
        with open(config_file, 'rb') as f:
            config = pickle.load(f)

    if args.saved_model_path.endswith(".pt"):
        status = torch.load(args.saved_model_path, map_location=get_device())
        config = status['config']
        if not torch.cuda.is_available():
            config.device = torch.device('cpu')
        model = STCNNT_MRI(config=config)
        model.load_state_dict(status['model'])
    elif args.saved_model_path.endswith(".pts"):
        model = torch.jit.load(args.saved_model_path, map_location=get_device())
    else:
        model, _ = load_model_onnx(model_dir="", model_file=args.saved_model_path, use_cpu=True)
    return model, config

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def process_a_batch(args, model, config, images, selected_cases, gmaps, device):
    for ind in range(len(images)):
        case_dir = selected_cases[ind]
        print(f"-----------> Process {selected_cases[ind]} <-----------")

        image = images[ind]
        gmap = gmaps[ind]
        output = apply_model(image.astype(np.complex64), model, gmap.astype(np.float32), config=config, scaling_factor=args.scaling_factor, device=device)

        case = os.path.basename(case_dir)
        output_dir = os.path.join(args.output_dir, case)
        os.makedirs(output_dir, exist_ok=True)

        save_inference_results(image, output, gmap, output_dir)

        print("--" * 30)

def main():

    args = check_args(arg_parser())
    print(args)
    
    print(f"{Fore.YELLOW}Load in model file - {args.saved_model_path}")
    model, config = load_model(args)
    
    patch_size_inference = args.patch_size_inference
              
    config.pad_time = args.pad_time
    config.ddp = False
    
    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference
    
    device=get_device()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # load the cases
    case_dirs = fast_scandir(args.input_dir)
    
    selected_cases = []
    images = []
    gmaps = []
    
    with tqdm(total=len(case_dirs), bar_format=get_bar_format()) as pbar:
        for c in case_dirs:
            fname = os.path.join(c, f"{args.input_fname}_real.npy")
            if os.path.isfile(fname):    
                image = np.load(os.path.join(c, f"{args.input_fname}_real.npy")) + np.load(os.path.join(c, f"{args.input_fname}_imag.npy")) * 1j
                image /= args.im_scaling

                if len(image.shape) == 2:
                    image = image[:,:,np.newaxis,np.newaxis]
                elif len(image.shape) == 3:
                    image = image[:,:,:,np.newaxis]

                if(image.shape[3]>20):
                    image = np.transpose(image, (0, 1, 3, 2))

                RO, E1, frames, slices = image.shape
                print(f"{c}, images - {image.shape}")

                gmap = np.load(f"{c}/gfactor.npy")
                gmap /= args.gmap_scaling

                if(gmap.ndim==2):
                    gmap = np.expand_dims(gmap, axis=2)

                if gmap.shape[2] != slices:
                    continue
                else:
                    images.append(image)
                    gmaps.append(gmap)
                    selected_cases.append(c)
            
            if len(images)>0 and len(images)%args.batch_size==0:                
                process_a_batch(args, model, config, images, selected_cases, gmaps, device)            
                selected_cases = []
                images = []
                gmaps = []
    
            pbar.update(1)                

    # process left over cases
    if len(images) > 0:
        process_a_batch(args, model, config, images, selected_cases, gmaps, device)

if __name__=="__main__":
    main()
