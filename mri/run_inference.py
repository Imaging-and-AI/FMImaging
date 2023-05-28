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

    parser.add_argument("--input_dir", default=None, help="folder to load the data")
    parser.add_argument("--output_dir", default=None, help="folder to save the data")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="scaling factor to adjust model strength; higher scaling means lower strength")
    parser.add_argument("--im_scaling", type=float, default=1.0, help="extra scaling applied to image")
    parser.add_argument("--gmap_scaling", type=float, default=1.0, help="extra scaling applied to gmap")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    
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
        model = STCNNT_MRI(config=config)
        model.load_state_dict(status['model'])
    elif args.saved_model_path.endswith(".pts"):
        model = torch.jit.load(args.saved_model_path)
    else:
        model = load_model_onnx(model_dir="", model_file=args.saved_model_path, use_cpu=True)
    return model, config

# -------------------------------------------------------------------------------------------------

def apply_model(data, model, gmap, config, scaling_factor):
    '''
    Input 
        data : [RO E1 N SLC], remove any extra scaling
        gmap : [RO E1 SLC], no scaling added
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (0.5 is more smoothing than 1.0)
    '''

    t0 = time()

    device = get_device()

    if(data.ndim==2):
        data = data[:,:,np.newaxis,np.newaxis]

    if(data.ndim<4):
        data = np.expand_dims(data, axis=3)

    RO, E1, PHS, SLC = data.shape

    if(gmap.ndim==2):
        gmap = np.expand_dims(gmap, axis=2)

    if(gmap.shape[0]!=RO or gmap.shape[1]!=E1 or gmap.shape[2]!=SLC):
        gmap = np.ones(RO, E1, SLC)

    print(f"---> apply_model, preparation took {time()-t0} seconds ")
    print(f"---> apply_model, input array {data.shape}")
    print(f"---> apply_model, gmap array {gmap.shape}")
    print(f"---> apply_model, pad_time {config.pad_time}")
    print(f"---> apply_model, height and width {config.height, config.width}")
    
    c = config
    
    try:
        for k in range(SLC):
            imgslab = data[:,:,:,k]
            gmapslab = gmap[:,:,k]
            
            H, W, T = imgslab.shape
            
            x = np.transpose(imgslab, [2, 0, 1]).reshape([1, T, 1, H, W])
            g = np.repeat(gmapslab[np.newaxis, np.newaxis, np.newaxis, :, :], T, axis=1)
            
            input = np.concatenate((x.real, x.imag, g), axis=2)
            
            with torch.inference_mode():
                if not c.pad_time:
                    cutout = (T, c.height[-1], c.width[-1])
                    overlap = (0, c.height[-1]//2, c.width[-1]//2)
                else:
                    cutout = (c.time, c.height[-1], c.width[-1])
                    overlap = (c.time//2, c.height[-1]//4, c.width[-1]//4)
    
                try:
                    _, output = running_inference(model, input, cutout=cutout, overlap=overlap, device=device)
                except:
                    print(f"{Fore.YELLOW}---> call inference on cpu ...")
                    _, output = running_inference(model, input, cutout=cutout, overlap=overlap, device="cpu")

                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()            
            
            output = np.transpose(output, (3, 4, 2, 1, 0)).squeeze()
            
            if(k==0):
                data_filtered = np.zeros((output.shape[0], output.shape[1], PHS, SLC), dtype=data.dtype)
            
            data_filtered[:,:,:,k] = output[:,:,0,:] + 1j*output[:,:,1,:]

        t1 = time()
        print(f"---> apply_model took {t1-t0} seconds ")

    except Exception as e:
        print(e)
        data_filtered = copy.deepcopy(data)

    return data_filtered

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

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
    
    setup_run(config, dirs=["log_path"])

    # load the data
    image = np.load(os.path.join(args.input_dir, f"im_real.npy")) + np.load(os.path.join(args.input_dir, f"im_imag.npy")) * 1j
    image /= args.im_scaling

    if len(image.shape) == 3:
        image = image[:,:,:,np.newaxis]

    if(image.shape[3]>20):
        image = np.transpose(image, (0, 1, 3, 2))

    RO, E1, frames, slices = image.shape
    print(f"{args.input_dir}, images - {image.shape}")

    gmap = np.load(f"{args.input_dir}/gfactor.npy")
    gmap /= args.gmap_scaling

    if(gmap.ndim==2):
        gmap = np.expand_dims(gmap, axis=2)

    assert gmap.shape[2] == slices
           
    output = apply_model(image.astype(np.complex64), model, gmap.astype(np.float32), config=config, scaling_factor=args.scaling_factor)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    res_name = os.path.join(args.output_dir, 'input_real.npy')
    print(res_name)
    np.save(res_name, image.real)
    res_name = os.path.join(args.output_dir, 'input_imag.npy')
    print(res_name)
    np.save(res_name, image.imag)
        
    res_name = os.path.join(args.output_dir, 'gmap.npy')
    print(res_name)
    np.save(res_name, gmap)
    
    res_name = os.path.join(args.output_dir, 'output_real.npy')
    print(res_name)
    np.save(res_name, output.real)
    res_name = os.path.join(args.output_dir, 'output_imag.npy')
    print(res_name)
    np.save(res_name, output.imag)
            
if __name__=="__main__":
    main()
