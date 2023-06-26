"""
Run MRI inference data for snr test
"""
import json
import wandb
import logging
import argparse
import copy
from time import time
from colorama import Fore, Back, Style

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
from trainer_mri import apply_model, load_model
from noise_augmentation import *

# -------------------------------------------------------------------------------------------------
# setup for testing from cmd

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("STCNNT MRI SNR test. The noise level will de")

    parser.add_argument("--input_dir", default='/export/Lab-Xue/projects/mri/validation/retro_cine/case1', help="folder to load the data")
    parser.add_argument("--output_dir", default=None, help="folder to save the results; if None, $input_dir/res will be the output directory")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="scaling factor to adjust model strength; higher scaling means lower strength")
    parser.add_argument("--im_scaling", type=float, default=1.0, help="extra scaling applied to image")
    parser.add_argument("--gmap_scaling", type=float, default=1.0, help="extra scaling applied to gmap")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')

    parser.add_argument("--input_fname", type=str, default="im", help='input file name')
    parser.add_argument("--gmap_fname", type=str, default="gfactor", help='gmap input file name')

    parser.add_argument("--noise_level", nargs='+', type=float, default=[1.0, 30.0, 30], help='min/max/num_steps for noise sigma')
    parser.add_argument("--num_rep", type=int, default=30, help='number of repetition test')

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

    # -------------------------------------------------------------------------
    # load model    
    # -------------------------------------------------------------------------
    print(f"{Fore.YELLOW}Load in model file - {args.saved_model_path}")
    model, config = load_model(args.saved_model_path, args.saved_model_config)
    
    patch_size_inference = args.patch_size_inference
              
    config.pad_time = args.pad_time
    config.ddp = False
    
    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference
        
    # -------------------------------------------------------------------------
    # load the data
    # -------------------------------------------------------------------------
    image = np.load(os.path.join(args.input_dir, f"{args.input_fname}_real.npy")) + np.load(os.path.join(args.input_dir, f"{args.input_fname}_imag.npy")) * 1j    
    image /= args.im_scaling

    if len(image.shape) == 2:
        image = image[:,:,np.newaxis,np.newaxis]

    if len(image.shape) == 3:
        image = image[:,:,:,np.newaxis]

    if(image.shape[3]>20):
        image = np.transpose(image, (0, 1, 3, 2))

    RO, E1, frames, slices = image.shape
    print(f"{args.input_dir}, images - {image.shape}")

    gmap = np.load(f"{args.input_dir}/{args.gmap_fname}.npy")
    gmap /= args.gmap_scaling

    if(gmap.ndim==2):
        assert slices == 1
        gmap = np.expand_dims(gmap, axis=2)

    if gmap.ndim==3 and gmap.shape[2] == slices:
        gmap = np.expand_dims(gmap, axis=2)
        
    elif gmap.ndim==3 and gmap.shape[2] == frames:
        gmap = np.expand_dims(gmap, axis=3)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'res')
        
    os.makedirs(args.output_dir, exist_ok=True)

    image = image.astype(np.complex64)
    gmap = gmap.astype(np.float32)

    print(f"{args.input_dir}, gmap - {gmap.shape}, median gmap {np.median(gmap)}")
    
    device = get_device()
    
    mse_loss_func = MSE_Loss(complex_i=config.complex_i)
    l1_loss_func = L1_Loss(complex_i=config.complex_i)
    ssim_loss_func = SSIM_Loss(complex_i=config.complex_i, device=device)
    ssim3D_loss_func = SSIM3D_Loss(complex_i=config.complex_i, device=device)
    psnr_loss_func = PSNR_Loss(range=2048)
    psnr_func = PSNR(range=2048)
    
    # -------------------------------------------------------------------------
    # generate noises
    # -------------------------------------------------------------------------
    N = int(args.noise_level[2])
    noisy_image = np.zeros((RO, E1, frames, slices, args.num_rep, N), dtype=image.dtype)
    pred_image = np.zeros((RO, E1, frames, slices, args.num_rep, N), dtype=image.dtype)
    
    record = []
    
    sigmas = np.linspace(args.noise_level[0], args.noise_level[1], int(args.noise_level[2]))
    for rep in tqdm(range(args.num_rep)):
        for ii, sigma in enumerate(sigmas):
            nns, noise_sigma = generate_3D_MR_correlated_noise(T=frames, RO=RO, E1=E1, REP=slices, 
                                                min_noise_level=sigma, 
                                                max_noise_level=sigma, 
                                                kspace_filter_sigma=[0, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
                                                pf_filter_ratio=[1.0, 0.875, 0.75, 0.625, 0.55],
                                                kspace_filter_T_sigma=[0, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25],
                                                phase_resolution_ratio=[1.0, 0.85, 0.7, 0.65, 0.55],
                                                readout_resolution_ratio=[1.0, 0.85, 0.7, 0.65, 0.55],
                                                rng=np.random.Generator(np.random.PCG64(8754132)),
                                                verbose=False)
        
            assert abs(sigma-noise_sigma)<0.00001
        
            if nns.ndim==3:
                nns = np.expand_dims(nns, axis=3)
                
            nns = np.transpose(nns, (1, 2, 0, 3)) # RO, E1, frames, slices
        
            nns *= gmap
        
            if ii < N-1:
                data = nns + np.copy(image)
            else:
                data = nns
                
            y = image / noise_sigma
            data /= noise_sigma
            noisy_image[:,:,:,:,rep,ii] = data
            
            gmap_used = np.copy(gmap[:,:,0,:].squeeze())
            if gmap_used.ndim==2:
                gmap_used = gmap_used[:,:,np.newaxis]
                            
            model.eval()
            output = apply_model(data.astype(np.complex64), model, gmap_used.astype(np.float32), config=config, scaling_factor=args.scaling_factor, device=get_device())
            
            pred_image[:,:,:,:,rep,ii] = output * noise_sigma
            
            noise = (data - image/noise_sigma) / gmap
            for slc in range(slices):
                print(f"rep - {rep}, sigma - {sigma:.2f}, data noise std - {np.mean(np.std(np.real(noise[:,:,:,slc]), axis=2))} - {np.mean(np.std(np.imag(noise[:,:,:,slc]), axis=2))}")
            
            y_hat = np.expand_dims(np.transpose(output, (3, 2, 0, 1)), axis=2)            
            y_hat = np.concatenate((np.real(y_hat), np.imag(y_hat)), axis=2)
            
            y = np.expand_dims(np.transpose(y, (3, 2, 0, 1)), axis=2)            
            y = np.concatenate((np.real(y), np.imag(y)), axis=2)                    
            
            y_hat = torch.from_numpy(y_hat).to(device=device)
            y = torch.from_numpy(y).to(device=device)
            
            y *= noise_sigma
            y_hat *= noise_sigma
            
            mse_loss = mse_loss_func(y_hat, y).item()
            l1_loss = l1_loss_func(y_hat, y).item()
            ssim_loss = ssim_loss_func(y_hat, y).item()
            ssim3D_loss = ssim3D_loss_func(y_hat, y).item()
            psnr_loss = psnr_loss_func(y_hat, y).item()
            psnr = psnr_func(y_hat, y).item()
                
            print(f"rep - {rep}, sigma - {sigma}, mse_loss {mse_loss:.4f}, l1_loss {l1_loss:.4f}, ssim_loss {ssim_loss:.4f}, ssim3D_loss {ssim3D_loss:.4f}, psnr_loss {psnr_loss:.4f}, psnr {psnr:.4f}")
            
            record.append([rep, sigma, mse_loss, l1_loss, ssim_loss, ssim3D_loss, psnr_loss, psnr])
            
    print(record)
    print(f"{args.input_dir}, noisy_image - {noisy_image.shape}")
    
    np.save(os.path.join(args.output_dir, 'noisy_image_no_scaling_real.npy'), noisy_image.real)
    np.save(os.path.join(args.output_dir, 'noisy_image_no_scaling_imag.npy'), noisy_image.imag)
    np.save(os.path.join(args.output_dir, 'noisy_image_no_scaling.npy'), np.abs(noisy_image))
    
    np.save(os.path.join(args.output_dir, 'sigmas.npy'), sigmas)
    
    for rep in tqdm(range(args.num_rep)):
        for ii, sigma in enumerate(sigmas):
            rec = record[ii+rep*sigmas.shape[0]]
            print(f"{Fore.GREEN}rep - {rep}, sigma - {sigma}, {Fore.YELLOW}mse_loss {rec[2]:.4f}, l1_loss {rec[3]:.4f}, ssim_loss {rec[4]:.4f}, ssim3D_loss {rec[5]:.4f}, psnr_loss {rec[6]:.4f}, psnr {rec[7]:.4f}{Style.RESET_ALL}")
            
    # -------------------------------------------------------------------------
    # apply the model
    # -------------------------------------------------------------------------
    
    # pred_image = np.zeros((RO, E1, frames, slices, args.num_rep, N), dtype=image.dtype)
    
    # for rep in range(args.num_rep):
    #     for ii, sigma in enumerate(sigmas):
            
    #         im = np.copy(noisy_image[:,:,:,:,rep,ii])
    #         gmap_used = np.copy(gmap[:,:,1,:].squeeze())
    #         if gmap_used.ndim==2:
    #             gmap_used = gmap_used[:,:,np.newaxis]
                
    #         print(f"{Fore.GREEN}--> process noise sigma {sigma}, rep {rep}, im - {im.shape}, gmap - {gmap_used.shape} - {np.median(gmap_used)} ...{Style.RESET_ALL}")
            
    #         output = apply_model(im, model, gmap_used, config=config, scaling_factor=args.scaling_factor, device=get_device())
            
    #         # input = np.flip(im, axis=0)
    #         # output2 = apply_model(input, model, np.flip(gmap_used, axis=0), config=config, scaling_factor=args.scaling_factor, device=get_device())
    #         # output2 = np.flip(output2, axis=0)

    #         # input = np.flip(im, axis=1)
    #         # output3 = apply_model(input, model, np.flip(gmap_used, axis=1), config=config, scaling_factor=args.scaling_factor, device=get_device())
    #         # output3 = np.flip(output3, axis=1)

    #         # input = np.transpose(im, axes=(1, 0, 2, 3))
    #         # output4 = apply_model(input, model, np.transpose(gmap_used, axes=(1, 0, 2)), config=config, scaling_factor=args.scaling_factor, device=get_device())
    #         # output4 = np.transpose(output4, axes=(1, 0, 2, 3))

    #         # res = output + output2 + output3 + output4
    #         # output = res / 4
            
    #         pred_image[:,:,:,:,rep,ii] = output * sigma
    #         print(f"{Fore.GREEN}--> =========================================================================== <--{Style.RESET_ALL}")

    print(f"{args.output_dir}, images - {image.shape}, gmap - {gmap.shape}, pred - {pred_image.shape}")

    # -------------------------------------------------------------------------
    # save results
    # -------------------------------------------------------------------------
       
    save_inference_results(image, pred_image, gmap, args.output_dir, noisy_image)

if __name__=="__main__":
    main()
