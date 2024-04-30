"""
Run micrsocopy inference for snr pseudo-replica test
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

import torch.multiprocessing as mp

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

    parser.add_argument("--added_noise_sd", type=float, default=0.1, help="add noise sigma")
    parser.add_argument("--rep", type=int, default=32, help="number of repetition")

    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help='devices for inference')

    return parser.parse_args()

def run_a_case(i, images, rank, config, args):

    if rank >=0:
        config.device = torch.device(f'cuda:{rank}')

    print(f"{Fore.RED}--> Processing on device {config.device}{Style.RESET_ALL}")

    model, _ = load_model(args.saved_model_path)

    for ind, v in enumerate(images):
        noisy_im, clean_im, f_name = v
        print(f"--> Processing {f_name}, {noisy_im.shape}, {ind} out of {len(images)}")

        start = time.time()
        noisy_im_N = create_pseudo_replica(noisy_im, added_noise_sd=args.added_noise_sd/args.scaling_vals[1], N=args.rep)
        print(f"--> create_pseudo_replica {time.time()-start:.1f} seconds")

        start = time.time()
        np.save(os.path.join(args.output_dir, f"{f_name}_pesudo_replica_x.npy"), np.transpose(np.squeeze(noisy_im_N), [1, 2, 0, 3])*args.scaling_vals[1])
        print(f"--> save pseudo_replica input data {time.time()-start:.1f} seconds")

        predi_im_N = noisy_im_N

        total_time_in_seconds = 0
        for n in range(args.rep):
            start = time.time()
            a_pred_im = apply_model(model, noisy_im_N[:,:,:,:,:,n], config, config.device, overlap=args.overlap, batch_size=args.batch_size)
            predi_im_N[:,:,:,:,:,n] = a_pred_im.cpu().numpy()
            total_time_in_seconds += time.time()-start
            print(f"----> process rep {n}, out of {args.rep}, {total_time_in_seconds}s")

        print(f"--> Total processing time is {total_time_in_seconds:.1f} seconds")

        predi_im_N *= args.scaling_vals[1]
        if clean_im: clean_im *= args.scaling_vals[1]

        predi_im_N = np.transpose(np.squeeze(predi_im_N), [1, 2, 0, 3])
        if clean_im: clean_im = np.transpose(np.squeeze(clean_im), [1, 2, 0])

        np.save(os.path.join(args.output_dir, f"{f_name}_pesudo_replica_output.npy"), predi_im_N)
        if clean_im:
            np.save(os.path.join(args.output_dir, f"{f_name}_clean_im.npy"), clean_im)

    print(f"{Fore.RED}--> Processing on device {config.device} -- completed {Style.RESET_ALL}")

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

    os.makedirs(args.output_dir, exist_ok=True)

    print("Start inference and saving")

    if args.cuda_devices is None:
        rank = -1
        run_a_case(0, images, rank, config, args)
    else:
        print(f"Perform inference on devices {args.cuda_devices}")

        num_devices = len(args.cuda_devices) if len(args.cuda_devices) < len(images) else len(images)

        def chunkify(lst,n):
            return [lst[i::n] for i in range(n)]

        tasks = chunkify(images, num_devices)

        for ind, a_task in enumerate(tasks):
            mp.spawn(run_a_case, args=(a_task, args.cuda_devices[ind], config, args), nprocs=1, join=False, daemon=False, start_method='spawn')
            print(f"{Fore.RED}--> Spawn task {ind}{Style.RESET_ALL}")

        print(f"Perform inference on devices {args.cuda_devices} for {len(images)} cases -- completed")

    print("End pesudo-replica test and saving")

if __name__=="__main__":
    main()
