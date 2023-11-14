"""
Run QPerf inference
"""

import argparse
import copy
import numpy as np
import scipy
import time

import os
import sys
import logging

from colorama import Fore, Back, Style
import nibabel as nib

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[3].resolve()
sys.path.append(str(REPO_DIR))

# Default functions
from utils.status import get_device, support_bfloat16

# Custom functions
from trainer import *
from qperf_parser import qperf_parser
from qperf_data import QPerfDataSet, denormalize_data
from qperf_loss import qperf_loss
from qperf_model import QPerfModel, QPerfBTEXModel, QPerfParamsModel
from qperf_metrics import QPerfMetricManager
from qperf_trainer import QPerfTrainManager, get_rank_str

# -------------------------------------------------------------------------------------------------
# setup for testing from cmd

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for QPerf evaluation")

    parser.add_argument("--data_folder", default=None, help="input mat data folder")
    parser.add_argument("--output_dir", default=None, help="folder to save the data")
    parser.add_argument("--max_samples", type=int, default=-1, help='max number of samples used in testing')
    parser.add_argument("--model", type=str, default=None, help="Qperf model")

    return parser.parse_args()

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    args = arg_parser()
    print(args)

    print(f"---> support bfloat16 is {support_bfloat16(device=get_device())}")

    pre_model = args.model + "_pre.pth"
    backbone_model = args.model + "_backbone.pth"
    post_model = args.model + "_post.pth"

    print(f"{Fore.YELLOW}Load in model file - {pre_model}, {backbone_model}, {post_model}")
    status = torch.load(pre_model)
    config = status['config']

    model = QPerfParamsModel(config=config, 
                       n_layer=config.n_layer[0], 
                       input_D=2, 
                       output_myo_D=1, 
                       num_params=5, 
                       T=config.qperf_T, 
                       is_causal=False, 
                       use_pos_embedding=config.use_pos_embedding, 
                       n_embd=config.n_embd, 
                       n_head=config.n_head, 
                       dropout_p=config.dropout_p, 
                       att_dropout_p=config.att_dropout_p, 
                       residual_dropout_p=config.residual_dropout_p)

    model.load_entire_model(save_path=None, save_file_name=args.model)

    test_set = QPerfDataSet(data_folder=args.data_folder, 
                        max_load=-1, max_samples=args.max_samples,
                        T=config.qperf_T, 
                        foot_to_end=config.foot_to_end, 
                        min_noise_level=config.min_noise_level, 
                        max_noise_level=config.max_noise_level,
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=True,
                        add_noise=[False, False],
                        cache_folder=None,
                        load_cache=False)

    N = len(test_set)
    print(test_set)

    data_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False, sampler=None,
                                    num_workers=8, prefetch_factor=config.prefetch_factor, drop_last=True,
                                    persistent_workers=False, pin_memory=False)

    total_iters = len(data_loader)
    data_loader_iter = iter(data_loader)

    dtype = torch.float32

    device = get_device()
    model.to(device=device)
    model.eval()

    with torch.inference_mode():
        with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:
            for idx in range(total_iters):

                loader_outputs = next(data_loader_iter, None)
                x, y, p = loader_outputs

                x = x.to(device=device, dtype=dtype)
                y = y.to(device, dtype=dtype)
                p = p.to(device, dtype=dtype)

                myo_est, p_est = model(x)

                if idx == 0:
                    B = x.shape[0]
                    perf_data = np.zeros((N, x.shape[1], 3)) # aif, myo, myo_est
                    p_data = np.zeros((N, 5, 2)) # p and p_est

                x = x.to(dtype=torch.float32).detach().cpu().numpy()
                y = y.to(dtype=torch.float32).detach().cpu().numpy()
                myo_est = myo_est.to(dtype=torch.float32).detach().cpu().numpy()
                p = p.to(dtype=torch.float32).detach().cpu().numpy()
                p_est = p_est.to(dtype=torch.float32).detach().cpu().numpy()

                x[:, :, 0] += 2.0
                x[:, :, 1] += 0.5
                y += 0.5
                myo_est += 0.5

                perf_data[idx*B:(idx+1)*B,:,0] = x[:, :, 0]
                perf_data[idx*B:(idx+1)*B,:,1] = x[:, :, 1]

                perf_data[idx*B:(idx+1)*B,:,2] = np.squeeze(myo_est)

                p_data[idx*B:(idx+1)*B,:,0] = p[:, :5]
                p_data[idx*B:(idx+1)*B,:,1] = p_est

                pbar.update(1)

    os.makedirs(args.output_dir, exist_ok=True)

    fname = os.path.join(args.output_dir, 'perf_data.npy')
    print(f"--> save to {fname}")
    np.save(fname, perf_data)

    fname = os.path.join(args.output_dir, 'p_data.npy')
    print(f"--> save to {fname}")
    np.save(fname, p_data)

if __name__=="__main__":
    main()
