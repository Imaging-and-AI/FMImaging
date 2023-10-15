"""
Run QPerf inference
"""

import argparse
import copy
import numpy as np
import scipy
from time import time

import os
import sys
import logging

from colorama import Fore, Back, Style
import nibabel as nib

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from trainer import *
from utils.status import model_info, start_timer, end_timer, support_bfloat16
from setup import setup_logger, Nestedspace
from optim.optim_utils import compute_total_steps

# Default functions
from setup import parse_config_and_setup_run, config_to_yaml
from optim.optim_base import OptimManager
from utils.status import get_device

# Custom functions
from qperf_parser import qperf_parser
from qperf_data import QPerfDataSet
from qperf_loss import qperf_loss
from qperf_model import QPerfModel
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

    parser.add_argument("--input_data", default=None, help="input mat data")
    parser.add_argument("--output_dir", default=None, help="folder to save the data")

    parser.add_argument("--model", type=str, default=None, help="Qperf model")

    return parser.parse_args()

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    args = arg_parser()
    print(args)

    print(f"---> support bfloat16 is {support_bfloat16(device=get_device())}")

    print(f"{Fore.YELLOW}Load in model file - {args.model}")
    status = torch.load(args.model)
    config = status['config']

    foot_to_end = config.foot_to_end

    model = QPerfModel(config=config, 
                       n_layer=config.n_layer, 
                       input_D=2, 
                       output_myo_D=1, 
                       num_params=5, 
                       T=config.T, 
                       is_causal=False, 
                       use_pos_embedding=config.use_pos_embedding, 
                       n_embd=config.n_embd, 
                       n_head=config.n_head, 
                       dropout_p=config.dropout_p, 
                       att_dropout_p=config.att_dropout_p, 
                       residual_dropout_p=config.residual_dropout_p)

    # load data
    t0 = time.time()
    mat = scipy.io.loadmat(args.input_data)
    t1 = time.time()
    print(f"Load mat file {args.input_data} takes {(t1-t0):.2f}s ...")

    N = mat['out'].shape[1]
    print(f"--> Find {N} cases - {args.input_data} ... ")

    T = config.qpref_T
    x = np.zeros((N, T, 2), dytpe=np.float32)

    pbar = tqdm(total=N)
    for ind in range(mat['out'].shape[1]):
        params = mat['out'][0,ind]['params'][0,0].flatten()
        aif = mat['out'][0,ind]['aif'][0,0].flatten().astype(np.float32)
        myo = mat['out'][0,ind]['myo'][0,0].flatten().astype(np.float32)

        a_x = np.concatenate((aif, myo), axis=1)
        L = a_x.shape[0]

        foot = int(params[6])

        if foot_to_end and foot < L/2 and foot > 3:
            a_x = a_x[foot:, :]
            L = a_x.shape[0]

        if L >= T:
            a_x = a_x[:T, :]
        else:
            a_x = np.append(a_x, a_x[L-1: 2] * np.ones( (T-N, 2)), axis=0)

        x[ind] = a_x

    # run inference
    device = get_device()
    model.to(device=device)
    x.to(device=device)

    t0 = time.time()
    with torch.inference_mode():
        output = model(x)
    print(f"Running model takes {(t1-t0):.2f}s ...")

    y, p_est = output
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    p_est = p_est.cpu().numpy()

    fname = os.path.join(args.output_dir, 'aif_myo.npy')
    print(f"--> save to {fname}")
    np.save(fname, x)

    fname = os.path.join(args.output_dir, 'myo_pred.npy')
    print(f"--> save to {fname}")
    np.save(fname, y)

    fname = os.path.join(args.output_dir, 'params_pred.npy')
    print(f"--> save to {fname}")
    np.save(fname, p_est)

if __name__=="__main__":
    main()
