"""
qperf run
"""


import copy
import numpy as np
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
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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
from qperf_data import QPerfDataSet, load_qperf_data
from qperf_loss import qperf_btex_loss
from qperf_model import QPerfBTEXModel
from qperf_metrics import QPerfBTEXMetricManager
from qperf_trainer import QPerfBTEXTrainManager, get_rank_str

# -------------------------------------------------------------------------------------------------

def main():

    # -----------------------------------------------

    config = parse_config_and_setup_run(qperf_parser) 

    # -----------------------------------------------

    if config.ddp:
        rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        device = torch.device(f"cuda:{rank}")
        config.device = device
    else:
        rank = -1
        global_rank = -1
        device = get_device()

    rank_str = get_rank_str(rank)

    # -----------------------------------------------

    # Save config to yaml file
    if rank<=0:
        yaml_file = config_to_yaml(config,os.path.join(config.log_dir, config.run_name))
        config.yaml_file = yaml_file

    # -----------------------------------------------

    train_set, val_set, test_set = load_qperf_data(config)

    # -----------------------------------------------

    loss_f = qperf_btex_loss(config=config)

    # -----------------------------------------------

    ddp = config.ddp

    # -----------------------------------------------

    pre_model_load_path = config.pre_model_load_path
    backbone_model_load_path = config.backbone_model_load_path
    post_model_load_path = config.post_model_load_path

    if config.pre_model_load_path is not None:
        status = torch.load(config.pre_model_load_path)
        config_for_model = status['config']
        config_for_model.device = device
        config_for_model.ddp = ddp
    else:
        config_for_model = config

    # -----------------------------------------------

    model = QPerfBTEXModel(config=config_for_model, 
                    n_layer=config_for_model.n_layer[0], 
                    input_D=1, 
                    output_myo_D=1, 
                    num_params=5, 
                    T=config.qperf_T, 
                    is_causal=False, 
                    use_pos_embedding=config_for_model.use_pos_embedding, 
                    n_embd=config_for_model.n_embd, 
                    n_head=config_for_model.n_head, 
                    dropout_p=config.dropout_p, 
                    att_dropout_p=config.att_dropout_p, 
                    residual_dropout_p=config.residual_dropout_p)

    # -----------------------------------------------

    if pre_model_load_path is not None:
        model.load_pre(pre_model_load_path, device=device)

    if backbone_model_load_path is not None:
        model.load_backbone(backbone_model_load_path, device=device)

    if post_model_load_path is not None:
        model.load_post(post_model_load_path, device=device)

    # -----------------------------------------------

    model = model.to(device)

    optim_manager = OptimManager(config=config, model_manager=model, train_set=train_set)

    print(f"{rank_str}, after initializing model, the config for running - {config}")
    print(f"{rank_str}, after initializing model, config.use_amp for running - {config.use_amp}")
    print(f"{rank_str}, after initializing model, config.optim for running - {config.optim}")
    print(f"{rank_str}, after initializing model, config.scheduler_type for running - {config.scheduler_type}")
    print(f"{rank_str}, after initializing model, config.num_workers for running - {config.num_workers}")

    print(f"{rank_str}, after initializing model, optim_manager.curr_epoch for running - {optim_manager.curr_epoch}")
    print(f"{rank_str}, {Fore.RED}after initializing model, model.device - {model.device}{Style.RESET_ALL}")
    print(f"{rank_str}, {Fore.WHITE}=============================================================={Style.RESET_ALL}")

    if config.ddp:
        dist.barrier()

    # -----------------------------------------------

    metric_manager = QPerfBTEXMetricManager(config=config)

    # -----------------------------------------------

    trainer = QPerfBTEXTrainManager(config=config,
                            train_sets=[train_set],
                            val_sets=[val_set],
                            test_sets=[test_set],
                            loss_f=loss_f,
                            model_manager=model,
                            optim_manager=optim_manager,
                            metric_manager=metric_manager)

    # -----------------------------------------------

    trainer.train()

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()
