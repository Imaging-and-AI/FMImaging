import pytest
import os

import shutil
import copy
import numpy as np
import time

import os
import sys
import logging
import pickle
import json

from colorama import Fore, Back, Style
import nibabel as nib

import subprocess

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
from setup import setup_logger, Nestedspace, set_seed
from optim.optim_utils import compute_total_steps

# Default functions
from setup import parse_config_and_setup_run, config_to_yaml
from optim.optim_base import OptimManager
from utils.status import get_device

# Custom functions
from mri.mri_parser import mri_parser
from mri.mri_data import MRIDenoisingDatasetTrain, load_mri_data
from mri.mri_loss import mri_loss 
from mri.mri_model import STCNNT_MRI, MRI_hrnet, MRI_double_net, omnivore_MRI, create_model
from mri.LSUV import LSUVinit
from mri.mri_metrics import MriMetricManager
from mri.mri_trainer import MRITrainManager, get_rank_str
from mri.inference import apply_model, load_model, apply_model_3D, load_model_pre_backbone_post

# --------------------------------------------------------
def get_test_folders():
    if 'FMI_DATA_ROOT' in os.environ:
        data_root = os.environ['FMI_DATA_ROOT'] + "/mri"
    else:
        data_root = "/export/Lab-Xue/projects/data/mri/data"

    if 'FMI_LOG_ROOT' in os.environ:
        log_root = os.environ['FMI_LOG_ROOT'] + "/mri"
    else:
        log_root = "/export/Lab-Xue/projects/logs/mri"

    assert os.path.exists(data_root)
    os.makedirs(log_root, exist_ok=True)

    return data_root, log_root

# --------------------------------------------------------

num_epochs=10
tra_ratio=10
val_ratio=10
test_ratio=10

class Test_MRI_Tra(object):

    @classmethod
    def setup_class(cls):
        set_seed(23564)
        torch.set_printoptions(precision=10)
        assert torch.cuda.is_available() and torch.cuda.device_count()>=2

    @classmethod
    def teardown_class(cls):
        os.system("kill -9 $(ps aux | grep torchrun | grep -v grep | awk '{print $2}') ")
        os.system("kill -9 $(ps aux | grep wandb | grep -v grep | awk '{print $2}') ")

    def run_training(self, data_root, log_root, cmd_run, run_folder):

        shutil.rmtree(log_root + "/" + run_folder, ignore_errors=True)

        logging.info(f"--> test run, {data_root}, {log_root}")
        logging.info(f"{Fore.YELLOW}{cmd_run}{Style.RESET_ALL}", flush=True)

        logging.info("===" * 20)
        logging.info(f"{Fore.GREEN}{cmd_run}{Style.RESET_ALL}")
        logging.info("--" * 20)
        logging.info(f"Running command:\n{Fore.WHITE}{Back.BLUE}{' '.join(cmd_run)}{Style.RESET_ALL}")
        time.sleep(3)
        fname = log_root+"/run.log"
        logging.info(f"Running command:\n{Fore.YELLOW}{Back.BLUE}{fname}{Style.RESET_ALL}")

        with open(fname, 'w') as f: 
            subprocess.run(cmd_run, stdout=f)

        logging.info("===" * 20)

        # check the test scores
        logging.info(f"--> read in from {run_folder}")

        metric_file = log_root + "/" + run_folder + "/test_metrics.json"
        with open(metric_file, 'r') as f: 
            metrics = json.load(f)

        logging.info(f"{Fore.YELLOW}{Back.RED}{metrics}{Style.RESET_ALL} ")

        return metrics

    def test_hrnet_TLG_TLG(self):

        data_root, log_root = get_test_folders()

        cmd_run = ["python3", str(Project_DIR)+"/projects/mri/inference/run_mri.py", 
                   "--standalone", 
                   "--nproc_per_node", f"{torch.cuda.device_count()}", 
                   "--use_amp", 
                   "--num_epochs", f"{num_epochs}", 
                   "--batch_size", "8", 
                   "--data_root", data_root, 
                   "--log_root", log_root, 
                   "--run_extra_note", "test_hrnet_TLG_TLG", 
                   "--num_workers", "32", 
                   "--model_backbone", "STCNNT_HRNET", 
                   "--model_type", "STCNNT_MRI", 
                   "--model_block_str", "T1L1G1", "T1L1G1", 
                   "--mri_height", "32", "64", 
                   "--mri_width", "32", "64", 
                   "--global_lr", "1e-4",
                   "--lr_pre", "1e-4", 
                   "--lr_post", "1e-4", 
                   "--lr_backbone", "1e-4", 
                   "--run_list", "0",  
                   "--tra_ratio", f"{tra_ratio}", 
                   "--val_ratio", f"{val_ratio}", 
                   "--test_ratio", f"{test_ratio}",
                   "--scheduler_factor", "0.5", 
                   "--ut_mode", "--scheduler_type", "OneCycleLR",
                   "--losses", "mse", "perpendicular", "perceptual", "charbonnier", "gaussian3D", "--loss_weights", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0",
                   "--backbone_C", "32", "--add_salt_pepper", "--add_possion", "--weighted_loss_snr",
                   "--project", "FM-UT-MRI"]

        metrics = self.run_training(data_root, log_root, cmd_run, 'FM-UT-MRI-test_hrnet_TLG_TLG_STCNNT_HRNET_T1L1G1_T1L1G1_STCNNT_MRI_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1')

        # assert metrics['mse'] < 150
        # assert metrics['l1'] < 12
        assert metrics['ssim'] > 0.3
        assert metrics['psnr'] > 42

        # =======================================================

    def test_unet_TLG_TLG(self):

        data_root, log_root = get_test_folders()

        cmd_run = ["python3", str(Project_DIR)+"/projects/mri/inference/run_mri.py", 
                   "--standalone", 
                   "--nproc_per_node", f"{torch.cuda.device_count()}", 
                   "--use_amp", 
                   "--num_epochs", f"{num_epochs}", 
                   "--batch_size", "8", 
                   "--data_root", data_root, 
                   "--log_root", log_root, 
                   "--run_extra_note", "test_unet_TLG_TLG", 
                   "--num_workers", "32", 
                   "--model_backbone", "STCNNT_UNET", 
                   "--model_type", "STCNNT_MRI", 
                   "--model_block_str", "T1L1G1", "T1L1G1", 
                   "--mri_height", "32", "64", 
                   "--mri_width", "32", "64", 
                   "--global_lr", "1e-4",
                   "--lr_pre", "1e-4", 
                   "--lr_post", "1e-4", 
                   "--lr_backbone", "1e-4", 
                   "--run_list", "0",  
                   "--tra_ratio", f"{tra_ratio}", 
                   "--val_ratio", f"{val_ratio}", 
                   "--test_ratio", f"{test_ratio}",
                   "--scheduler_factor", "0.5", 
                   "--ut_mode", "--scheduler_type", "OneCycleLR",
                   "--losses", "mse", "perpendicular", "perceptual", "charbonnier", "gaussian3D", "--loss_weights", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0",
                   "--backbone_C", "32", "--add_salt_pepper", "--add_possion", "--weighted_loss_snr",
                   "--project", "FM-UT-MRI"]

        metrics = self.run_training(data_root, log_root, cmd_run, 'FM-UT-MRI-test_unet_TLG_TLG_STCNNT_UNET_T1L1G1_T1L1G1_STCNNT_MRI_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1')

        assert metrics['mse'] < 150
        assert metrics['l1'] < 12
        assert metrics['ssim'] > 0.55
        assert metrics['psnr'] > 50

        # =======================================================

    def test_hrnet_C3C3C3_C3C3C3(self):

        data_root, log_root = get_test_folders()

        cmd_run = ["python3", str(Project_DIR)+"/projects/mri/inference/run_mri.py", 
                   "--standalone", 
                   "--nproc_per_node", f"{torch.cuda.device_count()}", 
                   "--use_amp", 
                   "--num_epochs", f"{num_epochs}", 
                   "--batch_size", "8", 
                   "--data_root", data_root, 
                   "--log_root", log_root, 
                   "--run_extra_note", "test_hrnet_C3C3C3_C3C3C3", 
                   "--num_workers", "32", 
                   "--model_backbone", "STCNNT_HRNET", 
                   "--model_type", "STCNNT_MRI", 
                   "--model_block_str", "C3C3C3", "C3C3C3", 
                   "--mri_height", "32", "64", 
                   "--mri_width", "32", "64", 
                   "--global_lr", "1e-4",
                   "--lr_pre", "1e-4", 
                   "--lr_post", "1e-4", 
                   "--lr_backbone", "1e-4", 
                   "--run_list", "0",  
                   "--tra_ratio", f"{tra_ratio}", 
                   "--val_ratio", f"{val_ratio}", 
                   "--test_ratio", f"{test_ratio}",
                   "--scheduler_factor", "0.8", 
                   "--ut_mode", "--scheduler_type", "OneCycleLR",
                   "--losses", "mse", "perpendicular", "perceptual", "charbonnier", "gaussian3D", "--loss_weights", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0",
                   "--backbone_C", "32", "--add_salt_pepper", "--add_possion", "--weighted_loss_snr",
                   "--project", "FM-UT-MRI"]

        metrics = self.run_training(data_root, log_root, cmd_run, 'FM-UT-MRI-test_hrnet_C3C3C3_C3C3C3_STCNNT_HRNET_C3C3C3_C3C3C3_STCNNT_MRI_C-64-1_amp-True_complex_residual-C3C3C3_C3C3C3')

        assert metrics['mse'] < 420
        assert metrics['l1'] < 17.5
        assert metrics['ssim'] > 0.42
        assert metrics['psnr'] > 45

        # =======================================================

if __name__ == "__main__":
    pass
