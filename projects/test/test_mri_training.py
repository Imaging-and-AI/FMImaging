import time
import pytest
import os

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

class Test_MRI_Tra(object):

    @classmethod
    def setup_class(cls):
        set_seed(23564)
        torch.set_printoptions(precision=10)

    @classmethod
    def teardown_class(cls):
        pass

    def test_hrnet_TLG_TLG(self):

        device = get_device()

        if 'FMI_DATA_ROOT' in os.environ:
            self.data_root = os.environ['FMI_DATA_ROOT']
        else:
            self.data_root = "/export/Lab-Xue/projects/data"

        assert os.path.exists(self.data_root)

        

        # =======================================================

    def test_unet_TLG_TLG(self):

        device = get_device()

        # =======================================================

    def test_hrnet_C3C3C3_C3C3C3(self):

        device = get_device()

        # =======================================================

if __name__ == "__main__":
    pass
