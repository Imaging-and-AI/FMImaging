"""
Main file for STCNNT MRI denoising
"""
import logging
import argparse

import torchvision as tv
from torchvision import transforms
import torchvision.transforms as T
import torch.multiprocessing as mp
import torch.distributed as dist

import sys
from colorama import Fore, Style
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from trainer_mri import trainer
from model_mri import STCNNT_MRI
from data_mri import load_mri_data
from trainer_base import Trainer_Base

# -------------------------------------------------------------------------------------------------
# Extra args on top of shared args

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT MRI")
    parser.add_argument("--data_root", type=str, default=None, help='root folder for the data')
    parser.add_argument("--train_files", type=str, nargs='+', default=["train_3D_3T_retro_cine_2020_small.h5"], help='list of train h5files')
    parser.add_argument("--test_files", type=none_or_str, nargs='+', default=["train_3D_3T_retro_cine_2020_small_2DT_test.h5"], help='list of test h5files')
    parser.add_argument("--train_data_types", type=str, nargs='+', default=["2dt"], help='the type of each train file: "2d", "2dt", "3d"')
    parser.add_argument("--test_data_types", type=str, nargs='+', default=["2dt"], help='the type of each test file: "2d", "2dt", "3d"')
    parser.add_argument("--max_load", type=int, default=-1, help='number of samples to load into the disk, if <0, samples will be read from the disk while training')
    
    parser = add_backbone_STCNNT_args(parser=parser)

    # Noise Augmentation arguments
    parser.add_argument("--min_noise_level", type=float, default=2.0, help='minimum noise sigma to add')
    parser.add_argument("--max_noise_level", type=float, default=14.0, help='maximum noise sigma to add')
    parser.add_argument('--matrix_size_adjust_ratio', type=float, nargs='+', default=[0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], help='down/upsample the image, keeping the fov')
    parser.add_argument('--kspace_filter_sigma', type=float, nargs='+', default=[0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], help='sigma for kspace filter')
    parser.add_argument('--kspace_T_filter_sigma', type=float, nargs='+', default=[0.25, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25], help='sigma for T filter')
    parser.add_argument('--pf_filter_ratio', type=float, nargs='+', default=[1.0, 0.875, 0.75, 0.625, 0.55], help='pf filter ratio')
    parser.add_argument('--phase_resolution_ratio', type=float, nargs='+', default=[1.0, 0.85, 0.7, 0.65, 0.55], help='phase resolution ratio')
    parser.add_argument('--readout_resolution_ratio', type=float, nargs='+', default=[1.0, 0.85, 0.7, 0.65, 0.55], help='readout resolution ratio')
    parser.add_argument("--snr_perturb_prob", type=float, default=0.1, help='prob to add snr perturbation')
    parser.add_argument("--snr_perturb", type=float, default=0.15, help='strength of snr perturbation')    
    parser.add_argument("--with_data_degrading", action="store_true", help='if true, degrade data for reduced resolution, temporal smoothing etc.')
    parser.add_argument("--not_add_noise", action="store_true", help='if set, not add noise.')

    # 2d/3d dataset arguments
    parser.add_argument('--twoD_num_patches_cutout', type=int, default=1, help='for 2D usecase, number of patches per frame')
    parser.add_argument("--twoD_patches_shuffle", action="store_true", help='shuffle 2D patches to break spatial consistency')
    parser.add_argument('--threeD_cutout_jitter', nargs='+', type=float, default=[-1, 0.5, 0.75, 1.0], help='cutout jitter range, relative to the cutout_shape')
    parser.add_argument("--threeD_cutout_shuffle_time", action="store_true", help='shuffle along time to break temporal consistency; for 2D+T, should not set this option')

    # inference
    parser.add_argument("--pad_time", action="store_true", help='whehter to pad along time when doing inference; if False, the entire series is inputted')
    
    # loss for mri
    parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1"], help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D", "psnr", "msssim", "perpendicular", "gaussian", "gaussian3D" ')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0, 1.0], help='to balance multiple losses, weights can be supplied')
    parser.add_argument("--complex_i", action="store_true", help='whether we are dealing with complex images or not')
    parser.add_argument("--residual", action="store_true", help='add long term residual connection')
    parser.add_argument("--weighted_loss", action="store_true", help='if set, weight loss by gfactor and noise values')

    # training
    parser.add_argument('--num_uploaded', type=int, default=12, help='number of images uploaded to wandb')

    ns = Nestedspace()
    args = parser.parse_args(namespace=ns)
    
    return args

# -------------------------------------------------------------------------------------------------

class MriTrainer(Trainer_Base):
    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
        """
        super().__init__(config)
        self.project = 'mri'

    def check_args(self):
        """
        checks the cmd args to make sure they are correct
        """
        
        super().check_args()
        
        self.config.C_in = 3 if self.config.complex_i else 2
        self.config.C_out = 2 if self.config.complex_i else 1

        if self.config.data_root is None:
            self.config.data_root = "/export/Lab-Xue/projects/mri/data"
            
    def set_up_config_for_sweep(self, wandb_config):
        super().set_up_config_for_sweep(wandb_config=wandb_config)
        
        self.config.backbone = wandb_config.backbone
        
        self.config.optim = wandb_config.optim
        
        self.config.height = wandb_config.width
        self.config.width = wandb_config.width
        
        self.config.train_files = wandb_config.train_files[0]
        self.config.train_data_types = wandb_config.train_files[1]
        
        self.config.test_files = None
        self.config.test_data_types = None
        
    def run_task_trainer(self, rank=-1, global_rank=-1, wandb_run=None):
        trainer(rank=rank, global_rank=global_rank, config=self.config, wandb_run=wandb_run)
        
# -------------------------------------------------------------------------------------------------
def main():

    config_default = arg_parser()
    
    trainer = MriTrainer(config_default)
    trainer.train()

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()
