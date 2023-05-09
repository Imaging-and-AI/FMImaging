"""
Main file for STCNNT MRI denoising
"""
import logging
import argparse
import torch.multiprocessing as mp

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from trainer_mri import trainer
from model_mri import STCNNT_MRI
from data_mri import load_mri_data

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
    parser.add_argument("--test_files", type=str, nargs='+', default=["train_3D_3T_retro_cine_2020_small_2DT_test.h5"], help='list of test h5files')
    parser.add_argument("--train_data_types", type=str, nargs='+', default=["2dt"], help='the type of each train file: "2d", "2dt", "3d"')
    parser.add_argument("--test_data_types", type=str, nargs='+', default=["2dt"], help='the type of each test file: "2d", "2dt", "3d"')
    parser = add_backbone_STCNNT_args(parser=parser)

    # Noise Augmentation arguments
    parser.add_argument("--min_noise_level", type=float, default=3.0, help='minimum noise sigma to add')
    parser.add_argument("--max_noise_level", type=float, default=6.0, help='maximum noise sigma to add')
    parser.add_argument('--matrix_size_adjust_ratio', type=float, nargs='+', default=[0.5, 0.75, 1.0, 1.25, 1.5], help='down/upsample the image, keeping the fov')
    parser.add_argument('--kspace_filter_sigma', type=float, nargs='+', default=[0.8, 1.0, 1.5, 2.0, 2.25], help='sigma for kspace filter')
    parser.add_argument('--pf_filter_ratio', type=float, nargs='+', default=[1.0, 0.875, 0.75, 0.625], help='pf filter ratio')
    parser.add_argument('--phase_resolution_ratio', type=float, nargs='+', default=[1.0, 0.75, 0.65, 0.55], help='phase resolution ratio')
    parser.add_argument('--readout_resolution_ratio', type=float, nargs='+', default=[1.0, 0.75, 0.65, 0.55], help='readout resolution ratio')

    # 2d/3d dataset arguments
    parser.add_argument('--twoD_num_patches_cutout', type=int, default=1, help='for 2D usecase, number of patches per frame')
    parser.add_argument("--twoD_patches_shuffle", action="store_true", help='shuffle 2D patches to break spatial consistency')
    parser.add_argument('--threeD_cutout_jitter', nargs='+', type=float, default=[-1, 0.5, 0.75, 1.0], help='cutout jitter range, relative to the cutout_shape')
    parser.add_argument("--threeD_cutout_shuffle_time", action="store_true", help='shuffle along time to break temporal consistency; for 2D+T, should not set this option')

    # loss for mri
    parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1"], help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D"')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0, 1.0], help='to balance multiple losses, weights can be supplied')
    parser.add_argument("--complex_i", action="store_true", help='whether we are dealing with complex images or not')
    parser.add_argument("--residual", action="store_true", help='add long term residual connection')

    ns = Nestedspace()
    args = parser.parse_args(namespace=ns)
    
    return args

def check_args(config):
    """
    checks the cmd args to make sure they are correct
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - config (Namespace): the checked and updated argparse for MRI
    """
    assert config.run_name is not None, f"Please provide a \"--run_name\" for wandb"
    assert config.data_root is not None, f"Please provide a \"--data_root\" to load the data"
    assert config.log_path is not None, f"Please provide a \"--log_path\" to save the logs in"
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.model_path is not None, f"Please provide a \"--model_path\" to save the final model in"
    assert config.check_path is not None, f"Please provide a \"--check_path\" to save the checkpoints in"

    config.C_in = 3 if config.complex_i else 2
    config.C_out = 2 if config.complex_i else 1

    return config

# -------------------------------------------------------------------------------------------------
# main function. spawns threads if going for distributed data parallel

def main():

    config = check_args(arg_parser())
    setup_run(config)

    train_set, val_set, test_set = load_mri_data(config=config)


    num_samples = sum([len(tset) for tset in train_set])
    
    config.batch_size = num_samples if config.batch_size>=num_samples else config.batch_size
    total_steps = (num_samples//config.batch_size + 1) * config.num_epochs
    if config.ddp: 
        total_steps //= torch.cuda.device_count()
        total_steps += 1
    
    model = STCNNT_MRI(config=config, total_steps=total_steps)

    # model summary
    model_summary = model_info(model, config)
    logging.info(f"Configuration for this run:\n{config}")
    logging.info(f"Model Summary:\n{str(model_summary)}")

    if not config.ddp: # run in main process
        trainer(rank=-1, model=model, config=config,
                train_set=train_set, val_set=val_set, test_set=test_set)
    else: # spawn a process for each gpu
        mp.spawn(trainer, args=(model, config, train_set, val_set, test_set),
                    nprocs=config.world_size)

if __name__=="__main__":
    main()
