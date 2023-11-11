"""
self.parser for the MRI project
"""

import argparse
import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from setup import none_or_str

class mri_parser(object):
    """
    MRI self.parser for project specific arguments
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser("Argument self.parser for STCNNT MRI")

        self.parser.add_argument("--data_root", type=str, default=None, help='root folder for the data')
        self.parser.add_argument("--train_files", type=str, nargs='+', default=["train_3D_3T_retro_cine_2020_small.h5"], help='list of train h5files')
        self.parser.add_argument("--test_files", type=none_or_str, nargs='+', default=["train_3D_3T_retro_cine_2020_small_2DT_test.h5"], help='list of test h5files')
        self.parser.add_argument("--train_data_types", type=str, nargs='+', default=["2dt"], help='the type of each train file: "2d", "2dt", "3d"')
        self.parser.add_argument("--test_data_types", type=str, nargs='+', default=["2dt"], help='the type of each test file: "2d", "2dt", "3d"')
        self.parser.add_argument("--max_load", type=int, default=-1, help='number of samples to load into the disk, if <0, samples will be read from the disk while training')

        self.parser.add_argument("--mri_height", nargs='+', type=int, default=[32, 64], help='heights of the training images')
        self.parser.add_argument("--mri_width", nargs='+', type=int, default=[32, 64], help='widths of the training images')

        # dataset arguments
        self.parser.add_argument("--ratio", nargs='+', type=float, default=[90,10,100], help='Ratio (as a percentage) for train/val/test divide of given data. Does allow for using partial dataset')    

        # Noise Augmentation arguments
        self.parser.add_argument("--min_noise_level", type=float, default=2.0, help='minimum noise sigma to add')
        self.parser.add_argument("--max_noise_level", type=float, default=24.0, help='maximum noise sigma to add')
        self.parser.add_argument('--matrix_size_adjust_ratio', type=float, nargs='+', default=[0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], help='down/upsample the image, keeping the fov')
        self.parser.add_argument('--kspace_filter_sigma', type=float, nargs='+', default=[0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], help='sigma for kspace filter')
        self.parser.add_argument('--kspace_T_filter_sigma', type=float, nargs='+', default=[0.25, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25], help='sigma for T filter')
        self.parser.add_argument('--pf_filter_ratio', type=float, nargs='+', default=[1.0, 0.875, 0.75, 0.625, 0.55], help='pf filter ratio')
        self.parser.add_argument('--phase_resolution_ratio', type=float, nargs='+', default=[1.0, 0.85, 0.7, 0.65, 0.55], help='phase resolution ratio')
        self.parser.add_argument('--readout_resolution_ratio', type=float, nargs='+', default=[1.0, 0.85, 0.7, 0.65, 0.55], help='readout resolution ratio')
        self.parser.add_argument("--snr_perturb_prob", type=float, default=0.0, help='prob to add snr perturbation')
        self.parser.add_argument("--snr_perturb", type=float, default=0.15, help='strength of snr perturbation')    
        self.parser.add_argument("--with_data_degrading", action="store_true", help='if true, degrade data for reduced resolution, temporal smoothing etc.')
        self.parser.add_argument("--not_add_noise", action="store_true", help='if set, not add noise.')
        self.parser.add_argument("--only_white_noise", action="store_true", help='if set, only add white noise.')
        self.parser.add_argument("--ignore_gmap", action="store_true", help='if set, do not use gmap for training.')

        # 2d/3d dataset arguments
        self.parser.add_argument('--twoD_num_patches_cutout', type=int, default=1, help='for 2D usecase, number of patches per frame')
        self.parser.add_argument("--twoD_patches_shuffle", action="store_true", help='shuffle 2D patches to break spatial consistency')
        self.parser.add_argument('--threeD_cutout_jitter', nargs='+', type=float, default=[-1, 0.5, 0.75, 1.0], help='cutout jitter range, relative to the cutout_shape')
        self.parser.add_argument("--threeD_cutout_shuffle_time", action="store_true", help='shuffle along time to break temporal consistency; for 2D+T, should not set this option')

        # super-resolution
        self.parser.add_argument("--super_resolution", action="store_true", help='if set, upsample image by x2 along H and W')

        # inference
        self.parser.add_argument("--pad_time", action="store_true", help='whether to pad along time when doing inference; if False, the entire series is inputted')

        # loss for mri
        self.parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1"], help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D", "psnr", "msssim", "perpendicular", "gaussian", "gaussian3D", "spec", "dwt" ')
        self.parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0, 1.0], help='to balance multiple losses, weights can be supplied')
        self.parser.add_argument("--complex_i", action="store_true", help='whether we are dealing with complex images or not')
        self.parser.add_argument("--residual", action="store_true", help='add long term residual connection')

        self.parser.add_argument("--weighted_loss_snr", action="store_true", help='if set, weight loss by the original signal levels')
        self.parser.add_argument("--weighted_loss_temporal", action="store_true", help='if set, weight loss by temporal/slice signal variation')
        self.parser.add_argument("--weighted_loss_added_noise", action="store_true", help='if set, weight loss by added noise strength')

        self.parser.add_argument("--disable_LSUV", action="store_true", help='if set, do not perform LSUV initialization.')

        # learn rate for pre/backbone/post, if < 0, using the global lr
        self.parser.add_argument("--lr_pre", type=float, default=-1, help='learning rate for pre network')
        self.parser.add_argument("--lr_backbone", type=float, default=-1, help='learning rate for backbone network')
        self.parser.add_argument("--lr_post", type=float, default=-1, help='learning rate for post network')

        self.parser.add_argument("--not_load_pre", action="store_true", help='if set, pre module will not be loaded.')
        self.parser.add_argument("--not_load_backbone", action="store_true", help='if set, backbone module will not be loaded.')
        self.parser.add_argument("--not_load_post", action="store_true", help='if set, pre module will not be loaded.')
        self.parser.add_argument("--post_model_of_1st_net", type=str, default=None, help='if not None and model_type is double net, load post for the 1st net here.')

        # self.parser.add_argument("--disable_pre", action="store_true", help='if set, pre module will have require_grad_(False).')
        # self.parser.add_argument("--disable_backbone", action="store_true", help='if set, backbone module will have require_grad_(False).')
        # self.parser.add_argument("--disable_post", action="store_true", help='if set, post module will have require_grad_(False).')

        self.parser.add_argument('--post_backbone', type=str, default="STCNNT_HRNET", choices=["STCNNT_HRNET", "STCNNT_mUNET"], help="model for post module, 'STCNNT_HRNET', 'STCNNT_mUNET' ")
        self.parser.add_argument('--post_hrnet.block_str', dest='post_hrnet.block_str', nargs='+', type=str, default=['T1L1G1', 'T1L1G1'], help="hrnet MR post network block string, from the low resolution level to high resolution level.")
        self.parser.add_argument('--post_hrnet.separable_conv', dest='post_hrnet.separable_conv', action="store_true", help="post network, whether to use separable convolution.")

        self.parser.add_argument('--post_mixed_unetr.num_resolution_levels', dest='post_mixed_unetr.num_resolution_levels', type=int, default=2, help="number of resolution levels for post mixed unetr")
        self.parser.add_argument('--post_mixed_unetr.block_str', dest='post_mixed_unetr.block_str', nargs='+', type=str, default=['T1L1G1', 'T1L1G1', 'T1L1G1'], help="block string for the post mixed unetr")
        self.parser.add_argument('--post_mixed_unetr.use_unet_attention', dest='post_mixed_unetr.use_unet_attention', type=int, default=1, help="whether to add unet attention between resolution levels")
        self.parser.add_argument('--post_mixed_unetr.transformer_for_upsampling', dest='post_mixed_unetr.transformer_for_upsampling', type=int, default=0, help="whether to use transformer for upsampling branch")
        self.parser.add_argument("--post_mixed_unetr.n_heads", dest='post_mixed_unetr.n_heads', nargs='+', type=int, default=[32, 32, 32], help='number of heads in each resolution layer')
        self.parser.add_argument('--post_mixed_unetr.use_conv_3d', dest='post_mixed_unetr.use_conv_3d', type=int, default=1, help="whether to use 3D convolution")
        self.parser.add_argument('--post_mixed_unetr.use_window_partition', dest='post_mixed_unetr.use_window_partition', type=int, default=0, help="whether to add window partition on input tensors")
        self.parser.add_argument('--post_mixed_unetr.separable_conv', dest='post_mixed_unetr.separable_conv', action="store_true", help="post network, whether to use separable convolution.")

        self.parser.add_argument("--continued_training", action="store_true", help='if set, it means a continued training loaded from checkpoints (optim and scheduler will be loaded); if not set, it mean a new stage of training.')

        # training
        self.parser.add_argument('--num_uploaded', type=int, default=12, help='number of images uploaded to wandb')
        self.parser.add_argument("--num_saved_samples", type=int, default=32, help='number of samples to save')
        self.parser.add_argument("--model_type", type=str, default="STCNNT_MRI", choices=["STCNNT_MRI", "MRI_hrnet", "MRI_double_net", "omnivore_MRI"],  help="STCNNT_MRI or MRI_hrnet, MRI_double_net, omnivore_MRI")

    