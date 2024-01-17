"""
self.parser for the Microscopy project
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

class microscopy_parser(object):
    """
    Microscopy self.parser for project specific arguments
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser("Argument parser for STCNNT Mirco")

        self.parser.add_argument("--data_set", type=str, default="microscopy", help='general purpose argument')
        self.parser.add_argument("--data_root", type=str, default=None, help='root folder containing h5 data files')
        self.parser.add_argument("--train_files", type=str, nargs='+', default=[], help='list of train h5files. If empty then all files present in the folder are used')
        self.parser.add_argument("--test_files", type=str, nargs='+', default=[], help='list of test h5files. Either complete paths, or file in data root')
        self.parser.add_argument("--max_load", type=int, default=-1, help='number of samples to load into the disk, if <0, samples will be read from the disk while training')

        # dataset arguments
        self.parser.add_argument("--ratio", nargs='+', type=float, default=[90,10,100], help='Ratio (as a percentage) for train/val/test divide of given data. Does allow for using partial dataset')
        self.parser.add_argument("--micro_time", type=int, default=16, help='time points of the training images')
        self.parser.add_argument("--micro_height", nargs='+', type=int, default=[32, 64], help='heights of the training images')
        self.parser.add_argument("--micro_width", nargs='+', type=int, default=[32, 64], help='widths of the training images')
        self.parser.add_argument("--scaling_type", type=str, default="val", help='scaling type: "val" for scaling with a static value or "per" for scaling with a percentile')
        self.parser.add_argument("--scaling_vals", type=float, nargs='+', default=[0,65536], help='min max values to scale with respect to the scaling type')
        self.parser.add_argument("--valu_thres", type=float, default=0.002, help='threshold of pixel value between background and foreground')
        self.parser.add_argument("--area_thres", type=float, default=0.25, help='percentage threshold of area that needs to be foreground')

        # loss for microscopy
        self.parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1"], help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D", "psnr", "msssim", "gaussian", "gaussian3D" ')
        self.parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0, 1.0], help='to balance multiple losses, weights can be supplied')
        self.parser.add_argument("--complex_i", action="store_true", help='whether we are dealing with complex images or not')
        self.parser.add_argument("--residual", action="store_true", help='add long term residual connection')

        # learn rate for pre/backbone/post, if < 0, using the global lr
        self.parser.add_argument("--lr_pre", type=float, default=-1, help='learning rate for pre network')
        self.parser.add_argument("--lr_backbone", type=float, default=-1, help='learning rate for backbone network')
        self.parser.add_argument("--lr_post", type=float, default=-1, help='learning rate for post network')

        self.parser.add_argument("--not_load_pre", action="store_true", help='if set, pre module will not be loaded.')
        self.parser.add_argument("--not_load_backbone", action="store_true", help='if set, backbone module will not be loaded.')
        self.parser.add_argument("--not_load_post", action="store_true", help='if set, pre module will not be loaded.')

        self.parser.add_argument("--disable_pre", action="store_true", help='if set, pre module will have require_grad_(False).')
        self.parser.add_argument("--disable_backbone", action="store_true", help='if set, backbone module will have require_grad_(False).')
        self.parser.add_argument("--disable_post", action="store_true", help='if set, post module will have require_grad_(False).')

        self.parser.add_argument("--disable_LSUV", action="store_true", help='if set, do not perform LSUV initialization.')

        self.parser.add_argument('--post_backbone', type=str, default="hrnet", help="model for post module, 'hrnet', 'mixed_unetr' ")
        self.parser.add_argument('--post_hrnet.block_str', dest='post_hrnet.block_str', nargs='+', type=str, default=['T1L1G1', 'T1L1G1'], help="hrnet MR post network block string, from the low resolution level to high resolution level.")
        self.parser.add_argument('--post_hrnet.separable_conv', dest='post_hrnet.separable_conv', action="store_true", help="post network, whether to use separable convolution.")

        self.parser.add_argument("--training_step", type=int, default=0, help='step number for muti-step training')

        # training
        self.parser.add_argument("--model_type", type=str, default="STCNNT_Microscopy", help='"STCNNT_Microscopy" or "Microscopy_double_net"')
        self.parser.add_argument("--train_only", action="store_true", help='focus on training only. no val or test')
        self.parser.add_argument('--train_samples', type=int, default=0, help='number of images to train/finetune with. First n are taken from the train set if n>0')
        self.parser.add_argument('--samples_per_image', type=int, default=32, help='samples to take from a single image per epoch')
        self.parser.add_argument('--num_uploaded', type=int, default=12, help='number of images uploaded to wandb')

        # inference
        self.parser.add_argument("--pad_time", action="store_true", help='whether to pad along time when doing inference; if False, the entire series is inputted')
