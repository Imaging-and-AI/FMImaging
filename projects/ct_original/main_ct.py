"""
Main file for STCNNT CT denoising
"""

import argparse

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from trainer_base import Trainer_Base
from ct.trainer_ct import trainer

# -------------------------------------------------------------------------------------------------
# Extra args on top of shared args

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT Mirco")
    parser.add_argument("--data_set", type=str, default="ct", help='general purpose argument')
    parser.add_argument("--data_root", type=str, default=None, help='root folder containing h5 data files')
    parser.add_argument("--train_files", type=str, nargs='+', default=[], help='list of train h5files. If empty then all files present in the folder are used')
    parser.add_argument("--test_files", type=str, nargs='+', default=[], help='list of test h5files. Either complete paths, or file in data root')
    parser.add_argument("--max_load", type=int, default=-1, help='number of samples to load into the disk, if <0, samples will be read from the disk while training')

    parser = add_backbone_STCNNT_args(parser=parser)

    # loss for ct
    parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1"], help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D", "psnr", "msssim", "gaussian", "gaussian3D" ')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0, 1.0], help='to balance multiple losses, weights can be supplied')
    parser.add_argument("--residual", action="store_true", help='add long term residual connection')

    # learn rate for pre/backbone/post, if < 0, using the global lr
    parser.add_argument("--lr_pre", type=float, default=-1, help='learning rate for pre network')
    parser.add_argument("--lr_backbone", type=float, default=-1, help='learning rate for backbone network')
    parser.add_argument("--lr_post", type=float, default=-1, help='learning rate for post network')

    parser.add_argument("--not_load_pre", action="store_true", help='if set, pre module will not be loaded.')
    parser.add_argument("--not_load_backbone", action="store_true", help='if set, backbone module will not be loaded.')
    parser.add_argument("--not_load_post", action="store_true", help='if set, pre module will not be loaded.')

    parser.add_argument("--disable_pre", action="store_true", help='if set, pre module will have require_grad_(False).')
    parser.add_argument("--disable_backbone", action="store_true", help='if set, backbone module will have require_grad_(False).')
    parser.add_argument("--disable_post", action="store_true", help='if set, post module will have require_grad_(False).')

    parser.add_argument("--continued_training", action="store_true", help='if set, it means a continued training loaded from checkpoints (optim and scheduler will be loaded); if not set, it mean a new stage of training.')
    parser.add_argument("--disable_LSUV", action="store_true", help='if set, do not perform LSUV initialization.')

    parser.add_argument('--post_backbone', type=str, default="hrnet", help="model for post module, 'hrnet', 'mixed_unetr' ")
    parser.add_argument('--post_hrnet.block_str', dest='post_hrnet.block_str', nargs='+', type=str, default=['T1L1G1', 'T1L1G1'], help="hrnet MR post network block string, from the low resolution level to high resolution level.")
    parser.add_argument('--post_hrnet.separable_conv', dest='post_hrnet.separable_conv', action="store_true", help="post network, whether to use separable convolution.")

    parser.add_argument("--training_step", type=int, default=0, help='step number for muti-step training')

    # training
    parser.add_argument("--model_type", type=str, default="STCNNT_CT", help='"STCNNT_CT" or "STCNNT_double"')
    parser.add_argument("--train_only", action="store_true", help='focus on training only. no val or test')
    parser.add_argument('--train_samples', type=int, default=0, help='number of images to train/finetune with. First n are taken from the train set if n>0')
    parser.add_argument('--samples_per_image', type=int, default=32, help='samples to take from a single image per epoch')
    parser.add_argument('--num_uploaded', type=int, default=12, help='number of images uploaded to wandb')

    # inference
    parser.add_argument("--pad_time", action="store_true", help='whether to pad along time when doing inference; if False, the entire series is inputted')

    ns = Nestedspace()
    args = parser.parse_args(namespace=ns)

    return args

# -------------------------------------------------------------------------------------------------

class CTTrainer(Trainer_Base):
    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
        """
        super().__init__(config)
        self.project = 'ct'

    def check_args(self):
        """
        checks the cmd args to make sure they are correct
        """

        super().check_args()

        if self.config.data_root is None:
            self.config.data_root = "/export/Lab-Xue/projects/ct/data"

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

    trainer = CTTrainer(config_default)
    trainer.train()

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()
