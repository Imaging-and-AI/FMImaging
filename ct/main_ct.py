"""
Main file for STCNNT Microscopy denoising
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
    parser.add_argument("--data_set", type=str, default="microscopy", help='general purpose argument')
    parser.add_argument("--data_root", type=str, default=None, help='root folder containing h5 data files')
    parser.add_argument("--train_files", type=str, nargs='+', default=[], help='list of train h5files. If empty then all files present in the folder are used')
    parser.add_argument("--test_files", type=str, nargs='+', default=[], help='list of test h5files. Either complete paths, or file in data root')
    parser.add_argument("--max_load", type=int, default=-1, help='number of samples to load into the disk, if <0, samples will be read from the disk while training')

    parser = add_backbone_STCNNT_args(parser=parser)

    # loss for micro
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

    # training
    parser.add_argument("--model_type", type=str, default="STCNNT_Micro", help="STCNNT_Micro only for now")
    parser.add_argument("--train_only", action="store_true", help='focus on training only. no val or test')
    parser.add_argument('--train_samples', type=int, default=0, help='number of images to train/finetune with. First n are taken from the train set if n>0')
    parser.add_argument('--samples_per_image', type=int, default=8, help='samples to take from a single image per epoch')
    parser.add_argument('--num_uploaded', type=int, default=12, help='number of images uploaded to wandb')
    parser.add_argument('--scaling_type', type=str, default="val", help='scaling type: "val" for scaling with a static value or "per" for scaling with a percentile')
    parser.add_argument("--scaling_vals", type=float, nargs='+', default=[0,65536], help='min max values to scale with respect to the scaling type')
    parser.add_argument("--valu_thres", type=float, default=0.002, help='threshold of pixel value between background and foreground')
    parser.add_argument("--area_thres", type=float, default=0.25, help='percentage threshold of area that needs to be foreground')

    # inference
    parser.add_argument("--pad_time", action="store_true", help='whether to pad along time when doing inference; if False, the entire series is inputted')

    ns = Nestedspace()
    args = parser.parse_args(namespace=ns)

    return args

# -------------------------------------------------------------------------------------------------

class MircoTrainer(Trainer_Base):
    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
        """
        super().__init__(config)
        self.project = 'micro'

    def check_args(self):
        """
        checks the cmd args to make sure they are correct
        """

        super().check_args()

        if self.config.data_root is None:
            self.config.data_root = "/export/Lab-Xue/projects/microscopy/data"

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

    trainer = MircoTrainer(config_default)
    trainer.train()

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()
