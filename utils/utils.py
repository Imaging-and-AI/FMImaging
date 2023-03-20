"""
General utility file for STCNNT

Provides following utilities:
    - add_shared_args
    - setup_run
    - get_device
    - get_number_of_params
    - AverageMeter (class)
"""
import os
import torch
import logging
import argparse
import numpy as np

from datetime import datetime

# -------------------------------------------------------------------------------------------------
# parser for commonly shared args (subject to change over time)

def add_shared_args(parser=argparse.ArgumentParser("Argument parser for STCNNT")):
    """
    Add shared arguments between trainers
    @args:
        parser (argparse, optional): parser object
    @rets:
        parser : new/modified parser
    """
    # common paths
    parser.add_argument("--log_path", type=str, default=None, help='directory for log files')
    parser.add_argument("--model_path", type=str, default=None, help='directory for saving the final model')
    parser.add_argument("--check_path", type=str, default=None, help='directory for saving checkpoints (model weights)')

    # wandb
    parser.add_argument("--project", type=str, default='STCNNT', help='project name')
    parser.add_argument("--run_name", type=str, default=None, help='current run name')
    parser.add_argument("--run_notes", type=str, default=None, help='notes for the current run')
    parser.add_argument("--wandb_entity", type=str, default="gadgetron", help='wandb entity to link with')
    parser.add_argument("--sweep_id", type=str, default="none", help='sweep id for hyper parameter searching')

    # dataset arguments
    parser.add_argument("--ratio", nargs='+', type=int, default=[90,5,5], help='Ratio (as a percentage) for train/val/test divide of given data. Does allow for using partial dataset')
    parser.add_argument("--complex_i", action="store_true", help='whether we are dealing with complex images or not')
    parser.add_argument("--time", type=int, default=16, help='the max time series length of the input cutout')
    parser.add_argument("--height", nargs='+', type=int, default=[64, 128], help='list of heights of the image patch cutout')
    parser.add_argument("--width", nargs='+', type=int, default=[64, 128], help='list of widths of the image patch cutout')

    # dataloader arguments
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers for dataloading')
    parser.add_argument("--prefetch_factor", type=int, default=4, help='number of batches loaded in advance by each worker')

    # trainer arguments
    parser.add_argument("--num_epochs", type=int, default=30, help='number of epochs to train for')
    parser.add_argument("--batch_size", type=int, default=8, help='size of each batch')
    parser.add_argument("--save_cycle", type=int, default=5, help='Number of epochs between saving model weights')
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help='gradient norm clip, if <=0, no clipping')

    # base model arguments
    parser.add_argument("--channels", nargs='+', type=int, default=[16,32,64], help='number of channels in each layer')
    parser.add_argument("--att_types", nargs='+', type=str, default=["temporal", "temporal"], help='types of attention modules "temporal", "local", "global". stacked in this order')
    parser.add_argument("--C_in", type=int, default=3, help='number of channels in the input')
    parser.add_argument("--C_out", type=int, default=16, help='number of channels in the output')
    parser.add_argument("--a_type", type=str, default="conv", help='type of attention in the spatial attention modules')
    parser.add_argument("--window_size", type=int, default=16, help='size of window for local and global att')
    parser.add_argument("--n_head", type=int, default=8, help='number of transformer heads')
    parser.add_argument("--kernel_size", type=int, default=3, help='size of the square kernel for CNN')
    parser.add_argument("--stride", type=int, default=1, help='stride for CNN (equal x and y)')
    parser.add_argument("--padding", type=int, default=1, help='padding for CNN (equal x and y)')
    parser.add_argument("--dropout_p", type=float, default=0.1, help='pdrop regulization in transformer')
    parser.add_argument("--norm_mode", type=str, default="instance2d", help='normalization mode: "layer", "batch2d", "instance2d", "batch3d", "instance3d"')
    parser.add_argument("--with_mixer", type=str, default="all", help='add a mixer to each attention cell. "all", "last", "none"')
    parser.add_argument("--residual", action="store_true", help='add long term residual connection')
    parser.add_argument("--is_causal", action="store_true", help='treat timed data as causal and mask future entries')
    parser.add_argument("--interp_align_c", action="store_true", help='align corners while interpolating')

    # loss, optimizer, and scheduler arguments
    parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1"], help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D"')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0, 1.0], help='to balance multiple losses, weights can be supplied')

    parser.add_argument("--optim", type=str, default="adamw", help='what optimizer to use, "adamw", "nadam", "sgd"')
    parser.add_argument("--global_lr", type=float, default=5e-4, help='step size for the optimizer')
    parser.add_argument("--beta1", type=float, default=0.90, help='beta1 for the default optimizer')
    parser.add_argument("--beta2", type=float, default=0.95, help='beta2 for the default optimizer')

    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau", help='"ReduceLROnPlateau", "StepLR", or "OneCycleLR"')
    parser.add_argument("--weight_decay", type=float, default=0.1, help='weight decay for regularization')
    parser.add_argument("--no_w_decay", action="store_true", help='option of having norm and bias params not have weight decay')

    # misc arguments
    parser.add_argument("--seed", type=int, default=3407, help='seed for randomization')
    parser.add_argument("--device", type=str, default=None, help='device to train on')
    parser.add_argument("--load_path", type=str, default=None, help='path to load model weights from')
    parser.add_argument("--debug", "-D", action="store_true", help='option to run in debug mode')

    return parser

# -------------------------------------------------------------------------------------------------
# setup the run

def setup_run(config, dirs={"log_path", "model_path", "check_path"}):
    """
    sets up datetime, logging, seed and ddp
    @args:
        - config (Namespace): runtime namespace for setup
        - dirs (Set): the directory to be created
    """
    # get current date
    now = datetime.now()
    now = now.strftime("%m-%d-%Y_T%H-%M-%S")
    config["date"] = now

    # setup logging
    os.makedirs(config.log_path, exist_ok=True)
    log_file_name = os.path.join(config.log_path, f"{config.run_name}_{now}.log")
    level = logging.INFO
    format = "%(asctime)s [%(levelname)s] %(message)s"
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(logging.Formatter(format))
    stream_handler = logging.StreamHandler()

    logging.basicConfig(level=level, format=format, handlers=[file_handler, stream_handler])

    file_only_logger = logging.getLogger("file_only") # seperate logger for files only
    file_only_logger.addHandler(file_handler)
    file_only_logger.setLevel(logging.INFO)
    file_only_logger.propagate=False

    # create relevent directories
    dirs.update(["log_path", "model_path", "check_path"])
    for dir in dirs:
        os.makedirs(config[dir], exist_ok=True)
        logging.info(f"Run:{config.run_name}, {dir} is {config[dir]}")
    
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # setup dp/ddp
    device = get_device(config.device)
    if config.device is None: config.update({"device":device}, allow_val_change=True)
    world_size = torch.cuda.device_count()
    config["ddp"] = config.device == "cuda" and world_size > 1
    config["world_size"] = world_size if config.ddp else -1
    logging.info(f"Training on {get_device(config.device)} with dp set to {config.ddp}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # pytorch loader fix
    if config.num_workers==0: config.update({"prefetch_factor":2}, allow_val_change=True)

# -------------------------------------------------------------------------------------------------
# wrapper around getting device

def get_device(device=None):
    """
    @args:
        - device (torch.device): if not None this device will be returned
            otherwise check if cuda is available
    @rets:
        - device (torch.device): the device to be used
    """

    return device if device is not None else \
            "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------------------------------
# total model parameters

def get_number_of_params(model):
    """
    Count the total number of parameters
    @args:
        - model (torch model): the model to check parameters of
    @rets:
        - trainable_params (int): the number of trainable params in the model
        - total_params (int): the total number of params in the model
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    return trainable_params, total_params

# -------------------------------------------------------------------------------------------------
# average metric tracker

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
