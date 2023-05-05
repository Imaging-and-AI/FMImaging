"""
General utility file for STCNNT

Provides following utilities:
    - add_shared_args
    - add_shared_STCNNT_args
    - setup_run
    - get_device
    - model_info
    - AverageMeter (class)
    ...
"""
import os
import torch
import logging
import argparse
import numpy as np

from collections import OrderedDict

from datetime import datetime
from torchinfo import summary

# -------------------------------------------------------------------------------------------------
# from https://stackoverflow.com/questions/18668227/argparse-subcommands-with-nested-namespaces
class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value
            
# -------------------------------------------------------------------------------------------------
# parser for commonly shared args (subject to change over time)
def add_shared_args(parser=argparse.ArgumentParser("Argument parser for transformer projects")):
    """
    Add shared arguments between trainers
    @args:
        parser (argparse, optional): parser object
    @rets:
        parser : new/modified parser
    """
    # common paths
    parser.add_argument("--log_path", type=str, default=None, help='directory for log files')
    parser.add_argument("--results_path", type=str, default=None, help='folder to save results in')
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

    # dataloader arguments
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers for data loading')
    parser.add_argument("--prefetch_factor", type=int, default=4, help='number of batches loaded in advance by each worker')

    # trainer arguments
    parser.add_argument("--num_epochs", type=int, default=30, help='number of epochs to train for')
    parser.add_argument("--batch_size", type=int, default=8, help='size of each batch')
    parser.add_argument("--save_cycle", type=int, default=5, help='Number of epochs between saving model weights')
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help='gradient norm clip, if <=0, no clipping')

    # loss, optimizer, and scheduler arguments
    parser.add_argument("--optim", type=str, default="adamw", help='what optimizer to use, "adamw", "nadam", "sgd"')
    parser.add_argument("--global_lr", type=float, default=5e-4, help='step size for the optimizer')
    parser.add_argument("--beta1", type=float, default=0.90, help='beta1 for the default optimizer')
    parser.add_argument("--beta2", type=float, default=0.95, help='beta2 for the default optimizer')

    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau", help='"ReduceLROnPlateau", "StepLR", or "OneCycleLR"')
    parser.add_argument('--scheduler.ReduceLROnPlateau.patience', dest='scheduler.ReduceLROnPlateau.patience', type=int, default=3, help="number of epochs to wait for further lr adjustment")
    parser.add_argument('--scheduler.ReduceLROnPlateau.cooldown', dest='scheduler.ReduceLROnPlateau.cooldown', type=int, default=1, help="after adjusting the lr, number of epochs to wait before another adjustment")
    parser.add_argument('--scheduler.ReduceLROnPlateau.min_lr', dest='scheduler.ReduceLROnPlateau.min_lr', type=float, default=1e-8, help="minimal lr")
    parser.add_argument('--scheduler.ReduceLROnPlateau.factor', dest='scheduler.ReduceLROnPlateau.factor', type=float, default=0.8, help="lr reduction factor, multiplication")
        
    parser.add_argument('--scheduler.StepLR.step_size', dest='scheduler.StepLR.step_size', type=int, default=5, help="number of epochs to reduce lr")
    parser.add_argument('--scheduler.StepLR.gamma', dest='scheduler.StepLR.gamma', type=float, default=0.8, help="multiplicative factor of learning rate decay")
        
    parser.add_argument("--weight_decay", type=float, default=0.1, help='weight decay for regularization')
    parser.add_argument("--all_w_decay", action="store_true", help='option of having all params have weight decay. By default norms and embeddings do not')
    parser.add_argument("--use_amp", action="store_true", help='whether to train with mixed precision')
    parser.add_argument("--ddp", action="store_true", help='whether to train with ddp')
    
    parser.add_argument("--iters_to_accumulate", type=int, default=1, help='Number of iterations to accumulate gradients; if >1, gradient accumulation')

    # misc arguments
    parser.add_argument("--seed", type=int, default=3407, help='seed for randomization')
    parser.add_argument("--device", type=str, default=None, help='device to train on')
    parser.add_argument("--load_path", type=str, default=None, help='path to load model weights from')
    parser.add_argument("--debug", "-D", action="store_true", help='option to run in debug mode')
    parser.add_argument("--summary_depth", type=int, default=5, help='depth to print the model summary till')

    return parser

def add_shared_STCNNT_args(parser=argparse.ArgumentParser("Argument parser for STCNNT")):
    """
    Add shared arguments for all STCNNT models
    @args:
        parser (argparse, optional): parser object
    @rets:
        parser : new/modified parser
    """
        
    # base model arguments
    parser.add_argument("--cell_type", type=str, default="sequential", help='cell type, sequential or parallel')
    
    parser.add_argument("--C_in", type=int, default=3, help='number of channels in the input')
    parser.add_argument("--C_out", type=int, default=16, help='number of channels in the output')
    parser.add_argument("--time", type=int, default=16, help='training time series length')
    parser.add_argument("--height", nargs='+', type=int, default=[64, 128], help='heights of the training images')
    parser.add_argument("--width", nargs='+', type=int, default=[64, 128], help='widths of the training images')
        
    parser.add_argument("--a_type", type=str, default="conv", help='type of attention in the spatial attention modules')
    
    parser.add_argument("--window_size", type=int, default=64, help='size of window for spatial attention. This is the number of pixels in a window. Given image height and weight H and W, number of windows is H/windows_size * W/windows_size')
    parser.add_argument("--patch_size", type=int, default=16, help='size of patch for spatial attention. This is the number of pixels in a patch. An image is first split into windows. Every window is further split into patches.')
    
    parser.add_argument("--window_sizing_method", type=str, default="mixed", help='method to adjust window_size betweem resolution levels, "keep_window_size", "keep_num_window", "mixed".\
                        "keep_window_size" means number of pixels in a window is kept after down/upsample the image; \
                        "keep_num_window" means the number of windows is kept after down/upsample the image; \
                        "mixed" means interleave both methods.')
    
    parser.add_argument("--n_head", type=int, default=8, help='number of transformer heads')
    parser.add_argument("--kernel_size", type=int, default=3, help='size of the square kernel for CNN')
    parser.add_argument("--stride", type=int, default=1, help='stride for CNN (equal x and y)')
    parser.add_argument("--padding", type=int, default=1, help='padding for CNN (equal x and y)')
    parser.add_argument("--stride_t", type=int, default=2, help='stride for temporal attention cnn (equal x and y)')   
    parser.add_argument("--normalize_Q_K", action="store_true", help='whether to normalize Q and K before computing attention matrix')

    parser.add_argument("--att_dropout_p", type=float, default=0.0, help='pdrop for the attention coefficient matrix')
    parser.add_argument("--dropout_p", type=float, default=0.1, help='pdrop regulization in transformer')
    
    parser.add_argument("--att_with_output_proj", type=bool, default=True, help='whether to add output projection in attention layer')
    parser.add_argument("--scale_ratio_in_mixer", type=float, default=4.0, help='the scaling ratio to increase/decrease dimensions in the mixer of an attention layer')
    
    parser.add_argument("--norm_mode", type=str, default="instance2d", help='normalization mode: "layer", "batch2d", "instance2d", "batch3d", "instance3d"')
        
    parser.add_argument("--is_causal", action="store_true", help='treat timed data as causal and mask future entries')
    parser.add_argument("--interp_align_c", action="store_true", help='align corners while interpolating')
    
    parser = add_shared_args(parser)

    return parser

def add_backbone_STCNNT_args(parser=argparse.ArgumentParser("Argument parser for backbone models")):
    """
    Add backbone model specific parameters
    """
    
    parser.add_argument('--backbone', type=str, default="unet", help="which backbone model to use, 'hrnet', 'unet', 'LLM', 'small_unet' ")
    
    # hrnet
    parser.add_argument('--backbone_hrnet.C', dest='backbone_hrnet.C', type=int, default=16, help="number of channels in main body of hrnet")
    parser.add_argument('--backbone_hrnet.num_resolution_levels', dest='backbone_hrnet.num_resolution_levels', type=int, default=2, help="number of resolution levels; image size reduce by x2 for every level")
    parser.add_argument('--backbone_hrnet.block_str', dest='backbone_hrnet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")    
    parser.add_argument('--backbone_hrnet.use_interpolation', dest='backbone_hrnet.use_interpolation', type=bool, default=True, help="whether to use interpolation in downsample layer; if False, use stride convolution")
    
    # unet            
    parser.add_argument('--backbone_unet.C', dest='backbone_unet.C', type=int, default=16, help="number of channels in main body of unet")
    parser.add_argument('--backbone_unet.num_resolution_levels', dest='backbone_unet.num_resolution_levels', type=int, default=2, help="number of resolution levels for unet; image size reduce by x2 for every level")
    parser.add_argument('--backbone_unet.block_str', dest='backbone_unet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")    
    parser.add_argument('--backbone_unet.use_unet_attention', dest='backbone_unet.use_unet_attention', type=bool, default=True, help="whether to add unet attention between resolution levels")
    parser.add_argument('--backbone_unet.use_interpolation', dest='backbone_unet.use_interpolation', type=bool, default=True, help="whether to use interpolation in downsample layer; if False, use stride convolution")
    parser.add_argument('--backbone_unet.with_conv', dest='backbone_unet.with_conv', type=bool, default=True, help="whether to add conv in down/upsample layers; if False, only interpolation is performed")
    
    # LLMs
    parser.add_argument('--backbone_LLM.C', dest='backbone_LLM.C', type=int, default=16, help="number of channels in main body of LLM net")
    parser.add_argument('--backbone_LLM.num_stages', dest='backbone_LLM.num_stages', type=int, default=2, help="number of stages")
    parser.add_argument('--backbone_LLM.block_str', dest='backbone_LLM.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in stages; if multiple strings are given, each is for a stage.")    
    parser.add_argument('--backbone_LLM.add_skip_connections', dest='backbone_LLM.add_skip_connections', type=bool, default=True, help="whether to add skip connections between stages; if True, densenet type connections are added; if False, LLM type network is created.")
                     
    # small unet
    parser.add_argument("--backbone_small_unet.channels", dest='backbone_small_unet.channels', nargs='+', type=int, default=[16,32,64], help='number of channels in each layer')
    parser.add_argument('--backbone_small_unet.block_str', dest='backbone_small_unet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in stages; if multiple strings are given, each is for a stage.")   
       
    parser = add_shared_STCNNT_args(parser=parser)
            
    return parser
    
# -------------------------------------------------------------------------------------------------
# setup logger

def setup_logger(config):
    """
    logger setup to be called from any process
    """
    os.makedirs(config.log_path, exist_ok=True)
    log_file_name = os.path.join(config.log_path, f"{config.run_name}_{config.date}.log")
    level = logging.INFO
    format = "%(asctime)s [%(levelname)s] %(message)s"
    file_handler = logging.FileHandler(log_file_name, 'a', 'utf-8')
    file_handler.setFormatter(logging.Formatter(format))
    stream_handler = logging.StreamHandler()

    logging.basicConfig(level=level, format=format, handlers=[file_handler,stream_handler])

    file_only_logger = logging.getLogger("file_only") # separate logger for files only
    file_only_logger.addHandler(file_handler)
    file_only_logger.setLevel(logging.INFO)
    file_only_logger.propagate=False

# -------------------------------------------------------------------------------------------------
# setup the run

def setup_run(config, dirs=["log_path", "results_path", "model_path", "check_path"]):
    """
    sets up datetime, logging, seed and ddp
    @args:
        - config (Namespace): runtime namespace for setup
        - dirs (str list): the directories from config to be created
    """
    # get current date
    now = datetime.now()
    now = now.strftime("%m-%d-%Y_T%H-%M-%S")
    config.date = now

    # setup logging
    setup_logger(config)

    # create relevant directories
    try:
        config_dict = dict(config)
    except TypeError:
        config_dict = vars(config)
    for dir in dirs:
        os.makedirs(config_dict[dir], exist_ok=True)
        logging.info(f"Run:{config.run_name}, {dir} is {config_dict[dir]}")
    
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # setup dp/ddp
    config.device = get_device(config.device)
    world_size = torch.cuda.device_count()
    
    if config.ddp:
        if config.device == torch.device('cpu') or world_size <= 1:
            config.ddp = False
        
    config.world_size = world_size if config.ddp else -1
    logging.info(f"Training on {config.device} with ddp set to {config.ddp}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # pytorch loader fix
    if config.num_workers==0: config.prefetch_factor = 2

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
def get_gpu_ram_usage(device='cuda:0'):
    """
    Get info regarding memory usage of a device
    @args:
        - device (torch.device): the device to get info about
    @rets:
        - result_string (str): a string containing the info
    """
    result_string = f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(device=device)/1024/1024/1024:.3}GB\n" + \
                    f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(device=device)/1024/1024/1024:.3f}GB\n" + \
                    f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(device=device)/1024/1024/1024:.3f}GB"

    return result_string
    
# -------------------------------------------------------------------------------------------------    

def create_generic_class_str(obj : object, exclusion_list=[torch.nn.Module, OrderedDict]) -> str:
    """
    Create a generic name of a class
    @args:
        - obj (object): the class to make string of
        - exclusion_list (object list): the objects to exclude from the class string
    @rets:
        - class_str (str): the generic class string
    """
    name = type(obj).__name__

    vars_list = []
    for key, value in vars(obj).items():
        valid = True
        for type_e in exclusion_list:
            if isinstance(value, type_e) or key.startswith('_'):
                valid = False
                break
        
        if valid:
            vars_list.append(f'{key}={value!r}')
            
    vars_str = ',\n'.join(vars_list)
    return f'{name}({vars_str})'

# -------------------------------------------------------------------------------------------------
# model info

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

def model_info(model, config):
    """
    Prints model info and sets total and trainable parameters in the config
    @args:
        - model (torch model): the model to check parameters of
        - config (Namespace): runtime namespace for setup
    @rets:
        - model_summary (ModelStatistics object): the model summary
            see torchinfo/model_statistics.py for more information.
    """
    c = config
    input_size = (c.batch_size, c.time, c.C_in, c.height[-1], c.width[-1])
    col_names=("num_params", "params_percent", "mult_adds", "input_size", "output_size", "trainable")
    row_settings=["var_names", "depth"]
    dtypes=[torch.float32]

    model_summary = summary(model, verbose=0, mode="train", depth=c.summary_depth,\
                            input_size=input_size, col_names=col_names,\
                            row_settings=row_settings, dtypes=dtypes,\
                            device=config.device)

    c.trainable_params = model_summary.trainable_params
    c.total_params = model_summary.total_params

    torch.cuda.empty_cache()

    return model_summary

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

if __name__=="__main__":
    
    parser = add_backbone_STCNNT_args()
    
    ns = Nestedspace()
    args = parser.parse_args(namespace=ns)
    
    print(args)
    
    print(args.backbone_hrnet.C)
    print(args.backbone_unet.block_str)