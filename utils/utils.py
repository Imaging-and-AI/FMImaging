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
import cv2
import wandb
import torch
import logging
import argparse
import tifffile
import numpy as np

from collections import OrderedDict
from skimage.util import view_as_blocks

from datetime import datetime
from torchinfo import summary

import torch.distributed as dist

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
    parser.add_argument("--log_path", type=str, default='/export/Lab-Xue/projects/log', help='directory for log files')
    parser.add_argument("--results_path", type=str, default='/export/Lab-Xue/projects/results', help='folder to save results in')
    parser.add_argument("--model_path", type=str, default='/export/Lab-Xue/projects/models', help='directory for saving the final model')
    parser.add_argument("--check_path", type=str, default='/export/Lab-Xue/projects/checkpoints', help='directory for saving checkpoints (model weights)')

    # wandb
    parser.add_argument("--project", type=str, default='STCNNT', help='project name')
    parser.add_argument("--run_name", type=str, default='cifar', help='current run name')
    parser.add_argument("--run_notes", type=str, default='cifar_train', help='notes for the current run')
    parser.add_argument("--wandb_entity", type=str, default="gadgetron", help='wandb entity to link with')
    parser.add_argument("--sweep_id", type=str, default="none", help='sweep id for hyper parameter searching')

    # dataset arguments
    parser.add_argument("--ratio", nargs='+', type=int, default=[100,100,0], help='Ratio (as a percentage) for train/val/test divide of given data. Does allow for using partial dataset')    

    # dataloader arguments
    parser.add_argument("--num_workers", type=int, default=8, help='number of workers for data loading')
    parser.add_argument("--prefetch_factor", type=int, default=8, help='number of batches loaded in advance by each worker')

    # trainer arguments
    parser.add_argument("--num_epochs", type=int, default=150, help='number of epochs to train for')
    parser.add_argument("--batch_size", type=int, default=128, help='size of each batch')
    parser.add_argument("--save_cycle", type=int, default=5, help='Number of epochs between saving model weights')
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help='gradient norm clip, if <=0, no clipping')

    # loss, optimizer, and scheduler arguments
    parser.add_argument("--optim", type=str, default="adamw", help='what optimizer to use, "adamw", "nadam", "sgd"')
    parser.add_argument("--global_lr", type=float, default=5e-4, help='step size for the optimizer')
    parser.add_argument("--beta1", type=float, default=0.90, help='beta1 for the default optimizer')
    parser.add_argument("--beta2", type=float, default=0.95, help='beta2 for the default optimizer')

    parser.add_argument("--scheduler_type", type=str, default="ReduceLROnPlateau", help='"ReduceLROnPlateau", "StepLR", or "OneCycleLR"')
    parser.add_argument('--scheduler.ReduceLROnPlateau.patience', dest='scheduler.ReduceLROnPlateau.patience', type=int, default=2, help="number of epochs to wait for further lr adjustment")
    parser.add_argument('--scheduler.ReduceLROnPlateau.cooldown', dest='scheduler.ReduceLROnPlateau.cooldown', type=int, default=2, help="after adjusting the lr, number of epochs to wait before another adjustment")
    parser.add_argument('--scheduler.ReduceLROnPlateau.min_lr', dest='scheduler.ReduceLROnPlateau.min_lr', type=float, default=1e-7, help="minimal lr")
    parser.add_argument('--scheduler.ReduceLROnPlateau.factor', dest='scheduler.ReduceLROnPlateau.factor', type=float, default=0.9, help="lr reduction factor, multiplication")
        
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
    
    parser.add_argument("--block_dense_connection", type=int, default=1, help='whether to add dense connections between cells in a block')
        
    parser.add_argument("--a_type", type=str, default="conv", help='type of attention in the spatial attention modules')
    parser.add_argument("--mixer_type", type=str, default="conv", help='conv or lin, type of mixer in the spatial attention modules; only conv is possible for the temporal attention')
    
    parser.add_argument("--window_size", nargs='+', type=int, default=[64, 64], help='size of window for spatial attention. This is the number of pixels in a window. Given image height and weight H and W, number of windows is H/windows_size * W/windows_size')
    parser.add_argument("--patch_size", nargs='+', type=int, default=[16, 16], help='size of patch for spatial attention. This is the number of pixels in a patch. An image is first split into windows. Every window is further split into patches.')
    
    parser.add_argument("--window_sizing_method", type=str, default="mixed", help='method to adjust window_size between resolution levels, "keep_window_size", "keep_num_window", "mixed".\
                        "keep_window_size" means number of pixels in a window is kept after down/upsample the image; \
                        "keep_num_window" means the number of windows is kept after down/upsample the image; \
                        "mixed" means interleave both methods.')
    
    parser.add_argument("--n_head", type=int, default=8, help='number of transformer heads')
    parser.add_argument("--kernel_size", type=int, default=3, help='size of the square kernel for CNN')
    parser.add_argument("--stride", type=int, default=1, help='stride for CNN (equal x and y)')
    parser.add_argument("--padding", type=int, default=1, help='padding for CNN (equal x and y)')
    parser.add_argument("--stride_t", type=int, default=2, help='stride for temporal attention cnn (equal x and y)') 
    
    parser.add_argument("--mixer_kernel_size", type=int, default=5, help='conv kernel size for the mixer')
    parser.add_argument("--mixer_stride", type=int, default=1, help='stride for the mixer')
    parser.add_argument("--mixer_padding", type=int, default=2, help='padding for the mixer')
      
    parser.add_argument("--normalize_Q_K", action="store_true", help='whether to normalize Q and K before computing attention matrix')
    parser.add_argument("--cosine_att", type=int, default=0, help='whether to use cosine attention; if True, normalize_Q_K is ignored')   
    parser.add_argument("--att_with_relative_postion_bias", type=int, default=1, help='whether to use relative position bias')   
            
    parser.add_argument("--att_dropout_p", type=float, default=0.0, help='pdrop for the attention coefficient matrix')
    parser.add_argument("--dropout_p", type=float, default=0.1, help='pdrop regulization for stochastic residual connections')
    
    parser.add_argument("--att_with_output_proj", type=int, default=1, help='whether to add output projection in attention layer')
    parser.add_argument("--scale_ratio_in_mixer", type=float, default=4.0, help='the scaling ratio to increase/decrease dimensions in the mixer of an attention layer')
    
    parser.add_argument("--norm_mode", type=str, default="instance2d", help='normalization mode: "layer", "batch2d", "instance2d", "batch3d", "instance3d"')
    
    parser.add_argument("--shuffle_in_window", type=int, default=0, help='whether to shuffle patches in a window for the global attention')    
    
    parser.add_argument("--is_causal", action="store_true", help='treat timed data as causal and mask future entries')
    parser.add_argument("--interp_align_c", action="store_true", help='align corners while interpolating')
    
    parser = add_shared_args(parser)

    return parser

def add_backbone_STCNNT_args(parser=argparse.ArgumentParser("Argument parser for backbone models")):
    """
    Add backbone model specific parameters
    """
    
    parser.add_argument('--backbone', type=str, default="hrnet", help="which backbone model to use, 'hrnet', 'unet', 'LLM', 'small_unet' ")
    
    # hrnet
    parser.add_argument('--backbone_hrnet.C', dest='backbone_hrnet.C', type=int, default=16, help="number of channels in main body of hrnet")
    parser.add_argument('--backbone_hrnet.num_resolution_levels', dest='backbone_hrnet.num_resolution_levels', type=int, default=2, help="number of resolution levels; image size reduce by x2 for every level")
    parser.add_argument('--backbone_hrnet.block_str', dest='backbone_hrnet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")    
    parser.add_argument('--backbone_hrnet.use_interpolation', dest='backbone_hrnet.use_interpolation', type=int, default=1, help="whether to use interpolation in downsample layer; if False, use stride convolution")
    
    # unet            
    parser.add_argument('--backbone_unet.C', dest='backbone_unet.C', type=int, default=16, help="number of channels in main body of unet")
    parser.add_argument('--backbone_unet.num_resolution_levels', dest='backbone_unet.num_resolution_levels', type=int, default=2, help="number of resolution levels for unet; image size reduce by x2 for every level")
    parser.add_argument('--backbone_unet.block_str', dest='backbone_unet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")    
    parser.add_argument('--backbone_unet.use_unet_attention', dest='backbone_unet.use_unet_attention', type=int, default=1, help="whether to add unet attention between resolution levels")
    parser.add_argument('--backbone_unet.use_interpolation', dest='backbone_unet.use_interpolation', type=int, default=1, help="whether to use interpolation in downsample layer; if False, use stride convolution")
    parser.add_argument('--backbone_unet.with_conv', dest='backbone_unet.with_conv', type=int, default=1, help="whether to add conv in down/upsample layers; if False, only interpolation is performed")
    
    # LLMs
    parser.add_argument('--backbone_LLM.C', dest='backbone_LLM.C', type=int, default=16, help="number of channels in main body of LLM net")
    parser.add_argument('--backbone_LLM.num_stages', dest='backbone_LLM.num_stages', type=int, default=2, help="number of stages")
    parser.add_argument('--backbone_LLM.block_str', dest='backbone_LLM.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in stages; if multiple strings are given, each is for a stage.")    
    parser.add_argument('--backbone_LLM.add_skip_connections', dest='backbone_LLM.add_skip_connections', type=int, default=1, help="whether to add skip connections between stages; if True, densenet type connections are added; if False, LLM type network is created.")
                     
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
    now = now.strftime("%m-%d-%Y") # make sure in ddp, different nodes have the save file name
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
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
       
    # pytorch loader fix
    if config.num_workers==0: config.prefetch_factor = 2

# -------------------------------------------------------------------------------------------------
def compute_total_steps(config, num_samples):
    if config.ddp: 
        num_samples /= dist.get_world_size()

    total_steps = int(np.ceil(num_samples/(config.batch_size*config.iters_to_accumulate))*config.num_epochs)
    
    return total_steps

# -------------------------------------------------------------------------------------------------
# # wrapper around getting device

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

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
                        
# -------------------------------------------------------------------------------------------------

def get_cuda_info(device):
	return {
		"PyTorch_version": torch.__version__,
		"CUDA_version": torch.version.cuda,
		"cuDNN_version": torch.backends.cudnn.version(),
		"Arch_version": torch._C._cuda_getArchFlags(),
		"device_count": torch.cuda.device_count(),
		"device_name": torch.cuda.get_device_name(device=device),
		"device_id": torch.cuda.current_device(),
		"cuda_capability": torch.cuda.get_device_capability(device=device),
		"device_properties": torch.cuda.get_device_properties(device=device),
		"reserved_memory": torch.cuda.memory_reserved(device=device) / 1024**3,
		"allocated_memory": torch.cuda.memory_allocated(device=device) / 1024**3,
		"max_allocated_memory": torch.cuda.max_memory_allocated(device=device) / 1024**3
	}
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
# 2D image patch and repatch

def cut_into_patches(image_list, cutout):
    """
    Cuts a 2D image into non-overlapping patches.
    Assembles patches in the time dimension
    Pads up to the required length by symmetric padding.
    @args:
        - image_list (5D torch.Tensor list): list of image to cut
        - cutout (2-tuple): the 2D cutout shape of each patch
    @reqs:
        - all images should have the same shape besides 3rd dimension (C)
        - 1st and 2nd dimension should be 1 (B,T)
    @rets:
        - final_image_list (5D torch.Tensor list): patch images
        - original_shape (5-tuple): the original shape of the image
        - patch_shape (10-tuple): the shape of patch array
    """
    original_shape = image_list[0].shape

    final_image_list = []

    for image in image_list:
        assert image.ndim==5, f"Image should have 5 dimensions"
        assert image.shape[0]==1, f"Batch size should be 1"
        assert image.shape[1]==1, f"Time size should be 1"
        assert image.shape[-2:]==original_shape[-2:], f"Image should have the same H,W"
        final_image_list.append(image.numpy(force=True))

    B,T,C,H,W = image_list[0].shape
    pad_H = (-1*H)%cutout[0]
    pad_W = (-1*W)%cutout[1]
    pad_shape = ((0,0),(0,0),(0,0),(0,pad_H),(0,pad_W))

    for i, image in enumerate(final_image_list):
        C = image.shape[2]
        image_i = np.pad(image, pad_shape, 'symmetric')
        image_i = view_as_blocks(image_i, (1,1,C,*cutout))
        patch_shape = image_i.shape
        image_i = image_i.reshape(1,-1,*image_i.shape[-3:])
        final_image_list[i] = torch.from_numpy(image_i)

    return final_image_list, original_shape, patch_shape

def repatch(image_list, original_shape, patch_shape):
    """
    Reassembles the patched image into the complete image
    Assumes the patches are assembled in the time dimension
    @args:
        - image_list (5D torch.Tensor list): patched images
        - original_shape (5-tuple): the original shape of the images
        - patch_shape (10-tuple): the shape of image as patches
    @reqs:
        - 1st dimension to be 1 (B)
    @rets:
        - final_image_list (5D torch.Tensor list): repatched images
    """
    HO,WO = original_shape[-2:]
    H = patch_shape[3]*patch_shape[8]
    W = patch_shape[4]*patch_shape[9]

    final_image_list = []

    for image in image_list:
        C = image.shape[-3]
        patch_shape_x = (*patch_shape[:-3],C,*patch_shape[-2:])
        image = image.reshape(patch_shape_x)
        image = image.permute(0,5,1,6,2,7,3,8,4,9)
        image = image.reshape(1,1,C,H,W)[:,:,:,:HO,:WO]
        final_image_list.append(image)

    return final_image_list

# -------------------------------------------------------------------------------------------------

def save_image_local(path, complex_i, i, noisy, predi, clean):
    """
    Saves the image locally as a 4D tiff [T,C,H,W]
    3 channels: noisy, predicted, clean
    If complex image then save the magnitude using first 2 channels
    Else use just the first channel
    @args:
        - path (str): the directory to save the images in
        - complex_i (bool): complex images or not
        - i (int): index of the image
        - noisy (5D numpy array): the noisy image
        - predi (5D numpy array): the predicted image
        - clean (5D numpy array): the clean image
    """

    if complex_i:
        save_x = np.sqrt(np.square(noisy[0,:,0]) + np.square(noisy[0,:,1]))
        save_p = np.sqrt(np.square(predi[0,:,0]) + np.square(predi[0,:,1]))
        save_y = np.sqrt(np.square(clean[0,:,0]) + np.square(clean[0,:,1]))
    else:
        save_x = noisy[0,:,0]
        save_p = predi[0,:,0]
        save_y = clean[0,:,0]

    composed_channel_wise = np.transpose(np.array([save_x, save_p, save_y]), (1,0,2,3))

    tifffile.imwrite(os.path.join(path, f"Image_{i:03d}_{save_x.shape}.tif"),\
                        composed_channel_wise, imagej=True)

def save_image_wandb(title, complex_i, noisy, predi, clean):
    """
    Logs the image to wandb as a 4D gif [T,C,H,3*W]
    3 width: noisy, predicted, clean
    If complex image then save the magnitude using first 2 channels
    Else use just the first channel
    @args:
        - title (str): title to log image with
        - complex_i (bool): complex images or not
        - noisy (5D numpy array): the noisy image
        - predi (5D numpy array): the predicted image
        - clean (5D numpy array): the clean image
    """

    if complex_i:
        save_x = np.sqrt(np.square(noisy[0,:,0]) + np.square(noisy[0,:,1]))
        save_p = np.sqrt(np.square(predi[0,:,0]) + np.square(predi[0,:,1]))
        save_y = np.sqrt(np.square(clean[0,:,0]) + np.square(clean[0,:,1]))
    else:
        save_x = noisy[0,:,0]
        save_p = predi[0,:,0]
        save_y = clean[0,:,0]

    if save_x.ndim==2:
        save_x = np.expand_dims(save_x, axis=0)
        save_p = np.expand_dims(save_p, axis=0)
        save_y = np.expand_dims(save_y, axis=0)
        
    T, H, W = save_x.shape
    composed_res = np.zeros((T, H, 3*W))
    composed_res[:,:H,0*W:1*W] = save_x
    composed_res[:,:H,1*W:2*W] = save_p
    composed_res[:,:H,2*W:3*W] = save_y

    composed_res = np.clip(composed_res, a_min=0.5*np.mean(composed_res), a_max=0.85*np.mean(composed_res))

    temp = np.zeros_like(composed_res)
    composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

    wandbvid = wandb.Video(composed_res[:,np.newaxis,:,:].astype('uint8'), fps=8, format="gif")
    wandb.log({title: wandbvid})


def save_image_batch_wandb(title, complex_i, noisy, predi, clean):
    """
    Logs the image to wandb as a 5D gif [B,T,C,H,W]
    If complex image then save the magnitude using first 2 channels
    Else use just the first channel
    @args:
        - title (str): title to log image with
        - complex_i (bool): complex images or not
        - noisy (5D numpy array): the noisy image [B, T, C+1, H, W]
        - predi (5D numpy array): the predicted image [B, T, C, H, W]
        - clean (5D numpy array): the clean image [B, T, C, H, W]
    """

    if noisy.ndim == 4:
        noisy = np.expand_dims(noisy, axis=0)
        predi = np.expand_dims(predi, axis=0)
        clean = np.expand_dims(clean, axis=0)

    if complex_i:
        save_x = np.sqrt(np.square(noisy[:,:,0,:,:]) + np.square(noisy[:,:,1,:,:]))
        save_p = np.sqrt(np.square(predi[:,:,0,:,:]) + np.square(predi[:,:,0,:,:]))
        save_y = np.sqrt(np.square(clean[:,:,0,:,:]) + np.square(clean[:,:,0,:,:]))
    else:
        save_x = noisy[:,:,0,:,:]
        save_p = predi[:,:,0,:,:]
        save_y = clean[:,:,0,:,:]
       
    B, T, H, W = save_x.shape
    if B>2:
        composed_res = np.zeros((T, 3*H, B*W))
        for b in range(B):
            for t in range(T):
                composed_res[t, :H, b*W:(b+1)*W] = save_x[b,t,:,:].squeeze()
                composed_res[t, H:2*H, b*W:(b+1)*W] = save_p[b,t,:,:].squeeze()
                composed_res[t, 2*H:3*H, b*W:(b+1)*W] = save_y[b,t,:,:].squeeze()
    else:
        composed_res = np.zeros((T, B*H, 3*W))
        for b in range(B):
            for t in range(T):
                composed_res[t, b*H:(b+1)*H, :W] = save_x[b,t,:,:].squeeze()
                composed_res[t, b*H:(b+1)*H, W:2*W] = save_p[b,t,:,:].squeeze()
                composed_res[t, b*H:(b+1)*H, 2*W:3*W] = save_y[b,t,:,:].squeeze()
                
    composed_res = np.clip(composed_res, a_min=0.5*np.median(composed_res), a_max=np.percentile(composed_res, 90))

    temp = np.zeros_like(composed_res)
    composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

    wandbvid = wandb.Video(np.repeat(composed_res[:,np.newaxis,:,:].astype('uint8'), 3, axis=1), fps=1, format="gif")
    wandb.log({title: wandbvid})
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