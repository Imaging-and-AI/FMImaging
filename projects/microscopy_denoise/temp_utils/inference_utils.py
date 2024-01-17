"""
Inference utils for microscopy
These should be common utils so can look into how to make them common
"""

import os
import sys
import copy
import pickle
import GPUtil
import logging
import numpy as np
import nibabel as nib
import onnxruntime as ort

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from time import time
from colorama import Fore, Back, Style
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Microscopy_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Microscopy_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler(sys.stderr)

log_format = logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)s() ] - %(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
logger.addHandler(c_handler)

from setup import yaml_to_config, Nestedspace
from utils import start_timer, end_timer, get_device
from microscopy_model import microscopy_ModelManager
from running_inference import running_inference

# -------------------------------------------------------------------------------------------------

def apply_model(model, x, config, device, overlap=None, batch_size=4, verbose=False):
    """
    Apply the model inference to given single image

    @args:
        - model (torch model): model to use
        - x (5D numpy.array/torch.tensor): input image as [B,T,C,H,W] #TODO: confirm the order
        - config (Namespace): the config of the model during train
        - device (str / torch.device): the device to run on
        - overlap (int 3-tuple): the overlaps in the three dimensions
        - batch_size (int): the number of patches to infer at the same time
        - verbose (bool): optional additional printing
    @rets:
        - output (5D torch.tensor): the output image
    """
    c = config

    B, T, C, H, W = x.shape

    if not c.pad_time:
        cutout = (T, c.height[-1], c.width[-1])
        if overlap is None: overlap = (0, c.height[-1]//2, c.width[-1]//2)
    else:
        cutout = (c.time, c.height[-1], c.width[-1])
        if overlap is None: overlap = (c.time//2, c.height[-1]//2, c.width[-1]//2)

    try:
        _, output = running_inference(model, x, cutout=cutout, overlap=overlap, batch_size=batch_size, device=device, verbose=verbose)
    except Exception as e:
        print(e)
        print(f"{Fore.YELLOW}---> call inference on cpu ...")
        _, output = running_inference(model, x, cutout=cutout, overlap=overlap, batch_size=batch_size, device=torch.device('cpu'), verbose=verbose)

    return output

# -------------------------------------------------------------------------------------------------

def load_model_onnx(model_dir, model_file, use_cpu=False):
    """Load onnx format model

    Args:
        model_dir (str): folder to store the model; if None, only model file is used
        model_file (str): model file name, can be the full path to the model
        use_cpu (bool): if True, only use CPU
        
    Returns:
        model: loaded model
        
    If GPU is avaiable, model will be loaded to CUDAExecutionProvider; otherwise, CPUExecutionProvider
    """
    
    m = None
    has_gpu = False
    
    try:
        if(model_dir is not None):
            model_full_file = os.path.join(model_dir, model_file)
        else:
            model_full_file = model_file

        logger.info("Load model : %s" % model_full_file)
        t0 = time()
        
        try:
            deviceIDs = GPUtil.getAvailable()
            has_gpu = True
            
            GPUs = GPUtil.getGPUs()
            logger.info(f"Found GPU, with memory size {GPUs[0].memoryTotal} Mb")
            
            if(GPUs[0].memoryTotal<8*1024):
                logger.info(f"At least 8GB GPU RAM are needed ...")    
                has_gpu = False
        except: 
            has_gpu = False
            
        if(not use_cpu and (ort.get_device()=='GPU' and has_gpu)):
            providers = [
                            ('CUDAExecutionProvider', 
                                {
                                    'arena_extend_strategy': 'kNextPowerOfTwo',
                                    'gpu_mem_limit': 16 * 1024 * 1024 * 1024,
                                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                                    'do_copy_in_default_stream': True,
                                    "cudnn_conv_use_max_workspace": '1'
                                }
                             ),
                            'CPUExecutionProvider'
                        ]
            
            m = ort.InferenceSession(model_full_file, providers=providers)
            logger.info("model is loaded into the onnx GPU ...")
        else:

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = os.cpu_count() // 2
            sess_options.inter_op_num_threads = os.cpu_count() // 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            m = ort.InferenceSession(model_full_file, sess_options=sess_options, providers=['CPUExecutionProvider'])
            logger.info("model is loaded into the onnx CPU ...")
        t1 = time()
        logger.info("Model loading took %f seconds " % (t1-t0))

        c_handler.flush()
    except Exception as e:
        logger.exception(e, exc_info=True)

    return m

def load_model_complete(saved_model_path):
    """
    load a the model. Either .pt or .pth or .onnx
    @args:
        - saved_model_path (str): the complete path of the complete model
    @rets:
        - model (torch model): the model ready for inference
    """

    model = None
    config = None

    if saved_model_path.endswith(".pt") or saved_model_path.endswith(".pth"):

        status = torch.load(saved_model_path, map_location='cpu')
        config = status['config']

        if not torch.cuda.is_available():
            config.device = torch.device('cpu')

        model = microscopy_ModelManager(config)

        print(f"{Fore.YELLOW}Load in model complete{Style.RESET_ALL}")
        model.load_state_dict(status['model_state'])
    elif saved_model_path.endswith("onnx"): 
        model = load_model_onnx(model_dir=None, model_file=saved_model_path)
        # yaml_fname = os.path.splitext(saved_model_path)[0]+'.yaml'
        # config = yaml_to_config(yaml_fname, '/tmp', 'inference')
        config_fname = os.path.splitext(saved_model_path)[0]+'.config'
        with open(config_fname, 'rb') as fid:
            config = pickle.load(fid)
    return model, config

def load_model_pre_backbone_post(saved_model_path):
    """
    load a model in parts.
    @args:
        - saved_model_path (str): the path to model not including the sub string (_pre.pth)
    @rets:
        - model (torch model): the model ready for inference
    """

    model = None
    config = None

    pre_name = saved_model_path+"_pre.pth"

    status = torch.load(pre_name, map_location=get_device())
    config = status['config']

    model = microscopy_ModelManager(config, config.model_type)
    model.config.device = get_device()

    print(f"{Fore.YELLOW}Load in model from parts{Style.RESET_ALL}")
    model.load(saved_model_path)

    return model, config

def load_model(saved_model_path):
    """
    loads a model from given path.
    Tries to load both from parts and as whole.
    Read above descriptions for input details.
    """

    try:
        return load_model_pre_backbone_post(saved_model_path)
    except:
        return load_model_complete(saved_model_path)

# -------------------------------------------------------------------------------------------------

def tests():
    pass    

if __name__=="__main__":
    tests()
