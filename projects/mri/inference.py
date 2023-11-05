"""
Inference code for MRI
"""

import copy
import pickle
import numpy as np
from time import time

import os
import sys
import logging

from colorama import Fore, Back, Style
import nibabel as nib

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler(sys.stderr)

log_format = logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)s() ] - %(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
logger.addHandler(c_handler)

from time import time

import onnxruntime as ort
import GPUtil

from utils.status import model_info, start_timer, end_timer, support_bfloat16, get_device
from mri_model import create_model
from running_inference import running_inference

# ---------------------------------------------------------

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

    return m, has_gpu

# -------------------------------------------------------------------------------------------------
def _apply_model(model, x, g, scaling_factor, config, device, overlap=None):
    """Apply the inference

    Input
        x : [1, T, 1, H, W], attention is alone T
        g : [1, T, 1, H, W]

    Output
        res : [1, T, Cout, H, W]
    """
    c = config

    x *= scaling_factor

    B, T, C, H, W = x.shape

    if config.complex_i:
        input = np.concatenate((x.real, x.imag, g), axis=2)
    else:
        input = np.concatenate((np.abs(x), g), axis=2)

    if not c.pad_time:
        cutout = (T, c.height[-1], c.width[-1])
        if overlap is None: overlap = (0, c.height[-1]//2, c.width[-1]//2)
    else:
        cutout = (c.time, c.height[-1], c.width[-1])
        if overlap is None: overlap = (c.time//2, c.height[-1]//2, c.width[-1]//2)

    try:
        _, output = running_inference(model, input, cutout=cutout, overlap=overlap, batch_size=1, device=device)
    except Exception as e:
        print(e)
        print(f"{Fore.YELLOW}---> call inference on cpu ...")
        _, output = running_inference(model, input, cutout=cutout, overlap=overlap, device=torch.device('cpu'))

    x /= scaling_factor
    output /= scaling_factor

    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    return output

# -------------------------------------------------------------------------------------------------

def apply_model(data, model, gmap, config, scaling_factor, device=torch.device('cpu'), overlap=None, verbose=False):
    '''
    Input 
        data : [H, W, T, SLC], remove any extra scaling
        gmap : [H, W, SLC], no scaling added
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (0.5 is more smoothing than 1.0)
        overlap (T, H, W): number of overlap between patches, can be (0, 0, 0)
    Output
        res: [H, W, T, SLC]
    '''

    t0 = time()

    if(data.ndim==2):
        data = data[:,:,np.newaxis,np.newaxis]

    if(data.ndim<4):
        data = np.expand_dims(data, axis=3)

    H, W, T, SLC = data.shape

    if(gmap.ndim==2):
        gmap = np.expand_dims(gmap, axis=2)

    if(gmap.shape[0]!=H or gmap.shape[1]!=W or gmap.shape[2]!=SLC):
        gmap = np.ones(H, W, SLC)

    if verbose:
        print(f"---> apply_model, preparation took {time()-t0} seconds ")
        print(f"---> apply_model, input array {data.shape}")
        print(f"---> apply_model, gmap array {gmap.shape}")
        print(f"---> apply_model, pad_time {config.pad_time}")
        print(f"---> apply_model, height and width {config.height, config.width}")
        print(f"---> apply_model, complex_i {config.complex_i}")
        print(f"---> apply_model, scaling_factor {scaling_factor}")
        print(f"---> apply_model, overlap {overlap}")

    c = config

    try:
        for k in range(SLC):
            imgslab = data[:,:,:,k]
            gmapslab = gmap[:,:,k]

            H, W, T = imgslab.shape

            x = np.transpose(imgslab, [2, 0, 1]).reshape([1, T, 1, H, W])
            g = np.repeat(gmapslab[np.newaxis, np.newaxis, np.newaxis, :, :], T, axis=1)

            print(f"---> running_inference, input {x.shape} for slice {k}")
            output = _apply_model(model, x, g, scaling_factor, config, device, overlap)

            output = np.transpose(output, (3, 4, 2, 1, 0))

            if(k==0):
                if config.complex_i:
                    data_filtered = np.zeros((output.shape[0], output.shape[1], T, SLC), dtype=data.dtype)
                else:
                    data_filtered = np.zeros((output.shape[0], output.shape[1], T, SLC), dtype=np.float32)

            if config.complex_i:
                data_filtered[:,:,:,k] = output[:,:,0,:,0] + 1j*output[:,:,1,:,0]
            else:
                data_filtered[:,:,:,k] = output.squeeze()

    except Exception as e:
        print(e)
        data_filtered = copy.deepcopy(data)

    t1 = time()
    print(f"---> apply_model took {t1-t0} seconds ")

    return data_filtered

# -------------------------------------------------------------------------------------------------

def apply_model_3D(data, model, gmap, config, scaling_factor, device='cpu', overlap=None, verbose=False):
    '''
    Input 
        data : [H W SLC], remove any extra scaling
        gmap : [H W SLC], no scaling added
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (0.5 is more smoothing than 1.0)
    Output
        res : [H W SLC]
    '''

    t0 = time()

    H, W, SLC = data.shape

    if(gmap.shape[0]!=H or gmap.shape[1]!=W or gmap.shape[2]!=SLC):
        gmap = np.ones(H, W, SLC)

    if verbose:
        print(f"---> apply_model_3D, preparation took {time()-t0} seconds ")
        print(f"---> apply_model_3D, input array {data.shape}")
        print(f"---> apply_model_3D, gmap array {gmap.shape}")
        print(f"---> apply_model_3D, pad_time {config.pad_time}")
        print(f"---> apply_model_3D, height and width {config.height, config.width}")
        print(f"---> apply_model_3D, complex_i {config.complex_i}")
        print(f"---> apply_model_3D, scaling_factor {scaling_factor}")

    c = config

    try:
        x = np.transpose(data, [2, 0, 1]).reshape([1, SLC, 1, H, W])
        g = np.transpose(gmap, [2, 0, 1]).reshape([1, SLC, 1, H, W])

        print(f"---> running_inference, input {x.shape} for volume")
        output = _apply_model(model, x, g, scaling_factor, config, device, overlap)

        output = np.transpose(output, (3, 4, 2, 1, 0)) # [H, W, Cout, SLC, 1]

        if config.complex_i:
            data_filtered = output[:,:,0,:,0] + 1j*output[:,:,1,:,0]
        else:
            data_filtered = output

        data_filtered = np.reshape(data_filtered, (H, W, SLC))

    except Exception as e:
        print(e)
        data_filtered = copy.deepcopy(data)

    t1 = time()
    print(f"---> apply_model_3D took {t1-t0} seconds ")

    return data_filtered

# -------------------------------------------------------------------------------------------------

def apply_model_2D(data, model, gmap, config, scaling_factor, device='cpu', overlap=None, verbose=False):
    '''
    Input 
        data : [H W SLC], remove any extra scaling
        gmap : [H W SLC], no scaling added
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (0.5 is more smoothing than 1.0)
    Output
        res : [H W SLC]
        
    Attention is performed within every 2D image.
    '''

    t0 = time()

    H, W, SLC = data.shape

    if(gmap.shape[0]!=H or gmap.shape[1]!=W or gmap.shape[2]!=SLC):
        gmap = np.ones(H, W, SLC)

    if verbose:
        print(f"---> apply_model_2D, preparation took {time()-t0} seconds ")
        print(f"---> apply_model_2D, input array {data.shape}")
        print(f"---> apply_model_2D, gmap array {gmap.shape}")
        print(f"---> apply_model_2D, pad_time {config.pad_time}")
        print(f"---> apply_model_2D, height and width {config.height, config.width}")
        print(f"---> apply_model_2D, complex_i {config.complex_i}")
        print(f"---> apply_model_2D, scaling_factor {scaling_factor}")

    c = config

    try:
        x = np.transpose(data, [2, 0, 1]).reshape([SLC, 1, 1, H, W])
        g = np.transpose(gmap, [2, 0, 1]).reshape([SLC, 1, 1, H, W])

        output = np.zeros([SLC, 1, 1, H, W])

        print(f"---> running_inference, input {x.shape} for 2D")
        for slc in range(SLC):
            output[slc] = _apply_model(model, x[slc], g[slc], scaling_factor, config, device, overlap)

        output = np.transpose(output, (3, 4, 2, 1, 0)) # [H, W, Cout, 1, SLC]

        if config.complex_i:
            data_filtered = output[:,:,0,:,:] + 1j*output[:,:,1,:,:]
        else:
            data_filtered = output

        data_filtered = np.reshape(data_filtered, (H, W, SLC))

    except Exception as e:
        print(e)
        data_filtered = copy.deepcopy(data)

    t1 = time()
    print(f"---> apply_model_2D took {t1-t0} seconds ")

    return data_filtered

# -------------------------------------------------------------------------------------------------

def compare_model(config, model, model_jit, model_onnx, device='cpu', x=None):
    """
    Compare onnx, pts and pt models
    """
    c = config

    C = 3 if config.complex_i else 2

    if x is None:
        x = np.random.randn(1, 12, C, 128, 128).astype(np.float32)

    B, T, C, H, W = x.shape

    model.to(device=device)
    model.eval()

    cutout_in = (c.time, c.mri_height[-1], c.mri_width[-1])
    overlap_in = (c.time//2, c.mri_height[-1]//2, c.mri_width[-1]//2)

    # tm = start_timer(enable=True)    
    # y, y_model = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, batch_size=2, device=device)
    # end_timer(enable=True, t=tm, msg="torch model took")

    tm = start_timer(enable=True)
    y_onnx, y_model_onnx = running_inference(model_onnx, x, cutout=cutout_in, overlap=overlap_in, batch_size=2, device=torch.device('cpu'))
    end_timer(enable=True, t=tm, msg="onnx model took")

    tm = start_timer(enable=True)
    y_jit, y_model_jit = running_inference(model_jit, x, cutout=cutout_in, overlap=overlap_in, batch_size=2, device=torch.device('cpu'))
    end_timer(enable=True, t=tm, msg="torch script model took")

    diff = np.linalg.norm(y-y_onnx)
    print(f"--> {Fore.GREEN}Onnx model difference is {diff} ... {Style.RESET_ALL}", flush=True)

    diff = np.linalg.norm(y-y_jit)
    print(f"--> {Fore.GREEN}Jit model difference is {diff} ... {Style.RESET_ALL}", flush=True)

    diff = np.linalg.norm(y_onnx-y_jit)
    print(f"--> {Fore.GREEN}Jit - onnx model difference is {diff} ... {Style.RESET_ALL}", flush=True)

# -------------------------------------------------------------------------------------------------

def load_model(saved_model_path):
    """
    load a ".pth"model
    @rets:
        - model (torch model): the model ready for inference
    """

    model = None
    config = None

    if saved_model_path.endswith(".pt") or saved_model_path.endswith(".pth"):

        status = torch.load(saved_model_path)
        config = status['config']

        if not torch.cuda.is_available():
            config.device = torch.device('cpu')

        model = create_model(config, config.model_type)

        print(f"{Fore.YELLOW}Load in model {Style.RESET_ALL}")
        model.load_state_dict(status['model_state'])

    return model, config

def load_model(saved_model_path):
    """
    load a ".pth"model
    @rets:
        - model (torch model): the model ready for inference
    """

    model = None
    config = None

    if saved_model_path.endswith(".pt") or saved_model_path.endswith(".pth"):

        status = torch.load(saved_model_path)
        config = status['config']

        if not torch.cuda.is_available():
            config.device = torch.device('cpu')

        model = create_model(config, config.model_type)

        print(f"{Fore.YELLOW}Load in model {Style.RESET_ALL}")
        model.load_state_dict(status['model_state'])

    return model, config

def load_model_pre_backbone_post(saved_model_path):
    """
    load a ".pth"model
    @rets:
        - model (torch model): the model ready for inference
    """

    model = None
    config = None

    pre_name = saved_model_path+"_pre.pth"

    status = torch.load(pre_name, map_location=get_device())
    config = status['config']

    model = create_model(config, config.model_type)
    model.config.device = get_device()

    print(f"{Fore.YELLOW}Load in model {Style.RESET_ALL}")
    model.load(saved_model_path)

    return model, config

# -------------------------------------------------------------------------------------------------

def tests():
    pass    

if __name__=="__main__":
    tests()
