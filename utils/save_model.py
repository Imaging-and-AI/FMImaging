"""
shared file to save general model
provides the following functions:
    - generate_model_file_name
    - save_final_model
"""
import os
import sys
import json
import pickle
import torch

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler(sys.stderr)
f_handler = logging.FileHandler('/tmp/fsi.log')

log_format = logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)s() ] - %(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)

from time import time

import onnxruntime as ort
import GPUtil

# -------------------------------------------------------------------------------------------------
# basic file name generator using the config

def generate_model_file_name(config):
    """
    generate file name from the config
    @args:
        - config (Namespace): runtime namespace for setup
    @args (from config):
        - run_name (str): the name of the run
        - data (str): the datetime when the run was started
    @rets:
        - model_file_name (str): the generated file name
    """
    run_name = config.run_name
    date = config.date

    return f"{run_name}_{date}"

# -------------------------------------------------------------------------------------------------
# function to save generic model

def save_final_model(model, config, best_model_wts, only_pt=False):
    """
    save the model as ".pt+.json" and ".pts", and save again after loading the best_model_wts
    @args:
        - model (torch model): model to be saved
        - config (Namespace): runtime namespace for setup
        - best_model_wts (model.state_dict): the best model weight during the run
    @args (from config):
        - time, C_in (int): for model input
        - height, width (int list): for model input
        - model_path (str): the directory to save in
    @rets:
        - last_model_name, best_model_name (str):
            the path the last and the best saved models respectively
    """
    c = config # shortening due to numerous uses

    model.eval()
    
    model_input = torch.randn(1, c.time, c.C_in, c.height[-1], c.width[-1], requires_grad=False)
    model_input = model_input.to('cpu')
    model.to('cpu')

    model_file_name = os.path.join(c.model_path, generate_model_file_name(config))

    # -----------------------------------------------
    def save_model_instance(model, name, only_pt=False):
        # save an instance of the model using the given name
        logging.info(f"Saving model weights and config at: {name}.pt")
        torch.save({"model":model.state_dict(), "config":config}, f"{name}.pt")

        if not only_pt:
            logging.info(f"Saving torchscript model at: {name}.pts")
            model_scripted = torch.jit.trace(model, model_input, strict=False)
            model_scripted.save(f"{name}.pts")

        with open(f"{name}.config", 'wb') as fid:
            pickle.dump(config, fid)

        if not only_pt:
            torch.onnx.export(model, model_input, f"{name}.onnx", 
                                export_params=True, 
                                opset_version=16, 
                                training =torch.onnx.TrainingMode.TRAINING,
                                do_constant_folding=False,
                                input_names = ['input'], 
                                output_names = ['output'], 
                                dynamic_axes={'input' : {0:'batch_size', 1: 'time', 3: 'H', 4: 'W'}, 
                                                'input' : {0:'batch_size', 1: 'time', 3: 'H', 4: 'W'}
                                                }
                                )

    # -----------------------------------------------
    last_model_name = f"{model_file_name}_last"
    save_model_instance(model, name=last_model_name, only_pt=only_pt)
    
    best_model_name = f"{model_file_name}_best"
    model.load_state_dict(best_model_wts)
    save_model_instance(model, name=best_model_name, only_pt=only_pt)

    logging.info(f"All saving complete")

    return last_model_name, best_model_name

# -------------------------------------------------------------------------------------------------
def load_model(model_dir, model_file):
    '''
    model_name: NN model
    '''

    if(model_dir is not None):
        model_file_name = os.path.join(model_dir, model_file)
    else:
        model_file_name = model_file

    try:
        print("---> Load model  ", model_file_name, file=sys.stderr)
        t0 = time()
        model = torch.jit.load(model_file_name)
        t1 = time()
        print("---> Model loading took %f seconds " % (t1-t0), file=sys.stderr)

        sys.stderr.flush()

    except Exception as e:
        print("Error happened in load_model for %s" % model_file_name, file=sys.stderr)
        print(e)
        model = None

    return model

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
            m = ort.InferenceSession(model_full_file, providers=['CPUExecutionProvider'])
            logger.info("model is loaded into the onnx CPU ...")
        t1 = time()
        logger.info("Model loading took %f seconds " % (t1-t0))

        c_handler.flush()
    except Exception as e:
        logger.exception(e, exc_info=True)

    return m, has_gpu

# -------------------------------------------------------------------------------------------------