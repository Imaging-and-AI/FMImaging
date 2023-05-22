"""
Utility functions to set up the training.
"""
import os
import torch
import logging
import numpy as np
from collections import OrderedDict
from datetime import datetime
from torchinfo import summary
import torch.distributed as dist

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
    input_size = (c.batch_size, c.time, c.C_in, c.height[0], c.width[0])
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

if __name__=="__main__":
    pass