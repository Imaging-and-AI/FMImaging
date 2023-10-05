"""
Utility functions to set up the training
"""

import os
import sys
import torch
import logging
import numpy as np
import random
from datetime import datetime
from torchinfo import summary
import torch.distributed as dist
from collections import OrderedDict
from colorama import Fore, Style

from pathlib import Path
Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

from utils.status import get_device


# -------------------------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# -------------------------------------------------------------------------------------------------
def setup_logger(config):
    """
    logger setup to be called from any process
    """
    
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.join(config.log_dir,config.run_name), exist_ok=True)
    
    log_file_name = os.path.join(config.log_dir, config.run_name, f"{config.date}.log")
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
def setup_ddp():
    
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    print(f"{Fore.YELLOW}---> dist.init_process_group on local rank {rank}, global rank {global_rank}, world size {world_size}, local World size {local_world_size} <---{Style.RESET_ALL}", flush=True)

    if not dist.is_initialized():
        torch.cuda.set_device(torch.device(f'cuda:{rank}'))
        if torch.cuda.is_available():
            dist.init_process_group(backend=torch.distributed.Backend.NCCL, rank=global_rank, world_size=world_size)
        else:
            dist.init_process_group(backend=torch.distributed.Backend.GLOO, rank=global_rank, world_size=world_size)
    
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'  

# -------------------------------------------------------------------------------------------------
def setup_run(config):
    """
    sets up datetime, logging, seed and ddp
    @args:
        - config (Namespace): runtime namespace for setup
        - dirs (str list): the directories from config to be created
    """
    # get current date
    now = datetime.now()
    now = now.strftime("%H-%M-%S-%Y%m%d") # make sure in ddp, different nodes have the save file name
    config.date = now

    # setup logging
    setup_logger(config)

    # create relevant directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.join(config.log_dir,config.run_name), exist_ok=True)
    logging.info(f"Run:{config.run_name}, {dir} is {os.path.join(config.log_dir,config.run_name)}")

    # set seed
    if config.seed is None:
        config.seed = np.random.randint(2000, 1000000)
        logging.info(f"config.seed:{config.seed}")
    set_seed(config.seed)

    # setup dp/ddp 
    if not dist.is_initialized():
        config.device = get_device(config.device)
        world_size = torch.cuda.device_count()

        if config.ddp:
            if config.device == torch.device('cpu') or world_size <= 1:
                config.ddp = False

        config.world_size = world_size if config.ddp else -1
    else:
        world_size = int(os.environ["WORLD_SIZE"])
        config.world_size = world_size

    if config.ddp: setup_ddp()
    logging.info(f"Training on {config.device} with ddp set to {config.ddp}")

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

if __name__=="__main__":
    pass