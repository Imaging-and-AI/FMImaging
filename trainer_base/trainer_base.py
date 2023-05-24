"""
A base model for training, supporting multi-node, multi-gpu training
"""

import os
import sys
import logging
import argparse
import pprint
import pickle
import abc 
from abc import ABC
from colorama import Fore, Style
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torchvision import transforms
import torchvision.transforms as T
import torch.multiprocessing as mp
import torch.distributed as dist

from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *

import wandb

__all__ = ['Trainer_Base']

#torch.multiprocessing.set_sharing_strategy('file_system')
#torch.multiprocessing.set_start_method('spawn')

# -------------------------------------------------------------------------------------------------
# Base model for training
# to handle the multi-node, multi-gpu training
#
# support wandb sweep.
# Note: wandb sweep does not work with torchrun due to multi-processiong conflicts.
# 
# this class supports:
# - single node, single process, single gpu training
# - single node, multiple process, multiple gpu training
# - multiple nodes, multiple processes, multiple gpu training

class Trainer_Base(ABC):
    """
    Base Runtime model for training.
    """
    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
        """
        super().__init__()
        self.config = config

    
    @abc.abstractmethod
    def run_task_trainer(self, rank=-1, wandb_run=None):
        """
        Main training function
        @args:
            - rank (int): local rank for the multi-processing; rank=-1 means 
            the single processing run.
            
        @output:
            None
        """
        config = self.config
        pass
    
    def check_args(self):
        """
        checks the cmd args to make sure they are correct
        @args:
            - config (Namespace): runtime namespace for setup
        @rets:
            - config (Namespace): the checked and updated argparse for Cifar10
        """
        
        if "FMIMAGING_PROJECT_BASE" in os.environ:
            project_base_dir = os.environ['FMIMAGING_PROJECT_BASE']
        else:
            if sys.platform == 'win32':
                project_base_dir = r'\\hl-share\sbc\Lab-Xue\projects'
            else:
                project_base_dir = '/export/Lab-Xue/projects'
            
        if self.config.data_root is None: self.config.data_root = os.path.join(project_base_dir, self.config.data_set, "data")
        if self.config.check_path is None: self.config.check_path = os.path.join(project_base_dir, self.config.data_set, "checkpoints")
        if self.config.model_path is None: self.config.model_path = os.path.join(project_base_dir, self.config.data_set, "models")
        if self.config.log_path is None: self.config.log_path = os.path.join(project_base_dir, self.config.data_set, "logs")
        if self.config.results_path is None: self.config.results_path = os.path.join(project_base_dir, self.config.data_set, "results")
            
        if self.config.run_name is None: self.config.run_name = self.config.data_set + '_' + datetime.now().strftime("%H-%M-%S")
        
        
    def set_up_config_for_sweep(self, wandb_config):
        """Update the config with the parameters from wandb
        """
        
        self.config.num_epochs = wandb_config.num_epochs
        self.config.batch_size = wandb_config.batch_size
        self.config.global_lr = wandb_config.global_lr
        self.config.window_size = wandb_config.window_size
        self.config.patch_size = wandb_config.patch_size
        self.config.weight_decay = wandb_config.weight_decay
        self.config.scheduler_type = wandb_config.scheduler_type
        self.config.use_amp = wandb_config.use_amp
        self.config.a_type = wandb_config.a_type
        self.config.cell_type = wandb_config.cell_type
        self.config.n_head = wandb_config.n_head
        self.config.scale_ratio_in_mixer = wandb_config.scale_ratio_in_mixer
        
        self.config.mixer_type = wandb_config.mixer_type
        self.config.normalize_Q_K = wandb_config.normalize_Q_K
        self.config.cosine_att = wandb_config.cosine_att
        self.config.att_with_relative_postion_bias = wandb_config.att_with_relative_postion_bias
        
        self.config.backbone_hrnet.C = wandb_config.C
        self.config.backbone_hrnet.num_resolution_levels = wandb_config.num_resolution_levels
        self.config.backbone_hrnet.block_str = wandb_config.block_str
        
        return self.config
    
    def run_training(self):
        """
        Function to run training.
        This function will set up the wandb run and 
        initialize/destory the torch run process group.
        
        To support the wandb sweep, the parameters are read from wandb and 
        broadcasted to all processes.
        """
      
        # -------------------------------------------------------
        # get the rank and runtime info
        if self.config.ddp:
            rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            rank = -1
            global_rank = -1
            world_size = 1
            
        print(f"{Fore.RED}---> Run start on local rank {rank} - global rank {global_rank} <---{Style.RESET_ALL}", flush=True)
            
        # -------------------------------------------------------
        # if sweeping, update the config with parameters from wandb
        # perform the wandb.init to create a run
        # only the rank=0 process get the wandb parameters
        wandb_run = None    
        if(self.config.sweep_id != 'none'):
            if rank<=0:
                print(f"---> get the config from wandb on local rank {rank}", flush=True)
                wandb_run = wandb.init()
                config = self.set_up_config_for_sweep(wandb_run.config)   
                #config.run_name = wandb_run.name
                print(f"---> wandb run is {wandb_run.name} on local rank {rank}", flush=True)
            else:
                config = self.config
        else:
            # if not sweep, use the inputted parameters
            # Config is a variable that holds and saves hyperparameters and inputs
            config = self.config
            if rank<=0:
                wandb_run = wandb.init(project=config.project, 
                        entity=config.wandb_entity, 
                        config=config, 
                        name=config.run_name, 
                        notes=config.run_notes)

        # -------------------------------------------------------
        # if ddp is used, broadcast the parameters from rank0 to all other ranks
        if self.config.ddp:                                        
            if(self.config.sweep_id != 'none'):
                
                if rank<=0:
                    c_list = [config]
                    print(f"{Fore.RED}--->before, on local rank {rank}, {c_list[0].run_name}{Style.RESET_ALL}", flush=True)
                else:
                    c_list = [None]
                
                if world_size > 1:
                    torch.distributed.broadcast_object_list(c_list, src=0, group=None, device=rank)
                    
                print(f"{Fore.RED}--->after, on local rank {rank}, {c_list[0].run_name}{Style.RESET_ALL}", flush=True)
                if rank>0:
                    config = c_list[0]
                            
            print(f"---> config synced for the local rank {rank}")                        
            if world_size > 1: dist.barrier()        
            print(f"{Fore.RED}---> Ready to run on local rank {rank}, {config.run_name}{Style.RESET_ALL}", flush=True)
            
        # -------------------------------------------------------
        # run the training for current rank and wandb run
        try: 
            self.run_task_trainer(rank=rank, wandb_run=wandb_run)
                                    
            if rank<=0:
                wandb_run.finish()                                
                    
            print(f"{Fore.RED}---> Run finished on local rank {rank} <---{Style.RESET_ALL}", flush=True)
                    
        except KeyboardInterrupt:
            print('Interrupted')

            if self.config.ddp:
                torch.distributed.destroy_process_group()
                            
            # make sure the runtime is cleaned, by brutelly removing processes
            os.system("kill -9 $(ps aux | grep torchrun | grep -v grep | awk '{print $2}') ")
            os.system("kill -9 $(ps aux | grep wandb | grep -v grep | awk '{print $2}') ")
            os.system("kill -9 $(ps aux | grep python3 | grep -v grep | awk '{print $2}') ")
        
    def train(self):
           
        # -------------------------------------------------------
        # get the rank and runtime info
        if self.config.ddp:
            if not dist.is_initialized():            
                dist.init_process_group("nccl")
                                
            rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            
            print(f"{Fore.YELLOW}---> dist.init_process_group on local rank {rank}, global rank{global_rank}, world size {world_size}, local World size {local_world_size} <---{Style.RESET_ALL}", flush=True)
        else:
            rank = -1
            global_rank = -1        
            print(f"---> ddp is off <---", flush=True)
        
        # -------------------------------------------------------
        # set up and check the run arguments
        self.check_args()
        setup_run(self.config)                
                
        print(f"--------> run training on local rank {rank}", flush=True)
                                
        # -------------------------------------------------------
        # note the sweep_id is used to control the condition
        # if doing sweep, get the parameters from wandb
        sweep_id = self.config.sweep_id
        print("get sweep id : ", sweep_id, flush=True)
        if (sweep_id != "none"):
            print("start sweep runs ...", flush=True)
                        
            if rank<=0:
                wandb.agent(sweep_id, self.run_training, project=self.project, count=self.config.sweep_count)
            else:
                print(f"--> local rank {rank} - not start another agent", flush=True)
                self.run_training()             
        else:
            # if not doing sweep, start a regular run
            print("start a regular run ...", flush=True)        
            self.run_training()
                    
        # -------------------------------------------------------
        # after the run, release the process groups
        if self.config.ddp:         
            if dist.is_initialized():
                print(f"---> dist.destory_process_group on local rank {rank}", flush=True)
                dist.destroy_process_group()
# -------------------------------------------------------------------------------------------------

def tests():
    pass    

if __name__=="__main__":
    tests()
