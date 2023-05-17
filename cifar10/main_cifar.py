"""
Main file for STCNNT Cifar10
"""
import torchvision as tv
from torchvision import transforms
import torchvision.transforms as T
import torch.multiprocessing as mp
import torch.distributed as dist

import wandb
import logging
import argparse
import pprint
import pickle

# from colorama import Fore, Style

from datetime import datetime, timedelta

import os
import sys
from pathlib import Path
Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from trainer_cifar import *
from model_cifar import STCNNT_Cifar

# # True class names
# classes = ['airplane', 'automobile', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# -------------------------------------------------------------------------------------------------
# Extra args on top of shared args

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - parser (ArgumentParser): the argparse for STCNNT Cifar10
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT Cifar10")
    parser.add_argument("--data_root", type=str, default=None, help='root folder for the data')
    parser.add_argument("--data_set", type=str, default="cifar10", help='choice of dataset: "cifar10", "cifar100", "imagenet"')
    parser.add_argument("--head_channels", nargs='+', type=int, default=[8,128,10], help='number of channels for cifar head')
    
    parser = add_backbone_STCNNT_args(parser=parser)

    ns = Nestedspace()
    args = parser.parse_args(namespace=ns)
    
    return args

def check_args(config):
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
        
    if config.data_root is None: config.data_root = os.path.join(project_base_dir, config.data_set, "data")
    if config.check_path is None: config.check_path = os.path.join(project_base_dir, config.data_set, "checkpoints")
    if config.model_path is None: config.model_path = os.path.join(project_base_dir, config.data_set, "models")
    if config.log_path is None: config.log_path = os.path.join(project_base_dir, config.data_set, "logs")
    if config.results_path is None: config.results_path = os.path.join(project_base_dir, config.data_set, "results")
        
    if config.run_name is None: config.run_name = config.data_set + '_' + datetime.now().strftime("%H-%M-%S")

    if config.data_set == 'cifar10':
        config.time = 1
        config.height = [32]
        config.width = [32]
        config.num_classes = 10
    
    if config.data_set == 'cifar10' and config.data_root is None:
        config.data_root = "/export/Lab-Xue/projects/cifar10/data"
        
    if config.data_set == 'cifar100' and config.data_root is None:
        config.data_root = "/export/Lab-Xue/projects/cifar100/data"

    if config.data_set == "imagenet":
        config.height = [256]
        config.width = [256]
        config.num_classes = 1000
        if config.data_root is None:
            config.data_root = "/export/Lab-Xue/projects/imagenet/data"
    
    return config
   

# -------------------------------------------------------------------------------------------------
   
config_default = arg_parser()

# -------------------------------------------------------------------------------------------------

def run_training():
    
    global config_default
       
    if config_default.ddp:
        rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = -1
        global_rank = -1
        world_size = 1
        
    print(f"{Fore.RED}---> Run start on local rank {rank} - global rank {global_rank} <---{Style.RESET_ALL}", flush=True)
           
    wandb_run = None    
    if(config_default.sweep_id != 'none'):
        if rank<=0:
            print(f"---> get the config from wandb on local rank {rank}", flush=True)
            wandb_run = wandb.init()
            config = set_up_config_for_sweep(wandb_run.config, config_default)   
            config.run_name = wandb_run.name
            print(f"---> wandb run is {wandb_run.name} on local rank {rank}", flush=True)
        else:
            config = config_default
    else:
        # Config is a variable that holds and saves hyperparameters and inputs
        config = config_default
        if rank<=0:
            wandb_run = wandb.init(project=config.project, 
                    entity=config.wandb_entity, 
                    config=config, 
                    name=config.run_name, 
                    notes=config.run_notes)

    if config_default.ddp:
                                    
        if(config_default.sweep_id != 'none'):
            
            #is_master = rank<=0
            #print(f"---> dist.TCPStore on local rank {rank}, is_master {is_master}", flush=True)
            #store=dist.TCPStore("localhost", 9001, dist.get_world_size(), is_master=is_master, timeout=timedelta(seconds=30), wait_for_workers=True)

            # if rank<=0:
            #     # add parameter to the store
            #     print(f"---> set the config to key store on local rank {rank}", flush=True)
            #     config_str = pickle.dumps(config)     
            #     store.set("config", config_str)    
            # else:
            #     print(f"---> get the config from key store on local rank {rank}", flush=True)
            #     config_str = store.get("config")
            #     config = pickle.loads(config_str)

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
          
    try: 
        trainer(rank=rank, config=config, wandb_run=wandb_run)
                                  
        if rank<=0:
            wandb_run.finish()                                
                
        print(f"{Fore.RED}---> Run finished on local rank {rank} <---{Style.RESET_ALL}", flush=True)
                
    except KeyboardInterrupt:
        print('Interrupted')

        if config_default.ddp:
            torch.distributed.destroy_process_group()            

        os.system("kill $(ps aux | grep torchrun | grep -v grep | awk '{print $2}') ")
        os.system("kill $(ps aux | grep wandb | grep -v grep | awk '{print $2}') ")
    
# -------------------------------------------------------------------------------------------------
# main function. spawns threads if going for distributed data parallel

def main():
    
    global config_default
    
    if config_default.ddp:
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
    
    config_default = check_args(config_default)
    setup_run(config_default)                
               
    print(f"--------> run training on local rank {rank}", flush=True)
                            
    # note the sweep_id is used to control the condition
    sweep_id = config_default.sweep_id
    print("get sweep id : ", sweep_id, flush=True)
    if (sweep_id != "none"):
        print("start sweep runs ...", flush=True)
                    
        if rank<=0:
            wandb.agent(sweep_id, run_training, project="cifar", count=50)
        else:
            print(f"--> local rank {rank} - not start another agent", flush=True)
            run_training()             
    else:
        print("start a regular run ...", flush=True)        
        run_training()
                   
    if config_default.ddp:         
        if dist.is_initialized():
            print(f"---> dist.destory_process_group on local rank {rank}", flush=True)
            dist.destroy_process_group()

if __name__=="__main__":    
    main()
