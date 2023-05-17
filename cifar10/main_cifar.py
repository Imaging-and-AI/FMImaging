"""
Main file for STCNNT Cifar10
"""
import wandb
import logging
import argparse
import pprint
import pickle

from colorama import Fore, Style

from datetime import datetime, timedelta

import torchvision as tv
from torchvision import transforms
import torchvision.transforms as T
import torch.multiprocessing as mp
import torch.distributed as dist

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from trainer_cifar import *
from model_cifar import STCNNT_Cifar

# True class names
classes = ['airplane', 'automobile', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
    assert config.run_name is not None, f"Please provide a \"--run_name\" for wandb"
    assert config.log_path is not None, f"Please provide a \"--log_path\" to save the logs in"
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.model_path is not None, f"Please provide a \"--model_path\" to save the final model in"
    assert config.check_path is not None, f"Please provide a \"--check_path\" to save the checkpoints in"

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

config_default = check_args(arg_parser())
setup_run(config_default)
    
# -------------------------------------------------------------------------------------------------

def run_training():
    
    if config_default.ddp:
        rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        rank = -1
        global_rank = -1
        
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
            
            torch.distributed.broadcast_object_list(c_list, src=0, group=None, device=rank)
            print(f"{Fore.RED}--->after, on local rank {rank}, {c_list[0].run_name}{Style.RESET_ALL}", flush=True)
            if rank>0:
                config = c_list[0]
                        
        print(f"---> config synced for the local rank {rank}")                        
        dist.barrier()        
        print(f"{Fore.RED}---> Ready to run on local rank {rank}, {config.run_name}{Style.RESET_ALL}", flush=True)
          
    try: 
        trainer(rank=rank, config=config, wandb_run=wandb_run)
                                  
        if rank<=0:
            wandb_run.finish()                                
                
        print(f"{Fore.RED}---> Run finished on local rank {rank} <---{Style.RESET_ALL}", flush=True)
                
    except KeyboardInterrupt:
        print('Interrupted')
        try: 
            torch.distributed.destroy_process_group()
        except KeyboardInterrupt: 
            os.system("kill $(ps aux | grep torchrun | grep -v grep | awk '{print $2}') ")
            os.system("kill $(ps aux | grep wandb | grep -v grep | awk '{print $2}') ")
    
# -------------------------------------------------------------------------------------------------
# main function. spawns threads if going for distributed data parallel

def main():
    
    sweep_id = config_default.sweep_id

    if config_default.ddp:        
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank=-1
        
    # note the sweep_id is used to control the condition
    print("get sweep id : ", sweep_id)
    if (sweep_id != "none"):
        print("start sweep runs ...")
        
        if config_default.ddp:
            if not dist.is_initialized():
                print(f"---> dist.init_process_group on local rank {rank}", flush=True)
                dist.init_process_group("nccl", timeout=timedelta(seconds=18000))
                    
        if rank<=0:
            wandb.agent(sweep_id, run_training, project="cifar", count=50)
        else:
            print(f"--> local rank {rank} - not start another agent")
            run_training() 
            
        if config_default.ddp:
            if dist.is_initialized():
                print(f"---> dist.destory_process_group on local rank {rank}", flush=True)
                dist.destroy_process_group()
    else:
        print("start a regular run ...")        
        run_training()

if __name__=="__main__":
    main()
