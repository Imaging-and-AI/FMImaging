"""
Main file for STCNNT Cifar10
"""
import wandb
import logging
import argparse
import pprint

import torchvision as tv
from torchvision import transforms
import torchvision.transforms as T
import torch.multiprocessing as mp

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
    assert config.data_root is not None, f"Please provide a \"--data_root\" to load the data"
    assert config.log_path is not None, f"Please provide a \"--log_path\" to save the logs in"
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.model_path is not None, f"Please provide a \"--model_path\" to save the final model in"
    assert config.check_path is not None, f"Please provide a \"--check_path\" to save the checkpoints in"

    config.time = 1
    config.height = [32]
    config.width = [32]
    config.num_classes = 10

    if config.data_set == "imagenet":
        config.height = [256]
        config.width = [256]
        config.num_classes = 1000
    
    return config

# -------------------------------------------------------------------------------------------------

config_default = check_args(arg_parser())
setup_run(config_default)
   
def set_up_config_for_sweep(wandb_config, config):
    config = config_default
    config.num_epochs = wandb_config.num_epochs
    config.batch_size = wandb_config.batch_size
    config.global_lr = wandb_config.global_lr
    config.weight_decay = wandb_config.weight_decay
    config.scheduler_type = wandb_config.scheduler_type
    config.use_amp = wandb_config.use_amp
    config.a_type = wandb_config.a_type
    config.n_head = wandb_config.n_head
    config.scale_ratio_in_mixer = wandb_config.scale_ratio_in_mixer
    
    config.backbone_hrnet.C = wandb_config.C
    config.backbone_hrnet.num_resolution_levels = wandb_config.num_resolution_levels
    config.backbone_hrnet.block_str = wandb_config.block_str
    
    return config

# ------------------------------------------------------------------------------------------------- run training
def run_training():
    
    if(config_default.sweep_id != 'none'):
        print("---> get the config from wandb ")
        wandb.init(project=config_default.project)
        config = set_up_config_for_sweep(wandb.config, config_default)        
    else:
        # Config is a variable that holds and saves hyperparameters and inputs
        config = config_default
        wandb.init(project=config.project, 
                   entity=config.wandb_entity, 
                   config=config_default, 
                   name=config.run_name, 
                   notes=config.run_notes)
        
    pprint.pprint(config)

    train_set, val_set = create_dataset(config=config)

    total_steps = compute_total_steps(config, len(train_set))

    model = STCNNT_Cifar(config=config, total_steps=total_steps)

    # model summary
    model_summary = model_info(model, config)
    logging.info(f"Configuration for this run:\n{config}")
    logging.info(f"Model Summary:\n{str(model_summary)}")
            
    if not config.ddp: # run in main process
        trainer(rank=-1, model=model, config=config, train_set=train_set, val_set=val_set)
    else: # spawn a process for each gpu        
        try: 
            mp.spawn(trainer, args=(model, config, train_set, val_set), nprocs=config.world_size)
        except KeyboardInterrupt:
            print('Interrupted')
            try: 
                torch.distributed.destroy_process_group()
            except KeyboardInterrupt: 
                os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
                os.system("kill $(ps aux | grep wandb | grep -v grep | awk '{print $2}') ")
    
# -------------------------------------------------------------------------------------------------
# main function. spawns threads if going for distributed data parallel

def main():
    
    sweep_id = config_default.sweep_id

    # note the sweep_id is used to control the condition
    print("get sweep id : ", sweep_id)
    if (sweep_id != "none"):
        print("start sweep runs ...")
        wandb.agent(sweep_id, run_training, project="cifar", count=50)
    else:
        print("start a regular run ...")        
        run_training()

if __name__=="__main__":
    main()
