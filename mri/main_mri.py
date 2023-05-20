"""
Main file for STCNNT MRI denoising
"""
import logging
import argparse

import torchvision as tv
from torchvision import transforms
import torchvision.transforms as T
import torch.multiprocessing as mp
import torch.distributed as dist

import sys
from colorama import Fore, Style
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from trainer_mri import trainer, set_up_config_for_sweep
from model_mri import STCNNT_MRI
from data_mri import load_mri_data

# -------------------------------------------------------------------------------------------------
# Extra args on top of shared args

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT MRI")
    parser.add_argument("--data_root", type=str, default=None, help='root folder for the data')
    parser.add_argument("--train_files", type=str, nargs='+', default=["train_3D_3T_retro_cine_2020_small.h5"], help='list of train h5files')
    parser.add_argument("--test_files", type=str, nargs='+', default=["train_3D_3T_retro_cine_2020_small_2DT_test.h5"], help='list of test h5files')
    parser.add_argument("--train_data_types", type=str, nargs='+', default=["2dt"], help='the type of each train file: "2d", "2dt", "3d"')
    parser.add_argument("--test_data_types", type=str, nargs='+', default=["2dt"], help='the type of each test file: "2d", "2dt", "3d"')
    parser = add_backbone_STCNNT_args(parser=parser)

    # Noise Augmentation arguments
    parser.add_argument("--min_noise_level", type=float, default=3.0, help='minimum noise sigma to add')
    parser.add_argument("--max_noise_level", type=float, default=6.0, help='maximum noise sigma to add')
    parser.add_argument('--matrix_size_adjust_ratio', type=float, nargs='+', default=[0.5, 0.75, 1.0, 1.25, 1.5], help='down/upsample the image, keeping the fov')
    parser.add_argument('--kspace_filter_sigma', type=float, nargs='+', default=[0.8, 1.0, 1.5, 2.0, 2.25], help='sigma for kspace filter')
    parser.add_argument('--pf_filter_ratio', type=float, nargs='+', default=[1.0, 0.875, 0.75, 0.625], help='pf filter ratio')
    parser.add_argument('--phase_resolution_ratio', type=float, nargs='+', default=[1.0, 0.75, 0.65, 0.55], help='phase resolution ratio')
    parser.add_argument('--readout_resolution_ratio', type=float, nargs='+', default=[1.0, 0.75, 0.65, 0.55], help='readout resolution ratio')

    # 2d/3d dataset arguments
    parser.add_argument('--twoD_num_patches_cutout', type=int, default=1, help='for 2D usecase, number of patches per frame')
    parser.add_argument("--twoD_patches_shuffle", action="store_true", help='shuffle 2D patches to break spatial consistency')
    parser.add_argument('--threeD_cutout_jitter', nargs='+', type=float, default=[-1, 0.5, 0.75, 1.0], help='cutout jitter range, relative to the cutout_shape')
    parser.add_argument("--threeD_cutout_shuffle_time", action="store_true", help='shuffle along time to break temporal consistency; for 2D+T, should not set this option')

    # loss for mri
    parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1"], help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D"')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0, 1.0], help='to balance multiple losses, weights can be supplied')
    parser.add_argument("--complex_i", action="store_true", help='whether we are dealing with complex images or not')
    parser.add_argument("--residual", action="store_true", help='add long term residual connection')

    ns = Nestedspace()
    args = parser.parse_args(namespace=ns)
    
    return args

def check_args(config):
    """
    checks the cmd args to make sure they are correct
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - config (Namespace): the checked and updated argparse for MRI
    """
    assert config.run_name is not None, f"Please provide a \"--run_name\" for wandb"
    assert config.data_root is not None, f"Please provide a \"--data_root\" to load the data"
    assert config.log_path is not None, f"Please provide a \"--log_path\" to save the logs in"
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.model_path is not None, f"Please provide a \"--model_path\" to save the final model in"
    assert config.check_path is not None, f"Please provide a \"--check_path\" to save the checkpoints in"

    config.C_in = 3 if config.complex_i else 2
    config.C_out = 2 if config.complex_i else 1

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
