"""
Python script to run bash scripts in batches
"""
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from trainer_base import *

# -------------------------------------------------------------

class imagenet_ddp_base(run_ddp_base):
    
    def __init__(self, project, script_to_run) -> None:
        super().__init__(project, script_to_run)
        
    def set_up_constants(self, config):
        
        super().set_up_constants(config)
        
        self.cmd.extend([
        "--data_set", "imagenet",
        
        "--num_epochs", "150",
        "--batch_size", "32",

        "--window_size", "32", "32",
        "--patch_size", "8", "8",

        "--n_head", "32",

        "--global_lr", "1e-4",
        "--clip_grad_norm", "1.0",
        "--weight_decay", "1.0",
        "--use_amp", 

        "--iters_to_accumulate", "1",

        "--num_workers", "16",
        
        "--scheduler_type", "OneCycleLR",
                
        # hrnet
        "--backbone_hrnet.num_resolution_levels", "3",
        
        # unet            
        "--backbone_unet.num_resolution_levels", "3",
        
        # LLMs
        "--backbone_LLM.num_stages", "3",
        "--backbone_LLM.add_skip_connections", "1",
                        
        # small unet
        "--backbone_small_unet.channels", "16", "32", "64",   
        "--backbone_small_unet.block_str", "T1L1G1", "T1L1G1", "T1L1G1",
        
        "--ratio", "100", "100", "100"
        ])

    def set_up_variables(self, config):
        
        vars = dict()
        
        vars['backbone'] = ['hrnet']
        vars['cell_types'] = ["sequential", "parallel"]
        vars['Q_K_norm'] = [True]
        vars['cosine_atts'] = ["1"]
        vars['att_with_relative_postion_biases'] = ["1"]
        vars['a_types'] = ["conv"]

        vars['larger_mixer_kernels'] = [False]
        vars['mixer_types'] = ["conv"]
        vars['shuffle_in_windows'] = ["0"]
        vars['block_dense_connections'] = ["1"]
        vars['norm_modes'] = ["layer"]
        vars['C'] = [64]
        vars['scale_ratio_in_mixers'] = [4.0]

        vars['block_strs'] = [
                        [["T1L1G1", "T1L1G1", "T1L1G1"], ["T1T1T1", "T1T1T1", "T1T1T1"] ]
                    ]
    
        return vars

# -------------------------------------------------------------

def main():
    
    ddp_run = imagenet_ddp_base(project="imagenet", script_to_run='./cifar10/main_cifar.py')
    ddp_run.run()

# -------------------------------------------------------------
if __name__=="__main__":
    main()