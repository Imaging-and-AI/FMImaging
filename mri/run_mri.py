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

class mri_ddp_base(run_ddp_base):
    
    def __init__(self, project, script_to_run) -> None:
        super().__init__(project, script_to_run)
        
    def set_up_constants(self, config):
        
        super().set_up_constants(config)
        
        self.cmd.extend([       
       
        "--num_epochs", "30",
        "--batch_size", "32",

        "--window_size", "8", "8",
        "--patch_size", "2", "2",

        "--n_head", "32",

        "--global_lr", "1e-4",

        "--clip_grad_norm", "1.0",
        "--weight_decay", "1.0",

        "--use_amp", 

        "--iters_to_accumulate", "1",

        "--num_workers", f"{os.cpu_count()//(2*config.nproc_per_node)}",
        "--prefetch_factor", "4",
        
        "--scheduler_type", "ReduceLROnPlateau",
        "--scheduler.ReduceLROnPlateau.patience", "0",
        "--scheduler.ReduceLROnPlateau.cooldown", "0",
        
        # hrnet
        "--backbone_hrnet.num_resolution_levels", "2",
        
        # unet            
        "--backbone_unet.num_resolution_levels", "3",
        
        # LLMs
        "--backbone_LLM.num_stages", "3",
                        
        # small unet
        "--backbone_small_unet.channels", "16", "32", "64",   
        "--backbone_small_unet.block_str", "T1L1G1", "T1L1G1", "T1L1G1",
        
        "--min_noise_level", "2.0",
        "--max_noise_level", "8.0",
        "--complex_i",
        "--residual",
        "--losses", "mse", "l1",
        "--loss_weights", "1.0", "1.0",
        "--height", "32", "64",
        "--width", "32", "64",
        "--time", "12",
        #"--max_load", "10000",
        
        "--train_files", "train_3D_3T_retro_cine_2018.h5", "train_3D_3T_perf_2021.h5", "/export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2019.h5", "train_3D_3T_retro_cine_2020.h5",
        "--train_data_types", "2dt", "2dt", "3d", "2d",
        
        "--test_files", "train_3D_3T_retro_cine_2020_small_3D_test.h5", "train_3D_3T_retro_cine_2020_small_2DT_test.h5", "train_3D_3T_retro_cine_2020_small_2D_test.h5"
        "--test_data_types", "3d", "2dt", "2d"
        ])
        
        if config.tra_ratio > 0 and config.tra_ratio<=100:
            self.cmd.extend(["--ratio", f"{int(config.tra_ratio)}", f"{int(config.val_ratio)}", "5"])
            
        self.cmd.extend(["--max_load", f"{int(config.max_load)}"])

    def set_up_variables(self, config):
        
        vars = dict()
                
        vars['optim'] = ['sophia']
        
        vars['backbone'] = ['hrnet']
        vars['cell_types'] = ["parallel", "sequential"]
        vars['Q_K_norm'] = [True]
        vars['cosine_atts'] = ["1"]
        vars['att_with_relative_postion_biases'] = ["1"]
        vars['a_types'] = ["conv"]

        vars['larger_mixer_kernels'] = [False]
        vars['mixer_types'] = ["conv"]
        vars['shuffle_in_windows'] = ["0"]
        vars['block_dense_connections'] = ["0"]
        vars['norm_modes'] = ["batch2d", "instance2d"]
        vars['C'] = [32]
        vars['scale_ratio_in_mixers'] = [1.0]

        vars['block_strs'] = [
                        [ 
                            ["T1L1G1", "T1L1G1", "T1L1G1", "T1L1G1"],
                            ["T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"],                            
                            ["T1T1T1", "T1T1T1", "T1T1T1", "T1T1T1"],
                            ["T1T1T1T1T1T1", "T1T1T1T1T1T1", "T1T1T1T1T1T1", "T1T1T1T1T1T1"]
                         ]
                    ]

        return vars

    def arg_parser(self):
        parser = super().arg_parser()
        parser.add_argument("--max_load", type=int, default=-1, help="number of max loaded samples into the RAM")
        return parser
    
# -------------------------------------------------------------

def main():
    
    os.system("ulimit -n 65536")
    
    ddp_run = mri_ddp_base(project="mri", script_to_run='./mri/main_mri.py')
    ddp_run.run()

# -------------------------------------------------------------

if __name__=="__main__":
    main()