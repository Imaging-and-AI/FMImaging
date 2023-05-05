"""
Python script to run bash scripts in batches
"""

import itertools
import subprocess
import os

# base command to run a file
cmd = ["python3", "cifar10/main_cifar.py"]

if "FMIMAGING_PROJECT_BASE" in os.environ:
    project_base_dir = os.environ['FMIMAGING_PROJECT_BASE']
else:
    project_base_dir = '/export/Lab-Xue/projects'

# unchanging paths
cmd.extend([
    "--data_set", "cifar10",
    "--data_root", os.path.join(project_base_dir, "cifar10", "data"),
    "--check_path", os.path.join(project_base_dir, "cifar10", "checkpoints"),
    "--model_path", os.path.join(project_base_dir, "cifar10", "models"),
    "--log_path", os.path.join(project_base_dir, "cifar10", "logs"),
    "--results_path", os.path.join(project_base_dir, "cifar10", "results")
])

# unchanging commands
cmd.extend([
    "--num_epochs", "100",
    "--batch_size", "128",
    "--device", "cuda",
    "--norm_mode", "instance2d",
    "--window_size", "8",
    "--patch_size", "2",
    "--global_lr", "1e-3",
    "--clip_grad_norm", "1.0",
    "--weight_decay", "0.0",
    "--use_amp", "--ddp", 
    "--iters_to_accumulate", "1",
    "--project", "cifar",
    "--num_workers", "8",
    "--scheduler", "ReduceLROnPlateau",
    
    "--backbone", "unet",
    
    "--backbone_hrnet.C", "32",
    "--backbone_hrnet.num_resolution_levels", "2",
    "--backbone_hrnet.block_str", "T1L1G1", "T1L1G1",
    "--backbone_hrnet.use_interpolation", "True",
    
    # unet            
    "--backbone_unet.C", "32",
    "--backbone_unet.num_resolution_levels", "2",
    "--backbone_unet.block_str", "T1L1G1", "T1L1G1", "T1L1G1",
    "--backbone_unet.use_unet_attention", "True",
    "--backbone_unet.use_interpolation", "True",
    "--backbone_unet.with_conv", "True",
    
    # LLMs
    "--backbone_LLM.C", "32",
    "--backbone_LLM.num_stages", "3",
    "--backbone_LLM.block_str", "T1L1G1", "T1L1G1", "T1L1G1",
    "--backbone_LLM.add_skip_connections", "True",
                     
    # small unet
    "--backbone_small_unet.channels", "16", "32", "64",   
    "--backbone_small_unet.block_str", "T1L1G1", "T1L1G1", "T1L1G1"    
])

# test backbones
backbone = ['unet', 'hrnet', 'LLM', 'small_unet']
block_strs = [
                [["T1L1G1", "T1L1G1", "T1L1G1"], ["T1T1T1", "T1T1T1", "T1T1T1"], ["L1G1", "L1G1", "L1G1"], ["L1L1", "L1L1", "L1L1"], ["G1G1", "G1G1", "G1G1"] ], 
                [["T1L1G1", "T1L1G1"], ["T1T1T1", "T1T1T1"], ["L1G1", "L1G1"] ], 
                [["T1L1G1", "T1L1G1"], ["T1T1T1", "T1T1T1"], ["L1G1", "L1G1"] ] , 
                [["T1L1G1", "T1L1G1", "T1L1G1"], ["T1T1T1", "T1T1T1", "T1T1T1"], ["L1G1", "L1G1", "L1G1"] ], 
            ]

a_types = ["conv", "lin"]
cell_types = ["sequential", "parallel"]
Q_K_norm = ["True", "False"]

for k, bk in enumerate(backbone):    
        block_str = block_strs[k]

        for a_type, cell_type, q_k_norm in itertools.product(a_types, cell_types, Q_K_norm):
            for bs in block_str:
                
                run_str = f"{a_type}-{cell_type}-qknorm-{q_k_norm}-block_str-{'_'.join(bs)}"
                
                cmd_run = cmd.copy()
                cmd_run.extend([
                    "--run_name", f"cifar-{bk}-{run_str}",
                    "--run_notes", f"cifar-{bk}-{run_str}",
                    "--backbone", f"{bk}",
                    "--a_type", f"{a_type}",
                    "--cell_type", f"{cell_type}"
                ])

                if q_k_norm:
                    cmd_run.extend(["--normalize_Q_K"])
                    
                cmd_run.extend([f"--backbone_{bk}.block_str", *bs])
                    
                print(f"Running command:\n{' '.join(cmd_run)}")

                subprocess.run(cmd_run)
