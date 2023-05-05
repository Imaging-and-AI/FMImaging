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
    "--num_epochs", "150",
    "--batch_size", "512",
    "--device", "cuda",
    "--norm_mode", "instance2d",
    "--window_size", "8",
    "--patch_size", "2",
    "--global_lr", "1e-3",
    "--clip_grad_norm", "1.0",
    "--normalize_Q_K", "False",
    "--weight_decay", "0.0",
    "--use_amp", "--ddp", 
    "--iters_to_accumulate", "2",
    "--project", "cifar",
    "--num_workers", "8",
    "--scheduler", "OneCycleLR"
])

# commands to iterate over
att_types = ['T1L1G1T1L1G1T1L1G1', 'L1G1L1G1L1G1', "T1T1T1", "L1L1L1", "G1G1G1", "T0T0T0", "L0L0L0", "G0G0G0"]
a_types = ["conv", "lin"]
cells_in_a_block = 3
for att_type, a_type in itertools.product(att_types, a_types):
    if 'T' in att_type and a_type == "lin": continue

    cmd_run = cmd.copy()
    cmd_run.extend([
        "--run_name", f"cifar_{att_type}_{cells_in_a_block}_{a_type}_inst_n",
        "--run_notes", f"{cells_in_a_block}_{att_type}_cells_in_a_block_w_mixer_atype_{a_type}_w_instance_n",
        "--a_type", f"{a_type}",
        "--att_types", f"{att_type}"
    ])

    print(f"Running command:\n{' '.join(cmd_run)}")

    subprocess.run(cmd_run)
