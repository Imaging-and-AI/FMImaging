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
    "--norm_mode", "batch2d",
    "--window_size", "8",
])

# commands to iterate over
att_types = ["temporal", "local", "global"]
att_types = ["T0T0T0", "T1T1T1", "L0L0L0", "L1L1L1", "G0G0G0", "G1G1G1"]
a_types = ["conv", "lin"]
cells_in_a_block = 3
for att_type, a_type in itertools.product(att_types, a_types):
    if 'T' in att_type and a_type == "lin": continue

    cmd_run = cmd.copy()
    if att_type == "temporal":
        cmd_run.extend([
            "--run_name", f"{att_type}_{cells_in_a_block}",
            "--run_notes", f"{cells_in_a_block}_{att_type}_cells_in_a_block_w_mixer",
            "--att_types", f"{att_type}"
        ])
    else:
        cmd_run.extend([
            "--run_name", f"{att_type}_{cells_in_a_block}_{a_type}_inst_n",
            "--run_notes", f"{cells_in_a_block}_{att_type}_cells_in_a_block_w_mixer_atype_{a_type}_w_instance_n",
            "--a_type", f"{a_type}",
            "--att_types", f"{att_type}"
        ])

    print(f"Running command:\n{cmd_run}")

    subprocess.run(cmd_run)
