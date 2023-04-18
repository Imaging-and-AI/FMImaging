"""
Python script to run bash scripts in batches
"""

import itertools
import subprocess

# base command to run a file
cmd = ["python3", "cifar10/main_cifar.py"]

# unchanging paths
cmd.extend([
    "--data_set", "cifar10",
    "--data_root", "/home/rehmana2/projects/STCNNT_2/cifar10",
    "--check_path", "/home/rehmana2/projects/STCNNT_2/checkpoints",
    "--model_path", "/home/rehmana2/projects/STCNNT_2/models",
    "--log_path", "/home/rehmana2/projects/STCNNT_2/logs",
    "--results_path", "/home/rehmana2/projects/STCNNT_2/results"
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
a_types = ["conv", "lin"]
mixers = ["all", "none"]
cells_in_a_block = 3
for att_type, a_type, mixer in itertools.product(att_types, a_types, mixers):
    if att_type == "temporal" and a_type == "lin": continue

    cmd_run = cmd.copy()
    if att_type == "temporal":
        cmd_run.extend([
            "--run_name", f"{att_type}_{cells_in_a_block}_{mixer}",
            "--run_notes", f"{cells_in_a_block}_{att_type}_cells_in_a_block_w_mixer_{mixer}",
            "--with_mixer", f"{mixer}",
            "--att_types"
        ])
    else:
        cmd_run.extend([
            "--run_name", f"{att_type}_{cells_in_a_block}_{a_type}_{mixer}_inst_n",
            "--run_notes", f"{cells_in_a_block}_{att_type}_cells_in_a_block_w_mixer_{mixer}_atype_{a_type}_w_instance_n",
            "--with_mixer", f"{mixer}",
            "--a_type", f"{a_type}",
            "--att_types"
        ])
    for i in range(cells_in_a_block): cmd_run.append(att_type)

    print(f"Running command:\n{cmd_run}")

    subprocess.run(cmd_run)
