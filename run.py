"""
Python script to run bash scripts in batches
"""

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
    "--no_w_decay"
])

# commands to iterate over
att_types = ["temporal", "linear", "global"]
cells_in_a_block = 12
for att_type in att_types:

    cmd_run = cmd.copy()
    cmd_run.extend([
        "--run_name", f"{att_type}_{cells_in_a_block}",
        "--run_notes", f"{cells_in_a_block}_{att_type}_cells_in_a_block",
        "--att_types"
    ])
    for i in range(cells_in_a_block): cmd_run.append(att_type)

    print(f"Running command:\n{cmd_run}")

    subprocess.run(cmd_run)
