"""
Python script to run bash scripts in batches
"""

import itertools
import subprocess

# base command to run a file
cmd = ["python3", "main_mri.py"]

# unchanging paths
cmd.extend([
    "--data_root", "/home/rehmana2/projects/STCNNT_2/mri/",
    "--check_path", "/home/rehmana2/projects/STCNNT_2/checkpoints",
    "--model_path", "/home/rehmana2/projects/STCNNT_2/models",
    "--log_path", "/home/rehmana2/projects/STCNNT_2/logs",
    "--results_path", "/home/rehmana2/projects/STCNNT_2/results"
])

# unchanging commands
cmd.extend([
    "--batch_size", "8",
    "--device", "cuda",
    "--time", "16",
    "--complex_i",
    "--residual",
    "--ratio", "100", "0", "0",
    "--losses", "mse", "l1",
    "--norm_mode", "instance2d",
    "--num_epochs", "100",
    "--height", "64", "96",
    "--width", "64", "96"
])

att_typess = ["T1T1T1T1", "L1L1L1L1", "T1L1T1L1", "L1T1L1T1"]

for att_types in att_typess:

    cmd_run = cmd.copy()

    cmd_run.extend([
        "--run_name", f"{att_types}_new_standard",
        "--run_notes", f"{att_types}_no_bias_same_c_mlp_strided_t",
        "--att_types", f"{att_types}"
    ])
    print(f"Running command:\n{cmd_run}")

    subprocess.run(cmd_run)
