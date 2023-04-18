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
    "--att_types", "T0T0T0",
    "--time", "16",
    "--complex_i",
    "--residual",
    "--ratio", "100", "0", "0",
    "--losses", "mse", "l1"
])

norm_modes = ["instance2d", "instance3d", "batch2d", "batch3d", "layer"]

for norm_mode in norm_modes:

    cmd_run = cmd.copy()

    if norm_mode == "layer":
        cmd_run.extend([
            "--num_epochs", "200",
            "--height", "96",
            "--width", "96"
        ])
    else:
        cmd_run.extend([
            "--num_epochs", "100",
            "--height", "48", "96",
            "--width", "48", "96"
        ])

    cmd_run.extend([
        "--run_name", f"{norm_mode}_norm_3_temporal",
        "--run_notes", f"{norm_mode}_norm_with_3_temporal_cell_per_block",
        "--norm_mode", f"{norm_mode}"
    ])
    print(f"Running command:\n{cmd_run}")

    subprocess.run(cmd_run)
