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
    "--results_path", "/home/rehmana2/projects/STCNNT_2/results",
    "--train_files", "train_3D_3T_retro_cine_2020_small.h5","train_3D_3T_retro_cine_2020_small.h5", "train_3D_3T_retro_cine_2020_small.h5",
    "--train_data_types", "2d", "2dt", "3d",
    "--test_files", "train_3D_3T_retro_cine_2020_small_2D_test.h5", "train_3D_3T_retro_cine_2020_small_2DT_test.h5", "train_3D_3T_retro_cine_2020_small_3D_test.h5",
    "--test_data_types", "2d", "2dt", "3d"
])

# unchanging commands
cmd.extend([
    "--batch_size", "8",
    "--device", "cuda",
    "--time", "16",
    "--complex_i",
    "--residual",
    "--ratio", "10", "0", "0",
    "--losses", "mse", "l1",
    "--norm_mode", "instance2d",
    "--num_epochs", "2",
    "--height", "64", "96",
    "--width", "64", "96",
    "--att_types", "L1T1L1T1"
])

runs = [(8,False),(1,False)]

for num_patches,shuffle in runs:

    cmd_run = cmd.copy()

    cmd_run.extend([
        "--run_name", f"patches_{num_patches}_shuffle_{shuffle}_final",
        "--run_notes", f"patches_{num_patches}_shuffle_{shuffle}",
        "--twoD_num_patches_cutout", f"{num_patches}"
    ])
    if shuffle: cmd_run.append(f"--twoD_patches_shuffle")
    print(f"Running command:\n{cmd_run}")

    subprocess.run(cmd_run)
