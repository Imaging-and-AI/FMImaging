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
    "--train_files", "train_3D_3T_retro_cine_2020_small.h5", "train_3D_3T_retro_cine_2020_small.h5", "train_3D_3T_retro_cine_2020_small.h5",
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
    "--height", "96",
    "--width", "96",
])

models = ["hrnet", "unet", "LLM", "small_unet"]
optims = ["adamw", "nadam", "sgd"]
scheds = ["ReduceLROnPlateau", "StepLR", "OneCycleLR"]

for model, optim, sched in itertools.product(models, optims, scheds):

    cmd_run = cmd.copy()

    cmd_run.extend([
        "--run_name", f"test_{model}_{optim}_{sched}",
        "--run_notes", f"test_{model}_{optim}_{sched}_base",
        "--backbone", f"{model}",
        f"--backbone_{model}.block_str", f"T1L1G1V1",
        "--optim", f"{optim}",
        "--scheduler_type", f"{sched}"
    ])
    print(f"Running command:\n{cmd_run}")

    subprocess.run(cmd_run)
