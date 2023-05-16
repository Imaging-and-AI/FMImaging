"""
Python script to run bash scripts in batches
"""

import argparse
import itertools
import subprocess
import os


def arg_parser():
    """
    @args:
        - No args
    @rets:
        - parser (ArgumentParser): the argparse for torchrun of imagenet
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT Cifar10")   
    parser.add_argument("--standalone", action="store_true", help='whether to run in the standalone mode')
    parser.add_argument("--nproc-per-node", type=int, default=1, help="number of processes per node")
    parser.add_argument("--nnodes", type=int, default=1, help="number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="current node rank")
    parser.add_argument("--rdzv_id", type=int, default=100, help="run id")
    parser.add_argument("--rdzv_backend", type=str, default="c10d", help="backend of torchrun")
    parser.add_argument("--rdzv_endpoint", type=str, default="localhost:9001", help="master node endpoint")
    parser.add_argument("--port", type=int, default=9001, help="port for standalone mode")
    
    args = parser.parse_args()
    
    return args

# -------------------------------------------------------------

def create_cmd_run(cmd_run, 
                   bk='hrnet', 
                   a_type='conv', 
                   cell_type='sequential', 
                   norm_mode='batch2d', 
                   block_dense_connection=1, 
                   c=32, 
                   q_k_norm=True, 
                   cosine_att=1, 
                   att_with_relative_postion_bias=1, 
                   bs=['T1G1L1', 'T1G1L1', 'T1G1L1', 'T1G1L1'],
                   larger_mixer_kernel=True,
                   mixer_type="conv",
                   shuffle_in_window=0,
                   scale_ratio_in_mixer=2.0
                ):
    
    run_str = f"{a_type}-{cell_type}-{norm_mode}-C-{c}-mixer-{mixer_type}-{larger_mixer_kernel}-{scale_ratio_in_mixer}-{int(scale_ratio_in_mixer)}-block_dense-{block_dense_connection}-qknorm-{q_k_norm}-cosine_att-{cosine_att}-shuffle_in_window-{shuffle_in_window}-att_with_relative_postion_bias-{att_with_relative_postion_bias}-block_str-{'_'.join(bs)}"
                                        
    cmd_run.extend([
        "--run_name", f"cifar-{bk}-{run_str}",
        "--run_notes", f"cifar-{bk}-{run_str}",
        "--backbone", f"{bk}",
        "--a_type", f"{a_type}",
        "--cell_type", f"{cell_type}",
        "--cosine_att", f"{cosine_att}",
        "--att_with_relative_postion_bias", f"{att_with_relative_postion_bias}",
        "--backbone_hrnet.C", f"{c}",
        "--backbone_unet.C", f"{c}",
        "--backbone_LLM.C", f"{c}",
        "--block_dense_connection", f"{block_dense_connection}",
        "--norm_mode", f"{norm_mode}",
        "--mixer_type", f"{mixer_type}",
        "--shuffle_in_window", f"{shuffle_in_window}",
        "--scale_ratio_in_mixer", f"{scale_ratio_in_mixer}"
    ])
    
    if larger_mixer_kernel:
        cmd_run.extend(["--mixer_kernel_size", "5", "--mixer_padding", "2", "--mixer_stride", "1"])
    else:
        cmd_run.extend(["--mixer_kernel_size", "3", "--mixer_padding", "1", "--mixer_stride", "1"])

    if q_k_norm:
        cmd_run.extend(["--normalize_Q_K"])
        
    cmd_run.extend([f"--backbone_{bk}.block_str", *bs])
        
    print(f"Running command:\n{' '.join(cmd_run)}")

    return cmd_run

# -------------------------------------------------------------

def main():
    
    config = arg_parser()

    # -------------------------------------------------------------

    # base command to run a file
    cmd = ["torchrun"]

    cmd.extend(["--nproc_per_node", f"{config.nproc_per_node}"])

    if config.standalone:
        cmd.extend(["--standalone"])
    else:
        cmd.extend(["--nnodes", f"{config.nnodes}", 
                    "--node_rank", f"{config.node_rank}", 
                    "--rdzv_id", f"{config.rdzv_id}", 
                    "--rdzv_backend", f"{config.rdzv_backend}", 
                    "--rdzv_endpoint", f"{config.rdzv_endpoint}"])

    cmd.extend(["cifar10/main_cifar.py"])

    # -------------------------------------------------------------    
    if "FMIMAGING_PROJECT_BASE" in os.environ:
        project_base_dir = os.environ['FMIMAGING_PROJECT_BASE']
    else:
        project_base_dir = '/export/Lab-Xue/projects'

    # unchanging paths
    cmd.extend([
        "--data_set", "imagenet",
        "--data_root", os.path.join(project_base_dir, "imagenet", "data"),
        "--check_path", os.path.join(project_base_dir, "imagenet", "checkpoints"),
        "--model_path", os.path.join(project_base_dir, "imagenet", "models"),
        "--log_path", os.path.join(project_base_dir, "imagenet", "logs"),
        "--results_path", os.path.join(project_base_dir, "imagenet", "results")
    ])

    # -------------------------------------------------------------

    # unchanging commands
    cmd.extend([
        "--summary_depth", "6",
        "--save_cycle", "200",
        
        "--num_epochs", "150",
        "--batch_size", "32",
        "--device", "cuda",
        "--window_size", "32", "32",
        "--patch_size", "8", "8",
        "--n_head", "32",
        "--global_lr", "1e-4",
        "--clip_grad_norm", "1.0",
        "--weight_decay", "1.0",
        "--use_amp", 
        "--ddp", 
        "--iters_to_accumulate", "1",
        "--project", "imagenet",
        "--num_workers", "16",
        
        "--scheduler_type", "OneCycleLR",
        
        "--scheduler.ReduceLROnPlateau.patience", "2",
        "--scheduler.ReduceLROnPlateau.cooldown", "2",
        "--scheduler.ReduceLROnPlateau.min_lr", "1e-7",
        "--scheduler.ReduceLROnPlateau.factor", "0.9",
            
        "--scheduler.StepLR.step_size", "5",
        "--scheduler.StepLR.gamma", "0.8",
        
        # hrnet
        "--backbone_hrnet.num_resolution_levels", "3",
        "--backbone_hrnet.use_interpolation", "1",
        
        # unet            
        "--backbone_unet.num_resolution_levels", "3",
        "--backbone_unet.use_unet_attention", "1",
        "--backbone_unet.use_interpolation", "1",
        "--backbone_unet.with_conv", "1",
        
        # LLMs
        "--backbone_LLM.num_stages", "3",
        "--backbone_LLM.add_skip_connections", "1",
                        
        # small unet
        "--backbone_small_unet.channels", "16", "32", "64",   
        "--backbone_small_unet.block_str", "T1L1G1", "T1L1G1", "T1L1G1"    
    ])
    
    # test backbones
    backbone = ['hrnet']
    cell_types = ["sequential", "parallel"]
    Q_K_norm = [True]
    cosine_atts = ["1"]
    att_with_relative_postion_biases = ["1"]
    a_types = ["conv"]

    larger_mixer_kernels = [False]
    mixer_types = ["conv"]
    shuffle_in_windows = ["0"]
    block_dense_connections = ["1"]
    norm_modes = ["batch2d"]
    C = [64]
    scale_ratio_in_mixers = [1.0]

    block_strs = [
                    [["T1L1G1", "T1L1G1", "T1L1G1"], ["T1T1T1", "T1T1T1", "T1T1T1"] ]
                ]

    # -------------------------------------------------------------

    for k, bk in enumerate(backbone):    
            block_str = block_strs[k]
            
            for bs in block_str:
                for a_type, cell_type in itertools.product(a_types, cell_types):
                    for q_k_norm in Q_K_norm:
                        for cosine_att in cosine_atts:
                            for att_with_relative_postion_bias in att_with_relative_postion_biases:
                                for c in C:
                                    for block_dense_connection in block_dense_connections:
                                        for norm_mode in norm_modes:
                                            for larger_mixer_kernel in larger_mixer_kernels:
                                                for shuffle_in_window in shuffle_in_windows:
                                                    for mixer_type in mixer_types:
                                                        for scale_ratio_in_mixer in scale_ratio_in_mixers:
                                                            
                                                            # -------------------------------------------------------------
                                                            cmd_run = create_cmd_run(cmd.copy(), 
                                                                            bk=bk, 
                                                                            a_type=a_type, 
                                                                            cell_type=cell_type,
                                                                            norm_mode=norm_mode, 
                                                                            block_dense_connection=block_dense_connection,
                                                                            c=c,
                                                                            q_k_norm=q_k_norm, 
                                                                            cosine_att=cosine_att, 
                                                                            att_with_relative_postion_bias=att_with_relative_postion_bias, 
                                                                            bs=bs,
                                                                            larger_mixer_kernel=larger_mixer_kernel,
                                                                            mixer_type=mixer_type,
                                                                            shuffle_in_window=shuffle_in_window,
                                                                            scale_ratio_in_mixer=scale_ratio_in_mixer)
                                                            # -------------------------------------------------------------
                                                            
                                                            print(cmd_run)
                                                            subprocess.run(cmd_run)

                                                            
# -------------------------------------------------------------

if __name__=="__main__":
    main()