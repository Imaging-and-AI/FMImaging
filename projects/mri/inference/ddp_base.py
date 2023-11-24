"""
Base class to support run ddp experiments for projects
"""

import sys
import argparse
import itertools
import subprocess
import os
import shutil
import pickle
import copy
import itertools
import time
from colorama import Fore, Back, Style

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from setup import none_or_str

class run_ddp_base(object):
    
    def __init__(self, project, script_to_run) -> None:
        super().__init__()
        self.project = project
        self.script_to_run = script_to_run
        self.cmd = []
        
    def set_up_torchrun(self, config):
        self.cmd = ["torchrun"]

        self.cmd.extend(["--nproc_per_node", f"{config.nproc_per_node}", "--max_restarts", "1", "--master_port", f"{config.master_port}"])

        if config.standalone:
            self.cmd.extend(["--standalone"])
        else:
            self.cmd.extend(["--nnodes", config.nnodes, 
                        "--node_rank", f"{config.node_rank}", 
                        "--rdzv_id", f"{config.rdzv_id}", 
                        "--rdzv_backend", f"{config.rdzv_backend}", 
                        "--rdzv_endpoint", f"{config.rdzv_endpoint}"])

        self.cmd.extend([self.script_to_run])

    
    def set_up_run_path(self, config):
        if "FMIMAGING_PROJECT_BASE" in os.environ:
            project_base_dir = os.environ['FMIMAGING_PROJECT_BASE']
        else:
            project_base_dir = '/export/Lab-Xue/projects'

        # unchanging paths

        ckp_path = os.path.join(project_base_dir, config.project, "checkpoints")
        log_path = os.path.join(project_base_dir, config.project, "logs")

        if config.load_path is None:
            if config.clean_checkpoints:
                print(f"--> clean {ckp_path}")
                shutil.rmtree(ckp_path, ignore_errors=True)
                os.mkdir(ckp_path)
                
                print(f"--> clean {log_path}")
                shutil.rmtree(log_path, ignore_errors=True)
                os.mkdir(log_path)

        data_root = config.data_root if config.data_root is not None else os.path.join(project_base_dir, config.project, "data")
        log_root = config.log_root if config.log_root is not None else os.path.join(project_base_dir, config.project, "logs")

        self.cmd.extend([
            "--data_dir", data_root,
            "--log_dir", log_root
        ])

        if config.with_timer:
            self.cmd.extend(["--with_timer"])

    def create_cmd_run(self, cmd_run, config, 
                        optim='adamw',
                        bk='STCNNT_HRNET', 
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
                        scale_ratio_in_mixer=2.0,
                        load_path=None
                        ):

        run_str = f"{a_type}-{cell_type}-{norm_mode}-{optim}-C-{c}-MIXER-{mixer_type}-{int(scale_ratio_in_mixer)}-{'_'.join(bs)}"

        if config.run_extra_note is not None:
            run_str += "_" 
            run_str += config.run_extra_note

        cmd_run.extend([
            "--run_name", f"{config.project}-{bk.upper()}-{run_str}",
            "--run_notes", f"{config.project}-{bk.upper()}-{run_str}",
            "--optim_type", f"{optim}",
            "--backbone_model", f"{bk}",
            "--a_type", f"{a_type}",
            "--cell_type", f"{cell_type}",
            "--cosine_att", f"{cosine_att}",
            "--att_with_relative_postion_bias", f"{att_with_relative_postion_bias}",
            "--block_dense_connection", f"{block_dense_connection}",
            "--norm_mode", f"{norm_mode}",
            "--mixer_type", f"{mixer_type}",
            "--shuffle_in_window", f"{shuffle_in_window}",
            "--scale_ratio_in_mixer", f"{scale_ratio_in_mixer}",
            "--stride_s", f"{config.stride_s}",
            "--stride_t", f"{config.stride_t}",
            "--wandb_dir", f"{config.wandb_dir}",
            "--override"
        ])

        if bk=='STCNNT_HRNET': 
            cmd_run.extend(["--backbone_hrnet.C", f"{c}", 
                            "--backbone_hrnet.use_interpolation", "1",
                            "--backbone_hrnet.num_resolution_levels", "2"
                        ])
        
 
        if bk=='STCNNT_UNET': 
            cmd_run.extend(["--backbone_unet.C", f"{c}",
                            "--backbone_unet.use_unet_attention", "1",
                            "--backbone_unet.use_interpolation", "1",
                            "--backbone_unet.with_conv", "1",
                            "--backbone_unet.num_resolution_levels", "2"
            ])
                   
            
        if bk=='STCNNT_mUNET': 
            cmd_run.extend(["--backbone_mixed_unetr.C", f"{c}",
                            "--backbone_mixed_unetr.num_resolution_levels", "2", 
                            "--backbone_mixed_unetr.use_unet_attention", "1", 
                            "--backbone_mixed_unetr.use_interpolation", "1", 
                            "--backbone_mixed_unetr.with_conv", "0", 
                            "--backbone_mixed_unetr.min_T", "16", 
                            "--backbone_mixed_unetr.encoder_on_skip_connection", "1", 
                            "--backbone_mixed_unetr.encoder_on_input", "1", 
                            "--backbone_mixed_unetr.transformer_for_upsampling", "0", 
                            "--backbone_mixed_unetr.n_heads", "32", "32", "32", 
                            "--backbone_mixed_unetr.use_conv_3d", "1",
                            "--backbone_mixed_unetr.use_window_partition", "0"
                        ])

        if larger_mixer_kernel:
            cmd_run.extend(["--mixer_kernel_size", "5", "--mixer_padding", "2", "--mixer_stride", "1"])
        else:
            cmd_run.extend(["--mixer_kernel_size", "3", "--mixer_padding", "1", "--mixer_stride", "1"])

        if q_k_norm:
            cmd_run.extend(["--normalize_Q_K"])

        if bk == "STCNNT_HRNET":
            cmd_run.extend([f"--backbone_hrnet.block_str", *bs])
        if bk == "STCNNT_UNET":
            cmd_run.extend([f"--backbone_unet.block_str", *bs])
        if bk == "STCNNT_mUNET":
            cmd_run.extend([f"--backbone_mixed_unetr.block_str", *bs])

        if load_path is not None:
            if not config.not_load_pre:
                cmd_run.extend(["--pre_model_load_path", f"{load_path}_pre.pth"])
            if not config.not_load_backbone: 
                cmd_run.extend(["--backbone_model_load_path", f"{load_path}_backbone.pth"])
            if not config.not_load_post:
                cmd_run.extend(["--post_model_load_path", f"{load_path}_post.pth"])

        if config.post_model_of_1st_net is not None:
            cmd_run.extend(["--post_model_of_1st_net", f"{config.post_model_of_1st_net}"])

        if config.freeze_pre:
            cmd_run.extend(["--freeze_pre", "True"])
        if config.freeze_backbone:
            cmd_run.extend(["--freeze_backbone", "True"])
        if config.freeze_post:
            cmd_run.extend(["--freeze_post", "True"])

        cmd_run.extend(["--optim.global_lr", f"{config.global_lr}"])
        cmd_run.extend(["--optim.lr", f"{config.lr_pre}", f"{config.lr_backbone}", f"{config.lr_post}"])

        if config.scheduler_type=='ReduceLROnPlateau': 
            cmd_run.extend(["--scheduler_type", "ReduceLROnPlateau",
                            "--scheduler.patience", "0",
                            "--scheduler.cooldown", "0",
                            "--scheduler.factor", f"{config.scheduler_factor}",
                            "--scheduler.min_lr", "1e-7"
                        ])
                       
        if config.scheduler_type=='OneCycleLR': 
            cmd_run.extend(["--scheduler_type", "OneCycleLR",
                            "--scheduler.pct_start", "0.2"
                        ])

        if config.seed is not None:
            cmd_run.extend(["--seed", f"{config.seed}"])

        if config.num_workers is not None:
            cmd_run.extend(["--num_workers", f"{config.num_workers}"])

        if config.continued_training:
            cmd_run.extend(["--continued_training", "True"])

        if config.only_eval:
            cmd_run.extend(["--train_model", "False"])

        if config.use_amp:
            cmd_run.extend(["--use_amp"])

        if config.separable_conv:
            cmd_run.extend(["--separable_conv"])

        if config.save_samples:
            cmd_run.extend(["--save_train_samples", "False", "--save_val_samples", "False", "--save_test_samples", "True"])
        else:
            cmd_run.extend(["--save_train_samples", "False", "--save_val_samples", "False", "--save_test_samples", "False"])

        #cmd_run.extend(["--window_sizing_method", "keep_num_window"])

        print(f"Running command:\n{' '.join(cmd_run)}")

        return cmd_run

    def set_up_constants(self, config):
        self.cmd.extend([
        "--summary_depth", "6",
        "--device", "cuda",
        "--ddp", 
        "--project", config.project
        ])

    def set_up_variables(self, config):

        vars = dict()

        vars['optim'] = ['sophia', 'adamw']

        vars['backbone'] = ['STCNNT_HRNET']
        vars['cell_types'] = ["sequential"]
        vars['Q_K_norm'] = [True]
        vars['cosine_atts'] = ["1"]
        vars['att_with_relative_postion_biases'] = ["1"]
        vars['a_types'] = ["conv"]

        vars['larger_mixer_kernels'] = [False]
        vars['mixer_types'] = ["conv"]
        vars['shuffle_in_windows'] = ["0"]
        vars['block_dense_connections'] = ["0"]
        #vars['norm_modes'] = ["batch2d"]
        vars['C'] = [64]
        vars['scale_ratio_in_mixers'] = [4.0]

        vars['block_strs'] = [
                        [["T1L1G1", "T1L1G1", "T1L1G1"], ["T1T1T1", "T1T1T1", "T1T1T1"] ]
                    ]

        return vars

    def run_vars(self, config, vars):

        cmd_runs = []

        vars['norm_modes'] = [config.norm_mode]

        for k, bk in enumerate(vars['backbone']):
                block_str = vars['block_strs'][k]

                for optim in vars['optim']:
                    for bs in block_str:
                        for a_type, cell_type in itertools.product(vars['a_types'], vars['cell_types']):
                            for q_k_norm in vars['Q_K_norm']:
                                for cosine_att in vars['cosine_atts']:  
                                    for att_with_relative_postion_bias in vars['att_with_relative_postion_biases']:
                                        for c in vars['C']:
                                            for block_dense_connection in vars['block_dense_connections']:
                                                for norm_mode in vars['norm_modes']:
                                                    for larger_mixer_kernel in vars['larger_mixer_kernels']:
                                                        for shuffle_in_window in vars['shuffle_in_windows']:
                                                            for mixer_type in vars['mixer_types']:
                                                                for scale_ratio_in_mixer in vars['scale_ratio_in_mixers']:

                                                                    # -------------------------------------------------------------
                                                                    cmd_run = self.create_cmd_run(cmd_run=self.cmd.copy(), 
                                                                                    config=config,
                                                                                    optim=optim,
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
                                                                                    scale_ratio_in_mixer=scale_ratio_in_mixer,
                                                                                    load_path=config.load_path)

                                                                    if cmd_run:
                                                                        print("---" * 20)
                                                                        print(cmd_run)
                                                                        print("---" * 20)
                                                                        #subprocess.run(cmd_run)
                                                                        cmd_runs.append(cmd_run)
        return cmd_runs

    def arg_parser(self):
        """
        @args:
            - No args
        @rets:
            - parser (ArgumentParser): the argparse for torchrun of mri
        """
        parser = argparse.ArgumentParser(prog=self.project)

        parser.add_argument("--project", type=str, default="mri_main", help="project name")

        parser.add_argument("--data_root", type=str, default=None, help="data folder; if None, use the project folder")
        parser.add_argument("--log_root", type=str, default=None, help="log folder; if None, use the project folder")
        parser.add_argument("--wandb_dir", type=str, default='/export/Lab-Xue/projects/mri/wandb', help='directory for saving wandb')

        parser.add_argument("--standalone", action="store_true", help='whether to run in the standalone mode')
        parser.add_argument("--nproc_per_node", type=int, default=4, help="number of processes per node")
        parser.add_argument("--nnodes", type=str, default="1", help="number of nodes")
        parser.add_argument("--node_rank", type=int, default=0, help="current node rank")
        parser.add_argument("--master_port", type=int, default=9050, help="torchrun port")
        parser.add_argument("--rdzv_id", type=int, default=100, help="run id")
        parser.add_argument("--rdzv_backend", type=str, default="c10d", help="backend of torchrun")
        parser.add_argument("--rdzv_endpoint", type=str, default="172.16.0.4", help="master node endpoint")

        parser.add_argument("--clean_checkpoints", action="store_true", help='whether to delete previous check point files')
        parser.add_argument("--with_timer", action="store_true", help='whether to train with timing')

        parser.add_argument("--load_path", type=str, default=None, help="check point file to load if provided")
        parser.add_argument("--post_model_of_1st_net", type=str, default=None, help="for double net, load post of the 1st model")
        parser.add_argument("--not_load_pre", action="store_true", help='if set, pre module will not be loaded.')
        parser.add_argument("--not_load_backbone", action="store_true", help='if set, backbone module will not be loaded.')
        parser.add_argument("--not_load_post", action="store_true", help='if set, pre module will not be loaded.')

        parser.add_argument("--freeze_pre", action="store_true", help='if set, pre module will have require_grad_(False).')
        parser.add_argument("--freeze_backbone", action="store_true", help='if set, backbone module will have require_grad_(False).')
        parser.add_argument("--freeze_post", action="store_true", help='if set, post module will have require_grad_(False).')

        parser.add_argument("--global_lr", type=float, default=0.0001, help='global learning rate')
        parser.add_argument("--lr_pre", type=float, default=1e-4, help='learning rate for pre network')
        parser.add_argument("--lr_backbone", type=float, default=1e-4, help='learning rate for backbone network')
        parser.add_argument("--lr_post", type=float, default=1e-4, help='learning rate for post network')

        parser.add_argument("--tra_ratio", type=float, default=90, help="percentage of training data used")
        parser.add_argument("--val_ratio", type=float, default=10, help="percentage of validation data used")
        parser.add_argument("--test_ratio", type=float, default=100, help="percentage of test data used")

        parser.add_argument("--stride_s", type=int, default=1, help='stride for spatial attention, q and k (equal x and y)') 
        parser.add_argument("--stride_t", type=int, default=2, help='stride for temporal attention, q and k (equal x and y)') 
        parser.add_argument("--separable_conv", action="store_true", help='if set, use separable conv')

        parser.add_argument("--scheduler_type", type=none_or_str, default="OneCycleLR", choices=["ReduceLROnPlateau", "StepLR", "OneCycleLR", None], help='Which LR scheduler to use')

        parser.add_argument("--seed", type=int, default=None, help='seed for randomization')

        parser.add_argument("--num_workers", type=int, default=None, help='number of total workers')

        parser.add_argument("--run_extra_note", type=str, default=None, help="extra notes for the runs")

        parser.add_argument("--run_list", type=int, nargs='+', default=[-1], help="run list")

        parser.add_argument("--continued_training", action="store_true", help='if set, it means a continued training loaded from checkpoints (optim and scheduler will be loaded); if not set, it mean a new stage of training.')
        parser.add_argument("--only_eval", action="store_true", help="If True, only perform evaluation")

        parser.add_argument("--use_amp", action="store_true", help='if set, use mixed precision training.')

        parser.add_argument("--save_samples", action="store_true", help='if set, save test samples.')

        parser.add_argument("--ut_mode", action="store_true", help='if set, this run is for unit test.')

        parser.add_argument('--scheduler_factor', type=float, default=0.9, help="LR reduction factor, multiplication")

        parser.add_argument("--norm_mode", type=str, default="instance2d", help='normalization mode, batch2d, instance2d, batch3d, instance3d')

        return parser

    def get_valid_runs(self, config):
        #config.project = self.project
        self.set_up_torchrun(config)
        self.set_up_run_path(config)
        self.set_up_constants(config)
        vars = self.set_up_variables(config)
        cmd_runs = self.run_vars(config, vars)

        valid_cmd_runs = cmd_runs
        return valid_cmd_runs

    def run(self):
        parser = self.arg_parser()
        config = parser.parse_args()
        valid_cmd_runs = self.get_valid_runs(config)

        run_lists = config.run_list
        print(f"Running run_lists: {Fore.GREEN}{run_lists}{Style.RESET_ALL}")

        if run_lists[0] < 0:
            run_lists = range(len(valid_cmd_runs))

        print("===" * 40)
        print(f"{Fore.WHITE}{Back.RED}run_lists is {run_lists}{Style.RESET_ALL}")

        for run_ind in run_lists:
            cmd_run = valid_cmd_runs[run_ind]
            print("\n\n")
            print("===" * 20)
            print(f"{Fore.YELLOW}Run - {run_ind} ...{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{cmd_run}{Style.RESET_ALL}")
            print("--" * 20)
            print(f"Running command:\n{Fore.WHITE}{Back.BLUE}{' '.join(cmd_run)}{Style.RESET_ALL}")
            time.sleep(3)
            subprocess.run(cmd_run)
            print("===" * 20)

            # run_completed = []
            # if os.path.isfile(self.run_record):
            #     with open(self.run_record, 'rb') as f:
            #         run_completed = pickle.load(f)

            # run_completed.append(cmd_run)
            # with open(self.run_record, 'wb') as f:
                #     pickle.dump(run_completed, f)

# -------------------------------------------------------------

def main():
    pass

# -------------------------------------------------------------

if __name__=="__main__":
    main()