"""
Python script to run microscopy bash scripts in batches
"""

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from trainer_base import *

import time
from datetime import datetime

# -------------------------------------------------------------------------------------------------

class micro_ddp_base(run_ddp_base):
    
    def __init__(self, project, script_to_run) -> None:
        super().__init__(project, script_to_run)

    def set_up_constants(self, config):
        
        super().set_up_constants(config)

        self.cmd.extend([

        "--num_epochs", "75",
        "--batch_size", "16",

        "--samples_per_image", "32",

        "--window_size", "8", "8",
        "--patch_size", "2", "2",

        "--global_lr", "0.0001",

        "--clip_grad_norm", "1.0",
        "--weight_decay", "1",
        "--iters_to_accumulate", "1",

        "--num_workers", "8",
        "--prefetch_factor", "4",

        "--scheduler_type", "ReduceLROnPlateau",
        #"--scheduler_type", "OneCycleLR",

        "--scheduler.ReduceLROnPlateau.patience", "0",
        "--scheduler.ReduceLROnPlateau.cooldown", "0",
        "--scheduler.ReduceLROnPlateau.factor", "0.9",

        "--scheduler.OneCycleLR.pct_start", "0.2",

        # hrnet
        "--backbone_hrnet.num_resolution_levels", "2",

        # unet
        "--backbone_unet.num_resolution_levels", "3",
        "--backbone_unet.C", "16",

        # LLMs
        # "--backbone_LLM.num_stages", "3",

        # small unet
        "--backbone_small_unet.channels", "16", "32", "64",   
        "--backbone_small_unet.block_str", "T1L1G1", "T1L1G1", "T1L1G1",

        #"--losses", "mse", "l1",
        #"--loss_weights", "1.0", "1.0",
        "--height", "128",
        "--width", "128",
        "--time", "12",
        "--C_in", "1",
        "--C_out", "1",
        "--num_uploaded", "12",

        "--train_files", "Base_Actin_train.h5",
        "--test_files", "Base_Actin_test.h5",
        
        "--ratio", "100", "10", "0"
        ])

        self.cmd.extend(["--max_load", f"{int(config.max_load)}"])

    def set_up_variables(self, config):

        vars = dict()

        vars['optim'] = ['sophia']

        vars['backbone'] = ['unet']
        vars['cell_types'] = ["sequential"]
        vars['Q_K_norm'] = [True]
        vars['cosine_atts'] = ["1"]
        vars['att_with_relative_postion_biases'] = ["0"]
        vars['a_types'] = ["conv"]

        vars['larger_mixer_kernels'] = [False]
        vars['mixer_types'] = ["conv"]
        vars['shuffle_in_windows'] = ["0"]
        vars['block_dense_connections'] = ["1"]
        vars['norm_modes'] = ["instance2d"]
        vars['C'] = [32]
        vars['scale_ratio_in_mixers'] = [1.0]

        vars['block_strs'] = [
                        [
                            ["T1T1T1"]
                         ]
                    ]

        vars['losses'] = [
            [['ssim'], ['1.0']],
        ]

        vars['residual'] = [True]

        vars['n_heads'] = [8]

        return vars
    
    def create_cmd_run(self, cmd_run, config, 
                        optim='adamw',
                        bk='hrnet', 
                        a_type='conv', 
                        cell_type='sequential', 
                        norm_mode='instance2d', 
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
                        load_path=None,
                        residual=True,
                        n_heads=32,
                        losses=['mse', 'l1'],
                        loss_weights=['1.0', '1.0']
                        ):

        if c < n_heads:
             return None

        cmd_run = super().create_cmd_run(cmd_run, config, 
                        optim, bk, a_type, cell_type, 
                        norm_mode, block_dense_connection, 
                        c, q_k_norm, cosine_att, att_with_relative_postion_bias, 
                        bs, larger_mixer_kernel, mixer_type, 
                        shuffle_in_window, scale_ratio_in_mixer,
                        load_path)

        curr_time = datetime.now()
        moment = curr_time.strftime('%Y%m%d_%H%M%S_%f')
        run_str = f"{config.model_type}_{moment}_C-{c}_amp-{config.use_amp}"
        #run_str = moment

        if config.run_extra_note is not None:
            run_str += "_" 
            run_str += config.run_extra_note

        if residual:
            cmd_run.extend(["--residual"])
            run_str += "_residual"

        if config.disable_LSUV:
            cmd_run.extend(["--disable_LSUV"])

        run_str += f"-{'_'.join(bs)}"

        cmd_run.extend(["--losses"])
        if config.losses is not None:
            cmd_run.extend(config.losses)
        else:
            cmd_run.extend(losses)

        cmd_run.extend(["--loss_weights"])
        if config.loss_weights is not None:
            cmd_run.extend([f"{lw}" for lw in config.loss_weights])
        else:
            cmd_run.extend(loss_weights)

        ind = cmd_run.index("--run_name")
        cmd_run.pop(ind)
        cmd_run.pop(ind)

        ind = cmd_run.index("--run_notes")
        cmd_run.pop(ind)
        cmd_run.pop(ind)
        
        cmd_run.extend([
            "--run_name", f"{config.project}-{run_str}",
            "--run_notes", f"{config.project}-{run_str}",
            "--n_head", f"{n_heads}"
        ])

        return cmd_run
    
    def run_vars(self, config, vars):

        cmd_runs = []

        for k, bk in enumerate(vars['backbone']):
            block_str = vars['block_strs'][k]

            for optim, \
                mixer_type, \
                shuffle_in_window, \
                larger_mixer_kernel, \
                norm_mode, \
                block_dense_connection, \
                att_with_relative_postion_bias, \
                cosine_att, \
                q_k_norm, \
                a_type, \
                cell_type,\
                residual, \
                n_heads, \
                c, \
                scale_ratio_in_mixer, \
                bs, \
                loss_and_weights \
                    in itertools.product( 
                                        vars['optim'],
                                        vars['mixer_types'], 
                                        vars['shuffle_in_windows'], 
                                        vars['larger_mixer_kernels'],
                                        vars['norm_modes'],
                                        vars['block_dense_connections'],
                                        vars['att_with_relative_postion_biases'],
                                        vars['cosine_atts'],
                                        vars['Q_K_norm'],
                                        vars['a_types'], 
                                        vars['cell_types'],
                                        vars['residual'],
                                        vars['n_heads'],
                                        vars['C'],
                                        vars['scale_ratio_in_mixers'],
                                        block_str,
                                        vars['losses']
                                        ):

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
                                    load_path=config.load_path,
                                    residual=residual,
                                    n_heads=n_heads,
                                    losses=loss_and_weights[0],
                                    loss_weights=loss_and_weights[1]
                                    )

                    if cmd_run:
                        print("---" * 20)
                        print(cmd_run)
                        print("---" * 20)
                        cmd_runs.append(cmd_run)
        return cmd_runs

    def arg_parser(self):

        parser = super().arg_parser()

        parser.add_argument("--max_load", type=int, default=-1, help='number of samples to load into the disk, if <0, samples will be read from the disk while training')
        parser.add_argument("--model_type", type=str, default="STCNNT_Micro", help="STCNNT_Micro only for now")

        parser.add_argument("--losses", nargs='+', type=str, default=["ssim"], help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D", "psnr", "msssim", "gaussian", "gaussian3D" ')
        parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0], help='to balance multiple losses, weights can be supplied')
        parser.add_argument("--disable_LSUV", action="store_true", help='if set, do not perform LSUV init.')

        return parser

# -------------------------------------------------------------

def main():

    os.system("ulimit -n 65536")

    ddp_run = micro_ddp_base(project="microscopy", script_to_run='./microscopy/main_micro.py')
    ddp_run.run()

# -------------------------------------------------------------

if __name__=="__main__":
    main()
