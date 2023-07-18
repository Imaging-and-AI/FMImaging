"""
Python script to run bash scripts in batches
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

# -------------------------------------------------------------

class mri_ddp_base(run_ddp_base):
    
    def __init__(self, project, script_to_run) -> None:
        super().__init__(project, script_to_run)

    def set_up_constants(self, config):
        
        super().set_up_constants(config)

        self.cmd.extend([

        "--num_epochs", "75",
        "--batch_size", "16",

        "--window_size", "8", "8",
        "--patch_size", "2", "2",

        "--global_lr", "0.0001",

        "--clip_grad_norm", "1.0",
        "--weight_decay", "1",

        #"--use_amp", 

        "--iters_to_accumulate", "1",

        "--num_workers", "64",
        "--prefetch_factor", "4",

        "--scheduler_type", "ReduceLROnPlateau",
        #"--scheduler_type", "OneCycleLR",

        "--scheduler.ReduceLROnPlateau.patience", "0",
        "--scheduler.ReduceLROnPlateau.cooldown", "0",
        "--scheduler.ReduceLROnPlateau.factor", "0.85",

        "--scheduler.OneCycleLR.pct_start", "0.2",

        # hrnet
        "--backbone_hrnet.num_resolution_levels", "2",

        # unet
        "--backbone_unet.num_resolution_levels", "2",

        # LLMs
        "--backbone_LLM.num_stages", "3",

        # small unet
        "--backbone_small_unet.channels", "16", "32", "64",   
        "--backbone_small_unet.block_str", "T1L1G1", "T1L1G1", "T1L1G1",

        #"--min_noise_level", "2.0",
        "--max_noise_level", "24.0",
        #"--complex_i",
        #"--residual",
        #"--losses", "mse", "l1",
        #"--loss_weights", "1.0", "1.0",
        "--height", "32", "64",
        "--width", "32", "64",
        "--time", "12",
        "--num_uploaded", "12",
        #"--snr_perturb_prob", "0.25",
        "--snr_perturb", "10.0",
        #"--weighted_loss",
        #"--max_load", "10000",

        #"--with_data_degrading",

        #"--save_samples",

        # "--train_files", "train_3D_3T_retro_cine_2018.h5",  "train_3D_3T_retro_cine_2019.h5", "train_3D_3T_retro_cine_2020.h5", "train_3D_3T_perf_2018.h5","train_3D_3T_perf_2019.h5", "train_3D_3T_perf_2020.h5","train_3D_3T_perf_2021.h5", 
        # "--train_data_types", "2dt", "2dt", "2dt", "2dt", "2dt", "2dt", "2d",

        "--post_hrnet.block_str", "T1L1G1T1L1G1", "T1L1G1T1L1G1",

        "--train_files", "train_3D_3T_retro_cine_2018.h5",  
                        "train_3D_3T_retro_cine_2019.h5", 
                        "train_3D_3T_retro_cine_2020.h5", 
                        "BARTS_RetroCine_3T_2023.h5", 
                        "BARTS_RetroCine_1p5T_2023.h5",
                        #"BWH_Perfusion_3T_2023.h5",
                        #"BWH_Perfusion_3T_2022.h5",
                        "MINNESOTA_UHVC_RetroCine_1p5T_2023.h5", 
                        "MINNESOTA_UHVC_RetroCine_1p5T_2022.h5",

        "--train_data_types", "2dt", "2dt", "2dt", "2dt", "2dt", "2dt", "2dt", "2dt", "3d",

        "--test_files", "train_3D_3T_retro_cine_2020_small_3D_test.h5", 
                        "train_3D_3T_retro_cine_2020_small_2DT_test.h5", 
                        "train_3D_3T_retro_cine_2020_small_2D_test.h5", 
                        "train_3D_3T_retro_cine_2020_500_samples.h5",

        "--test_data_types", "3d", "2dt", "2d", "2dt" 
        ])

        if config.tra_ratio > 0 and config.tra_ratio<=100:
            self.cmd.extend(["--ratio", f"{int(config.tra_ratio)}", f"{int(config.val_ratio)}", f"{int(config.test_ratio)}"])

        self.cmd.extend(["--max_load", f"{int(config.max_load)}"])

        self.cmd.extend(["--lr_pre", f"{config.lr_pre}"])
        self.cmd.extend(["--lr_backbone", f"{config.lr_backbone}"])
        self.cmd.extend(["--lr_post", f"{config.lr_post}"])

        self.cmd.extend(["--model_type", f"{config.model_type}"])

        if config.not_load_pre:
            self.cmd.extend(["--not_load_pre"])
        if config.not_load_backbone:
            self.cmd.extend(["--not_load_backbone"])
        if config.not_load_post:
            self.cmd.extend(["--not_load_post"])

        if config.disable_pre:
            self.cmd.extend(["--disable_pre"])
        if config.disable_backbone:
            self.cmd.extend(["--disable_backbone"])
        if config.disable_post:
            self.cmd.extend(["--disable_post"])

    def set_up_variables(self, config):

        vars = dict()

        vars['optim'] = ['sophia']

        vars['backbone'] = ['hrnet']
        vars['cell_types'] = ["parallel"]
        vars['Q_K_norm'] = [True]
        vars['cosine_atts'] = ["1"]
        vars['att_with_relative_postion_biases'] = ["0"]
        vars['a_types'] = ["conv"]

        vars['larger_mixer_kernels'] = [False]
        vars['mixer_types'] = ["conv"]
        vars['shuffle_in_windows'] = ["0"]
        vars['block_dense_connections'] = ["1"]
        vars['norm_modes'] = ["batch2d"]
        vars['C'] = [32]
        vars['scale_ratio_in_mixers'] = [1.0]

        vars['snr_perturb_prob'] = [0.0]

        vars['block_strs'] = [
                        [
                            #["T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"],
                            #["T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1"],
                            ["T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"],
                            ["T1L1G1", "T1L1G1", "T1L1G1", "T1L1G1"],
                            ["T1T1T1", "T1T1T1", "T1T1T1", "T1T1T1"],
                            ["T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1"],
                         ],

                        [
                            ["T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"],
                            ["T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"],
                            ["T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1"],
                            ["T1L1G1", "T1L1G1", "T1L1G1", "T1L1G1"],
                            ["T1T1T1", "T1T1T1", "T1T1T1", "T1T1T1"]
                         ]
                    ]

        vars['losses'] = [
            [["mse", "perpendicular", "psnr", "l1"], ['1.0', '1.0', '1.0', '1.0', '1.0']],
            [["mse", "perpendicular", "psnr", "l1", "gaussian", "gaussian3D", "ssim"], ['0.1', '1.0', '1.0', '1.0', '20.0', '20.0', "5.0"]],
            #[['perpendicular', 'ssim', 'psnr', 'l1'], ['1.0', '1.0', '1.0', '1.0', '1.0']],
            #[['psnr','l1', 'mse'], ['1.0', '1.0', '1.0', '1.0', '1.0']],
            #[['ssim', 'ssim3D', 'mse', 'l1', 'psnr'], ['0.1', '0.1', '1.0', '1.0', '1.0']], 
            #[['mse', 'l1'], ['1.0', '1.0']], 
            #[['ssim'], ['1.0']],
            #[['ssim', 'mse'], ['0.1', '1.0']], 
        ]

        vars['complex_i'] = [True]
        vars['residual'] = [True]
        vars['weighted_loss'] = [True]

        vars['n_heads'] = [32]

        return vars

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
                    snr_perturb_prob, \
                    n_heads, \
                    c, \
                    scale_ratio_in_mixer, \
                    complex_i,\
                    bs, \
                    weighted_loss, \
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
                                            vars['snr_perturb_prob'],
                                            vars['n_heads'],
                                            vars['C'],
                                            vars['scale_ratio_in_mixers'],
                                            vars['complex_i'],
                                            block_str,
                                            vars['weighted_loss'],
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
                                        complex_i=complex_i,
                                        residual=residual,
                                        weighted_loss=weighted_loss,
                                        snr_perturb_prob=snr_perturb_prob,
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

    def create_cmd_run(self, cmd_run, config, 
                        optim='adamw',
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
                        scale_ratio_in_mixer=2.0,
                        load_path=None,
                        complex_i=True,
                        residual=True,
                        weighted_loss=True,
                        snr_perturb_prob=0,
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
        run_str = f"{moment}_C-{c}-{int(scale_ratio_in_mixer)}_amp-{config.use_amp}"
        #run_str = moment

        if config.run_extra_note is not None:
            run_str += "_" 
            run_str += config.run_extra_note

        if complex_i:
            cmd_run.extend(["--complex_i"])
            run_str += "_complex"

        if residual:
            cmd_run.extend(["--residual"])
            run_str += "_residual"

        if weighted_loss:
            cmd_run.extend(["--weighted_loss"])
            run_str += "_weighted_loss"

        if config.with_data_degrading:
            cmd_run.extend(["--with_data_degrading"])
            run_str += "_with_data_degrading"

        if config.not_add_noise:
            cmd_run.extend(["--not_add_noise"])
            run_str += "_no_noise"

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

        cmd_run.extend(["--min_noise_level", f"{config.min_noise_level}"])

        cmd_run.extend([
            "--run_name", f"{config.project}-{bk.upper()}-{run_str}",
            "--run_notes", f"{config.project}-{bk.upper()}-{run_str}",
            "--snr_perturb_prob", f"{snr_perturb_prob}",
            "--n_head", f"{n_heads}"
        ])

        return cmd_run

    def arg_parser(self):

        parser = super().arg_parser()
        parser.add_argument("--max_load", type=int, default=-1, help="number of max loaded samples into the RAM")

        parser.add_argument("--model_type", type=str, default="STCNNT_MRI", help="STCNNT_MRI or MRI_hrnet or MRI_double_net")

        parser.add_argument("--losses", nargs='+', type=str, default=None, help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D", "psnr", "msssim", "perpendicular", "gaussian", "gaussian3D" ')
        parser.add_argument('--loss_weights', nargs='+', type=float, default=None, help='to balance multiple losses, weights can be supplied')

        parser.add_argument("--min_noise_level", type=float, default=2.0, help='minimal noise level')

        parser.add_argument("--lr_pre", type=float, default=-1, help='learning rate for pre network')
        parser.add_argument("--lr_backbone", type=float, default=-1, help='learning rate for backbone network')
        parser.add_argument("--lr_post", type=float, default=-1, help='learning rate for post network')

        parser.add_argument("--not_load_pre", action="store_true", help='if set, pre module will not be loaded.')
        parser.add_argument("--not_load_backbone", action="store_true", help='if set, backbone module will not be loaded.')
        parser.add_argument("--not_load_post", action="store_true", help='if set, pre module will not be loaded.')

        parser.add_argument("--disable_pre", action="store_true", help='if set, pre module will have require_grad_(False).')
        parser.add_argument("--disable_backbone", action="store_true", help='if set, backbone module will have require_grad_(False).')
        parser.add_argument("--disable_post", action="store_true", help='if set, post module will have require_grad_(False).')

        parser.add_argument("--not_add_noise", action="store_true", help='if set, will not add noise to images.')
        parser.add_argument("--with_data_degrading", action="store_true", help='if set, degrade image before adding noise.')

        return parser

# -------------------------------------------------------------

def main():

    os.system("ulimit -n 65536")

    ddp_run = mri_ddp_base(project="mri", script_to_run='./mri/main_mri.py')
    ddp_run.run()

# -------------------------------------------------------------

if __name__=="__main__":
    main()
