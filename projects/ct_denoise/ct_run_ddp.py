"""
Python script to run bash scripts in batches
"""

import os
import sys
import itertools
from pathlib import Path

CT_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(CT_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

import time
from datetime import datetime
from temp_utils.ddp_base import run_ddp_base


class ct_ddp_base(run_ddp_base):
    
    def __init__(self, project, script_to_run) -> None:
        super().__init__(project, script_to_run)

    def set_up_constants(self, config):
        
        super().set_up_constants(config)

        self.cmd.extend([

        #"--num_epochs", "50",
        #"--batch_size", "8",

        "--window_size", "8", "8",
        "--patch_size", "2", "2",

        #"--global_lr", "0.0001",

        "--clip_grad_norm", "1.0",
        "--optim.weight_decay", "1",

        "--dropout_p", "0.1",

        #"--use_amp", 
        
        "--activation_func", "prelu",

        "--iters_to_accumulate", "1",

        #"--num_workers", "48",
        "--prefetch_factor", "8",

        # "--scheduler_type", "ReduceLROnPlateau",
        # "--scheduler.patience", "0",
        # "--scheduler.cooldown", "0",
        # "--scheduler.min_lr", "1e-7",
        # "--scheduler.factor", "0.9",

        #"--scheduler_type", "OneCycleLR",

        #"--post_backbone", "STCNNT_mUNET", 
        #"--post_backbone", "STCNNT_HRNET", 

        #"--post_backbone", "mixed_unetr", 
        #"--post_backbone", "hrnet", 

        #"--complex_i",
        #"--residual",
        #"--losses", "mse", "l1",
        #"--loss_weights", "1.0", "1.0",
        # "--ct_height", "32", "64",
        # "--ct_width", "32", "64",
        "--ct_time", "16",
        "--num_uploaded", "32",
        "--samples_per_image", "256",

        "--no_in_channel", "1",
        "--no_out_channel", "1",

        #"--max_load", "10000",

        #"--save_samples",

        #"--seed", "593197",

        #"--post_hrnet.block_str", "T1L1G1", "T1L1G1",

        #"--post_hrnet.separable_conv",
        ])

        if config.tra_ratio > 0 and config.tra_ratio<=100:
            self.cmd.extend(["--ratio", f"{int(config.tra_ratio)}", f"{int(config.val_ratio)}", f"{int(config.test_ratio)}"])

        self.cmd.extend(["--max_load", f"{int(config.max_load)}"])

        self.cmd.extend(["--model_type", f"{config.model_type}"])
        # if 'omnivore' in config.model_type: 
        #     self.cmd.extend(["--omnivore.size", "custom"])
        #     self.cmd.extend(["--omnivore.patch_size", "1", "1", "1"])
        #     self.cmd.extend(["--omnivore.window_size", "14", "7", "7"])
        #     self.cmd.extend(["--omnivore.embed_dim", "24"])
        #     self.cmd.extend(["--omnivore.depths", "2", "2", "6", "2"])
        #     self.cmd.extend(["--omnivore.num_heads", "3","6", "12", "24"])

        # if config.add_salt_pepper:
        #     self.cmd.extend(["--add_salt_pepper"])

        # if config.add_possion:
        #     self.cmd.extend(["--add_possion"])

        # if config.super_resolution:
        #     self.cmd.extend([
        #                 "--super_resolution",

        #                 "--train_files", "train_3D_3T_retro_cine_2018_with_2x_resized.h5",  
        #                                     "train_3D_3T_retro_cine_2019_with_2x_resized.h5", 
        #                                     "train_3D_3T_retro_cine_2020_with_2x_resized.h5", 
        #                                     "BARTS_RetroCine_3T_2023_with_2x_resized.h5", 
        #                                     #"BARTS_RetroCine_1p5T_2023_with_2x_resized.h5",
        #                                     #"MINNESOTA_UHVC_RetroCine_1p5T_2023_with_2x_resized.h5", 
        #                                     #"MINNESOTA_UHVC_RetroCine_1p5T_2022_with_2x_resized.h5",
        #                                     #"VIDA_train_clean_0430_with_2x_resized.h5",

        #                 "--test_files", "train_3D_3T_retro_cine_2020_small_3D_test_with_2x_resized.h5", 
        #                                 "train_3D_3T_retro_cine_2020_small_2DT_test_with_2x_resized.h5", 
        #                                 "train_3D_3T_retro_cine_2020_small_2D_test_with_2x_resized.h5", 
        #                                 "train_3D_3T_retro_cine_2020_500_samples_with_2x_resized.h5",

        #                 "--train_data_types", "2dt", "2dt", "2dt", "2dt", "2dt", "2dt", "2dt", "2dt", "3d",
        #                 "--test_data_types", "3d", "2dt", "2d", "2dt",
        #                 ])

        if config.train_files is not None:
            self.cmd.extend(["--train_files"])
            self.cmd.extend(config.train_files)

        self.cmd.extend(["--eval_train_set", "False"])
        self.cmd.extend(["--eval_val_set", "False"])
        self.cmd.extend(["--eval_test_set", "True"])

    def set_up_variables(self, config):

        vars = dict()

        vars['optim'] = ['sophia']

        #vars['backbone'] = ['hrnet']
        vars['cell_types'] = ["parallel"]
        vars['Q_K_norm'] = [True]
        vars['cosine_atts'] = ["1"]
        vars['att_with_relative_postion_biases'] = ["0"]
        vars['a_types'] = ["conv"]

        vars['larger_mixer_kernels'] = [False]
        vars['mixer_types'] = ["conv"]
        vars['shuffle_in_windows'] = ["0"]
        vars['block_dense_connections'] = ["0"]
        #vars['norm_modes'] = ["instance2d"]
        #vars['C'] = [64]
        vars['scale_ratio_in_mixers'] = [1.0]

        vars['block_strs'] = [
                        [
                            #["C2C2C2", "C2C2C2", "C2C2C2", "C2C2C2"],
                            #["C3C3C3", "C3C3C3", "C3C3C3", "C3C3C3"],
                            #["C2C2C2", "C2C2C2C2C2C2", "C2C2C2", "C2C2C2"],
                            #["C3C3C3", "C3C3C3C3C3C3", "C3C3C3", "C3C3C3"],
                            ["T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"],
                            #["T1T1T1", "T1T1T1T1T1T1", "T1T1T1T1T1T1", "T1T1T1T1T1T1"],
                            ["T1L1G1", "T1L1G1", "T1L1G1", "T1L1G1"],
                            #["T1T1T1", "T1T1T1", "T1T1T1", "T1T1T1"],
                            #["T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1", "T1L1G1T1L1G1T1L1G1"],
                         ],
                    ]

        vars['losses'] = [
            [["mse", "psnr", "l1"], ['1.0', '1.0', '1.0', '1.0', '1.0']],
            #[["mse", "psnr", "l1", "gaussian", "gaussian3D", "ssim"], ['1.0', '1.0', '1.0', '1.0', '20.0', '20.0', "5.0"]],
            #[['ssim', 'psnr', 'l1'], ['1.0', '1.0', '1.0', '1.0', '1.0']],
            #[['psnr','l1', 'mse'], ['1.0', '1.0', '1.0', '1.0', '1.0']],
            #[['ssim', 'ssim3D', 'mse', 'l1', 'psnr'], ['0.1', '0.1', '1.0', '1.0', '1.0']], 
            #[['mse', 'l1'], ['1.0', '1.0']], 
            #[['ssim'], ['1.0']],
            #[['ssim', 'mse'], ['0.1', '1.0']], 
        ]

        vars['complex_i'] = [False]
        vars['residual'] = [True]

        vars['n_heads'] = [32]

        return vars

    def run_vars(self, config, vars):

        cmd_runs = []

        vars['backbone'] = [config.model_backbone]
        vars['norm_modes'] = [config.norm_mode]
        vars['C'] = [config.backbone_C]

        for k, bk in enumerate(vars['backbone']):

                if config.model_block_str is not None:
                    block_str = [config.model_block_str]
                else:
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
                    complex_i,\
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
                                            vars['complex_i'],
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
                                        complex_i=complex_i,
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

    def create_cmd_run(self, cmd_run, config, 
                        optim='sophia',
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
                        load_path=None,
                        complex_i=True,
                        residual=True,
                        n_heads=32,
                        losses=['mse', 'l1'],
                        loss_weights=['1.0', '1.0']
                        ):

        cmd_run = super().create_cmd_run(cmd_run, config, 
                        optim, bk, a_type, cell_type, 
                        norm_mode, block_dense_connection, 
                        c, q_k_norm, cosine_att, att_with_relative_postion_bias, 
                        bs, larger_mixer_kernel, mixer_type, 
                        shuffle_in_window, scale_ratio_in_mixer,
                        load_path)

        curr_time = datetime.now()
        moment = curr_time.strftime('%Y%m%d_%H%M%S_%f')
        run_str = f"{config.model_type}_C-{c}-{int(scale_ratio_in_mixer)}_amp-{config.use_amp}"
        if not config.ut_mode:
            run_str = f"{moment}_" + run_str
        #run_str = moment

        if config.run_extra_note is not None:
            run_str = config.run_extra_note + "_" + f"{bk}" + "_" + f"{'_'.join(bs)}" + "_" + run_str

        if complex_i:
            cmd_run.extend(["--complex_i"])
            run_str += "_complex"

        if residual:
            cmd_run.extend(["--residual"])
            run_str += "_residual"

        cmd_run.extend(["--num_epochs", f"{config.num_epochs}"])
        cmd_run.extend(["--batch_size", f"{config.batch_size}"])

        if config.disable_LSUV:
            cmd_run.extend(["--disable_LSUV"])

        cmd_run.extend(["--post_backbone", f"{config.post_backbone}"])
        cmd_run.extend([f"--post_hrnet.block_str", *config.post_block_str])

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

        cmd_run.extend(["--ct_height"])
        cmd_run.extend([f"{h}" for h in config.ct_height])

        cmd_run.extend(["--ct_width"])
        cmd_run.extend([f"{w}" for w in config.ct_width])

        cmd_run.extend([
            "--run_name", f"{config.project}-{run_str}",
            "--run_notes", f"{config.project}-{run_str}",
            "--n_head", f"{n_heads}"
        ])

        return cmd_run

    def arg_parser(self):

        parser = super().arg_parser()

        parser.add_argument("--cuda_devices", type=str, default="0,1", help='devices for cuda training') # TODO: move to common ddp

        parser.add_argument("--num_epochs", type=int, default=30, help='number of epochs to train for')
        parser.add_argument("--batch_size", type=int, default=8, help='size of each batch')

        parser.add_argument("--train_files", type=str, nargs='+', default=None, help='list of train h5files')
        parser.add_argument("--test_files", type=str, nargs='+', default=None, help='list of test h5files')
        parser.add_argument("--max_load", type=int, default=-1, help="number of max loaded samples into the RAM")
        parser.add_argument("--ct_height", nargs='+', type=int, default=[32, 64], help='heights of the training images')
        parser.add_argument("--ct_width", nargs='+', type=int, default=[32, 64], help='widths of the training images')

        parser.add_argument("--model_type", type=str, default="STCNNT_CT", help="STCNNT_CT or CT_double_net")
        parser.add_argument('--model_backbone', type=str, default="STCNNT_HRNET", help="which backbone model to use, 'STCNNT_HRNET', 'STCNNT_UNET', 'omnivore' ")
        parser.add_argument('--model_block_str', nargs='+', type=str, default=None, help="block string to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")
        parser.add_argument('--post_backbone', type=str, default="STCNNT_HRNET", help="model for post module, 'STCNNT_HRNET', 'STCNNT_mUNET' ")
        parser.add_argument('--post_block_str', nargs='+', type=str, default=['T1L1G1', 'T1L1G1'], help="hrnet MR post network block string, from the low resolution level to high resolution level.")

        parser.add_argument("--losses", nargs='+', type=str, default=None, help='Any combination of "mse", "l1", "sobel", "ssim", "ssim3D", "psnr", "msssim", "perpendicular", "gaussian", "gaussian3D", "spec", "dwt", "charbonnier", "perceptual" ')
        parser.add_argument('--loss_weights', nargs='+', type=float, default=None, help='to balance multiple losses, weights can be supplied')

        parser.add_argument("--disable_LSUV", action="store_true", help='if set, do not perform LSUV init.')

        return parser

# -------------------------------------------------------------------------------------------------

def main():

    os.system("ulimit -n 65536")

    ddp_run = ct_ddp_base(project="ct", script_to_run=str(REPO_DIR)+"/projects/ct_denoise/ct_run.py")
    ddp_run.run()

if __name__=="__main__":
    main()
