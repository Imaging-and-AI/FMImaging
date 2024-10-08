"""
parser for the qperf
"""

import argparse
import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from setup import none_or_str, str_to_bool

class qperf_parser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser("")

        self.parser.add_argument("--ratio", nargs='+', type=float, default=[100,100,100], help='Ratio (as a percentage) for train/val/test divide of given data. Does allow for using partial dataset')    

        self.parser.add_argument("--n_embd", type=int, default=512, help='embeding dimension')

        self.parser.add_argument("--n_layer", nargs='+', type=int, default=[16, 16], help='number of transformer layers for params model and betex model')

        self.parser.add_argument("--qperf_T", type=int, default=80, help='data length')
        self.parser.add_argument("--foot_to_end", action="store_true", help='if set, use data from foot to end')
        self.parser.add_argument("--use_pos_embedding", action="store_true", help='if set, use positional embedding')
        self.parser.add_argument("--residual_dropout_p", type=float, default=0.1, help='drop out on the mixer residual connection')

        self.parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1", "gauss", "max_ae"], help='Any combination of "mse", "l1", "gauss", "max_ae" ')
        self.parser.add_argument('--loss_weights', nargs='+', type=float, default=[10.0, 10.0, 10.0, 1.0], help='to balance multiple losses, weights can be supplied')
        self.parser.add_argument('--loss_weights_params', nargs='+', type=float, default=[1.0, 0.1, 0.1, 0.1, 5.0], help='weights for Fp, Vp, Visf, PS, Delay')

        self.parser.add_argument('--min_noise_level', nargs='+', type=float, default=[0.001, 0.001], help='min noise level added to aif and myo')
        self.parser.add_argument('--max_noise_level', nargs='+', type=float, default=[0.1, 0.15], help='max noise level added to aif and myo')

        self.parser.add_argument('--add_noise', nargs='+', type=str_to_bool, default=[True, True], help='max noise level added to aif and myo')

        self.parser.add_argument('--num_uploaded', type=int, default=16, help='number of samples uploaded to wandb')

        self.parser.add_argument("--max_samples", nargs='+', type=int, default=[-1, -1, -1], help='max number of samples used in tra/val/test')

        self.parser.add_argument("--disable_LSUV", action="store_true", help='if set, do not perform LSUV initialization.')

        self.parser.add_argument("--qperf_model_type", type=str, default="QPerfModel", choices=['QPerfModel', 'QPerfModel_double_net', 'QPerfBTEXModel'], help='model type')

        self.parser.add_argument("--model_btex_load_path", type=none_or_str, default=None, help='Path to load btex model, pre-trained')