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

from setup import none_or_str

class qperf_parser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser("")

        #self.parser.add_argument("--ratio", nargs='+', type=float, default=[90,10,100], help='Ratio (as a percentage) for train/val/test divide of given data. Does allow for using partial dataset')    

        self.parser.add_argument("--n_embd", type=int, default=512, help='embeding dimension')
        self.parser.add_argument("--n_layer", type=int, default=16, help='number of transformer layers')
        self.parser.add_argument("--qperf_T", type=int, default=80, help='data length')
        self.parser.add_argument("--use_pos_embedding", action="store_true", help='if set, use positional embedding')
        self.parser.add_argument("--residual_dropout_p", type=float, default=0.1, help='drop out on the mixer residual connection')
        self.parser.add_argument("--losses", nargs='+', type=str, default=["mse", "l1"], help='Any combination of "mse", "l1" ')
        self.parser.add_argument('--loss_weights', nargs='+', type=float, default=[1.0, 1.0], help='to balance multiple losses, weights can be supplied')
        self.parser.add_argument('--loss_weights_params', nargs='+', type=float, default=[2.0, 1.0, 1.0, 1.0, 2.0], help='weights for Fp, Vp, Visf, PS, Delay')