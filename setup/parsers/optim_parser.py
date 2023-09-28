import argparse
import sys
from pathlib import Path

Setup_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Setup_DIR))

from config_utils import *

class optim_parser(object):
    """
    Parser that contains args for optimizer
    @args:
        no args
    @rets:
        no rets; self.parser contains args
    """

    def __init__(self, optim_type):
        self.parser = argparse.ArgumentParser("")
        self.parser.add_argument("--optim.lr", type=float, default=1e-5, help='Step size for the optimizer')
        self.parser.add_argument("--optim.weight_decay", type=float, default=0.0, help='Weight decay regularization')
        self.parser.add_argument("--optim.all_w_decay", action="store_true", help='Option of having all params have weight decay. By default norms and embeddings do not')
        if optim_type in ['adamw','adam','nadam','sophia']:
            self.add_adam_optim_args()        

    def add_adam_optim_args(self):
        self.parser.add_argument("--optim.beta1", type=float, default=0.90, help='Beta1 for the adam optimizers')
        self.parser.add_argument("--optim.beta2", type=float, default=0.95, help='Beta2 for the adam optimizers')
