import argparse
import sys
from pathlib import Path

Setup_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Setup_DIR))

from config_utils import *

class sched_parser(object):
    """
    Parser that contains args for the scheduler
    @args:
        no args
    @rets:
        no rets; self.parser contains args
    """

    def __init__(self, scheduler_type):
        self.parser = argparse.ArgumentParser("")
        if scheduler_type=='ReduceLROnPlateau': 
            self.add_plateau_sched_args()
        if scheduler_type=='StepLR': 
            self.add_step_sched_args()
        if scheduler_type=='OneCycleLR': 
            self.add_cycle_sched_args()

    def add_plateau_sched_args(self):
        self.parser.add_argument('--scheduler.patience', type=int, default=0, help="Number of epochs to wait for further LR adjustment")
        self.parser.add_argument('--scheduler.cooldown', type=int, default=0, help="After adjusting the LR, number of epochs to wait before tracking loss")
        self.parser.add_argument('--scheduler.min_lr', type=float, default=1e-7, help="Minimum LR")
        self.parser.add_argument('--scheduler.factor', type=float, default=0.9, help="LR reduction factor, multiplication")

    def add_step_sched_args(self):
        self.parser.add_argument('--scheduler.step_size', type=int, default=5, help="Number of epochs to reduce LR")
        self.parser.add_argument('--scheduler.gamma', type=float, default=0.8, help="Multiplicative factor of learning rate decay")

    def add_cycle_sched_args(self):
        self.parser.add_argument('--scheduler.pct_start', type=float, default=0.3, help="The percentage of the cycle spent increasing the learning rate")
