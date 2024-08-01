
import argparse
import sys
from pathlib import Path

Setup_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Setup_DIR))

from config_utils import *

class task_parser(object):
    """
    Parser that contains args depending on the task type
    @args:
        no args
    @rets:
        no rets; self.parser contains args
    """

    def __init__(self, task_types):
        self.parser = argparse.ArgumentParser("")

        if 'ss_image_restoration' in task_types: 
            self.ss_image_restoration_args()

        if 'enhance' in task_types: 
            self.enhance_args()
        
    def ss_image_restoration_args(self):  
        self.parser.add_argument("--ss_image_restoration.mask_percent", type=float, default=0.5, help='If using image restoration as self-supervision, what percent of the image to mask out (set to 0 to not use masked image modeling)')
        self.parser.add_argument("--ss_image_restoration.mask_patch_size", nargs='+', type=int, default=[1,32,32], help='If using image restoration as self-supervision, what size patches to mask out, ordered as T, H, W')
        self.parser.add_argument("--ss_image_restoration.noise_std", type=float, default=1, help='If using image restoration as self-supervision, what std of Gaussian noise to add (set to 0 to not use noise modeling)')
        self.parser.add_argument("--ss_image_restoration.resolution_factor", type=int, default=2, help='If using image restoration as self-supervision, what resolution factor to use (set to 1 to not alter resolution)')

    def enhance_args(self):
        self.parser.add_argument("--enhance.data_range", type=float, nargs='+', default=[1], help='If training an enhancement task, the data range for computing PSNR and SSIM')
