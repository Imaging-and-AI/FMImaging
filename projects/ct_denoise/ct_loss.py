"""
Loss for ct, a combined loss
"""

import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from loss.loss_functions import Combined_Loss

class ct_loss(object):
    def __init__(self, config):

        self.config = config
        self.loss_f = Combined_Loss(self.config.losses, self.config.loss_weights,
                                    complex_i=self.config.complex_i, device=self.config.device)

    def __call__(self, outputs, targets, weights=None):

        loss_value = self.loss_f(outputs, targets, weights=weights)
        return loss_value
