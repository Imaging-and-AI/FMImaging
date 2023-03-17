"""
Model(s) used for cifar10
"""
import sys
import torch.nn as nn
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.base_models import *

# -------------------------------------------------------------------------------------------------
# Cifar model

class STCNNT_Cifar(STCNNT_Base_Runtime):
    """
    STCNNT for Cifar 10
    Build on top of CNNT Unet with additional layers for classification
    """

    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
        """
        super().__init__(config=config)

        self.model = nn.Sequential(CNNT_Unet(config=config, total_steps=total_steps),
                                    nn.Linear(config.C_out*config.height[0]*config.width[0], config.C_out),
                                    nn.LeakyReLU(),
                                    nn.Linear(config.C_out, 64),
                                    nn.LeakyReLU(),
                                    nn.Linear(64, 10),
                                   )
        
    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image
        """
        return self.model(x)
