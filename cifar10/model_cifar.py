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
    Built on top of CNNT Unet with additional layers for classification
    """
    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
        """
        super().__init__(config=config)

        self.unet = CNNT_Unet(config=config, total_steps=total_steps, load=False)
        self.head = nn.Sequential(nn.Flatten(start_dim=0, end_dim=1),
                                    nn.Conv2d(config.C_out, config.head_channels[0], kernel_size=3, stride=2, padding=1),
                                    nn.Flatten(start_dim=1, end_dim=-1),
                                    nn.LeakyReLU(),
                                    nn.Linear(config.head_channels[0]*config.height[0]//2*config.width[0]//2, config.head_channels[1]),
                                    nn.LeakyReLU(),
                                    nn.Linear(config.head_channels[1], config.head_channels[2]),
                                    nn.Softmax(dim=1))

        device = get_device(device=config.device)
        self.set_up_optim_and_scheduling(total_steps=total_steps)

        if config.load_path is not None:
            self.load(device=device)

    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image
        """
        return self.head(self.unet(x))
