"""
Model(s) used for cifar10
"""
import sys
import torch.nn as nn
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.backbone import *
from model_base.backbone.backbone_small_unet import *
from model_base.task_base import *
from utils.utils import get_device

# -------------------------------------------------------------------------------------------------
# Cifar model

class STCNNT_Cifar(STCNNT_Task_Base):
    """
    STCNNT for Cifar 10
    Built on top of CNNT Unet with additional layers for classification
    """
    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
            
        Task specific args:
            None
        """
        super().__init__(config=config)

        final_c = 10 if config.data_set == "cifar10" else 100

        self.unet = CNNT_Unet(config=config, total_steps=total_steps, load=False)
        self.head = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                    # nn.Conv2d(config.C_out, 128, kernel_size=3, stride=2, padding=1),
                                    # nn.LeakyReLU(),
                                    # nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
                                    # nn.LeakyReLU(),
                                    # nn.Flatten(start_dim=1, end_dim=-1),
                                    # #nn.Linear(config.head_channels[0]*config.height[0]//2*config.width[0]//2, config.head_channels[1]),
                                    # nn.Linear(64*8*8, 128),
                                    # nn.LeakyReLU(),
                                    nn.Linear(config.C_out*32*32, final_c))

        device = get_device(device=config.device)
        
        self.set_up_loss(device=device)
        self.set_up_optim_and_scheduling(total_steps=total_steps)

        if config.load_path is not None:
            self.load(device=device)

    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image
        """
        return self.head(self.unet(x))
    
    def set_up_loss(self, device="cpu"):
        """
        Sets up the loss
        @args:
            - device (torch.device): device to setup the loss on
            
        @output:
            self.loss_f : cross entropy loss
        """
        self.loss_f = nn.CrossEntropyLoss()
