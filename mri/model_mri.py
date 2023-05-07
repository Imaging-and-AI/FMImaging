"""
Model(s) used for MRI
"""
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.backbone import *
from model_base.backbone.backbone_small_unet import *
from utils.utils import get_device
from model_base.task_base import *
import losses

# -------------------------------------------------------------------------------------------------
# MRI model

class STCNNT_MRI(STCNNT_Task_Base):
    """
    STCNNT for MRI data
    Just the base CNNT with care to complex_i and residual
    """
    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
            
        Task specific args:
        
            - losses: 
            - loss_weights:
            - complex_i:
        """
        super().__init__(config=config)

        self.complex_i = config.complex_i
        self.residual = config.residual

        self.unet = CNNT_Unet(config=config, total_steps=total_steps, load=False)

        device = get_device(device=config.device)
        self.set_up_loss(device=device)
        self.set_up_optim_and_scheduling(total_steps=total_steps)

        if config.load_path is not None:
            self.load(device=device)

    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image (denoised)
        """
        output = self.unet(x)

        if self.residual:
            C = 2 if self.complex_i else 1
            output = x[:,:,:C] - output

        return output
    
    def set_up_loss(self, device="cpu"):
        """
        Sets up the combined loss
        @args:
            - device (torch.device): device to setup the loss on
        @args (from config):
            - losses (list of "ssim", "ssim3D", "l1", "mse"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether we are dealing with complex images or not
        """
        self.loss_f = losses.Combined_Loss(self.config.losses, self.config.loss_weights,
                                    complex_i=self.config.complex_i, device=device)
