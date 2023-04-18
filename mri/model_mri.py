"""
Model(s) used for MRI
"""
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.base_models import *

# -------------------------------------------------------------------------------------------------
# MRI model

class STCNNT_MRI(STCNNT_Base_Runtime):
    """
    STCNNT for MRI data
    Just the base CNNT with care to complex_i and residual
    """
    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
        """
        super().__init__(config=config)

        self.complex_i = config.complex_i
        self.residual = config.residual
        config.residual = False

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
