
"""
Requirements to use a custom ModelManager with the general codebase: 
- Inheret the ModelManager class and make adjustments, keeping the method inputs/outputs the same
- Note that the custom class model __init___ args can be customized via the custom run.py file
"""

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

from model.model_base import ModelManager

class custom_ModelManager(ModelManager):
    """
    Example custom ModelManager
    """
    def __init__(self, config):
        super().__init__(config=config)
        
    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image, B C D/T H W
        @rets:
            - output: final output from model for this task
        """

        pre_output = self.pre(x)
        backbone_output = self.backbone(pre_output[-1])
        post_output = self.post(backbone_output)
        return post_output[-1]