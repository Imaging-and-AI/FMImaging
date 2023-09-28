"""
Post heads for enhancement tasks
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from pathlib import Path

Model_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Model_DIR))

from imaging_attention import Conv2DExt

#----------------------------------------------------------------------------------------------------------------
class SimpleMultidepthConv(nn.Module):
    def __init__(
        self,
        config,
        feature_channels,
    ):
        """
        Takes in features from backbone model and produces an output of same size as input with no_out_channel 
        This is a very simple head that I made up, should be replaced by something bebtter
        @args:
            config (namespace): contains all parsed args
            feature_channels (List[int]): contains a list of the number of feature channels in each tensor returned by the backbone
            forward pass, x (List[tensor]): contains a list of torch tensors output by the backbone model, each five dimensional (B C* D* H* W*).
        @rets:
            forward pass, x (tensor): output from the enhancement task head
        """
        super().__init__()

        self.config = config
        if self.config.use_patches:
            self.input_size = (config.patch_time,config.patch_height,config.patch_width)
        else:
            self.input_size = (config.time,config.height,config.width)
        
        self.permute = torchvision.ops.misc.Permute([0,2,1,3,4])
        self.conv2d_1 = Conv2DExt(in_channels=feature_channels[-1], out_channels=config.no_out_channel, kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)
        self.conv2d_2 = Conv2DExt(in_channels=feature_channels[-2], out_channels=config.no_out_channel, kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)
        self.conv2d_3 = Conv2DExt(in_channels=feature_channels[-3], out_channels=config.no_out_channel, kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)
        self.conv2d_4 = Conv2DExt(in_channels=feature_channels[-4], out_channels=config.no_out_channel, kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        x_out = torch.zeros((x[0].shape[0], self.config.no_out_channel, *self.input_size)).to(device=x[0].device)
        for x_in, op in zip([x[-1],x[-2],x[-3],x[-4]], [self.conv2d_1,self.conv2d_2,self.conv2d_3,self.conv2d_4]):
            x_in = self.permute(x_in)
            x_in = op(x_in)
            x_in = self.permute(x_in)
            x_in = F.interpolate(x_in, size=self.input_size, mode='trilinear')
            x_out += x_in
            
        return x_out

