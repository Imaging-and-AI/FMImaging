"""
Post heads for classification tasks
"""

import sys
import torch
import torch.nn as nn
import torchvision
from torchvision.ops.misc import Permute

from pathlib import Path

Model_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Model_DIR))

from imaging_attention import Conv2DExt

#----------------------------------------------------------------------------------------------------------------
class NormPoolLinear(nn.Module):
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ):
        """
        Takes in features from backbone model and produces a classification vector using only the last feature tensor
        Originally used for baseline tests with swin and omnivore backbones.
        @args:
            config (namespace): contains all parsed args
            input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the backbone)
            output_feature_channels (List[int]): contains a list of the number of feature channels in each tensor expected to be returned by this task head
            forward pass, x (List[tensor]): contains a list of torch tensors output by the backbone model, each five dimensional (B C' D' H' W')
        @rets:
            forward pass, x (List(tensor)): output from the classification task head, each of size B x no_out_channel
        """
        super().__init__()

        self.norm = nn.LayerNorm(input_feature_channels[-1])
        self.permute1 = Permute([0, 2, 3, 4, 1])  # B C D H W -> B D H W C
        self.permute2 = Permute([0, 4, 1, 2, 3])  # B D H W C -> B C D H W
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(input_feature_channels[-1], output_feature_channels[-1])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.permute1(x[-1])
        x = self.norm(x)
        x = self.permute2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return [x]
    
#----------------------------------------------------------------------------------------------------------------
class ConvPoolLinear(nn.Module):
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ):
        """
        Takes in features from backbone model and produces a classification vector using only the last feature tensor
        Originally used by STCNNT code.
        @args:
            config (namespace): contains all parsed args
            input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the backbone)
            output_feature_channels (List[int]): contains a list of the number of feature channels in each tensor expected to be returned by this task head
            forward pass, x (List[tensor]): contains a list of torch tensors output by the backbone model, each five dimensional (B C' D' H' W')
        @rets:
            forward pass, x (List[tensor]): output from the classification task head, size B x no_out_channel
        """
        super().__init__()

        conv_dim = 2048 # Could set in config with arg

        self.permute = torchvision.ops.misc.Permute([0,2,1,3,4])
        self.conv2d = Conv2DExt(in_channels=input_feature_channels[-1], out_channels=conv_dim, kernel_size=[1,1], padding=[0, 0], stride=[1,1])
        self.avgpool = torch.nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(conv_dim, output_feature_channels[-1])        

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.permute(x[-1])
        x = self.conv2d(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return [x]