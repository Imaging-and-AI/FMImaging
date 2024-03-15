"""
Post heads for segmentation tasks
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
# Upernet head code from https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

class PSPModule2D(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule2D, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse2D(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse2D, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

class UperNet2D(nn.Module):
    """
    UperNet3D head, used for segmentation. Incorporates features from four different depths in backbone.
    @args:
        config (namespace): contains all args
        task_ind (int): index of the task this post component is associated with
        input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the backbone)
        output_feature_channels (List[int]): contains a list of the number of feature channels in each tensor expected to be returned by this task head
        forward pass, features (torch tensor): features we will process, size B C H W
    @rets:
        forward pass, x (torch tensor): output tensor, size B C H W

    """
    def __init__(self, 
                 config, 
                 task_ind,
                 input_feature_channels,
                 output_feature_channels):
        super(UperNet2D, self).__init__()


        self.config = config
        self.fpn_out = input_feature_channels[0]
        if self.config.use_patches[task_ind]:
            self.input_size = (config.patch_height[task_ind],config.patch_width[task_ind])
        else:
            self.input_size = (config.height[task_ind],config.width[task_ind])
        self.PPN = PSPModule2D(input_feature_channels[-1])
        self.FPN = FPN_fuse2D(input_feature_channels, fpn_out=self.fpn_out)
        self.head = nn.Conv2d(self.fpn_out, output_feature_channels[-1], kernel_size=3, padding=1)

    def forward(self, features):
        # features = [f[:,:,0,:,:] for f in features] # Remove time dim for 2d convoultional upernet
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))
        x = F.interpolate(x, size=self.input_size, mode='bilinear')
        x = torch.unsqueeze(x,2) # Add back in time dim for main codebase 
        return [x]

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


#----------------------------------------------------------------------------------------------------------------
class PSPModule3D(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule3D, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=1, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool3d(output_size=bin_sz)
        conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm3d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        d, h, w = features.size()[2], features.size()[3], features.size()[4]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(d, h, w), mode='trilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

def up_and_add3D(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3), y.size(4)), mode='trilinear', align_corners=True) + y

class FPN_fuse3D(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse3D, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv3d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv3d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv3d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add3D(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        D, H, W = P[0].size(2), P[0].size(3), P[0].size(4)
        P[1:] = [F.interpolate(feature, size=(D, H, W), mode='trilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

class UperNet3D(nn.Module):
    """
    UperNet3D head, used for segmentation. Incorporates features from four different depths in backbone.
    @args:
        config (namespace): contains all args
        task_ind (int): index of the task this post component is associated with
        input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the backbone)
        output_feature_channels (List[int]): contains a list of the number of feature channels in each tensor expected to be returned by this task head
        forward pass, features (torch tensor): features we will process, size B C D H W
    @rets:
        forward pass, x (torch tensor): output tensor, size B C D H W

    """
    def __init__(self, 
                 config, 
                 task_ind,
                 input_feature_channels,
                 output_feature_channels):
        super(UperNet3D, self).__init__()

        self.config = config
        self.fpn_out = input_feature_channels[0]
        if self.config.use_patches[task_ind]:
            self.input_size = (config.patch_time[task_ind],config.patch_height[task_ind],config.patch_width[task_ind])
        else:
            self.input_size = (config.time[task_ind],config.height[task_ind],config.width[task_ind])
        self.PPN = PSPModule3D(input_feature_channels[-1])
        self.FPN = FPN_fuse3D(input_feature_channels, fpn_out=self.fpn_out)
        self.head = nn.Conv3d(self.fpn_out, output_feature_channels[-1], kernel_size=3, padding=1)

    def forward(self, features, output_size=None):
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))
        if output_size is None:
            x = F.interpolate(x, size=self.input_size, mode='trilinear')
        else:
            x = F.interpolate(x, size=output_size, mode='trilinear')
        return [x]

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d): module.eval()

#----------------------------------------------------------------------------------------------------------------

class SimpleConv(nn.Module):
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels,
    ):
        """
        Takes in features from backbone model and produces an output of same size as input with output_feature_channels using only the last feature tensor
        Originally used with STCNNT experiments
        @args:
            config (namespace): contains all parsed args
            input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the backbone)
            output_feature_channels (List[int]): contains a list of the number of feature channels in each tensor expected to be returned by this task head
            forward pass, x (List[tensor]): contains a list of torch tensors output by the backbone model, each five dimensional (B C D H W).
            ** Note that this function requires the backbone output features of same resolution as input---this function does not interpolate features
        @rets:
            forward pass, x (tensor): output from the segmentation task head
        """
        super().__init__()

        self.permute = torchvision.ops.misc.Permute([0,2,1,3,4])
        self.conv2d = Conv2DExt(in_channels=input_feature_channels[-1], out_channels=output_feature_channels[-1], kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.permute(x[-1])
        x = self.conv2d(x)
        x = self.permute(x)
        return [x]
