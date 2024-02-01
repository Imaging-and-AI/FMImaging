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
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import optional_import

rearrange, _ = optional_import("einops", name="rearrange")

#----------------------------------------------------------------------------------------------------------------
class UNETR2D(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    UNETR code modified from monai
    """
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ) -> None:

        super().__init__()

        if input_feature_channels[0] % 12 != 0:
            raise ValueError("Features should be divisible by 12 to use current UNETR config.")

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.no_in_channel,
            out_channels=input_feature_channels[0],
            kernel_size=(1,3,3),
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[0],
            out_channels=input_feature_channels[0],
            kernel_size=(1,3,3),
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[1],
            out_channels=input_feature_channels[1],
            kernel_size=(1,3,3),
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[2],
            out_channels=input_feature_channels[2],
            kernel_size=(1,3,3),
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[4],
            out_channels=input_feature_channels[4],
            kernel_size=(1,3,3),
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[4],
            out_channels=input_feature_channels[3],
            kernel_size=(1,3,3),
            upsample_kernel_size=(1,2,2), #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[3],
            out_channels=input_feature_channels[2],
            kernel_size=(1,3,3),
            upsample_kernel_size=(1,2,2), #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[2],
            out_channels=input_feature_channels[1],
            kernel_size=(1,3,3),
            upsample_kernel_size=(1,2,2), #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[1],
            out_channels=input_feature_channels[0],
            kernel_size=(1,3,3),
            upsample_kernel_size=(1,2,2), #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[0],
            out_channels=input_feature_channels[0],
            kernel_size=(1,3,3),
            upsample_kernel_size=config.omnivore.patch_size, #This should be the patch embedding kernel size
            norm_name="instance",
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=input_feature_channels[0], out_channels=output_feature_channels[-1])

    def forward(self, x_in, backbone_features):
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(backbone_features[0])
        enc2 = self.encoder3(backbone_features[1])
        enc3 = self.encoder4(backbone_features[2])
        dec4 = self.encoder10(backbone_features[4])
        dec3 = self.decoder5(dec4, backbone_features[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        out = self.out(out)
        return [out]

#----------------------------------------------------------------------------------------------------------------
class UNETR3D(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    UNETR code modified from monai
    """
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ) -> None:

        super().__init__()

        if input_feature_channels[0] % 12 != 0:
            raise ValueError("Features should be divisible by 12 to use current UNETR config.")

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.no_in_channel,
            out_channels=input_feature_channels[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[0],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[1],
            out_channels=input_feature_channels[1],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[2],
            out_channels=input_feature_channels[2],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[4],
            out_channels=input_feature_channels[4],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[4],
            out_channels=input_feature_channels[3],
            kernel_size=3,
            upsample_kernel_size=(1,2,2), #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[3],
            out_channels=input_feature_channels[2],
            kernel_size=3,
            upsample_kernel_size=(1,2,2), #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[2],
            out_channels=input_feature_channels[1],
            kernel_size=3,
            upsample_kernel_size=(1,2,2), #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[1],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=(1,2,2), #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=input_feature_channels[0],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=config.omnivore.patch_size, #This should be the patch embedding kernel size
            norm_name="instance",
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=input_feature_channels[0], out_channels=output_feature_channels[-1])

    def forward(self, x_in, backbone_features):
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(backbone_features[0])
        enc2 = self.encoder3(backbone_features[1])
        enc3 = self.encoder4(backbone_features[2])
        dec4 = self.encoder10(backbone_features[4])
        dec3 = self.decoder5(dec4, backbone_features[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        out = self.out(out)
        return [out]

#----------------------------------------------------------------------------------------------------------------
class SimpleMultidepthConv(nn.Module):
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ):
        """
        Takes in features from backbone model and produces an output of same size as input with no_out_channel 
        This is a very simple head that I made up, should be replaced by something bebtter
        @args:
            config (namespace): contains all parsed args
            input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the backbone)
            output_feature_channels (List[int]): contains a list of the number of feature channels in each tensor expected to be returned by this task head
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
        self.conv2d_1 = Conv2DExt(in_channels=input_feature_channels[-1], out_channels=output_feature_channels[-1], kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)
        self.conv2d_2 = Conv2DExt(in_channels=input_feature_channels[-2], out_channels=output_feature_channels[-1], kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)
        self.conv2d_3 = Conv2DExt(in_channels=input_feature_channels[-3], out_channels=output_feature_channels[-1], kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)
        self.conv2d_4 = Conv2DExt(in_channels=input_feature_channels[-4], out_channels=output_feature_channels[-1], kernel_size=[1,1], padding=[0, 0], stride=[1,1], bias=True)

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
            
        return [x_out]

