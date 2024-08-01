"""
Post heads for enhancement tasks
"""

from __future__ import annotations

from collections.abc import Sequence

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

# from imaging_attention import Conv2DExt
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, UnetrPrUpBlock
from monai.utils import optional_import

rearrange, _ = optional_import("einops", name="rearrange")

#----------------------------------------------------------------------------------------------------------------
class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    UNETR code modified from monai
    """
    def __init__(
        self,
        config,
        task_ind,
        input_feature_channels,
        output_feature_channels
    ) -> None:

        super().__init__()

        if input_feature_channels[0] % 12 != 0:
            raise ValueError("Features should be divisible by 12 to use current UNETR config.")
        
        input_image_channels = config.no_in_channel[task_ind]
        if config.time[task_ind]==1:
            spatial_dims=2
            self.spatial_dims=2
            upsample_kernel_size=2
            if config.backbone_component=='omnivore': 
                mod_patch_size=config.omnivore.patch_size[1:]
            else:
                mod_patch_size=config.SWIN.patch_size[1:]
        else: 
            spatial_dims=3
            self.spatial_dims=3
            if config.backbone_component=='omnivore': 
                upsample_kernel_size=(1,2,2) #These all should reflect the patchmerging ops in the backbonev
                mod_patch_size=config.omnivore.patch_size
            else: 
                upsample_kernel_size=(2,2,2)
                mod_patch_size=config.SWIN.patch_size
            

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_image_channels,
            out_channels=input_feature_channels[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[0],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[1],
            out_channels=input_feature_channels[1],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[2],
            out_channels=input_feature_channels[2],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[4],
            out_channels=input_feature_channels[4],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[4],
            out_channels=input_feature_channels[3],
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size, #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[3],
            out_channels=input_feature_channels[2],
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size, #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[2],
            out_channels=input_feature_channels[1],
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size, #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[1],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size, #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[0],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=mod_patch_size, #This should be the patch embedding kernel size
            norm_name="instance",
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=input_feature_channels[0], out_channels=output_feature_channels[-1])

    def forward(self, input_data):
        if self.spatial_dims==2:
            input_data = [i.squeeze(2) for i in input_data]
        x_in = input_data[0]
        backbone_features = input_data[1:]
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
        if self.spatial_dims==2:
            out = out.unsqueeze(2)
        return [out]

#----------------------------------------------------------------------------------------------------------------
class ViTUNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    Code modified from monai
    """
    def __init__(
        self,
        config,
        task_ind,
        input_feature_channels,
        output_feature_channels
    ) -> None:

        super().__init__()
        
        feature_size = 32

        input_image_channels = config.no_in_channel[task_ind]
        hidden_size = config.ViT.hidden_size
        if config.time[task_ind]==1:
            spatial_dims=2
            self.spatial_dims=2
            img_size = [config.height[task_ind], config.width[task_ind]]
            patch_size = config.ViT.patch_size[1:]
        else: 
            spatial_dims=3
            self.spatial_dims=3
            img_size = [config.time[task_ind], config.height[task_ind], config.width[task_ind]]
            patch_size = config.ViT.patch_size
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = hidden_size

        # Look at the UNETR paper to adjust these, makes it clear whah each param is doing
        if config.ViT.patch_size[1]==2 and config.ViT.patch_size[2]==2:
            n_us2, n_us3, n_us4 = 0, 0, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 1, 1, 1, 2
        elif config.ViT.patch_size[1]==4 and config.ViT.patch_size[2]==4:
            n_us2, n_us3, n_us4 = 1, 1, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 1, 1, 2, 2
        elif config.ViT.patch_size[1]==8 and config.ViT.patch_size[2]==8:
            n_us2, n_us3, n_us4 = 2, 1, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 1, 2, 2, 2
        elif config.ViT.patch_size[1]==16 and config.ViT.patch_size[2]==16:
            n_us2, n_us3, n_us4 = 2, 1, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 2, 2, 2, 2
        elif config.ViT.patch_size[1]==32 and config.ViT.patch_size[2]==32:
            n_us2, n_us3, n_us4 = 2, 1, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 4, 2, 2, 2
        else:
            raise ValueError(f'ViT patch size {config.ViT.patch_size} not yet supported')
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_image_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=n_us2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=enc_us2,
            norm_name="instance",
            conv_block=True,
            res_block=True,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=n_us3,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=enc_us3,
            norm_name="instance",
            conv_block=True,
            res_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=n_us4,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=enc_us4,
            norm_name="instance",
            conv_block=True,
            res_block=True,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=dec_us4,
            norm_name="instance",
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=dec_us3,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=dec_us2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=dec_us1,
            norm_name="instance",
            res_block=True,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=output_feature_channels[-1])
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, input_data):
        x_in = input_data[0]
        if self.spatial_dims==2:
            x_in = x_in.squeeze(2)
        hidden_states_out = input_data[1:-1]
        x = input_data[-1]
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        out = self.out(out)
        if self.spatial_dims==2:
            out = out.unsqueeze(2)
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
        Takes in features from backbone model and produces an output of same size as input with output_feature_channels 
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

        x_out = torch.zeros((x[0].shape[0], self.output_feature_channels[-1], *self.input_size)).to(device=x[0].device)
        for x_in, op in zip([x[-1],x[-2],x[-3],x[-4]], [self.conv2d_1,self.conv2d_2,self.conv2d_3,self.conv2d_4]):
            x_in = self.permute(x_in)
            x_in = op(x_in)
            x_in = self.permute(x_in)
            x_in = F.interpolate(x_in, size=self.input_size, mode='trilinear')
            x_out += x_in
            
        return [x_out]

#----------------------------------------------------------------------------------------------------------------

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    Adapted from https://github.com/kuoweilai/pixelshuffle3d/blob/master/pixelshuffle3d.py
    '''
    def __init__(self, scale_depth, scale_height, scale_width):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale_depth = scale_depth
        self.scale_height = scale_height
        self.scale_width = scale_width

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // (self.scale_depth * self.scale_height * self.scale_width)

        out_depth = in_depth * self.scale_depth
        out_height = in_height * self.scale_height
        out_width = in_width * self.scale_width

        input_view = input.contiguous().view(batch_size, nOut, self.scale_depth, self.scale_height, self.scale_width, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class PixelShuffle2d(nn.Module):
    '''
    This class is a 2d version of pixelshuffle.
    Adapted from https://github.com/kuoweilai/pixelshuffle3d/blob/master/pixelshuffle3d.py
    '''
    def __init__(self, scale_height, scale_width):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale_height = scale_height
        self.scale_width = scale_width

    def forward(self, input):
        batch_size, channels, in_height, in_width = input.size()
        nOut = channels // (self.scale_height * self.scale_width)

        out_height = in_height * self.scale_height
        out_width = in_width * self.scale_width

        input_view = input.contiguous().view(batch_size, nOut, self.scale_height, self.scale_width, in_height, in_width)

        output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()

        return output.view(batch_size, nOut, out_height, out_width)
    
class ViTMAEHead(nn.Module):
    """
    Simple decoder for MAE pretraining; based off of SimMIM implementation
    Adapted from https://github.com/microsoft/SimMIM/blob/main/models/simmim.py
    """
    def __init__(
        self,
        config,
        task_ind,
        input_feature_channels,
        output_feature_channels
    ) -> None:
        
        super().__init__()

        if config.time[task_ind]==1:
            self.spatial_dims = 2
            if not config.use_patches[0]: self.input_size = [config.height[task_ind], config.width[task_ind]]
            else: self.input_size = [config.patch_height[task_ind], config.patch_width[task_ind]]
            if len(config.ViT.patch_size)==3: self.mod_patch_size = config.ViT.patch_size[1:]
            else: self.mod_patch_size = config.ViT.patch_size
            self.vit_decoder = nn.Sequential(
                                nn.Conv2d(
                                    in_channels=input_feature_channels[-1],
                                    out_channels=(self.mod_patch_size[0]*self.mod_patch_size[1]) * output_feature_channels[-1], kernel_size=1),
                                PixelShuffle2d(self.mod_patch_size[0], self.mod_patch_size[1]),
                )
            
        else:
            self.spatial_dims = 3
            if not config.use_patches[0]: self.input_size = [config.time[task_ind], config.height[task_ind], config.width[task_ind]]
            else: self.input_size = [config.patch_time[task_ind], config.patch_height[task_ind], config.patch_width[task_ind]]
            self.mod_patch_size = config.ViT.patch_size
            self.vit_decoder = nn.Sequential(
                                nn.Conv3d(
                                    in_channels=input_feature_channels[-1],
                                    out_channels=(self.mod_patch_size[0]*self.mod_patch_size[1]*self.mod_patch_size[2]) * output_feature_channels[-1], kernel_size=1),
                                PixelShuffle3d(self.mod_patch_size[0], self.mod_patch_size[1], self.mod_patch_size[2]),
                )
    
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(self.input_size, self.mod_patch_size))
        self.proj_axes = (0, self.spatial_dims + 1) + tuple(d + 1 for d in range(self.spatial_dims))
        
        
    def _reshape_vit_output(self, x):

        hidden_size = x.shape[-1]
        proj_view_shape = list(self.feat_size) + [hidden_size]
        
        new_view = [x.size(0)] + proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()

        return x

    def forward(self, x):
        x = self._reshape_vit_output(x[-1])        
        x = self.vit_decoder(x)
        if self.spatial_dims==2: 
            x = torch.unsqueeze(x,2)
        return [x]
    
class SwinMAEHead(nn.Module):
    """
    Simple decoder for MAE pretraining; based off of SimMIM implementation
    Adapted from https://github.com/microsoft/SimMIM/blob/main/models/simmim.py
    """
    def __init__(
        self,
        config,
        task_ind,
        input_feature_channels,
        output_feature_channels
    ) -> None:
        
        super().__init__()

        if config.time[task_ind]==1:
            self.spatial_dims = 2
            if len(config.SWIN.patch_size)==3: self.mod_patch_size = config.SWIN.patch_size[1:]
            else: self.mod_patch_size = config.SWIN.patch_size
            self.encoder_stride = (self.mod_patch_size[0] * 2**len(config.SWIN.depths), 
                                   self.mod_patch_size[1] * 2**len(config.SWIN.depths))
            self.swin_decoder = nn.Sequential(
                                nn.Conv2d(
                                    in_channels=input_feature_channels[-1],
                                    out_channels=(self.encoder_stride[0]*self.encoder_stride[1]) * output_feature_channels[-1], kernel_size=1),
                                PixelShuffle2d(self.encoder_stride[0], self.encoder_stride[1]),
                )
            
        else:
            self.spatial_dims = 3
            self.mod_patch_size = config.SWIN.patch_size
            self.encoder_stride = (self.mod_patch_size[0] * 2**len(config.SWIN.depths), 
                                   self.mod_patch_size[1] * 2**len(config.SWIN.depths),
                                   self.mod_patch_size[2] * 2**len(config.SWIN.depths))
            self.swin_decoder = nn.Sequential(
                                nn.Conv3d(
                                    in_channels=input_feature_channels[-1],
                                    out_channels=(self.encoder_stride[0]*self.encoder_stride[1]*self.encoder_stride[2]) * output_feature_channels[-1], kernel_size=1),
                                PixelShuffle3d(self.encoder_stride[0], self.encoder_stride[1], self.encoder_stride[2]),
                )

    def forward(self, x):
        x = x[-1]
        if self.spatial_dims==2: 
            x = torch.squeeze(x,2)
        x = self.swin_decoder(x)
        if self.spatial_dims==2: 
            x = torch.unsqueeze(x,2)
        return [x]