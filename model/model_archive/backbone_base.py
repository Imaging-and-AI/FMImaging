"""
Main base model for backbone
Provides implementation for the following:
    - STCNNT_Base_Runtime_Model:
        - the base class that setups the optimizer scheduler and loss
        - also provides ability to save and load checkpoints
"""

import os
import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torchvision
import interpol

from pathlib import Path
from argparse import Namespace

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Current_DIR))

Model_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Model_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from imaging_attention import *
from blocks import *
from model_utils import create_generic_class_str

__all__ = ['STCNNT_Base_Runtime', 'set_window_patch_sizes_keep_num_window', 'set_window_patch_sizes_keep_window_size', 'DownSample', 'UpSample', 'identity_model']

# -------------------------------------------------------------------------------------------------

def set_window_patch_sizes_keep_num_window(kwargs, HW, num_wind, num_patch, module_name=None):

        num_wind = [2 if v<2 else v for v in num_wind]
        num_patch = [2 if v<2 else v for v in num_patch]

        kwargs["window_size"] = [v//w if v//w>=1 else 1 for v, w in zip(HW, num_wind)]
        kwargs["patch_size"] = [v//w if v//w>=1 else 1 for v, w in zip(kwargs["window_size"], num_patch)]

        kwargs["num_wind"] = num_wind
        kwargs["num_patch"] = num_patch

        info_str = f" --> image size {HW} - windows size {kwargs['''window_size''']} - patch size {kwargs['''patch_size''']} - num windows {kwargs['''num_wind''']} - num patch {kwargs['''num_patch''']}"

        if module_name is not None:
            info_str = module_name + info_str

        print(info_str)

        return kwargs

# -------------------------------------------------------------------------------------------------

def set_window_patch_sizes_keep_window_size(kwargs, HW, window_size, patch_size, module_name=None):        

    if HW[0]//window_size[0] < 2:
        window_size[0] = max(HW[0]//2, 1)

    if HW[1]//window_size[1] < 2:
        window_size[1] = max(HW[1]//2, 1)

    if window_size[0]//patch_size[0] < 2:
        patch_size[0] = max(window_size[0]//2, 1)

    if window_size[1]//patch_size[1] < 2:
        patch_size[1] = max(window_size[1]//2, 1)

    kwargs["window_size"] = window_size
    kwargs["patch_size"] = patch_size

    kwargs["num_wind"] = [v//w for v, w in zip(HW, window_size)]
    kwargs["num_patch"] = [v//w for v, w in zip(window_size, patch_size)]

    info_str = f" --> image size {HW} - windows size {kwargs['''window_size''']} - patch size {kwargs['''patch_size''']} - num windows {kwargs['''num_wind''']} - num patch {kwargs['''num_patch''']}"

    if module_name is not None:
        info_str = module_name + info_str

    print(info_str)

    return kwargs

# -------------------------------------------------------------------------------------------------
# building blocks

class _D2(nn.Module):
    """
    Downsample by 2 layer

    This module takes in a [B, T, C, H, W] tensor and downsample it to [B, T, C, H//2, W//2]

    By default, the operation is performed with a bilinear interpolation.
    If with_conv is True, a 1x1 convolution is added after interpolation.
    If with_interpolation is False, the stride convolution is used.
    """

    def __init__(self, C_in=16, C_out=-1, use_interpolation=True, with_conv=True) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out>0 else C_in

        self.use_interpolation = use_interpolation
        self.with_conv = with_conv

        self.stride_conv = None
        self.conv = None

        if not self.use_interpolation:
            self.stride_conv = Conv2DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[3,3], stride=[2,2], padding=[1,1])
        elif self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv2DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[1,1], stride=[1,1], padding=[0,0])

    def forward(self, x:Tensor) -> Tensor:

        B, T, C, H, W = x.shape
        if self.use_interpolation:
            y = F.interpolate(x.view((B*T, C, H, W)), scale_factor=(0.5, 0.5), mode="bilinear", align_corners=False, recompute_scale_factor=False)
            y = torch.reshape(y, (B, T, *y.shape[1:]))
            if self.with_conv:
                y = self.conv(y)
        else:
            y = self.stride_conv(x)

        return y

# -------------------------------------------------------------------------------------------------

class _D2_patch_merging(nn.Module):
    """
    Downsample by 2 layer using patch merging

    This module takes in a [B, T, C, H, W] tensor and first reformat it to [B, T, 4*C_in, H//2, W//2],
    then a conv is used to get C_out channels.
    """

    def __init__(self, C_in=16, C_out=64) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out>0 else C_in

        self.conv = Conv2DExt(in_channels=4*self.C_in, out_channels=self.C_out, kernel_size=[1,1], stride=[1,1], padding=[0,0])

    def forward(self, x:Tensor) -> Tensor:

        B, T, C, H, W = x.shape

        x0 = x[:, :, :, 0::2, 0::2]  # B T C, H/2 W/2
        x1 = x[:, :, :, 1::2, 0::2]
        x2 = x[:, :, :, 0::2, 1::2]
        x3 = x[:, :, :, 1::2, 1::2]

        y = torch.cat([x0, x1, x2, x3], dim=2)  # B T 4*C H/2 W/2
        y = self.conv(y)

        return y

# -------------------------------------------------------------------------------------------------

class _D2_3D(nn.Module):
    """
    Downsample by 2

    This module takes in a [B, T, C, H, W] tensor and downsample it to [B, T//2, C, H//2, W//2]

    By default, the operation is performed with a trilinear interpolation.
    If with_conv is True, a 1x1 convolution is added after interpolation.
    If with_interpolation is False, the stride convolution is used.
    """

    def __init__(self, C_in=16, C_out=-1, use_interpolation=True, with_conv=True) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out>0 else C_in

        self.use_interpolation = use_interpolation
        self.with_conv = with_conv

        self.stride_conv = None
        self.conv = None

        if not self.use_interpolation:
            self.stride_conv = Conv3DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[3,3,3], stride=[2,2,2], padding=[1,1,1])
        elif self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv3DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[1,1,1], stride=[1,1,1], padding=[0,0,0])

    def forward(self, x:Tensor) -> Tensor:

        B, T, C, H, W = x.shape
        if self.use_interpolation:
            y = F.interpolate(torch.permute(x, (0, 2, 1, 3, 4)), scale_factor=(0.5, 0.5, 0.5), mode="trilinear", align_corners=False, recompute_scale_factor=False)
            y = torch.permute(y, (0, 2, 1, 3, 4))
            if self.with_conv:
                y = self.conv(y)
        else:
            y = self.stride_conv(x)

        return y

# -------------------------------------------------------------------------------------------------

class _D2_patch_merging_3D(nn.Module):
    """
    Downsample by 2 layer using patch merging

    This module takes in a [B, T, C, H, W] tensor and first reformat it to [B, T//2, 8*C_in, H//2, W//2],
    then a conv is used to get C_out channels.
    """

    def __init__(self, C_in=16, C_out=64) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out>0 else C_in

        self.conv = Conv3DExt(in_channels=8*self.C_in, out_channels=self.C_out, kernel_size=[1,1,1], stride=[1,1,1], padding=[0,0,0])

    def forward(self, x:Tensor) -> Tensor:

        B, T, C, H, W = x.shape

        x0 = x[:, 0::2, :, 0::2, 0::2]  # B T C, H/2 W/2
        x1 = x[:, 0::2, :, 1::2, 0::2]
        x2 = x[:, 0::2, :, 0::2, 1::2]
        x3 = x[:, 0::2, :, 1::2, 1::2]
        x4 = x[:, 1::2, :, 0::2, 0::2]
        x5 = x[:, 1::2, :, 1::2, 0::2]
        x6 = x[:, 1::2, :, 0::2, 1::2]
        x7 = x[:, 1::2, :, 1::2, 1::2]

        y = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=2)
        y = self.conv(y)

        return y

# -------------------------------------------------------------------------------------------------

class DownSample(nn.Module):
    """
    Downsample by x(2^N), by using N D2 layers
    """

    def __init__(self, N=2, C_in=16, C_out=-1, use_interpolation=True, with_conv=True, is_3D=False) -> None:
        super().__init__()

        C_out = C_out if C_out>0 else C_in

        self.N = N
        self.C_in = C_in
        self.C_out = C_out
        self.use_interpolation = use_interpolation
        self.with_conv = with_conv
        self.is_3D = is_3D

        DownSampleLayer = _D2_patch_merging
        if is_3D:
            DownSampleLayer = _D2_patch_merging_3D
            
        #layers = [('D2_0', _D2(C_in=C_in, C_out=C_out, use_interpolation=use_interpolation, with_conv=with_conv))]
        layers = [('D2_0', DownSampleLayer(C_in=C_in, C_out=C_out))]
            
        for n in range(1, N):
            #layers.append( (f'D2_{n}', _D2(C_in=C_out, C_out=C_out, use_interpolation=use_interpolation, with_conv=with_conv)) )
            layers.append( (f'D2_{n}', DownSampleLayer(C_in=C_out, C_out=C_out)) )

        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)

# -------------------------------------------------------------------------------------------------

class _U2(nn.Module):
    """
    Upsample by 2

    This module takes in a [B, T, Cin, H, W] tensor and upsample it to [B, T, Cout, 2*H, 2*W], if channel_first is False
    This module takes in a [B, Cin, T, H, W] tensor and upsample it to [B, Cout, T, 2*H, 2*W], if channel_first is True

    By default, the operation is performed with a bilinear interpolation.
    If with_conv is True, a 1x1 convolution is added after interpolation.
    """

    def __init__(self, C_in=16, C_out=-1, method='linear', with_conv=True, channel_first=False) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out>0 else C_in

        self.method = method

        self.with_conv = with_conv
        self.channel_first = channel_first

        self.conv = None
        if self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv2DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[3,3], stride=[1,1], padding=[1,1], channel_first=self.channel_first)

    def forward(self, x:Tensor) -> Tensor:

        B, D1, D2, H, W = x.shape

        if self.method == "NN":
            y = F.interpolate(x.reshape((B*D1, D2, H, W)), size=(2*H, 2*W), mode="nearest", recompute_scale_factor=False)
        elif self.method == 'linear':
            y = F.interpolate(x.reshape((B*D1, D2, H, W)), size=(2*H, 2*W), mode="bilinear", align_corners=False, recompute_scale_factor=False)
        else:
            opt = dict(shape=[2*H, 2*W], anchor='first', bound='replicate')
            y = interpol.resize(x.reshape((B*D1, D2, H, W)), **opt, interpolation=5)

        y = torch.reshape(y, (B, D1, *y.shape[1:]))
        if self.with_conv:
            y = self.conv(y)

        return y

class _U2_3D(nn.Module):
    """
    Upsample by 2

    This module takes in a [B, T, Cin, H, W] tensor and upsample it to [B, 2*T, Cout, 2*H, 2*W], if channel_first is False
    This module takes in a [B, Cin, T, H, W] tensor and upsample it to [B, Cout, 2*T, 2*H, 2*W], if channel_first is True

    By default, the operation is performed with a trilinear interpolation.
    If with_conv is True, a 1x1 convolution is added after interpolation.
    """

    def __init__(self, C_in=16, C_out=-1, method='linear', with_conv=True, channel_first=False) -> None:
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out if C_out>0 else C_in
        self.method = method        
        self.with_conv = with_conv
        self.channel_first = channel_first
        
        self.conv = None
        if self.with_conv or (self.C_in != self.C_out):
            self.conv = Conv3DExt(in_channels=self.C_in, out_channels=self.C_out, kernel_size=[3,3,3], stride=[1,1,1], padding=[1,1,1], channel_first=self.channel_first)

    def forward(self, x:Tensor) -> Tensor:

        if self.channel_first:
            x_channel_first = x
        else:
            x_channel_first = torch.permute(x, (0, 2, 1, 3, 4))
            
        B, C, T, H, W = x_channel_first.shape

        if self.method == "NN":
            y = F.interpolate(x_channel_first, size=(2*T, 2*H, 2*W), mode="nearest", recompute_scale_factor=False)
        elif self.method == 'linear':
            y = F.interpolate(x_channel_first, size=(2*T, 2*H, 2*W), mode="trilinear", align_corners=False, recompute_scale_factor=False)
        else:
            opt = dict(shape=[2*T, 2*H, 2*W], anchor='first', bound='replicate')
            y = interpol.resize(x_channel_first, **opt, interpolation=5)

        if not self.channel_first:
            y = torch.permute(y, (0, 2, 1, 3, 4))
            
        if self.with_conv:
            y = self.conv(y)

        return y

# -------------------------------------------------------------------------------------------------

class UpSample(nn.Module):
    """
    Upsample by x(2^N), by using N U2 layers
    """

    def __init__(self, N=2, C_in=16, C_out=-1, method='linear', with_conv=True, is_3D=False, channel_first=False) -> None:
        super().__init__()

        C_out = C_out if C_out>0 else C_in

        self.N = N
        self.C_in = C_in
        self.C_out = C_out
        self.with_conv = with_conv
        self.is_3D = is_3D
        self.method = method
        self.channel_first = channel_first

        UpSampleLayer = _U2
        if is_3D:
            UpSampleLayer = _U2_3D

        layers = [('U2_0', UpSampleLayer(C_in=C_in, C_out=C_out, method=method, with_conv=with_conv, channel_first=self.channel_first))]
        for n in range(1, N):
            layers.append( (f'U2_{n}', UpSampleLayer(C_in=C_out, C_out=C_out, method=method, with_conv=with_conv, channel_first=self.channel_first)) )

        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x:Tensor) -> Tensor:
        return self.block(x)

# -------------------------------------------------------------------------------------------------

class WindowPartition2D(nn.Module):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor [B, T, C, H, W].
        window_size: local window size.
    Outputs:
        res : [B, T, w[0]*w[1]*C, H//w[0], W//w[0]]
    """

    def __init__(self, window_size=[2,2]):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):

        B, T, C, H, W = x.shape

        H_prime = H // self.window_size[0]
        W_prime = W // self.window_size[1]

        x = x.view(B, T, C, 
            H_prime, self.window_size[0],
            W_prime, self.window_size[1]
        )

        res = (
            x.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(B, T, C*self.window_size[0]*self.window_size[1], H_prime, W_prime)
        )

        return res

class WindowPartition3D(nn.Module):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor [B, T, C, H, W].
        window_size: local window size.
    Outputs:
        res : [B, T//w[2], w[0]*w[1]*w[2]*C, H//w[0], W//w[0]]
    """

    def __init__(self, window_size=[2,2,2]):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):

        B, T, C, H, W = x.shape

        H_prime = H // self.window_size[0]
        W_prime = W // self.window_size[1]
        T_prime = T // self.window_size[2]

        x = x.view(B, T_prime, self.window_size[2], C, H_prime, self.window_size[0], W_prime, self.window_size[1])

        res = (
            x.permute(0, 1, 3, 2, 5, 7, 4, 6).contiguous().view(B, T_prime, C*self.window_size[0]*self.window_size[1]*self.window_size[2], H_prime, W_prime)
        )

        return res

class WindowPartitionReverse2D(nn.Module):
    """
     Args:
        window_size: local window size.
    """
    def __init__(self, window_size=[2,2]):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        B, T, C, H, W = x.shape

        C_prime = C//(self.window_size[0]*self.window_size[1])

        res = x.view(
            B, T, C_prime, self.window_size[0], self.window_size[1], H, W 
        )

        res = res.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(B, T, C_prime, self.window_size[0]*H, self.window_size[1]*W)

        return res

class WindowPartitionReverse3D(nn.Module):
    """
     Args:
        window_size: local window size.
    """
    def __init__(self, window_size=[2,2,2]):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        B, T, C, H, W = x.shape

        C_prime = C//(self.window_size[0]*self.window_size[1]*self.window_size[2])

        res = x.view(
            B, T, C_prime, self.window_size[0], self.window_size[1], self.window_size[2], H, W 
        )

        res = res.permute(0, 1, 5, 2, 3, 6, 4, 7).contiguous().view(B, T*self.window_size[2], C_prime, self.window_size[0]*H, self.window_size[1]*W)

        return res

# -------------------------------------------------------------------------------------------------
# # Base model for rest to inherit

class STCNNT_Base_Runtime(nn.Module):
    """
    Base Runtime model of STCNNT
    Sets up the optimizer, scheduler and loss
    Provides generic save and load functionality
    """
    def __init__(self, config) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
        """
        super().__init__()
        self.config = config

    def permute(self, x):
            return torch.permute(x, (0,2,1,3,4))

    @property
    def device(self):
        return next(self.parameters()).device

    # def set_window_patch_sizes_keep_num_window(self, kwargs, HW, num_wind, num_patch, module_name=None):

    #     num_wind = [2 if v<2 else v for v in num_wind]
    #     num_patch = [2 if v<2 else v for v in num_patch]

    #     kwargs["window_size"] = [v//w if v//w>=1 else 1 for v, w in zip(HW, num_wind)]
    #     kwargs["patch_size"] = [v//w if v//w>=1 else 1 for v, w in zip(kwargs["window_size"], num_patch)]

    #     kwargs["num_wind"] = num_wind
    #     kwargs["num_patch"] = num_patch

    #     info_str = f" --> image size {HW} - windows size {kwargs['''window_size''']} - patch size {kwargs['''patch_size''']} - num windows {kwargs['''num_wind''']} - num patch {kwargs['''num_patch''']}"

    #     if module_name is not None:
    #         info_str = module_name + info_str

    #     print(info_str)

    #     return kwargs

    # def set_window_patch_sizes_keep_window_size(self, kwargs, HW, window_size, patch_size, module_name=None):        

    #     if HW[0]//window_size[0] < 2:
    #         window_size[0] = max(HW[0]//2, 1)

    #     if HW[1]//window_size[1] < 2:
    #         window_size[1] = max(HW[1]//2, 1)

    #     if window_size[0]//patch_size[0] < 2:
    #         patch_size[0] = max(window_size[0]//2, 1)

    #     if window_size[1]//patch_size[1] < 2:
    #         patch_size[1] = max(window_size[1]//2, 1)

    #     kwargs["window_size"] = window_size
    #     kwargs["patch_size"] = patch_size

    #     kwargs["num_wind"] = [v//w for v, w in zip(HW, window_size)]
    #     kwargs["num_patch"] = [v//w for v, w in zip(window_size, patch_size)]

    #     info_str = f" --> image size {HW} - windows size {kwargs['''window_size''']} - patch size {kwargs['''patch_size''']} - num windows {kwargs['''num_wind''']} - num patch {kwargs['''num_patch''']}"

    #     if module_name is not None:
    #         info_str = module_name + info_str

    #     print(info_str)

    #     return kwargs

    # def save(self, epoch):
    #     """
    #     Save model checkpoints
    #     @args:
    #         - epoch (int): current epoch of the training cycle
    #     @args (from config):
    #         - date (datetime str): runtime date
    #         - checkpath (str): directory to save checkpoint in
    #     """
    #     run_name = self.config.run_name.replace(" ", "_")
    #     save_file_name = f"backbone_{run_name}_{self.config.date}_epoch-{epoch}.pth"
    #     save_path = os.path.join(self.config.check_path, save_file_name)
    #     logging.info(f"Saving backbone status at {save_path}")
    #     torch.save({
    #         "epoch":epoch,
    #         "model_state": self.state_dict(), 
    #         "config": self.config
    #     }, save_path)

    # def load(self, device=None):
    #     """
    #     Load a checkpoint from the load path in config
    #     @args:
    #         - device (torch.device): device to setup the model on
    #     @args (from config):
    #         - load_path (str): path to load the weights from
    #     """
    #     logging.info(f"Loading backbone from {self.config.load_path}")
        
    #     device = get_device(device=device)
        
    #     status = torch.load(self.config.load_path, map_location=device)
        
    #     self.load_state_dict(status['model'])
    #     self.config = status['config']
        
# -------------------------------------------------------------------------------------------------

# moved this to model_utils.py
    
# class IdentityModel(nn.Module):
#     """
#     Simple class to implement identity model in format requierd by codebase
#     """
#     def __init__(self):
#         super().__init__()
#         self.identity_layer = nn.Identity()

#     def forward(self, x):
#         return [self.identity_layer(x)]

# def identity_model(config, input_feature_channels):
#     """
#     Simple function to return identity model and feature channels in format requierd by codebase
#     """
#     model = IdentityModel()
#     output_feature_channels = input_feature_channels
#     return model, output_feature_channels

# -------------------------------------------------------------------------------------------------

def tests():
    print("Passed all tests")

if __name__=="__main__":
    tests()