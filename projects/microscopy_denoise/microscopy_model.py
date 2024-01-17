
"""
Microscopy models

- STCNNT_Microscopy: the pre-backbone-post model with a simple pre and post module
- Microscopy_double_net: a hrnet or mixed_unet backbone + a hrnet or mixed_unet post
"""

import os
import sys
import copy
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from colorama import Fore, Back, Style

import torch
import torch.nn as nn

from model.model_base import ModelManager
from model.backbone import identity_model, omnivore, STCNNT_HRnet_model, STCNNT_Unet_model, STCNNT_Mixed_Unetr_model
from model.backbone import STCNNT_HRnet, STCNNT_Mixed_Unetr, UpSample, set_window_patch_sizes_keep_num_window, set_window_patch_sizes_keep_window_size, STCNNT_Block
from model.imaging_attention import *
from model.task_heads import *

from setup import get_device, Nestedspace
from utils import model_info, get_gpu_ram_usage, start_timer, end_timer
from optim.optim_utils import divide_optim_into_groups

# input to the Microscopy models are channel-first, [B, C, T, H, W]
# output from the Microscopy models are channel-first, [B, C, T, H, W]
# inside the model, permutes are used as less as possible

# -------------------------------------------------------------------------------------------------

def microscopy_ModelManager(config):
    """
    Model selector based on cli args
    """
    config_copy = copy.deepcopy(config)
    if config.model_type == "STCNNT_Microscopy":
        model = STCNNT_Microscopy(config=config_copy)
    elif config.model_type == "Microscopy_double_net":
        model = Microscopy_double_net(config=config_copy)
    else:
        raise NotImplementedError(f"Microscopy model not implemented: {config.model_type}")

    return model

# -------------------------------------------------------------------------------------------------
# Microscopy model

class STCNNT_Microscopy(ModelManager):
    """
    STCNNT for Microscopy data
    Just the base STCNNT with care to complex_i and residual
    """
    def __init__(self, config):

        config.height = config.micro_height[-1]
        config.width = config.micro_width[-1]

        super().__init__(config)

        self.complex_i = config.complex_i
        self.residual = config.residual
        self.C_in = config.no_in_channel
        self.C_out = config.no_out_channel

        self.permute = lambda x : torch.permute(x, [0,2,1,3,4])

    def create_pre(self):

        config = self.config

        self.pre_feature_channels = [32]

        if self.config.backbone_model=='Identity':
            self.pre_feature_channels = [32]
        elif self.config.backbone_model=='omnivore' and self.config.omnivore.size == 'tiny':
            self.pre_feature_channels = [32]
        elif self.config.backbone_model=='omnivore' and self.config.omnivore.size == 'small':
            self.pre_feature_channels = [32]
        elif self.config.backbone_model=='omnivore' and self.config.omnivore.size == 'base':
            self.pre_feature_channels = [32]
        elif self.config.backbone_model=='omnivore' and self.config.omnivore.size == 'large':
            self.pre_feature_channels = [32]

        if self.config.backbone_model == "STCNNT_HRNET":
            self.pre_feature_channels = [config.backbone_hrnet.C]

        if self.config.backbone_model == "STCNNT_mUNET":
            self.pre_feature_channels = [config.backbone_mixed_unetr.C]

        if self.config.backbone_model == "STCNNT_UNET":
            self.pre_feature_channels = [config.backbone_unet.C]

        self.pre = nn.ModuleDict()
        self.pre["in_conv"] = Conv2DExt(config.no_in_channel, self.pre_feature_channels[0], kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)

    def create_post(self):

        config = self.config
        self.post = Conv2DExt(self.feature_channels[0], config.no_out_channel, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)

    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image (denoised)
        """
        # input x is [B, C, T, H, W]

        res_pre = self.pre["in_conv"](x)
        B, C, T, H, W = res_pre.shape

        if self.config.backbone_model=="STCNNT_HRNET":
            y_hat, _ = self.backbone(res_pre)
        else:
            y_hat = self.backbone(res_pre)[0]

        if self.residual:
            y_hat[:, :C, :, :, :] = res_pre + y_hat[:, :C, :, :, :]

        # channel first is True here
        # if self.config.super_resolution:
        #     res = self.post["o_upsample"](y_hat)
        #     res = self.post["o_nl"](res)
        #     logits = self.post["o_conv"](res)
        # else:
        logits = self.post(y_hat)

        return logits

# -------------------------------------------------------------------------------------------------
# Microscopy double net model

class Microscopy_double_net(STCNNT_Microscopy):
    """
    Microscopy_double_net
    Using the hrnet backbone, plus a unet post network
    """
    def __init__(self, config):
        assert config.backbone_model == 'STCNNT_HRNET' or config.backbone_model == 'STCNNT_mUNET' or config.backbone_model == 'STCNNT_UNET'
        assert config.post_backbone == 'STCNNT_HRNET' or config.post_backbone == 'STCNNT_mUNET'
        super().__init__(config=config)

    def get_backbone_C_out(self):
        config = self.config
        if config.backbone_model == 'STCNNT_HRNET':
            C = config.backbone_hrnet.C
            backbone_C_out = int(C * sum([np.power(2, k) for k in range(config.backbone_hrnet.num_resolution_levels)]))
        else:
            backbone_C_out = self.feature_channels[0]

        return backbone_C_out

    def create_post(self):

        config = self.config

        backbone_C_out = self.get_backbone_C_out()

        # original post
        self.post_1st = Conv2DExt(backbone_C_out, config.no_out_channel, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)

        self.post = torch.nn.ModuleDict()

        if self.config.super_resolution:
            # self.post["output_ps"] = PixelShuffle2DExt(2)
            # C_out = C_out // 4

            #self.post_2nd["o_upsample"] = UpSample(N=1, C_in=backbone_C_out, C_out=backbone_C_out//2, method='bspline', with_conv=True)
            #self.post_2nd["o_nl"] = nn.GELU(approximate="tanh")
            #self.post_2nd["o_conv"] = Conv2DExt(backbone_C_out//2, backbone_C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

            self.post["1st_upsample"] = UpSample(N=1, C_in=config.no_out_channel, C_out=config.no_out_channel, method='bspline', with_conv=False, channel_first=True)

            self.post["o_upsample"] = UpSample(N=1, C_in=backbone_C_out, C_out=backbone_C_out, method='bspline', with_conv=False, channel_first=True)
            self.post["o_nl"] = nn.GELU(approximate="tanh")
            self.post["o_conv"] = Conv2DExt(backbone_C_out, backbone_C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)

        if config.post_backbone == 'STCNNT_HRNET':
            config_post = copy.deepcopy(config)

            if config.backbone_model != 'STCNNT_HRNET':
                config_post.backbone_hrnet = Nestedspace()
                config_post.backbone_hrnet.num_resolution_levels = len(config.post_hrnet.block_str)
                config_post.backbone_hrnet.use_interpolation = True

            config_post.backbone_hrnet.block_str = config.post_hrnet.block_str
            config_post.separable_conv = config.post_hrnet.separable_conv

            config_post.no_in_channel = backbone_C_out
            config_post.backbone_hrnet.C = backbone_C_out

            if self.config.super_resolution:
                config_post.height *= 2
                config_post.width *= 2

            self.post['post_main'] = STCNNT_HRnet(config=config_post)

            C_out = int(config_post.backbone_hrnet.C * sum([np.power(2, k) for k in range(config_post.backbone_hrnet.num_resolution_levels)]))
        else:
            config_post = copy.deepcopy(config)
            config_post.separable_conv = config.post_mixed_unetr.separable_conv

            config_post.backbone_mixed_unetr.block_str = config.post_mixed_unetr.block_str
            config_post.backbone_mixed_unetr.num_resolution_levels = config.post_mixed_unetr.num_resolution_levels
            config_post.backbone_mixed_unetr.use_unet_attention = config.post_mixed_unetr.use_unet_attention
            config_post.backbone_mixed_unetr.transformer_for_upsampling = config.post_mixed_unetr.transformer_for_upsampling
            config_post.backbone_mixed_unetr.n_heads = config.post_mixed_unetr.n_heads
            config_post.backbone_mixed_unetr.use_conv_3d = config.post_mixed_unetr.use_conv_3d
            config_post.backbone_mixed_unetr.use_window_partition = config.post_mixed_unetr.use_window_partition
            config_post.backbone_mixed_unetr.num_resolution_levels = config.post_mixed_unetr.num_resolution_levels

            config_post.no_in_channel = backbone_C_out
            config_post.backbone_mixed_unetr.C = backbone_C_out

            if self.config.super_resolution:
                config_post.height *= 2
                config_post.width *= 2

            self.post['post_main'] = STCNNT_Mixed_Unetr(config=config_post)

            if config_post.backbone_mixed_unetr.use_window_partition:
                if config_post.backbone_mixed_unetr.encoder_on_input:
                    C_out = config_post.backbone_mixed_unetr.C * 5
                else:
                    C_out = config_post.backbone_mixed_unetr.C * 4
            else:
                C_out = config_post.backbone_mixed_unetr.C * 3


        # if self.config.super_resolution:
        #     # self.post["output_ps"] = PixelShuffle2DExt(2)
        #     # C_out = C_out // 4

        #     self.post["o_upsample"] = UpSample(N=1, C_in=C_out, C_out=C_out//2, method='bspline', with_conv=True)
        #     self.post["o_nl"] = nn.GELU(approximate="tanh")
        #     self.post["o_conv"] = Conv2DExt(C_out//2, C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

        self.post["output_conv"] = Conv2DExt(C_out, config_post.no_out_channel, kernel_size=config_post.kernel_size, stride=config_post.stride, padding=config_post.padding, bias=True, channel_first=True)

    def load_post_1st_net(self, load_path, device=None):
        print(f"{Fore.YELLOW}Loading post in the 1st network from {load_path}{Style.RESET_ALL}")

        if os.path.isfile(load_path):
            status = torch.load(load_path, map_location=self.config.device)
            self.post_1st.load_state_dict(status['post_model_state'])
        else:
            print(f"{Fore.YELLOW}{load_path} does not exist .... {Style.RESET_ALL}")

    def freeze_backbone(self):
        super().freeze_backbone()
        self.post_1st.requires_grad_(False)
        for param in self.post_1st.parameters():
            param.requires_grad = False


    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image
        """
        res_pre = self.pre["in_conv"](x)
        B, C, T, H, W = res_pre.shape

        if self.config.backbone_model == 'STCNNT_HRNET':
            y_hat, _ = self.backbone(res_pre)
        else:
            y_hat = self.backbone(res_pre)[0]

        if self.residual:
            y_hat[:, :C, :, :, :] = res_pre + y_hat[:, :C, :, :, :]

        logits_1st = self.post_1st(y_hat)

        if self.config.super_resolution:
            logits_1st = self.post["1st_upsample"](logits_1st)
            y_hat = self.post["o_upsample"](y_hat)
            y_hat = self.post["o_nl"](y_hat)
            y_hat = self.post["o_conv"](y_hat)

        if self.config.post_backbone == 'STCNNT_HRNET':
            res, _ = self.post['post_main'](y_hat)
        else:
            res = self.post['post_main'](y_hat)

        B, C, T, H, W = y_hat.shape
        if self.residual:
            res[:, :C, :, :, :] = res[:, :C, :, :, :] + y_hat

        logits = self.post["output_conv"](res)

        return logits, logits_1st
