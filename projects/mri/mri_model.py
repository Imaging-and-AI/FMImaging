
"""
MRI models

- STCNNT_MRI: the pre-backbone-post model with a simple pre and post module
- MRI_hrnet: a hrnet backbone + a hrnet post
- MRI_double_net: a hrnet or mixed_unet backbone + a hrnet or mixed_unet post
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
import torchvision

from model.model_base import ModelManager
from model.backbone import identity_model, omnivore_tiny, omnivore_small, omnivore_base, omnivore_large, STCNNT_HRnet_model, STCNNT_Unet_model, STCNNT_Mixed_Unetr_model
from model.backbone import STCNNT_HRnet, STCNNT_Mixed_Unetr, UpSample, set_window_patch_sizes_keep_num_window, set_window_patch_sizes_keep_window_size, STCNNT_Block
from model.imaging_attention import *
from model.task_heads import *

from setup import get_device, Nestedspace
from utils import model_info, get_gpu_ram_usage, start_timer, end_timer
from optim.optim_utils import divide_optim_into_groups

# input to the MRI models are channel-first, [B, C, T, H, W]
# output from the MRI models are channel-first, [B, C, T, H, W]
# inside the model, permutes are used as less as possible

# -------------------------------------------------------------------------------------------------

def create_model(config, model_type):
    config_copy = copy.deepcopy(config)
    if model_type == "STCNNT_MRI":
        model = STCNNT_MRI(config=config_copy)
    elif model_type == "MRI_hrnet":
        model = MRI_hrnet(config=config_copy)
    elif model_type == "omnivore_MRI":
        model = omnivore_MRI(config=config_copy)
    else:
        model = MRI_double_net(config=config_copy)

    return model

# -------------------------------------------------------------------------------------------------
# MRI model

class STCNNT_MRI(ModelManager):
    """
    STCNNT for MRI data
    Just the base CNNT with care to complex_i and residual
    """
    def __init__(self, config):

        config.height = config.mri_height[-1]
        config.width = config.mri_width[-1]

        super().__init__(config)

        self.complex_i = config.complex_i
        self.residual = config.residual
        self.C_in = config.no_in_channel
        self.C_out = config.no_out_channel

        self.permute = lambda x : torch.permute(x, [0,2,1,3,4])

        print(f"{Fore.BLUE}{Back.WHITE}===> MRI - done <==={Style.RESET_ALL}")

    def create_pre(self):

        config = self.config

        self.pre_feature_channels = [32]

        if self.config.backbone_model=='Identity':
            self.pre_feature_channels = [32]
        elif self.config.backbone_model=='omnivore_tiny':
            self.pre_feature_channels = [32]
        elif self.config.backbone_model=='omnivore_small':
            self.pre_feature_channels = [32]
        elif self.config.backbone_model=='omnivore_base':
            self.pre_feature_channels = [32]
        elif self.config.backbone_model=='omnivore_large':
            self.pre_feature_channels = [32]

        if self.config.backbone_model == "STCNNT_HRNET":
            self.pre_feature_channels = [config.backbone_hrnet.C]
            
        if self.config.backbone_model == "STCNNT_mUNET":
            self.pre_feature_channels = [config.backbone_mixed_unetr.C]
            
        if self.config.backbone_model == "STCNNT_UNET":
            self.pre_feature_channels = [config.backbone_unet.C]

        self.pre = nn.ModuleDict()
        self.pre["in_conv"] = Conv2DExt(config.no_in_channel, self.pre_feature_channels[0], kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)
        self.paras = torch.nn.ParameterDict()
        self.paras["a"] = torch.nn.Parameter(torch.tensor(5.0))
        self.paras["b"] = torch.nn.Parameter(torch.tensor(4.0))
        self.pre["paras"] = self.paras

    def create_post(self):

        config = self.config

        if self.config.super_resolution:
            self.post = nn.ModuleDict()
            #self.post.add_module("post_ps", PixelShuffle2DExt(2))
            #self.post.add_module("post_conv", Conv2DExt(self.feature_channels//4, config.no_out_channel, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True))
            self.post["o_upsample"] = UpSample(N=1, C_in=self.feature_channels[0], C_out=self.feature_channels[0]//2, method='bspline', with_conv=True, channel_first=True)
            self.post["o_nl"] = nn.GELU(approximate="tanh")
            self.post["o_conv"] = Conv2DExt(self.feature_channels[0]//2, config.no_out_channel, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)
        else:
            self.post = Conv2DExt(self.feature_channels[0], config.no_out_channel, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)


    def forward(self, x, snr=None, base_snr_t=None):
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
        if self.config.super_resolution:
            res = self.post["o_upsample"](y_hat)
            res = self.post["o_nl"](res)
            logits = self.post["o_conv"](res)
        else:
            logits = self.post(y_hat)

        if base_snr_t is not None:
            weights = self.compute_weights(snr=snr, base_snr_t=base_snr_t)
            return logits, weights
        else:
            return logits

    def compute_weights(self, snr, base_snr_t):
        weights = self.pre["paras"]["a"] - self.pre["paras"]["b"] * torch.sigmoid(snr-base_snr_t)
        return weights

# -------------------------------------------------------------------------------------------------
# MRI model

class omnivore_MRI(STCNNT_MRI):
    """
    omnivore for MRI data
    Just the base CNNT with care to complex_i and residual
    """
    def __init__(self, config):
        super().__init__(config)

    def create_backbone(self):

        self.backbone = nn.ModuleDict()

        if self.config.backbone_model=='omnivore_tiny':
            self.omnivore = omnivore_tiny(self.config, self.pre_feature_channels)
            self.backbone["omnivore"] = self.omnivore
            self.feature_channels = self.omnivore.feature_channels
        elif self.config.backbone_model=='omnivore_small':
            self.omnivore = omnivore_small(self.config, self.pre_feature_channels)
            self.backbone["omnivore"] = self.omnivore
            self.feature_channels = self.omnivore.feature_channels
        elif self.config.backbone_model=='omnivore_base':
            self.omnivore = omnivore_base(self.config, self.pre_feature_channels)
            self.backbone["omnivore"] = self.omnivore
            self.feature_channels = self.omnivore.feature_channels
        elif self.config.backbone_model=='omnivore_large':
            self.omnivore = omnivore_large(self.config, self.pre_feature_channels)
            self.backbone["omnivore"] = self.omnivore
            self.feature_channels = self.omnivore.feature_channels
        else:
            raise NotImplementedError(f"Backbone model not implemented: {self.config.backbone_model}")

        config = copy.deepcopy(self.config)
        config.no_out_channel = self.feature_channels[0]
        self.backbone["mixer"] = UperNet3D(config, feature_channels=self.feature_channels)

        # # upsample 3D along T, H, W
        # self.backbone["upsample_3d"] = UpSample(N=1, C_in=self.feature_channels[0], C_out=self.feature_channels[0], method='linear', with_conv=True, is_3D=True)

        # self.backbone["o_nl"] = nn.GELU(approximate="tanh")

        # # upsample 2D, along H, W
        # self.backbone["upsample_2d"] = UpSample(N=2, C_in=self.feature_channels[0], C_out=self.feature_channels[0], method='linear', with_conv=True, is_3D=False)

    def create_post(self):
        config = self.config
        self.post = Conv2DExt(self.feature_channels[0], config.no_out_channel, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)

    def forward(self, x, snr=None, base_snr_t=None):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image (denoised)
        """
        res_pre = self.pre["in_conv"](x)
        B, C, T, H, W = res_pre.shape
        
        res_backbone = self.backbone["omnivore"](res_pre)
        y_hat = self.backbone["mixer"](res_backbone, output_size=(T, H, W))

        if self.residual:
            y_hat[:, :C, :, :, :] = res_pre + y_hat[:, :C, :, :, :]

        logits = self.post(y_hat) # + x[:, :, :self.config.no_out_channel, :, :]

        if base_snr_t is not None:
            weights = self.compute_weights(snr=snr, base_snr_t=base_snr_t)
            return logits, weights
        else:
            return logits

# -------------------------------------------------------------------------------------------------
# MRI hrnet model

class MRI_hrnet(STCNNT_MRI):
    """
    MR hrnet
    Using the hrnet backbone, plus a unet type post module
    """
    def __init__(self, config):
        assert config.backbone_model == 'STCNNT_HRNET'
        super().__init__(config=config)

    def create_post(self):

        config = self.config
        assert config.backbone_hrnet.num_resolution_levels >= 1 and config.backbone_hrnet.num_resolution_levels<= 4

        c = config

        self.post = torch.nn.ModuleDict()

        self.num_wind = [c.height//c.window_size[0], c.width//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]

        C = config.backbone_hrnet.C

        kwargs = {
            "C_in": C,
            "C_out": C,
            "H":c.height,
            "W":c.width,
            "a_type":c.a_type,
            "window_size": c.window_size,
            "patch_size": c.patch_size,
            "is_causal":c.is_causal,
            "dropout_p":c.dropout_p,
            "n_head":c.n_head,

            "kernel_size":(c.kernel_size, c.kernel_size),
            "stride":(c.stride, c.stride),
            "padding":(c.padding, c.padding),

            "stride_s": (c.stride_s, c.stride_s),
            "stride_t":(c.stride_t, c.stride_t),

            "separable_conv": c.post_hrnet.separable_conv,


            "mixer_kernel_size":(c.mixer_kernel_size, c.mixer_kernel_size),
            "mixer_stride":(c.mixer_stride, c.mixer_stride),
            "mixer_padding":(c.mixer_padding, c.mixer_padding),

            "norm_mode":c.norm_mode,
            "interpolate":"none",
            "interp_align_c":c.interp_align_c,

            "cell_type": c.cell_type,
            "normalize_Q_K": c.normalize_Q_K, 
            "att_dropout_p": c.att_dropout_p,
            "att_with_output_proj": c.att_with_output_proj, 
            "scale_ratio_in_mixer": c.scale_ratio_in_mixer,
            "cosine_att": c.cosine_att,
            "att_with_relative_postion_bias": c.att_with_relative_postion_bias,
            "block_dense_connection": c.block_dense_connection,

            "num_wind": self.num_wind,
            "num_patch": self.num_patch,

            "mixer_type": c.mixer_type,
            "shuffle_in_window": c.shuffle_in_window,

            "use_einsum": c.use_einsum,
            "temporal_flash_attention": c.temporal_flash_attention
        }

        self.block_str = c.post_hrnet.block_str if len(c.post_hrnet.block_str)>=config.backbone_hrnet.num_resolution_levels else [c.post_hrnet.block_str[0] for n in range(config.backbone_hrnet.num_resolution_levels)]

        self.num_wind = [c.height//c.window_size[0], c.width//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]

        window_sizes = []
        patch_sizes = []

        if config.backbone_hrnet.num_resolution_levels == 1:

            kwargs["C_in"] = C
            kwargs["C_out"] = C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            if c.window_sizing_method == "keep_num_window":
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="P0")
            elif c.window_sizing_method == "keep_window_size":
                kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P0")
            else: # mixed
                kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]] , self.num_wind, self.num_patch, module_name="P0")

            window_sizes.append(kwargs["window_size"])
            patch_sizes.append(kwargs["patch_size"])

            kwargs["att_types"] = self.block_str[0]
            self.post["P0"] = STCNNT_Block(**kwargs)

            hrnet_C_out = 2*C

        if config.backbone_hrnet.num_resolution_levels == 2:
            kwargs["C_in"] = 2*C
            kwargs["C_out"] = 2*C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2

            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="P1")

            kwargs["att_types"] = self.block_str[0]
            self.post["P1"] = STCNNT_Block(**kwargs)

            self.post["up_1_0"] = UpSample(N=1, C_in=4*C, C_out=4*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 4*C
            kwargs["C_out"] = 2*C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P0")

            kwargs["att_types"] = self.block_str[1]
            self.post["P0"] = STCNNT_Block(**kwargs)
            # -----------------------------------------
            hrnet_C_out = 3*C

        if config.backbone_hrnet.num_resolution_levels == 3:
            kwargs["C_in"] = 4*C
            kwargs["C_out"] = 4*C
            kwargs["H"] = c.height // 4
            kwargs["W"] = c.width // 4

            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="P2")

            kwargs["att_types"] = self.block_str[0]
            self.post["P2"] = STCNNT_Block(**kwargs)

            self.post["up_2_1"] = UpSample(N=1, C_in=8*C, C_out=8*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 8*C
            kwargs["C_out"] = 4*C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2

            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P1")

            kwargs["att_types"] = self.block_str[1]
            self.post["P1"] = STCNNT_Block(**kwargs)

            self.post["up_1_0"] = UpSample(N=1, C_in=6*C, C_out=6*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 6*C
            kwargs["C_out"] = 3*C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], kwargs["window_size"], kwargs["patch_size"], module_name="P0")

            kwargs["att_types"] = self.block_str[2]
            self.post["P0"] = STCNNT_Block(**kwargs)
            # -----------------------------------------
            hrnet_C_out = 4*C

        if config.backbone_hrnet.num_resolution_levels == 4:
            kwargs["C_in"] = 8*C
            kwargs["C_out"] = 8*C
            kwargs["H"] = c.height // 8
            kwargs["W"] = c.width // 8

            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="P3")

            kwargs["att_types"] = self.block_str[0]
            self.post["P3"] = STCNNT_Block(**kwargs)

            self.post["up_3_2"] = UpSample(N=1, C_in=16*C, C_out=16*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 16*C
            kwargs["C_out"] = 8*C
            kwargs["H"] = c.height // 4
            kwargs["W"] = c.width // 4

            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P2")

            kwargs["att_types"] = self.block_str[1]
            self.post["P2"] = STCNNT_Block(**kwargs)

            self.post["up_2_1"] = UpSample(N=1, C_in=12*C, C_out=12*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 12*C
            kwargs["C_out"] = 6*C
            kwargs["H"] = c.height // 2
            kwargs["W"] = c.width // 2
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], kwargs["window_size"], kwargs["patch_size"], module_name="P1")

            kwargs["att_types"] = self.block_str[2]
            self.post["P1"] = STCNNT_Block(**kwargs)

            self.post["up_1_0"] = UpSample(N=1, C_in=8*C, C_out=8*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 8*C
            kwargs["C_out"] = 4*C
            kwargs["H"] = c.height
            kwargs["W"] = c.width
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P0")

            kwargs["att_types"] = self.block_str[3]
            self.post["P0"] = STCNNT_Block(**kwargs)
            # -----------------------------------------
            hrnet_C_out = 5*C

        if self.config.super_resolution:
            self.post["o_upsample"] = UpSample(N=1, C_in=hrnet_C_out, C_out=hrnet_C_out//2, method='bspline', with_conv=True, channel_first=True)
            self.post["o_nl"] = nn.GELU(approximate="tanh")
            self.post["o_conv"] = Conv2DExt(hrnet_C_out//2, hrnet_C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)
            # self.post["output_ps"] = PixelShuffle2DExt(2)
            # hrnet_C_out = hrnet_C_out // 4
            # self.post["o_conv"] = Conv2DExt(hrnet_C_out, 4*hrnet_C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)
            # hrnet_C_out = 4*hrnet_C_out

        self.post["output_conv"] = Conv2DExt(hrnet_C_out, config.no_out_channel, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True, channel_first=True)

    def forward(self, x, snr=-1, base_snr_t=None):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image
        """
        res_pre = self.pre["in_conv"](x)
        res_backbone = self.backbone(res_pre)
        
        if self.residual:
            res_backbone[1][0] = res_pre + res_backbone[1][0]

        res_backbone[1] = [self.permute(a) for a in res_backbone[1]] # from B, C, T, H, W to B, T, C, H, W

        # now, channel_first is False
        num_resolution_levels = self.config.backbone_hrnet.num_resolution_levels
        if num_resolution_levels == 1:
            res_0, _ = self.post["P0"](res_backbone[1][0])
            res = torch.cat((res_0, res_backbone[1][0]), dim=2)

        elif num_resolution_levels == 2:
            res_1, _ = self.post["P1"](res_backbone[1][1])
            res_1 = torch.cat((res_1, res_backbone[1][1]), dim=2)
            res_1 = self.post["up_1_0"](res_1)

            res_0, _ = self.post["P0"](res_1)
            res = torch.cat((res_0, res_backbone[1][0]), dim=2)

        elif num_resolution_levels == 3:

            res_2, _ = self.post["P2"](res_backbone[1][2])
            res_2 = torch.cat((res_2, res_backbone[1][2]), dim=2)
            res_2 = self.post["up_2_1"](res_2)

            res_1, _ = self.post["P1"](res_2)
            res_1 = torch.cat((res_1, res_backbone[1][1]), dim=2)
            res_1 = self.post["up_1_0"](res_1)

            res_0, _ = self.post["P0"](res_1)
            res = torch.cat((res_1, res_backbone[1][0]), dim=2)

        elif num_resolution_levels == 4:

            res_3, _ = self.post["P3"](res_backbone[1][3])
            res_3 = torch.cat((res_3, res_backbone[1][3]), dim=2)
            res_3 = self.post["up_3_2"](res_3)

            res_2, _ = self.post["P2"](res_3)
            res_2 = torch.cat((res_2, res_backbone[1][2]), dim=2)
            res_2 = self.post["up_2_1"](res_2)

            res_1, _ = self.post["P1"](res_2)
            res_1 = torch.cat((res_1, res_backbone[1][1]), dim=2)
            res_1 = self.post["up_1_0"](res_1)

            res_0, _ = self.post["P0"](res_1)
            res = torch.cat((res_1, res_backbone[1][0]), dim=2)

        res = self.permute(res) # go back to channel first

        # res = self.post["output"](res)
        if self.config.super_resolution:
            #res = self.post["output_ps"](res)
            res = self.post["o_upsample"](res)
            res = self.post["o_nl"](res)
            res = self.post["o_conv"](res)

        logits = self.post["output_conv"](res)

        if base_snr_t is not None:
            weights = self.compute_weights(snr=snr, base_snr_t=base_snr_t)
            return logits, weights
        else:
            return logits

# -------------------------------------------------------------------------------------------------
# MRI double net model

class MRI_double_net(STCNNT_MRI):
    """
    MRI_double_net
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


    def forward(self, x, snr=-1, base_snr_t=None):
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

        if base_snr_t is not None:
            weights = self.compute_weights(snr=snr, base_snr_t=base_snr_t)
            return logits, weights, logits_1st
        else:
            return logits, logits_1st