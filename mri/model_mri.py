"""
Model(s) used for MRI
"""
import os
import sys
import logging
import copy
import abc 
from abc import ABC
from colorama import Fore, Back, Style

import torch
import torch.nn as nn

from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.imaging_attention import *
from model_base.backbone import *
from model_base.backbone.backbone_small_unet import *

from utils import get_device, create_generic_class_str, optimizer_to

from model_base.task_base import *
from model_base.losses import *

# -------------------------------------------------------------------------------------------------
# MRI model

class STCNNT_MRI(STCNNT_Task_Base):
    """
    STCNNT for MRI data
    Just the base CNNT with care to complex_i and residual
    """
    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
            
        Task specific args:
        
            - losses: 
            - loss_weights:
            - complex_i:
        """
        super().__init__(config=config)

        self.complex_i = config.complex_i
        self.residual = config.residual
        self.C_in = config.C_in
        self.C_out = config.C_out

        print(f"{Fore.YELLOW}{Back.WHITE}===> MRI - create pre <==={Style.RESET_ALL}")
        self.create_pre()

        print(f"{Fore.GREEN}{Back.WHITE}===> MRI - backbone <==={Style.RESET_ALL}")
        if config.backbone == "small_unet":
            self.backbone = CNNT_Unet(config=config)

        if config.backbone == "hrnet":
            config.C_in = config.backbone_hrnet.C
            self.backbone = STCNNT_HRnet(config=config)
            config.C_in = self.C_in

        if config.backbone == "unet":
            self.backbone = STCNNT_Unet(config=config)

        if config.backbone == "mixed_unetr":
            config.C_in = config.backbone_mixed_unetr.C
            self.backbone = STCNNT_Mixed_Unetr(config=config)
            config.C_in = self.C_in
            
        if config.backbone == "LLM":
            self.backbone = STCNNT_LLMnet(config=config) 

        print(f"{Fore.RED}{Back.WHITE}===> MRI - post <==={Style.RESET_ALL}")
        self.create_post()

        # if use weighted loss
        self.a = torch.nn.Parameter(torch.tensor(5.0))
        self.b = torch.nn.Parameter(torch.tensor(4.0))

        device = get_device(device=config.device)
        self.set_up_loss(device=device)

        self.set_up_optim_and_scheduling(total_steps=total_steps)

        if config.load_path is not None:
            self.load(load_path=config.load_path, device=device)

        print(f"{Fore.BLUE}{Back.WHITE}===> MRI - done <==={Style.RESET_ALL}")

    def create_pre(self):

        config = self.config

        if self.config.backbone == "small_unet":
            self.pre = nn.Identity()

        if self.config.backbone == "hrnet":
            self.pre = Conv2DExt(config.C_in, config.backbone_hrnet.C, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

        if self.config.backbone == "mixed_unetr":
            self.pre = Conv2DExt(config.C_in, config.backbone_mixed_unetr.C, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)
            
        if self.config.backbone == "unet":
            self.pre = nn.Identity()

        if self.config.backbone == "LLM":
            self.pre = nn.Identity()


    def create_post(self):

        config = self.config

        if self.config.backbone == "small_unet":
            self.pre = nn.Identity()

        if self.config.backbone == "hrnet":
            hrnet_C_out = int(config.backbone_hrnet.C * sum([np.power(2, k) for k in range(config.backbone_hrnet.num_resolution_levels)]))
            if self.config.super_resolution:
                self.post = nn.ModuleDict()
                #self.post.add_module("post_ps", PixelShuffle2DExt(2))
                #self.post.add_module("post_conv", Conv2DExt(hrnet_C_out//4, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True))
                self.post["o_upsample"] = UpSample(N=1, C_in=hrnet_C_out, C_out=hrnet_C_out//2, method='bspline', with_conv=True)
                self.post["o_nl"] = nn.GELU(approximate="tanh")
                self.post["o_conv"] = Conv2DExt(hrnet_C_out//2, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)
            else:
                self.post = Conv2DExt(hrnet_C_out, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

        if self.config.backbone == "mixed_unetr":
            if config.backbone_mixed_unetr.use_window_partition:
                if config.backbone_mixed_unetr.encoder_on_input:
                    mixed_unetr_C_out = config.backbone_mixed_unetr.C * 5
                else:
                    mixed_unetr_C_out = config.backbone_mixed_unetr.C * 4
            else:
                mixed_unetr_C_out = config.backbone_mixed_unetr.C * 3

            if self.config.super_resolution:
                self.post = nn.ModuleDict()
                #self.post.add_module("post_ps", PixelShuffle2DExt(2))
                #self.post.add_module("post_conv", Conv2DExt(mixed_unetr_C_out//4, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True))

                self.post["o_upsample"] = UpSample(N=1, C_in=mixed_unetr_C_out, C_out=mixed_unetr_C_out//2, method='bspline', with_conv=True)
                self.post["o_nl"] = nn.GELU(approximate="tanh")
                self.post["o_conv"] = Conv2DExt(mixed_unetr_C_out//2, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)
            else:
                self.post = Conv2DExt(mixed_unetr_C_out, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

        if config.backbone == "unet":
            self.post = Conv2DExt(config.backbone_unet.C, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

        if config.backbone == "LLM":
            output_C = int(np.power(2, config.backbone_LLM.num_stages-2)) if config.backbone_LLM.num_stages>2 else config.backbone_LLM.C
            self.post = Conv2DExt(output_C,config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)


    def forward(self, x, snr=None, base_snr_t=None):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image (denoised)
        """
        res_pre = self.pre(x)

        B, T, C, H, W = res_pre.shape

        if self.config.backbone=="hrnet" or self.config.backbone=="mixed_unetr":

            if self.config.backbone=="hrnet":
                y_hat, _ = self.backbone(res_pre)
            else:
                y_hat = self.backbone(res_pre)

            if self.residual:
                y_hat[:,:, :C, :, :] = res_pre + y_hat[:,:, :C, :, :]

            if self.config.super_resolution:
                #res = self.post["post_ps"](y_hat)
                #logits = self.post["post_conv"](res)
                res = self.post["o_upsample"](y_hat)
                res = self.post["o_nl"](res)
                logits = self.post["o_conv"](res)
            else:
                logits = self.post(y_hat)

        else:
            res_backbone = self.backbone(res_pre)
            logits = self.post(res_backbone)

            if self.residual:
                C = 2 if self.complex_i else 1
                logits = x[:,:,:C] - logits

        if base_snr_t is not None:
            weights = self.compute_weights(snr=snr, base_snr_t=base_snr_t)
            return logits, weights, None
        else:
            return logits, None

    def compute_weights(self, snr, base_snr_t):
        weights = self.a - self.b * torch.sigmoid(snr-base_snr_t)
        return weights

    def set_up_loss(self, device="cpu"):
        """
        Sets up the combined loss
        @args:
            - device (torch.device): device to setup the loss on
        @args (from config):
            - losses (list of "ssim", "ssim3D", "l1", "mse", "psnr"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether we are dealing with complex images or not
        """
        self.loss_f = Combined_Loss(self.config.losses, self.config.loss_weights,
                                    complex_i=self.config.complex_i, device=device)


    def configure_optim_groups_lr(self, module, lr=-1):
        """
        @args:
            - module (torch module): module to create parameter groups for
            - lr (float): learning rate for the module m
            - weight_decay (float, from config): weight decay coefficient for regularization
        """
        decay, no_decay, param_dict = self.get_decay_no_decay_para_groups(module=module)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        if lr >= 0:
            optim_groups[0]['lr'] = lr
            optim_groups[1]['lr'] = lr

        return optim_groups

    def configure_optim_groups(self):
        """set optim groups for pre/post/backbone
        """

        pre_optim_groups = self.configure_optim_groups_lr(self.pre, lr=self.config.lr_pre)
        backbone_optim_groups = self.configure_optim_groups_lr(self.backbone, lr=self.config.lr_backbone)
        post_optim_groups = self.configure_optim_groups_lr(self.post, lr=self.config.lr_post)

        optim_groups = []
        optim_groups.extend(pre_optim_groups)
        optim_groups.extend(backbone_optim_groups)
        optim_groups.extend(post_optim_groups)

        return optim_groups

    def check_model_learnable_status(self, rank_str=""):
        num = 0
        num_learnable = 0
        for param in self.pre.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        print(f"{rank_str} model, pre, learnable tensors {num_learnable} out of {num} ...")

        num = 0
        num_learnable = 0
        for param in self.backbone.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        print(f"{rank_str} model, backbone, learnable tensors {num_learnable} out of {num} ...")

        num = 0
        num_learnable = 0
        for param in self.post.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        print(f"{rank_str} model, post, learnable tensors {num_learnable} out of {num} ...")

    def save(self, epoch, only_paras=False, save_file_name=None):
        """
        Save model checkpoints
        @args:
            - epoch (int): current epoch of the training cycle
        @args (from config):
            - date (datetime str): runtime date
            - checkpath (str): directory to save checkpoint in
        """
        if save_file_name is None:
            run_name = self.config.run_name.replace(" ", "_")
            save_file_name = f"{run_name}_epoch-{epoch}.pth"
            
        save_path = os.path.join(self.config.check_path, save_file_name)
        logging.info(f"{Fore.YELLOW}Saving model status at {save_path}{Style.RESET_ALL}")
        self.save_to_file(epoch, save_path, only_paras)
        return save_path

    def save_to_file(self, epoch, save_path, only_paras):
        if only_paras:
                torch.save({
                "epoch":epoch,
                "config": self.config,
                "pre_state": self.pre.state_dict(), 
                "backbone_state": self.backbone.state_dict(), 
                "post_state": self.post.state_dict(), 
                "a": self.a,
                "b": self.b
            }, save_path)
        else:
            torch.save({
                "epoch":epoch,
                "pre_state": self.pre.state_dict(), 
                "backbone_state": self.backbone.state_dict(), 
                "post_state": self.post.state_dict(), 
                "a": self.a,
                "b": self.b,
                "optimizer_state": self.optim.state_dict(), 
                "scheduler_state": self.sched.state_dict(),
                "config": self.config,
                "scheduler_type":self.stype
            }, save_path)

    def load_from_status(self, status, device=None, load_others=True):
        """
        Load the model from status; the config will not be updated
        @args:
            - status (dict): dict to hold model parameters etc.
        """

        if 'backbone_state' in status:
            self.pre.load_state_dict(status['pre_state'])
            self.backbone.load_state_dict(status['backbone_state'])
            self.post.load_state_dict(status['post_state'])
            self.a = status['a']
            self.b = status['b']
        else:
            self.load_state_dict(status['model'])

        if load_others:
            if 'optimizer_state' in status:
                self.optim.load_state_dict(status['optimizer_state'])
                if device is not None: optimizer_to(self.optim, device=device)

            if 'scheduler_state' in status:
                self.sched.load_state_dict(status['scheduler_state'])

            if 'scheduler_type' in status:
                self.stype = status['scheduler_type']

            if 'epoch' in status:
                self.curr_epoch = status['epoch']

    def load(self, load_path, device=None):
        """
        Load a checkpoint from the load path
        @args:
            - device (torch.device): device to setup the model on
        @args (from config):
            - load_path (str): path to load the weights from
        """
        logging.info(f"{Fore.YELLOW}Loading model from {load_path}{Style.RESET_ALL}")

        if os.path.isfile(load_path):
            status = torch.load(self.config.load_path)

            self.config = status['config']

            self.load_from_status(status, device=device, load_others=True)
                
            if device is not None:
                self.to(device=device)
        else:
            logging.warning(f"{Fore.YELLOW}{load_path} does not exist .... {Style.RESET_ALL}")

    def load_pre(self, status):
        self.pre.load_state_dict(status['pre_state'])

    def load_backbone(self, status):
        self.backbone.load_state_dict(status['backbone_state'])

    def load_post(self, status):
        self.post.load_state_dict(status['post_state'])

    def disable_pre(self):
        self.pre.requires_grad_(False)
        for param in self.pre.parameters():
            param.requires_grad = False

    def disable_backbone(self):
        self.backbone.requires_grad_(False)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def disable_post(self):
        self.post.requires_grad_(False)
        for param in self.post.parameters():
            param.requires_grad = False

# -------------------------------------------------------------------------------------------------
# MRI model with loading backbone

class MRI_hrnet(STCNNT_MRI):
    """
    MR hrnet
    Using the hrnet backbone, plus a unet type post module
    """
    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
        """
        assert config.backbone == 'hrnet'
        super().__init__(config=config, total_steps=1)

    def create_post(self):

        config = self.config
        assert config.backbone_hrnet.num_resolution_levels >= 1 and config.backbone_hrnet.num_resolution_levels<= 4

        c = config

        self.post = torch.nn.ModuleDict()

        self.num_wind = [c.height[0]//c.window_size[0], c.width[0]//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]

        C = config.backbone_hrnet.C

        kwargs = {
            "C_in": C,
            "C_out": C,
            "H":c.height[0],
            "W":c.width[0],
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

        self.num_wind = [c.height[0]//c.window_size[0], c.width[0]//c.window_size[1]]
        self.num_patch = [c.window_size[0]//c.patch_size[0], c.window_size[1]//c.patch_size[1]]

        window_sizes = []
        patch_sizes = []

        if config.backbone_hrnet.num_resolution_levels == 1:

            kwargs["C_in"] = C
            kwargs["C_out"] = C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
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
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2

            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="P1")

            kwargs["att_types"] = self.block_str[0]
            self.post["P1"] = STCNNT_Block(**kwargs)

            self.post["up_1_0"] = UpSample(N=1, C_in=4*C, C_out=4*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 4*C
            kwargs["C_out"] = 2*C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P0")

            kwargs["att_types"] = self.block_str[1]
            self.post["P0"] = STCNNT_Block(**kwargs)
            # -----------------------------------------
            hrnet_C_out = 3*C

        if config.backbone_hrnet.num_resolution_levels == 3:
            kwargs["C_in"] = 4*C
            kwargs["C_out"] = 4*C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4

            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="P2")

            kwargs["att_types"] = self.block_str[0]
            self.post["P2"] = STCNNT_Block(**kwargs)

            self.post["up_2_1"] = UpSample(N=1, C_in=8*C, C_out=8*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 8*C
            kwargs["C_out"] = 4*C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2

            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P1")

            kwargs["att_types"] = self.block_str[1]
            self.post["P1"] = STCNNT_Block(**kwargs)

            self.post["up_1_0"] = UpSample(N=1, C_in=6*C, C_out=6*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 6*C
            kwargs["C_out"] = 3*C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], kwargs["window_size"], kwargs["patch_size"], module_name="P0")

            kwargs["att_types"] = self.block_str[2]
            self.post["P0"] = STCNNT_Block(**kwargs)
            # -----------------------------------------
            hrnet_C_out = 4*C

        if config.backbone_hrnet.num_resolution_levels == 4:
            kwargs["C_in"] = 8*C
            kwargs["C_out"] = 8*C
            kwargs["H"] = c.height[0] // 8
            kwargs["W"] = c.width[0] // 8

            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], self.num_wind, self.num_patch, module_name="P3")

            kwargs["att_types"] = self.block_str[0]
            self.post["P3"] = STCNNT_Block(**kwargs)

            self.post["up_3_2"] = UpSample(N=1, C_in=16*C, C_out=16*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 16*C
            kwargs["C_out"] = 8*C
            kwargs["H"] = c.height[0] // 4
            kwargs["W"] = c.width[0] // 4

            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P2")

            kwargs["att_types"] = self.block_str[1]
            self.post["P2"] = STCNNT_Block(**kwargs)

            self.post["up_2_1"] = UpSample(N=1, C_in=12*C, C_out=12*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 12*C
            kwargs["C_out"] = 6*C
            kwargs["H"] = c.height[0] // 2
            kwargs["W"] = c.width[0] // 2
            kwargs = set_window_patch_sizes_keep_num_window(kwargs, [kwargs["H"],kwargs["W"]], kwargs["window_size"], kwargs["patch_size"], module_name="P1")

            kwargs["att_types"] = self.block_str[2]
            self.post["P1"] = STCNNT_Block(**kwargs)

            self.post["up_1_0"] = UpSample(N=1, C_in=8*C, C_out=8*C, with_conv=True)
            # -----------------------------------------
            kwargs["C_in"] = 8*C
            kwargs["C_out"] = 4*C
            kwargs["H"] = c.height[0]
            kwargs["W"] = c.width[0]
            kwargs = set_window_patch_sizes_keep_window_size(kwargs, [kwargs["H"],kwargs["W"]], c.window_size, c.patch_size, module_name="P0")

            kwargs["att_types"] = self.block_str[3]
            self.post["P0"] = STCNNT_Block(**kwargs)
            # -----------------------------------------
            hrnet_C_out = 5*C

        if self.config.super_resolution:
            self.post["o_upsample"] = UpSample(N=1, C_in=hrnet_C_out, C_out=hrnet_C_out//2, method='bspline', with_conv=True)
            self.post["o_nl"] = nn.GELU(approximate="tanh")
            self.post["o_conv"] = Conv2DExt(hrnet_C_out//2, hrnet_C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)
            # self.post["output_ps"] = PixelShuffle2DExt(2)
            # hrnet_C_out = hrnet_C_out // 4
            # self.post["o_conv"] = Conv2DExt(hrnet_C_out, 4*hrnet_C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)
            # hrnet_C_out = 4*hrnet_C_out

        self.post["output_conv"] = Conv2DExt(hrnet_C_out, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)


    def forward(self, x, snr=-1, base_snr_t=None):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image
        """
        res_pre = self.pre(x)
        res_backbone = self.backbone(res_pre)

        if self.residual:
            res_backbone[1][0] = res_pre + res_backbone[1][0]

        # if self.residual:
        #     C = 2 if self.complex_i else 1
        #     res_backbone[1][0] = x[:,:,:C] - res_backbone[1][0]

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

        # res = self.post["output"](res)
        if self.config.super_resolution:
            #res = self.post["output_ps"](res)
            res = self.post["o_upsample"](res)
            res = self.post["o_nl"](res)
            res = self.post["o_conv"](res)

        logits = self.post["output_conv"](res)

        # if self.residual:
        #     C = 2 if self.complex_i else 1
        #     logits = x[:,:,:C] - logits

        if base_snr_t is not None:
            weights = self.compute_weights(snr=snr, base_snr_t=base_snr_t)
            return logits, weights, None
        else:
            return logits, None

# -------------------------------------------------------------------------------------------------
# MRI model with loading backbone, double network

class MRI_double_net(STCNNT_MRI):
    """
    MRI_double_net
    Using the hrnet backbone, plus a unet post network
    """
    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR
        """
        assert config.backbone == 'hrnet' or config.backbone == 'mixed_unetr'
        assert config.post_backbone == 'hrnet' or config.post_backbone == 'mixed_unetr'
        super().__init__(config=config, total_steps=1)

    def get_backbone_C_out(self):
        config = self.config
        if config.backbone == 'hrnet':
            C = config.backbone_hrnet.C
            backbone_C_out = int(C * sum([np.power(2, k) for k in range(config.backbone_hrnet.num_resolution_levels)]))
        else:
            C = config.backbone_mixed_unetr.C

            if config.backbone_mixed_unetr.use_window_partition:
                if config.backbone_mixed_unetr.encoder_on_input:
                    backbone_C_out = config.backbone_mixed_unetr.C * 5
                else:
                    backbone_C_out = config.backbone_mixed_unetr.C * 4
            else:
                backbone_C_out = config.backbone_mixed_unetr.C * 3

        return backbone_C_out

    def create_post(self):

        config = self.config

        backbone_C_out = self.get_backbone_C_out()

        # original post
        self.post_1st = Conv2DExt(backbone_C_out, config.C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

        self.post = torch.nn.ModuleDict()

        if self.config.super_resolution:
            # self.post["output_ps"] = PixelShuffle2DExt(2)
            # C_out = C_out // 4

            #self.post_2nd["o_upsample"] = UpSample(N=1, C_in=backbone_C_out, C_out=backbone_C_out//2, method='bspline', with_conv=True)
            #self.post_2nd["o_nl"] = nn.GELU(approximate="tanh")
            #self.post_2nd["o_conv"] = Conv2DExt(backbone_C_out//2, backbone_C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

            self.post["1st_upsample"] = UpSample(N=1, C_in=config.C_out, C_out=config.C_out, method='bspline', with_conv=False)

            self.post["o_upsample"] = UpSample(N=1, C_in=backbone_C_out, C_out=backbone_C_out, method='bspline', with_conv=False)
            self.post["o_nl"] = nn.GELU(approximate="tanh")
            self.post["o_conv"] = Conv2DExt(backbone_C_out, backbone_C_out, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

        if config.post_backbone == 'hrnet':
            config_post = copy.deepcopy(config)
            config_post.backbone_hrnet.block_str = config.post_hrnet.block_str
            config_post.separable_conv = config.post_hrnet.separable_conv

            config_post.C_in = backbone_C_out
            config_post.backbone_hrnet.C = backbone_C_out

            if self.config.super_resolution:
                config_post.height[0] *= 2
                config_post.width[0] *= 2

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

            config_post.C_in = backbone_C_out
            config_post.backbone_mixed_unetr.C = backbone_C_out

            if self.config.super_resolution:
                config_post.height[0] *= 2
                config_post.width[0] *= 2

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

        self.post["output_conv"] = Conv2DExt(C_out, config_post.C_out, kernel_size=config_post.kernel_size, stride=config_post.stride, padding=config_post.padding, bias=True)

    def load_backbone(self, status):
        super().load_backbone(status)
        if 'post_1st_state' in status:
            self.post_1st.load_state_dict(status['post_1st_state'])
        else:
            self.post_1st.load_state_dict(status['post_state'])

    def disable_backbone(self):
        super().disable_backbone()
        self.post_1st.requires_grad_(False)
        for param in self.post_1st.parameters():
            param.requires_grad = False

    def load_from_status(self, status, device=None, load_others=True):
        super().load_from_status(status=status, device=device, load_others=load_others)
        self.post_1st.load_state_dict(status['post_1st_state'])

    def save_to_file(self, epoch, save_path, only_paras):
        if only_paras:
                torch.save({
                "epoch":epoch,
                "config": self.config,
                "pre_state": self.pre.state_dict(), 
                "backbone_state": self.backbone.state_dict(), 
                "post_state": self.post.state_dict(), 
                "post_1st_state": self.post_1st.state_dict(), 
                "a": self.a,
                "b": self.b
            }, save_path)
        else:
            torch.save({
                "epoch":epoch,
                "pre_state": self.pre.state_dict(), 
                "backbone_state": self.backbone.state_dict(), 
                "post_state": self.post.state_dict(), 
                "post_1st_state": self.post_1st.state_dict(), 
                "a": self.a,
                "b": self.b,
                "optimizer_state": self.optim.state_dict(), 
                "scheduler_state": self.sched.state_dict(),
                "config": self.config,
                "scheduler_type":self.stype
            }, save_path)


    def forward(self, x, snr=-1, base_snr_t=None):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image
        """
        res_pre = self.pre(x)
        B, T, C, H, W = res_pre.shape

        if self.config.backbone == 'hrnet':
            y_hat, _ = self.backbone(res_pre)
        else:
            y_hat = self.backbone(res_pre)

        if self.residual:
            y_hat[:,:, :C, :, :] = res_pre + y_hat[:,:, :C, :, :]

        logits_1st = self.post_1st(y_hat)

        if self.config.super_resolution:
            logits_1st = self.post["1st_upsample"](logits_1st)
            y_hat = self.post["o_upsample"](y_hat)
            y_hat = self.post["o_nl"](y_hat)
            y_hat = self.post["o_conv"](y_hat)

        if self.config.post_backbone == 'hrnet':
            res, _ = self.post['post_main'](y_hat)
        else:
            res = self.post['post_main'](y_hat)

        B, T, C, H, W = y_hat.shape
        # if self.residual:
        #     res[:,:, :C, :, :] = res[:,:, :C, :, :] + y_hat

        logits = self.post["output_conv"](res) + logits_1st

        if base_snr_t is not None:
            weights = self.compute_weights(snr=snr, base_snr_t=base_snr_t)
            return logits, weights, logits_1st
        else:
            return logits, logits_1st
