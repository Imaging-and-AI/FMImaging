"""
Model(s) used for Mircoscopy
"""
import os
import torch
import logging
from colorama import Fore, Back, Style

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.imaging_attention import *
from model_base.backbone import *

from utils import get_device, optimizer_to

from model_base.task_base import *
from model_base.losses import *

# -------------------------------------------------------------------------------------------------
# Micro model

class STCNNT_MICRO(STCNNT_Task_Base):
    """
    STCNNT for Micro data
    Just the base CNNT with care to residual
    """
    def __init__(self, config, total_steps=1) -> None:
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - total_steps (int): total training steps. used for OneCycleLR

        Task specific args:

            - losses:
            - loss_weights:
        """
        super().__init__(config=config)

        self.residual = config.residual
        self.C_in = config.C_in
        self.C_out = config.C_out

        logging.info(f"{Fore.YELLOW}{Back.WHITE}===> Micro - create pre <==={Style.RESET_ALL}")
        self.create_pre()

        logging.info(f"{Fore.GREEN}{Back.WHITE}===> Micro - backbone <==={Style.RESET_ALL}")
        if config.backbone == "small_unet":
            self.backbone = CNNT_Unet(config=config)

        if config.backbone == "hrnet":
            config.C_in = config.backbone_hrnet.C
            self.backbone = STCNNT_HRnet(config=config)
            config.C_in = self.C_in

        if config.backbone == "unet":
            config.C_in = config.backbone_unet.C
            self.backbone = STCNNT_Unet(config=config)
            config.C_in = self.C_in

        if config.backbone == "LLM":
            self.backbone = STCNNT_LLMnet(config=config)

        logging.info(f"{Fore.RED}{Back.WHITE}===> Micro - post <==={Style.RESET_ALL}")
        self.create_post()

        device = get_device(device=config.device)
        self.set_up_loss(device=device)

        self.set_up_optim_and_scheduling(total_steps=total_steps)

        if config.load_path is not None:
            self.load(load_path=config.load_path, device=device)

        logging.info(f"{Fore.BLUE}{Back.WHITE}===> Micro - done <==={Style.RESET_ALL}")

    def create_pre(self):

        c = self.config

        if self.config.backbone == "small_unet":
            self.pre = torch.nn.Identity()

        if self.config.backbone == "hrnet":
            self.pre = Conv2DExt(c.C_in, c.backbone_hrnet.C, kernel_size=c.kernel_size, stride=c.stride, padding=c.padding, bias=True)

        if self.config.backbone == "unet":
            self.pre = Conv2DExt(c.C_in, c.backbone_unet.C, kernel_size=c.kernel_size, stride=c.stride, padding=c.padding, bias=True)

        if self.config.backbone == "LLM":
            self.pre = torch.nn.Identity()


    def create_post(self):

        c = self.config

        if self.config.backbone == "small_unet":
            self.pre = torch.nn.Identity()

        if self.config.backbone == "hrnet":
            hrnet_C_out = int(c.backbone_hrnet.C * sum([np.power(2, k) for k in range(c.backbone_hrnet.num_resolution_levels)]))
            self.post = Conv2DExt(hrnet_C_out, c.C_out, kernel_size=c.kernel_size, stride=c.stride, padding=c.padding, bias=True)

        if c.backbone == "unet":
            self.post = Conv2DExt(c.backbone_unet.C, c.C_out, kernel_size=c.kernel_size, stride=c.stride, padding=c.padding, bias=True)

        if c.backbone == "LLM":
            output_C = int(np.power(2, c.backbone_LLM.num_stages-2)) if c.backbone_LLM.num_stages>2 else c.backbone_LLM.C
            self.post = Conv2DExt(output_C,c.C_out, kernel_size=c.kernel_size, stride=c.stride, padding=c.padding, bias=True)


    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image
        @rets:
            - output (5D torch.Tensor): output image (denoised)
        """
        res_pre = self.pre(x)

        B, T, C, H, W = res_pre.shape

        if self.config.backbone == "hrnet":

            y_hat, _ = self.backbone(res_pre)

            if self.residual:
                y_hat[:,:, :C, :, :] = res_pre + y_hat[:,:, :C, :, :]

            logits = self.post(y_hat)

        else:
            res_backbone = self.backbone(res_pre)
            logits = self.post(res_backbone)

            if self.residual:
                logits = x - logits

        return logits


    def set_up_loss(self, device="cpu"):
        """
        Sets up the combined loss
        @args:
            - device (torch.device): device to setup the loss on
        @args (from config):
            - losses (list of "ssim", "ssim3D", "msssim", "l1", "mse", "psnr", "gaussian", "gaussian3D"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
        """
        self.loss_f = Combined_Loss(self.config.losses, self.config.loss_weights, device=device)


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

        optim_groups = [*pre_optim_groups, *backbone_optim_groups, *post_optim_groups]

        return optim_groups

    def check_model_learnable_status(self, rank_str=""):
        num = 0
        num_learnable = 0
        for param in self.pre.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        logging.info(f"{rank_str} model, pre, learnable tensors {num_learnable} out of {num} ...")

        num = 0
        num_learnable = 0
        for param in self.backbone.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        logging.info(f"{rank_str} model, backbone, learnable tensors {num_learnable} out of {num} ...")

        num = 0
        num_learnable = 0
        for param in self.post.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        logging.info(f"{rank_str} model, post, learnable tensors {num_learnable} out of {num} ...")

    def save(self, epoch, only_paras=False, save_file_name=None):
        """
        Save model checkpoints
        @args:
            - epoch (int): current epoch of the training cycle
            - only_paras (bool): save only parameters or scheduler and optimizer as well
            - save_file_name (str): name to save the file with
        @args (from config):
            - date (datetime str): runtime date
            - checkpath (str): directory to save checkpoint in
        @rets:
            - save_path (str): path of the saved model
        """
        if save_file_name is None:
            run_name = self.config.run_name.replace(" ", "_")
            save_file_name = f"{run_name}_epoch-{epoch}.pth"

        save_path = os.path.join(self.config.check_path, save_file_name)
        logging.info(f"{Fore.YELLOW}Saving model status at {save_path}{Style.RESET_ALL}")
        if only_paras:
                torch.save({
                "epoch":epoch,
                "config": self.config,
                "pre_state": self.pre.state_dict(),
                "backbone_state": self.backbone.state_dict(),
                "post_state": self.post.state_dict(),
            }, save_path)
        else:
            torch.save({
                "epoch":epoch,
                "config": self.config,
                "pre_state": self.pre.state_dict(),
                "backbone_state": self.backbone.state_dict(),
                "post_state": self.post.state_dict(),
                "optimizer_state": self.optim.state_dict(),
                "scheduler_state": self.sched.state_dict(),
                "scheduler_type":self.stype
            }, save_path)

        return save_path

    def load_from_status(self, status, device=None, load_others=True):
        """
        Load the model from status; the config will not be updated
        @args:
            - status (dict): dict to hold model parameters etc
            - device (torch.device): device to load at
            - load_others (bool): on top of params, load scheduler and optimizer as well
        """

        if 'backbone_state' in status:
            self.pre.load_state_dict(status['pre_state'])
            self.backbone.load_state_dict(status['backbone_state'])
            self.post.load_state_dict(status['post_state'])
        else:
            self.load_state_dict(status['model_state'])

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
            - load_path (str): path to load the weights from
            - device (torch.device): device to setup the model on
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
