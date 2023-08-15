"""
Trainer for MRI denoising.
Provides the mian function to call for training:
    - trainer
"""
import copy
import wandb
import numpy as np
import pickle
from time import time
import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from model_base.losses import *
from utils.save_model import save_final_model
from utils.running_inference import running_inference

from model_mri import STCNNT_MRI, MRI_hrnet, MRI_double_net
from data_mri import MRIDenoisingDatasetTrain, load_mri_data

from colorama import Fore, Back, Style
import nibabel as nib

# -------------------------------------------------------------------------------------------------

class mri_trainer_meters(object):
    """
    A helper class to organize meters for training
    """

    def __init__(self, config, device):

        super().__init__()

        self.config = config

        c = config

        self.mse_meter = AverageMeter()
        self.l1_meter = AverageMeter()
        self.ssim_meter = AverageMeter()
        self.ssim3D_meter = AverageMeter()
        self.psnr_meter = AverageMeter()
        self.psnr_loss_meter = AverageMeter()
        self.perp_meter = AverageMeter()
        self.gaussian_meter = AverageMeter()
        self.gaussian3D_meter = AverageMeter()

        self.mse_loss_func = MSE_Loss(complex_i=c.complex_i)
        self.l1_loss_func = L1_Loss(complex_i=c.complex_i)
        self.ssim_loss_func = SSIM_Loss(complex_i=c.complex_i, device=device)
        self.ssim3D_loss_func = SSIM3D_Loss(complex_i=c.complex_i, device=device)
        self.psnr_func = PSNR(range=2048)
        self.psnr_loss_func = PSNR_Loss(range=2048)    
        self.perp_func = Perpendicular_Loss()
        self.gaussian_func = GaussianDeriv_Loss(sigmas=[0.5, 1.0, 1.5], complex_i=c.complex_i, device=device)
        self.gaussian3D_func = GaussianDeriv3D_Loss(sigmas=[0.5, 1.0, 1.5], sigmas_T=[0.5, 0.5, 0.5], complex_i=c.complex_i, device=device)

    def reset(self):

        self.mse_meter.reset()
        self.l1_meter.reset()
        self.ssim_meter.reset()
        self.ssim3D_meter.reset()
        self.psnr_meter.reset()
        self.psnr_loss_meter.reset()
        self.perp_meter.reset()
        self.gaussian_meter.reset()
        self.gaussian3D_meter.reset()

    def update(self, output, y):

        total = y.shape[0]

        mse_loss = self.mse_loss_func(output, y).item()
        l1_loss = self.l1_loss_func(output, y).item()
        ssim_loss = self.ssim_loss_func(output, y).item()
        ssim3D_loss = self.ssim3D_loss_func(output, y).item()
        psnr_loss = self.psnr_loss_func(output, y).item()
        psnr = self.psnr_func(output, y).item()
        if self.config.complex_i: perp = self.perp_func(output, y).item()
        gauss_loss = self.gaussian_func(output, y).item()
        gauss3D_loss = self.gaussian3D_func(output, y).item()

        self.mse_meter.update(mse_loss, n=total)
        self.l1_meter.update(l1_loss, n=total)
        self.ssim_meter.update(ssim_loss, n=total)
        self.ssim3D_meter.update(ssim3D_loss, n=total)
        self.psnr_loss_meter.update(psnr_loss, n=total)
        self.psnr_meter.update(psnr, n=total)
        if self.config.complex_i: self.perp_meter.update(perp, n=total)        
        self.gaussian_meter.update(gauss_loss, n=total)
        self.gaussian3D_meter.update(gauss3D_loss, n=total)

    def get_loss(self):
        # mse, l1, ssim, ssim3D, psnr_loss, psnr, perp,  gaussian, gaussian3D
        return self.mse_meter.avg, self.l1_meter.avg, self.ssim_meter.avg, self.ssim3D_meter.avg, self.psnr_loss_meter.avg, self.psnr_meter.avg, self.perp_meter.avg, self.gaussian_meter.avg, self.gaussian3D_meter.avg

# -------------------------------------------------------------------------------------------------
# trainer

def create_log_str(config, epoch, rank, data_shape, gmap_median, noise_sigma, loss, snr, loss_meters, curr_lr, role):
    if data_shape is not None:
        data_shape_str = f"{data_shape[-1]}, "
    else:
        data_shape_str = ""

    if curr_lr >=0:
        lr_str = f", lr {curr_lr:.8f}"
    else:
        lr_str = ""

    if role == 'tra':
        C = Fore.YELLOW
    else:
        C = Fore.GREEN

    if snr >=0:
        snr_str = f", snr {snr:.2f}"
    else:
        snr_str = ""

    mse, l1, ssim, ssim3D, psnr_loss, psnr, perp,  gaussian, gaussian3D = loss_meters.get_loss()

    str= f"{Fore.GREEN}Epoch {epoch}/{config.num_epochs}, {C}{role}, {Style.RESET_ALL}{rank}, " + data_shape_str + f"{Fore.BLUE}{Back.WHITE}{Style.BRIGHT}loss {loss:.4f},{Style.RESET_ALL} {Fore.WHITE}{Back.LIGHTBLUE_EX}{Style.NORMAL}gmap {gmap_median:.2f}, sigma {noise_sigma:.2f}{snr_str}{Style.RESET_ALL} {C}mse {mse:.4f}, l1 {l1:.4f}, perp {perp:.4f}, ssim {ssim:.4f}, ssim3D {ssim3D:.4f}, gaussian {gaussian:.4f}, gaussian3D {gaussian3D:.4f}, psnr loss {psnr_loss:.4f}, psnr {psnr:.4f}{Style.RESET_ALL}{lr_str}"

    return str

# -------------------------------------------------------------------------------------------------

def save_batch_samples(saved_path, fname, x, y, output, y_2x, y_degraded, gmap_median, noise_sigma):

    noisy_im = x.numpy()
    clean_im = y.numpy()
    pred_im = output.numpy()
    y_degraded = y_degraded.numpy()
    y_2x = y_2x.numpy()

    post_str = ""
    if gmap_median > 0 and noise_sigma > 0:
        post_str = f"_gmap_{gmap_median:.2f}_sigma_{noise_sigma:.2f}"

    fname += post_str

    np.save(os.path.join(saved_path, f"{fname}_x.npy"), noisy_im)
    np.save(os.path.join(saved_path, f"{fname}_y.npy"), clean_im)
    np.save(os.path.join(saved_path, f"{fname}_output.npy"), pred_im)
    np.save(os.path.join(saved_path, f"{fname}_y_degraded.npy"), y_degraded)
    np.save(os.path.join(saved_path, f"{fname}_y_2x.npy"), y_2x)

    B, T, C, H, W = x.shape

    noisy_im = np.transpose(noisy_im, [3, 4, 2, 1, 0])
    clean_im = np.transpose(clean_im, [3, 4, 2, 1, 0])
    pred_im = np.transpose(pred_im, [3, 4, 2, 1, 0])
    y_degraded = np.transpose(y_degraded, [3, 4, 2, 1, 0])
    y_2x = np.transpose(y_2x, [3, 4, 2, 1, 0])

    if C==3:
        x = noisy_im[:,:,0,:,:] + 1j * noisy_im[:,:,1,:,:]
        gmap = noisy_im[:,:,2,:,:]

        nib.save(nib.Nifti1Image(np.real(x), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_x_real.nii"))
        nib.save(nib.Nifti1Image(np.imag(x), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_x_imag.nii"))
        nib.save(nib.Nifti1Image(np.abs(x), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_x.nii"))

        y = clean_im[:,:,0,:,:] + 1j * clean_im[:,:,1,:,:]
        nib.save(nib.Nifti1Image(np.real(y), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_real.nii"))
        nib.save(nib.Nifti1Image(np.imag(y), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_imag.nii"))
        nib.save(nib.Nifti1Image(np.abs(y), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y.nii"))

        output = pred_im[:,:,0,:,:] + 1j * pred_im[:,:,1,:,:]
        nib.save(nib.Nifti1Image(np.real(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_output_real.nii"))
        nib.save(nib.Nifti1Image(np.imag(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_output_imag.nii"))
        nib.save(nib.Nifti1Image(np.abs(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_output.nii"))  

        output = y_degraded[:,:,0,:,:] + 1j * y_degraded[:,:,1,:,:]
        nib.save(nib.Nifti1Image(np.real(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_degraded_real.nii"))
        nib.save(nib.Nifti1Image(np.imag(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_degraded_imag.nii"))
        nib.save(nib.Nifti1Image(np.abs(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_degraded.nii"))

        output = y_2x[:,:,0,:,:] + 1j * y_2x[:,:,1,:,:]
        nib.save(nib.Nifti1Image(np.real(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_2x_real.nii"))
        nib.save(nib.Nifti1Image(np.imag(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_2x_imag.nii"))
        nib.save(nib.Nifti1Image(np.abs(output), affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_2x.nii"))
    else:
        x = noisy_im[:,:,0,:,:]
        gmap = noisy_im[:,:,1,:,:]

        nib.save(nib.Nifti1Image(x, affine=np.eye(4)), os.path.join(saved_path, f"{fname}_x.nii"))
        nib.save(nib.Nifti1Image(clean_im, affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y.nii"))
        nib.save(nib.Nifti1Image(pred_im, affine=np.eye(4)), os.path.join(saved_path, f"{fname}_output.nii"))
        nib.save(nib.Nifti1Image(y_degraded, affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_degraded.nii"))
        nib.save(nib.Nifti1Image(y_2x, affine=np.eye(4)), os.path.join(saved_path, f"{fname}_y_2x.nii"))

    nib.save(nib.Nifti1Image(gmap, affine=np.eye(4)), os.path.join(saved_path, f"{fname}_gmap.nii"))

# -------------------------------------------------------------------------------------------------

def distribute_learning_rates(rank, optim, src=0):

    N = len(optim.param_groups)
    new_lr = torch.zeros(N).to(rank)
    for ind in range(N):
        new_lr[ind] = optim.param_groups[ind]["lr"]

    dist.broadcast(new_lr, src=src)

    if rank != src:
        for ind in range(N):
            optim.param_groups[ind]["lr"] = new_lr[ind].item()

# -------------------------------------------------------------------------------------------------

def get_rank_str(rank):
    if rank == 0:
        return f"{Fore.BLUE}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 1:
        return f"{Fore.GREEN}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 2:
        return f"{Fore.YELLOW}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 3:
        return f"{Fore.MAGENTA}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 4:
        return f"{Fore.LIGHTYELLOW_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 5:
        return f"{Fore.LIGHTBLUE_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 6:
        return f"{Fore.LIGHTRED_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 7:
        return f"{Fore.LIGHTCYAN_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"

    return f"{Fore.WHITE}{Style.BRIGHT}rank {rank} {Style.RESET_ALL}"

def create_model(config, model_type, total_steps=-1):
    if model_type == "STCNNT_MRI":
        model = STCNNT_MRI(config=config, total_steps=total_steps)
    elif model_type == "MRI_hrnet":
        model = MRI_hrnet(config=config, total_steps=total_steps)
    else:
        model = MRI_double_net(config=config, total_steps=total_steps)

    return model

# -------------------------------------------------------------------------------------------------

def trainer(rank, global_rank, config, wandb_run):
    """
    The trainer cycle. Allows training on cpu/single gpu/multiple gpu(ddp)
    @args:
        - rank (int): for distributed data parallel (ddp)
            -1 if running on cpu or only one gpu
        - model (torch model): model to be trained
        - config (Namespace): runtime namespace for setup
        - train_set (torch Dataset list): the data to train on
        - val_set (torch Dataset list): the data to validate each epoch
        - test_set (torch Dataset list): the data to test model at the end
    """

    rank_str = get_rank_str(rank)

    # -----------------------------------------------

    start = time()
    train_set, val_set, test_set = load_mri_data(config=config)
    logging.info(f"{rank_str}, load_mri_data took {time() - start} seconds ...")

    total_num_samples = sum([len(s) for s in train_set])

    total_steps = compute_total_steps(config, total_num_samples)
    logging.info(f"{rank_str}, total_steps for this run: {total_steps}, len(train_set) {[len(s) for s in train_set]}, batch {config.batch_size}")

    # -----------------------------------------------
    if not config.disable_LSUV:
        if (config.load_path is None) or (not config.continued_training):
            t0 = time()
            num_samples = len(train_set[-1])
            sampled_picked = np.random.randint(0, num_samples, size=32)
            input_data  = torch.stack([train_set[-1][i][0] for i in sampled_picked])
            print(f"{rank_str}, LSUV prep data took {time()-t0 : .2f} seconds ...")

    # -----------------------------------------------

    if config.ddp:
        device = torch.device(f"cuda:{rank}")
        config.device = device
    else:
        device = config.device

    # -----------------------------------------------

    print(f"{rank_str}, {Style.BRIGHT}{Fore.RED}{Back.LIGHTWHITE_EX}RUN NAME - {config.run_name}{Style.RESET_ALL}")

    # -----------------------------------------------

    num_epochs = config.num_epochs
    batch_size = config.batch_size
    lr = config.global_lr
    optim = config.optim
    scheduler_type = config.scheduler_type
    losses = config.losses
    loss_weights = config.loss_weights
    weighted_loss_snr = config.weighted_loss_snr
    weighted_loss_temporal = config.weighted_loss_temporal
    weighted_loss_added_noise = config.weighted_loss_added_noise
    save_samples = config.save_samples
    num_saved_samples = config.num_saved_samples
    height = config.height
    width = config.width
    c_time = config.time
    use_amp = config.use_amp
    num_workers = config.num_workers
    lr_pre = config.lr_pre
    lr_backbone = config.lr_backbone
    lr_post = config.lr_post
    continued_training = config.continued_training
    disable_pre = config.disable_pre
    disable_backbone = config.disable_backbone
    disable_post = config.disable_post
    model_type = config.model_type
    not_load_pre = config.not_load_pre
    not_load_backbone = config.not_load_backbone
    not_load_post = config.not_load_post
    run_name = config.run_name
    run_notes = config.run_notes
    disable_LSUV = config.disable_LSUV
    super_resolution = config.super_resolution

    post_backbone = config.post_backbone

    post_hrnet_block_str = config.post_hrnet.block_str
    post_hrnet_separable_conv = config.post_hrnet.separable_conv

    post_mixed_unetr_num_resolution_levels = config.post_mixed_unetr.num_resolution_levels
    post_mixed_unetr_block_str = config.post_mixed_unetr.block_str
    post_mixed_unetr_use_unet_attention = config.post_mixed_unetr.use_unet_attention
    post_mixed_unetr_transformer_for_upsampling = config.post_mixed_unetr.transformer_for_upsampling
    post_mixed_unetr_n_heads = config.post_mixed_unetr.n_heads
    post_mixed_unetr_use_conv_3d = config.post_mixed_unetr.use_conv_3d
    post_mixed_unetr_use_window_partition = config.post_mixed_unetr.use_window_partition
    post_mixed_unetr_separable_conv = config.post_mixed_unetr.separable_conv

    ddp = config.ddp

    if config.load_path is not None:
        status = torch.load(config.load_path)
        config = status['config']

        # overwrite the config parameters with current settings
        config.device = device
        config.losses = losses
        config.loss_weights = loss_weights
        config.optim = optim
        config.scheduler_type = scheduler_type
        config.global_lr = lr
        config.num_epochs = num_epochs
        config.batch_size = batch_size
        config.weighted_loss_snr = weighted_loss_snr
        config.weighted_loss_temporal = weighted_loss_temporal
        config.weighted_loss_added_noise = weighted_loss_added_noise
        config.save_samples = save_samples
        config.num_saved_samples = num_saved_samples
        config.height = height
        config.width = width
        config.time = c_time
        config.use_amp = use_amp
        config.num_workers = num_workers
        config.lr_pre = lr_pre
        config.lr_backbone = lr_backbone
        config.lr_post = lr_post
        config.disable_pre = disable_pre
        config.disable_backbone = disable_backbone
        config.disable_post = disable_post
        config.not_load_pre = not_load_pre
        config.not_load_backbone = not_load_backbone
        config.not_load_post = not_load_post
        config.model_type = model_type
        config.run_name = run_name
        config.run_notes = run_notes
        config.disable_LSUV = disable_LSUV
        config.super_resolution = super_resolution

        config.post_backbone = post_backbone

        config.post_hrnet = Nestedspace()
        config.post_hrnet.block_str = post_hrnet_block_str
        config.post_hrnet.separable_conv = post_hrnet_separable_conv

        config.post_mixed_unetr = Nestedspace()
        config.post_mixed_unetr.num_resolution_levels = post_mixed_unetr_num_resolution_levels
        config.post_mixed_unetr.block_str = post_mixed_unetr_block_str
        config.post_mixed_unetr.use_unet_attention = post_mixed_unetr_use_unet_attention
        config.post_mixed_unetr.transformer_for_upsampling = post_mixed_unetr_transformer_for_upsampling
        config.post_mixed_unetr.n_heads = post_mixed_unetr_n_heads
        config.post_mixed_unetr.use_conv_3d = post_mixed_unetr_use_conv_3d
        config.post_mixed_unetr.use_window_partition = post_mixed_unetr_use_window_partition
        config.post_mixed_unetr.separable_conv = post_mixed_unetr_separable_conv
        #config.load_path = load_path

        print(f"{rank_str}, {Fore.WHITE}=============================================================={Style.RESET_ALL}")

        model = create_model(config, model_type, total_steps)

        if 'backbone_state' in status:
            print(f"{rank_str}, load saved model, continued_training - {continued_training}")
            if continued_training:
                model.load_from_status(status=status, device=device, load_others=continued_training)
            else: # new stage training
                model = model.to(device)
                if not config.disable_LSUV:
                    t0 = time()
                    LSUVinit(model, input_data.to(device=device), verbose=True, cuda=True)
                    print(f"{rank_str}, LSUVinit took {time()-t0 : .2f} seconds ...")

                # ------------------------------
                if not not_load_pre:
                    print(f"{rank_str}, {Fore.YELLOW}load saved model, pre_state{Style.RESET_ALL}")
                    model.pre.load_state_dict(status['pre_state'])
                else:
                    print(f"{rank_str}, {Fore.RED}load saved model, WITHOUT pre_state{Style.RESET_ALL}")

                if disable_pre:
                    print(f"{rank_str}, {Fore.YELLOW}load saved model, pre requires_grad_(False){Style.RESET_ALL}")
                    model.pre.requires_grad_(False)
                    for param in model.pre.parameters():
                        param.requires_grad = False
                else:
                    print(f"{rank_str}, {Fore.RED}load saved model, pre requires_grad_(True){Style.RESET_ALL}")
                # ------------------------------
                if not not_load_backbone:
                    print(f"{rank_str}, {Fore.YELLOW}load saved model, backbone_state{Style.RESET_ALL}")
                    model.backbone.load_state_dict(status['backbone_state'])
                else:
                    print(f"{rank_str}, {Fore.RED}load saved model, WITHOUT backbone_state{Style.RESET_ALL}")

                if disable_backbone:
                    print(f"{rank_str}, {Fore.YELLOW}load saved model, backbone requires_grad_(False){Style.RESET_ALL}")
                    model.backbone.requires_grad_(False)
                    for param in model.backbone.parameters():
                        param.requires_grad = False
                else:
                    print(f"{rank_str}, {Fore.RED}load saved model, backbone requires_grad_(True){Style.RESET_ALL}")
                # ------------------------------
                if not not_load_post:
                    print(f"{rank_str}, {Fore.YELLOW}load saved model, post_state{Style.RESET_ALL}")
                    model.post.load_state_dict(status['post_state'])
                else:
                    print(f"{rank_str}, {Fore.RED}load saved model, WITHOUT post_state{Style.RESET_ALL}")

                if disable_post:
                    print(f"{rank_str}, {Fore.YELLOW}load saved model, post requires_grad_(False){Style.RESET_ALL}")
                    model.post.requires_grad_(False)
                    for param in model.post.parameters():
                        param.requires_grad = False
                else:
                    print(f"{rank_str}, {Fore.RED}load saved model, post requires_grad_(True){Style.RESET_ALL}")
                # ------------------------------

                model.a = status['a']
                model.b = status['b']

                # ---------------------------------------------------

        model = model.to(device)

        config.ddp = ddp

        print(f"{rank_str}, after load saved model, the config for running - {config}")
        print(f"{rank_str}, after load saved model, config.use_amp for running - {config.use_amp}")
        print(f"{rank_str}, after load saved model, config.optim for running - {config.optim}")
        print(f"{rank_str}, after load saved model, config.scheduler_type for running - {config.scheduler_type}")
        print(f"{rank_str}, after load saved model, config.weighted_loss_snr for running - {config.weighted_loss_snr}")
        print(f"{rank_str}, after load saved model, config.weighted_loss_temporal for running - {config.weighted_loss_temporal}")
        print(f"{rank_str}, after load saved model, config.weighted_loss_added_noise for running - {config.weighted_loss_added_noise}")
        print(f"{rank_str}, after load saved model, config.num_workers for running - {config.num_workers}")
        print(f"{rank_str}, after load saved model, config.super_resolution for running - {config.super_resolution}")
        print(f"{rank_str}, after load saved model, config.post_backbone for running - {config.post_backbone}")
        print(f"{rank_str}, after load saved model, config.post_hrnet for running - {config.post_hrnet}")
        print(f"{rank_str}, after load saved model, config.post_mixed_unetr for running - {config.post_mixed_unetr}")

        print(f"{rank_str}, after load saved model, model.curr_epoch for running - {model.curr_epoch}")
        print(f"{rank_str}, {Fore.GREEN}after load saved model, model type - {config.model_type}{Style.RESET_ALL}")
        print(f"{rank_str}, {Fore.RED}after load saved model, model.device - {model.device}{Style.RESET_ALL}")
        print(f"{rank_str}, {Fore.WHITE}=============================================================={Style.RESET_ALL}")
    else:
        load_path = None
        model = create_model(config, config.model_type, total_steps)

        model = model.to(device)
        if not config.disable_LSUV:
            t0 = time()
            LSUVinit(model, input_data.to(device=device), verbose=True, cuda=True)
            print(f"{rank_str}, LSUVinit took {time()-t0 : .2f} seconds ...")

    if config.ddp:
        dist.barrier()

    c = config

    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True
    # torch._dynamo.config.skip_nnmodule_hook_guards=False
    # model = torch.compile(model, mode="reduce-overhead")

    if rank<=0:

        # model summary
        model_summary = model_info(model, config)
        logging.info(f"Configuration for this run:\n{config}")
        logging.info(f"Model Summary:\n{str(model_summary)}")

        if wandb_run is not None:
            logging.info(f"Wandb name: {wandb_run.name}")
            wandb_run.watch(model, log="parameters")
            wandb_run.log_code(".")

    # -----------------------------------------------

    if c.ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optim = model.module.optim
        sched = model.module.sched
        stype = model.module.stype
        loss_f = model.module.loss_f
        curr_epoch = model.module.curr_epoch
        samplers = [DistributedSampler(train_set_x, shuffle=True) for train_set_x in train_set]
        shuffle = False
    else:
        optim = model.optim
        sched = model.sched
        stype = model.stype
        loss_f = model.loss_f
        curr_epoch = model.curr_epoch
        samplers = [None for _ in train_set]
        shuffle = True

    if c.backbone == 'hrnet':
        model_str = f"heads {c.n_head}, {c.backbone_hrnet}"
    elif c.backbone == 'unet':
        model_str = f"heads {c.n_head}, {c.backbone_unet}"
    elif c.backbone == 'mixed_unetr':
        model_str = f"{c.backbone_mixed_unetr}"

    logging.info(f"{rank_str}, {Fore.RED}Local Rank:{rank}, global rank: {global_rank}, {c.backbone}, {c.a_type}, {c.cell_type}, {c.optim}, {c.global_lr}, {c.scheduler_type}, {c.losses}, {c.loss_weights}, weighted loss - snr {c.weighted_loss_snr} - temporal {c.weighted_loss_temporal} - added_noise {c.weighted_loss_added_noise}, data degrading {c.with_data_degrading}, snr perturb {c.snr_perturb_prob}, {c.norm_mode}, scale_ratio_in_mixer {c.scale_ratio_in_mixer}, amp {c.use_amp}, super resolution {c.super_resolution}, stride_s {c.stride_s}, separable_conv {c.separable_conv}, upsample method {c.upsample_method}, batch_size {c.batch_size}, {model_str}{Style.RESET_ALL}")

    # -----------------------------------------------

    num_workers_per_loader = c.num_workers//len(train_set)
    local_world_size = 1

    if c.ddp:
        # local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        # num_workers_per_loader = c.num_workers // local_world_size - 1
        # num_workers_per_loader = 4 if num_workers_per_loader<4 else num_workers_per_loader

        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        num_workers_per_loader = num_workers_per_loader // local_world_size
        num_workers_per_loader = 1 if num_workers_per_loader<1 else num_workers_per_loader
 
    logging.info(f"{rank_str}, {Fore.YELLOW}Local_world_size {local_world_size}, number of datasets {len(train_set)}, cpu {os.cpu_count()}, number of workers per loader is {num_workers_per_loader}{Style.RESET_ALL}")

    train_loader = [DataLoader(dataset=train_set_x, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers[i],
                                num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                persistent_workers=c.num_workers>0, pin_memory=False) for i, train_set_x in enumerate(train_set)]

    train_set_type = [train_set_x.data_type for train_set_x in train_set]

    # -----------------------------------------------

    if rank<=0: # main or master process
        if c.ddp: setup_logger(config) # setup master process logging

        if wandb_run is not None:
            wandb_run.summary["trainable_params"] = c.trainable_params
            wandb_run.summary["total_params"] = c.total_params
            wandb_run.summary["total_mult_adds"] = c.total_mult_adds 

            wandb_run.define_metric("epoch")

            wandb_run.define_metric("train_loss_avg", step_metric='epoch')
            wandb_run.define_metric("train_mse_loss", step_metric='epoch')
            wandb_run.define_metric("train_l1_loss", step_metric='epoch')
            wandb_run.define_metric("train_ssim_loss", step_metric='epoch')
            wandb_run.define_metric("train_ssim3D_loss", step_metric='epoch')
            wandb_run.define_metric("train_psnr_loss", step_metric='epoch')
            wandb_run.define_metric("train_psnr", step_metric='epoch')
            wandb_run.define_metric("train_snr", step_metric='epoch')
            wandb_run.define_metric("train_perp", step_metric='epoch')
            wandb_run.define_metric("train_gaussian_deriv", step_metric='epoch')
            wandb_run.define_metric("train_gaussian3D_deriv", step_metric='epoch')

            wandb_run.define_metric("val_loss_avg", step_metric='epoch')
            wandb_run.define_metric("val_mse_loss", step_metric='epoch')
            wandb_run.define_metric("val_l1_loss", step_metric='epoch')
            wandb_run.define_metric("val_ssim_loss", step_metric='epoch')
            wandb_run.define_metric("val_ssim3D_loss", step_metric='epoch')
            wandb_run.define_metric("val_psnr", step_metric='epoch')
            wandb_run.define_metric("val_perp", step_metric='epoch')
            wandb_run.define_metric("val_gaussian_deriv", step_metric='epoch')
            wandb_run.define_metric("val_gaussian3D_deriv", step_metric='epoch')

            # log a few training examples
            for i, train_set_x in enumerate(train_set):
                ind = np.random.randint(0, len(train_set_x), 4)
                x, y, y_degraded, y_2x, gmaps_median, noise_sigmas = train_set_x[ind[0]]
                x = np.expand_dims(x, axis=0)
                y = np.expand_dims(y, axis=0)
                y_degraded = np.expand_dims(y_degraded, axis=0)
                y_2x = np.expand_dims(y_2x, axis=0)
                for ii in range(1, len(ind)):
                    a_x, a_y, a_y_degraded, a_y_2x, gmaps_median, noise_sigmas = train_set_x[ind[ii]]
                    x = np.concatenate((x, np.expand_dims(a_x, axis=0)), axis=0)
                    y = np.concatenate((y, np.expand_dims(a_y, axis=0)), axis=0)
                    y_degraded = np.concatenate((y_degraded, np.expand_dims(a_y_degraded, axis=0)), axis=0)
                    y_2x = np.concatenate((y_2x, np.expand_dims(a_y_2x, axis=0)), axis=0)

                title = f"Tra_samples_{i}_Noisy_Noisy_GT_{x.shape}"
                vid = save_image_batch(c.complex_i, x, y_degraded, y, y_2x)
                wandb_run.log({title:wandb.Video(vid, caption=f"Tra sample {i}", fps=1, format='gif')})
                logging.info(f"{Fore.YELLOW}---> Upload tra sample - {title}, noise range {train_set_x.min_noise_level} to {train_set_x.max_noise_level}")

            logging.info(f"{Fore.YELLOW}---> noise range for validation {val_set[0].min_noise_level} to {val_set[0].max_noise_level}")

    # -----------------------------------------------
    # save best model to be saved at the end
    best_val_loss = np.inf
    best_model_wts = copy.deepcopy(model.module.state_dict() if c.ddp else model.state_dict())

    train_loss = AverageMeter()
    train_snr_meter = AverageMeter()
    loss_meters = mri_trainer_meters(config=c, device=device)    

    # -----------------------------------------------

    total_iters = sum([len(loader_x) for loader_x in train_loader])
    total_iters = total_iters if not c.debug else min(10, total_iters)

    # mix precision training
    scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp)

    optim.zero_grad(set_to_none=True)

    # -----------------------------------------------

    base_snr = 0
    beta_snr = 0.9
    beta_counter = 0
    if c.weighted_loss_snr:
        # get the base_snr
        mean_signal = list()
        median_signal = list()
        for i, train_set_x in enumerate(train_set):
            stat = train_set_x.get_stat()
            mean_signal.extend(stat['mean'])
            median_signal.extend(stat['median'])

        base_snr = np.abs(np.median(mean_signal)) / 2

        logging.info(f"{rank_str}, {Fore.YELLOW}base_snr {base_snr:.4f}, Mean signal {np.abs(np.median(mean_signal)):.4f}, median {np.abs(np.median(median_signal)):.4f}, from {len(mean_signal)} images {Style.RESET_ALL}")

    logging.info(f"{rank_str}, {Fore.GREEN}----------> Start training loop <----------{Style.RESET_ALL}")

    if c.ddp:
        model.module.check_model_learnable_status(rank_str)
    else:
        model.check_model_learnable_status(rank_str)

    for epoch in range(curr_epoch, c.num_epochs):
        logging.info(f"{Fore.GREEN}{'-'*20} Epoch:{epoch}/{c.num_epochs}, {rank_str}, global rank {global_rank} {'-'*20}{Style.RESET_ALL}")

        if config.save_samples:
            saved_path = os.path.join(config.log_path, config.run_name, f"tra_{epoch}")
            os.makedirs(saved_path, exist_ok=True)
            logging.info(f"{Fore.GREEN}saved_path - {saved_path}{Style.RESET_ALL}")

        train_loss.reset()
        train_snr_meter.reset()
        loss_meters.reset()

        model.train()
        if c.ddp: [loader_x.sampler.set_epoch(epoch) for loader_x in train_loader]

        images_saved = 0

        train_loader_iter = [iter(loader_x) for loader_x in train_loader]

        image_save_step_size = int(total_iters // config.num_saved_samples)
        if image_save_step_size == 0: image_save_step_size = 1

        curr_lr = 0

        all_lrs = [pg['lr'] for pg in optim.param_groups]
        logging.info(f"{rank_str}, {Fore.WHITE}{Style.BRIGHT}learning rate for epoch {epoch} - {all_lrs}{Style.RESET_ALL}")

        with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

            for idx in range(total_iters):

                loader_ind = idx % len(train_loader_iter)

                tm = start_timer(enable=c.with_timer)
                stuff = next(train_loader_iter[loader_ind], None)
                while stuff is None:
                    del train_loader_iter[loader_ind]
                    loader_ind = idx % len(train_loader_iter)
                    stuff = next(train_loader_iter[loader_ind], None)

                data_type = train_set_type[loader_ind]
                x, y, y_degraded, y_2x, gmaps_median, noise_sigmas = stuff
                end_timer(enable=c.with_timer, t=tm, msg="---> load batch took ")

                y_for_loss = y
                if config.super_resolution:
                    y_for_loss = y_2x

                tm = start_timer(enable=c.with_timer)
                x = x.to(device=device)
                y_for_loss = y_for_loss.to(device)
                noise_sigmas = noise_sigmas.to(device)
                gmaps_median = gmaps_median.to(device)

                B, T, C, H, W = x.shape

                if c.weighted_loss_temporal:
                    # compute temporal std
                    if C == 3:
                        std_t = torch.std(torch.abs(y[:,:,0,:,:] + 1j * y[:,:,1,:,:]), dim=1)
                    else:
                        std_t = torch.std(y(y[:,:,0,:,:], dim=1))

                    weights_t = torch.mean(std_t, dim=(-2, -1)).to(device)

                # compute snr
                signal = torch.mean(torch.linalg.norm(y, dim=2, keepdim=True), dim=(1, 2, 3, 4)).to(device)
                #snr = signal / (noise_sigmas*gmaps_median)
                snr = signal / gmaps_median
                #snr = snr.to(device)

                # base_snr : original snr in the clean patch
                # noise_sigmas: added noise
                # weighted_t: temporal/slice signal variation

                if c.weighted_loss_snr:
                    beta_counter += 1
                    base_snr = beta_snr * base_snr + (1-beta_snr) * torch.mean(snr).item()
                    base_snr_t = base_snr / (1 - np.power(beta_snr, beta_counter))
                else:
                    base_snr_t = -1

                noise_sigmas = torch.reshape(noise_sigmas, (B, 1, 1, 1, 1))

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=c.use_amp):
                    if c.weighted_loss_snr:
                        output, weights = model(x, snr, base_snr_t)
                        if c.weighted_loss_temporal:
                            weights *= weights_t
                    else:
                        output = model(x)
                        if c.weighted_loss_temporal:
                            weights = weights_t

                    if torch.sum(noise_sigmas).item() > 0:
                        if c.weighted_loss_snr or c.weighted_loss_temporal:
                            if c.weighted_loss_added_noise:
                                loss = loss_f(output*noise_sigmas, y_for_loss*noise_sigmas, weights=weights.to(device))
                            else:
                                loss = loss_f(output, y_for_loss, weights=weights.to(device))
                        else:
                            if c.weighted_loss_added_noise:
                                loss = loss_f(output*noise_sigmas, y_for_loss*noise_sigmas)
                            else:
                                loss = loss_f(output, y_for_loss)
                    else:
                        if c.weighted_loss_snr or c.weighted_loss_temporal:
                            loss = loss_f(output, y_for_loss, weights=weights.to(device))
                        else:
                            loss = loss_f(output, y_for_loss)

                    loss = loss / c.iters_to_accumulate

                end_timer(enable=c.with_timer, t=tm, msg="---> forward pass took ")

                if torch.isnan(loss):
                    print(f"Warning - loss is nan ... ")
                    optim.zero_grad()
                    continue

                tm = start_timer(enable=c.with_timer)
                scaler.scale(loss).backward()
                end_timer(enable=c.with_timer, t=tm, msg="---> backward pass took ")


                tm = start_timer(enable=c.with_timer)
                if (idx + 1) % c.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                    if(c.clip_grad_norm>0):
                        scaler.unscale_(optim)
                        nn.utils.clip_grad_norm_(model.parameters(), c.clip_grad_norm)

                    scaler.step(optim)
                    optim.zero_grad()
                    scaler.update()

                    if stype == "OneCycleLR": sched.step()
                end_timer(enable=c.with_timer, t=tm, msg="---> other steps took ")


                tm = start_timer(enable=c.with_timer)
                curr_lr = optim.param_groups[0]['lr']

                total=x.shape[0]
                train_loss.update(loss.item(), n=total)

                train_snr_meter.update(torch.mean(snr), n=total)

                output = output.to(x.dtype)

                loss_meters.update(output, y_for_loss)

                if rank<=0 and idx%image_save_step_size==0 and images_saved < config.num_saved_samples and config.save_samples:  
                    save_batch_samples(saved_path, f"tra_epoch_{epoch}_{images_saved}", x.cpu(), y.cpu(), output.detach().cpu(), y_for_loss.cpu(), y_degraded.cpu(), torch.mean(gmaps_median).item(), torch.mean(noise_sigmas).item())
                    images_saved += 1

                pbar.update(1)
                log_str = create_log_str(config, epoch, rank, 
                                         x.shape, 
                                         torch.mean(gmaps_median).cpu().item(),
                                         torch.mean(noise_sigmas).cpu().item(),
                                         train_loss.avg, 
                                         train_snr_meter.avg,
                                         loss_meters,
                                         curr_lr, 
                                         "tra")

                pbar.set_description_str(log_str)

                if wandb_run is not None:
                    wandb_run.log({"running_train_loss": loss.item()})
                    wandb_run.log({"running_train_snr": train_snr_meter.avg})
                    wandb_run.log({"running_train_mse": loss_meters.mse_meter.avg})
                    wandb_run.log({"running_train_ssim": loss_meters.ssim_meter.avg})
                    wandb_run.log({"running_train_l1": loss_meters.l1_meter.avg})
                    wandb_run.log({"running_train_perp": loss_meters.perp_meter.avg})
                    wandb_run.log({"lr": curr_lr})

                end_timer(enable=c.with_timer, t=tm, msg="---> logging and measuring took ")

            # ---------------------------------------
            log_str = create_log_str(c, epoch, rank, 
                                         None, 
                                         torch.mean(gmaps_median).cpu().item(),
                                         torch.mean(noise_sigmas).cpu().item(),
                                         train_loss.avg, 
                                         train_snr_meter.avg,
                                         loss_meters,
                                         curr_lr, 
                                         "tra")

            pbar.set_description_str(log_str)

            #print(f"--> mean SNR is {train_snr_meter.avg:.4f}")
            base_snr = train_snr_meter.avg

        # -------------------------------------------------------

        val_losses = eval_val(rank, model, c, val_set, epoch, device, wandb_run)

        # -------------------------------------------------------
        if rank<=0: # main or master process
            model_e = model.module if c.ddp else model
            saved_model = model_e.save(epoch)
            # model_loaded, config_loaded = load_model(saved_model)
            if val_losses[0] < best_val_loss:
                best_val_loss = val_losses[0]
                best_model_wts = copy.deepcopy(model_e.state_dict())
                run_name = config.run_name.replace(" ", "_")
                saved_model = model_e.save(epoch, only_paras=True, save_file_name=f"{run_name}_epoch-{epoch}_best.pth")
                # model_loaded, config_loaded = load_model(saved_model)
                if wandb_run is not None:
                    wandb_run.log({"epoch": epoch, "best_val_loss":best_val_loss})

            if wandb_run is not None:
                wandb_run.log(
                                {
                                    "epoch": epoch,
                                    "train_loss_avg": train_loss.avg,
                                    "train_mse_loss": loss_meters.mse_meter.avg,
                                    "train_l1_loss": loss_meters.l1_meter.avg,
                                    "train_ssim_loss": loss_meters.ssim_meter.avg,
                                    "train_ssim3D_loss": loss_meters.ssim3D_meter.avg,
                                    "train_psnr_loss": loss_meters.psnr_loss_meter.avg,
                                    "train_psnr": loss_meters.psnr_meter.avg,
                                    "train_snr": train_snr_meter.avg,
                                    "train_perp": loss_meters.perp_meter.avg,
                                    "train_gaussian_deriv": loss_meters.gaussian_meter.avg,
                                    "train_gaussian3D_deriv": loss_meters.gaussian3D_meter.avg,
                                    "val_loss_avg": val_losses[0],
                                    "val_mse_loss": val_losses[1],
                                    "val_l1_loss": val_losses[2],
                                    "val_ssim_loss": val_losses[3],
                                    "val_ssim3D_loss": val_losses[4],
                                    "val_psnr": val_losses[5],
                                    "val_perp": val_losses[6],
                                    "val_gaussian_deriv": val_losses[7],
                                    "val_gaussian3D_deriv": val_losses[8]
                                }
                              )

            if stype == "ReduceLROnPlateau":
                sched.step(val_losses[0])
            else: # stype == "StepLR"
                sched.step()

            if c.ddp:
                distribute_learning_rates(rank=rank, optim=optim, src=0)

        else: # child processes
            distribute_learning_rates(rank=rank, optim=optim, src=0)


    # test last model
    test_losses = eval_val(rank, model, config, test_set, epoch, device, wandb_run, id="test")
    if rank<=0:
        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = best_val_loss
            wandb_run.summary["last_val_loss"] = val_losses[0]

            wandb_run.summary["test_loss_last"] = test_losses[0]
            wandb_run.summary["test_mse_last"] = test_losses[1]
            wandb_run.summary["test_l1_last"] = test_losses[2]
            wandb_run.summary["test_ssim_last"] = test_losses[3]
            wandb_run.summary["test_ssim3D_last"] = test_losses[4]
            wandb_run.summary["test_psnr_last"] = test_losses[5]
            wandb_run.summary["test_perp_last"] = test_losses[6]
            wandb_run.summary["test_gaussian_deriv_last"] = test_losses[7]
            wandb_run.summary["test_gaussian3D_deriv_last"] = test_losses[8]

            model = model.module if c.ddp else model
            model.save(epoch)

            # save both models
            fname_last, fname_best = save_final_model(model, config, best_model_wts, only_pt=True)

            logging.info(f"--> {Fore.YELLOW}Save last mode at {fname_last}{Style.RESET_ALL}")
            logging.info(f"--> {Fore.YELLOW}Save best mode at {fname_best}{Style.RESET_ALL}")

    # test best model, reload the weights
    model = create_model(config, config.model_type, total_steps)

    model.load_state_dict(best_model_wts)
    model = model.to(device)

    if c.ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    test_losses = eval_val(rank, model, config, test_set, epoch, device, wandb_run, id="test")
    if rank<=0:
        if wandb_run is not None:
            wandb_run.summary["test_loss_best"] = test_losses[0]
            wandb_run.summary["test_mse_best"] = test_losses[1]
            wandb_run.summary["test_l1_best"] = test_losses[2]
            wandb_run.summary["test_ssim_best"] = test_losses[3]
            wandb_run.summary["test_ssim3D_best"] = test_losses[4]
            wandb_run.summary["test_psnr_best"] = test_losses[5]
            wandb_run.summary["test_perp_best"] = test_losses[6]
            wandb_run.summary["test_gaussian_deriv_best"] = test_losses[7]
            wandb_run.summary["test_gaussian3D_deriv_best"] = test_losses[8]

            wandb_run.save(fname_last+'.pt')
            #wandb_run.save(fname_last+'.pts')
            #wandb_run.save(fname_last+'.onnx')

            wandb_run.save(fname_best+'.pt')
            #wandb_run.save(fname_best+'.pts')
            #wandb_run.save(fname_best+'.onnx')

            # try:
            #     # # test the best model, reloading the saved model
            #     # model_jit = load_model(model_dir=None, model_file=fname_best+'.pts')
            #     # model_onnx, _ = load_model_onnx(model_dir=None, model_file=fname_best+'.onnx', use_cpu=True)

            #     # # pick a random case
            #     # a_test_set = test_set[np.random.randint(0, len(test_set))]
            #     # x, y, gmaps_median, noise_sigmas = a_test_set[np.random.randint(0, len(a_test_set))]

            #     # x = np.expand_dims(x, axis=0)
            #     # y = np.expand_dims(y, axis=0)

            #     #compare_model(config=config, model=model, model_jit=model_jit, model_onnx=model_onnx, device=device, x=x)
            # except:
            #     print(f"--> ignore the extra tests ...")

    if c.ddp:
        dist.barrier()
    print(f"--> run finished ...")

# -------------------------------------------------------------------------------------------------
# evaluate the val set

def eval_val(rank, model, config, val_set, epoch, device, wandb_run, id="val", scaling_factor=-1):
    """
    The validation evaluation.
    @args:
        - model (torch model): model to be validated
        - config (Namespace): runtime namespace for setup
        - val_set (torch Dataset list): the data to validate on
        - epoch (int): the current epoch
        - device (torch.device): the device to run eval on
    @rets:
        - val_loss_avg (float): the average val loss
        - val_mse_loss_avg (float): the average val mse loss
        - val_l1_loss_avg (float): the average val l1 loss
        - val_ssim_loss_avg (float): the average val ssim loss
        - val_ssim3D_loss_avg (float): the average val ssim3D loss
        - val_psnr_avg (float): the average val psnr
        - val_perp_avg (float): the average val perp loss
    """
    c = config

    shuffle = False

    try:
        if c.ddp:
            loss_f = model.module.loss_f
        else:
            loss_f = model.loss_f
    except:
        loss_f = None

    if c.ddp:
        sampler = [DistributedSampler(val_set_x, shuffle=False) for val_set_x in val_set]
    else:
        sampler = [None for _ in val_set]

    batch_size = c.batch_size if isinstance(val_set[0], MRIDenoisingDatasetTrain) else 1

    num_workers_per_loader = c.num_workers // len(val_set)

    if c.ddp:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        num_workers_per_loader = c.num_workers // (2 * local_world_size)
        num_workers_per_loader = 4 if num_workers_per_loader < 4 else num_workers_per_loader

    print(f"eval, num_workers is {num_workers_per_loader}")
    val_loader = [DataLoader(dataset=val_set_x, batch_size=batch_size, shuffle=False, sampler=sampler[i],
                                num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor,
                                persistent_workers=True) for i, val_set_x in enumerate(val_set)]

    val_loss_meter = AverageMeter()
    loss_meters = mri_trainer_meters(config=c, device=device) 

    model.eval()
    model.to(device)

    if rank <= 0 and epoch < 1:
        logging.info(f"Eval height and width is {c.height[-1]}, {c.width[-1]}")

    cutout = (c.time, c.height[-1], c.width[-1])
    overlap = (c.time//2, c.height[-1]//4, c.width[-1]//4)

    val_loader_iter = [iter(val_loader_x) for val_loader_x in val_loader]
    total_iters = sum([len(loader_x) for loader_x in val_loader])
    total_iters = total_iters if not c.debug else min(2, total_iters)

    images_logged = 0
    images_saved = 0
    if config.save_samples:
        if epoch >= 0:
            saved_path = os.path.join(config.log_path, config.run_name, id)
        else:
            saved_path = os.path.join(config.log_path, config.run_name, f"{id}_{epoch}")
        os.makedirs(saved_path, exist_ok=True)
        print(f"save path is {saved_path}")

    with torch.inference_mode():
        with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

            for idx in range(total_iters):

                loader_ind = idx % len(val_loader_iter)
                batch = next(val_loader_iter[loader_ind], None)
                while batch is None:
                    del val_loader_iter[loader_ind]
                    loader_ind = idx % len(val_loader_iter)
                    batch = next(val_loader_iter[loader_ind], None)
                x, y, y_degraded, y_2x, gmaps_median, noise_sigmas = batch

                if scaling_factor > 0:
                    x *= scaling_factor

                gmaps_median = gmaps_median.to(device=device, dtype=x.dtype)
                noise_sigmas = noise_sigmas.to(device=device, dtype=x.dtype)

                B = x.shape[0]
                noise_sigmas = torch.reshape(noise_sigmas, (B, 1, 1, 1, 1))

                if config.super_resolution:
                    y = y_2x

                x = x.to(device)
                y = y.to(device)

                if batch_size >1 and x.shape[-1]==c.width[-1]:
                    output = model(x)
                else:
                    B, T, C, H, W = x.shape

                    cutout_in = cutout
                    overlap_in = overlap
                    if not config.pad_time:
                        cutout_in = (T, c.height[-1], c.width[-1])
                        overlap_in = (0, c.height[-1]//2, c.width[-1]//2)

                    try:
                        _, output = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device=device)
                    except:
                        logging.info(f"{Fore.YELLOW}---> call inference on cpu ...")
                        _, output = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device="cpu")
                        y = y.to("cpu")

                if scaling_factor > 0:
                    output /= scaling_factor

                total = x.shape[0]

                if loss_f:
                    if torch.mean(noise_sigmas).item() > 0:
                        loss = loss_f(output*noise_sigmas, y*noise_sigmas)
                    else:
                        loss = loss_f(output, y)

                    val_loss_meter.update(loss.item(), n=total)

                # to help measure the performance, keep noise to be ~1
                output_scaled = output
                y_scaled = y

                loss_meters.update(output_scaled, y_scaled)

                if rank<=0 and images_logged < config.num_uploaded and wandb_run is not None:
                    images_logged += 1
                    title = f"{id.upper()}_{images_logged}_{x.shape}"
                    vid = save_image_batch(c.complex_i, x.numpy(force=True), output.numpy(force=True), y.numpy(force=True), y_2x.numpy(force=True))
                    wandb_run.log({title: wandb.Video(vid, 
                                                      caption=f"epoch {epoch}, gmap {torch.mean(gmaps_median).item():.2f}, noise {torch.mean(noise_sigmas).item():.2f}, mse {loss_meters.mse_meter.avg:.2f}, ssim {loss_meters.ssim_meter.avg:.2f}, psnr {loss_meters.psnr_meter.avg:.2f}", 
                                                      fps=1, format="gif")})

                if rank<=0 and images_saved < config.num_saved_samples and config.save_samples:
                    save_batch_samples(saved_path, f"{id}_epoch_{epoch}_{images_saved}", x.cpu(), y.cpu(), output.cpu(), y_2x.cpu(), y_degraded.cpu(), torch.mean(gmaps_median).item(), torch.mean(noise_sigmas).item())
                    images_saved += 1

                pbar.update(1)
                log_str = create_log_str(c, epoch, rank, 
                                         x.shape, 
                                         torch.mean(gmaps_median).cpu().item(),
                                         torch.mean(noise_sigmas).cpu().item(),
                                         val_loss_meter.avg, 
                                         -1,
                                         loss_meters,
                                         -1, 
                                         id)

                pbar.set_description_str(log_str)

            # -----------------------------------
            log_str = create_log_str(c, epoch, rank, 
                                         None, 
                                         torch.mean(gmaps_median).cpu().item(),
                                         torch.mean(noise_sigmas).cpu().item(),
                                         val_loss_meter.avg, 
                                         -1,
                                         loss_meters,
                                         -1,
                                         id)

            pbar.set_description_str(log_str)

    if c.ddp:
        val_loss = torch.tensor(val_loss_meter.avg).to(device=device)
        dist.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)

        val_mse = torch.tensor(loss_meters.mse_meter.avg).to(device=device)
        dist.all_reduce(val_mse, op=torch.distributed.ReduceOp.AVG)

        val_l1 = torch.tensor(loss_meters.l1_meter.avg).to(device=device)
        dist.all_reduce(val_l1, op=torch.distributed.ReduceOp.AVG)

        val_ssim = torch.tensor(loss_meters.ssim_meter.avg).to(device=device)
        dist.all_reduce(val_ssim, op=torch.distributed.ReduceOp.AVG)

        val_ssim3D = torch.tensor(loss_meters.ssim3D_meter.avg).to(device=device)
        dist.all_reduce(val_ssim3D, op=torch.distributed.ReduceOp.AVG)

        val_psnr_loss = torch.tensor(loss_meters.psnr_loss_meter.avg).to(device=device)
        dist.all_reduce(val_psnr_loss, op=torch.distributed.ReduceOp.AVG)

        val_psnr = torch.tensor(loss_meters.psnr_meter.avg).to(device=device)
        dist.all_reduce(val_psnr, op=torch.distributed.ReduceOp.AVG)

        val_perp = torch.tensor(loss_meters.perp_meter.avg).to(device=device)
        dist.all_reduce(val_perp, op=torch.distributed.ReduceOp.AVG)

        val_gaussian = torch.tensor(loss_meters.gaussian_meter.avg).to(device=device)
        dist.all_reduce(val_gaussian, op=torch.distributed.ReduceOp.AVG)

        val_gaussian3D = torch.tensor(loss_meters.gaussian3D_meter.avg).to(device=device)
        dist.all_reduce(val_gaussian3D, op=torch.distributed.ReduceOp.AVG)
    else:
        val_loss = val_loss_meter.avg
        val_mse = loss_meters.mse_meter.avg
        val_l1 = loss_meters.l1_meter.avg
        val_ssim = loss_meters.ssim_meter.avg
        val_ssim3D = loss_meters.ssim3D_meter.avg
        val_psnr_loss = loss_meters.psnr_loss_meter.avg
        val_psnr = loss_meters.psnr_meter.avg
        val_perp = loss_meters.perp_meter.avg
        val_gaussian = loss_meters.gaussian_meter.avg
        val_gaussian3D = loss_meters.gaussian3D_meter.avg

    if rank<=0:

        log_str = create_log_str(c, epoch, rank, 
                                None, 
                                -1, -1,
                                val_loss, 
                                -1,
                                loss_meters, 
                                -1, 
                                id)

        logging.info(log_str)

    return val_loss, val_mse, val_l1, val_ssim, val_ssim3D, val_psnr, val_perp, val_gaussian, val_gaussian3D

# -------------------------------------------------------------------------------------------------
def _apply_model(model, x, g, scaling_factor, config, device, overlap=None):
    """Apply the inference

    Input
        x : [1, T, 1, H, W], attention is alone T
        g : [1, T, 1, H, W]

    Output
        res : [1, T, Cout, H, W]
    """
    c = config

    x *= scaling_factor

    B, T, C, H, W = x.shape

    if config.complex_i:
        input = np.concatenate((x.real, x.imag, g), axis=2)
    else:
        input = np.concatenate((np.abs(x), g), axis=2)

    if not c.pad_time:
        cutout = (T, c.height[-1], c.width[-1])
        if overlap is None: overlap = (0, c.height[-1]//2, c.width[-1]//2)
    else:
        cutout = (c.time, c.height[-1], c.width[-1])
        if overlap is None: overlap = (c.time//2, c.height[-1]//2, c.width[-1]//2)

    try:
        _, output = running_inference(model, input, cutout=cutout, overlap=overlap, batch_size=2, device=device)
    except Exception as e:
        print(e)
        print(f"{Fore.YELLOW}---> call inference on cpu ...")
        _, output = running_inference(model, input, cutout=cutout, overlap=overlap, device=torch.device('cpu'))

    x /= scaling_factor
    output /= scaling_factor

    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    return output

# -------------------------------------------------------------------------------------------------

def apply_model(data, model, gmap, config, scaling_factor, device=torch.device('cpu'), overlap=None, verbose=False):
    '''
    Input 
        data : [H, W, T, SLC], remove any extra scaling
        gmap : [H, W, SLC], no scaling added
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (0.5 is more smoothing than 1.0)
        overlap (T, H, W): number of overlap between patches, can be (0, 0, 0)
    Output
        res: [H, W, T, SLC]
    '''

    t0 = time()

    if(data.ndim==2):
        data = data[:,:,np.newaxis,np.newaxis]

    if(data.ndim<4):
        data = np.expand_dims(data, axis=3)

    H, W, T, SLC = data.shape

    if(gmap.ndim==2):
        gmap = np.expand_dims(gmap, axis=2)

    if(gmap.shape[0]!=H or gmap.shape[1]!=W or gmap.shape[2]!=SLC):
        gmap = np.ones(H, W, SLC)

    if verbose:
        print(f"---> apply_model, preparation took {time()-t0} seconds ")
        print(f"---> apply_model, input array {data.shape}")
        print(f"---> apply_model, gmap array {gmap.shape}")
        print(f"---> apply_model, pad_time {config.pad_time}")
        print(f"---> apply_model, height and width {config.height, config.width}")
        print(f"---> apply_model, complex_i {config.complex_i}")
        print(f"---> apply_model, scaling_factor {scaling_factor}")
        print(f"---> apply_model, overlap {overlap}")

    c = config

    try:
        for k in range(SLC):
            imgslab = data[:,:,:,k]
            gmapslab = gmap[:,:,k]

            H, W, T = imgslab.shape

            x = np.transpose(imgslab, [2, 0, 1]).reshape([1, T, 1, H, W])
            g = np.repeat(gmapslab[np.newaxis, np.newaxis, np.newaxis, :, :], T, axis=1)

            print(f"---> running_inference, input {x.shape} for slice {k}")
            output = _apply_model(model, x, g, scaling_factor, config, device, overlap)

            output = np.transpose(output, (3, 4, 2, 1, 0))

            if(k==0):
                if config.complex_i:
                    data_filtered = np.zeros((output.shape[0], output.shape[1], T, SLC), dtype=data.dtype)
                else:
                    data_filtered = np.zeros((output.shape[0], output.shape[1], T, SLC), dtype=np.float32)

            if config.complex_i:
                data_filtered[:,:,:,k] = output[:,:,0,:,0] + 1j*output[:,:,1,:,0]
            else:
                data_filtered[:,:,:,k] = output.squeeze()

    except Exception as e:
        print(e)
        data_filtered = copy.deepcopy(data)

    t1 = time()
    print(f"---> apply_model took {t1-t0} seconds ")

    return data_filtered

# -------------------------------------------------------------------------------------------------

def apply_model_3D(data, model, gmap, config, scaling_factor, device='cpu', overlap=None, verbose=False):
    '''
    Input 
        data : [H W SLC], remove any extra scaling
        gmap : [H W SLC], no scaling added
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (0.5 is more smoothing than 1.0)
    Output
        res : [H W SLC]
    '''

    t0 = time()

    H, W, SLC = data.shape

    if(gmap.shape[0]!=H or gmap.shape[1]!=W or gmap.shape[2]!=SLC):
        gmap = np.ones(H, W, SLC)

    if verbose:
        print(f"---> apply_model_3D, preparation took {time()-t0} seconds ")
        print(f"---> apply_model_3D, input array {data.shape}")
        print(f"---> apply_model_3D, gmap array {gmap.shape}")
        print(f"---> apply_model_3D, pad_time {config.pad_time}")
        print(f"---> apply_model_3D, height and width {config.height, config.width}")
        print(f"---> apply_model_3D, complex_i {config.complex_i}")
        print(f"---> apply_model_3D, scaling_factor {scaling_factor}")

    c = config

    try:
        x = np.transpose(data, [2, 0, 1]).reshape([1, SLC, 1, H, W])
        g = np.transpose(gmap, [2, 0, 1]).reshape([1, SLC, 1, H, W])

        print(f"---> running_inference, input {x.shape} for volume")
        output = _apply_model(model, x, g, scaling_factor, config, device, overlap)

        output = np.transpose(output, (3, 4, 2, 1, 0)) # [H, W, Cout, SLC, 1]

        if config.complex_i:
            data_filtered = output[:,:,0,:,0] + 1j*output[:,:,1,:,0]
        else:
            data_filtered = output

        data_filtered = np.reshape(data_filtered, (H, W, SLC))

    except Exception as e:
        print(e)
        data_filtered = copy.deepcopy(data)

    t1 = time()
    print(f"---> apply_model_3D took {t1-t0} seconds ")

    return data_filtered

# -------------------------------------------------------------------------------------------------

def apply_model_2D(data, model, gmap, config, scaling_factor, device='cpu', overlap=None, verbose=False):
    '''
    Input 
        data : [H W SLC], remove any extra scaling
        gmap : [H W SLC], no scaling added
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (0.5 is more smoothing than 1.0)
    Output
        res : [H W SLC]
        
    Attention is performed within every 2D image.
    '''

    t0 = time()

    H, W, SLC = data.shape

    if(gmap.shape[0]!=H or gmap.shape[1]!=W or gmap.shape[2]!=SLC):
        gmap = np.ones(H, W, SLC)

    if verbose:
        print(f"---> apply_model_2D, preparation took {time()-t0} seconds ")
        print(f"---> apply_model_2D, input array {data.shape}")
        print(f"---> apply_model_2D, gmap array {gmap.shape}")
        print(f"---> apply_model_2D, pad_time {config.pad_time}")
        print(f"---> apply_model_2D, height and width {config.height, config.width}")
        print(f"---> apply_model_2D, complex_i {config.complex_i}")
        print(f"---> apply_model_2D, scaling_factor {scaling_factor}")

    c = config

    try:
        x = np.transpose(data, [2, 0, 1]).reshape([SLC, 1, 1, H, W])
        g = np.transpose(gmap, [2, 0, 1]).reshape([SLC, 1, 1, H, W])

        output = np.zeros([SLC, 1, 1, H, W])

        print(f"---> running_inference, input {x.shape} for 2D")
        for slc in range(SLC):
            output[slc] = _apply_model(model, x[slc], g[slc], scaling_factor, config, device, overlap)

        output = np.transpose(output, (3, 4, 2, 1, 0)) # [H, W, Cout, 1, SLC]

        if config.complex_i:
            data_filtered = output[:,:,0,:,:] + 1j*output[:,:,1,:,:]
        else:
            data_filtered = output

        data_filtered = np.reshape(data_filtered, (H, W, SLC))

    except Exception as e:
        print(e)
        data_filtered = copy.deepcopy(data)

    t1 = time()
    print(f"---> apply_model_2D took {t1-t0} seconds ")

    return data_filtered

# -------------------------------------------------------------------------------------------------

def compare_model(config, model, model_jit, model_onnx, device='cpu', x=None):
    """
    Compare onnx, pts and pt models
    """
    c = config

    C = 3 if config.complex_i else 2

    if x is None:
        x = np.random.randn(1, 12, C, 128, 128).astype(np.float32)

    B, T, C, H, W = x.shape

    model.to(device=device)
    model.eval()

    cutout_in = (c.time, c.height[-1], c.width[-1])
    overlap_in = (c.time//2, c.height[-1]//2, c.width[-1]//2)

    tm = start_timer(enable=True)    
    y, y_model = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device=device)
    end_timer(enable=True, t=tm, msg="torch model took")

    tm = start_timer(enable=True)
    y_onnx, y_model_onnx = running_inference(model_onnx, x, cutout=cutout_in, overlap=overlap_in, device=device)
    end_timer(enable=True, t=tm, msg="onnx model took")

    diff = np.linalg.norm(y-y_onnx)
    print(f"--> {Fore.GREEN}Onnx model difference is {diff} ... {Style.RESET_ALL}", flush=True)

    tm = start_timer(enable=True)
    y_jit, y_model_jit = running_inference(model_jit, x, cutout=cutout_in, overlap=overlap_in, device=device)
    end_timer(enable=True, t=tm, msg="torch script model took")

    diff = np.linalg.norm(y-y_jit)
    print(f"--> {Fore.GREEN}Jit model difference is {diff} ... {Style.RESET_ALL}", flush=True)

    diff = np.linalg.norm(y_onnx-y_jit)
    print(f"--> {Fore.GREEN}Jit - onnx model difference is {diff} ... {Style.RESET_ALL}", flush=True)

# -------------------------------------------------------------------------------------------------

def load_model(saved_model_path, saved_model_config=None, model_type=None):
    """
    load a ".pt" or ".pts" model
    @rets:
        - model (torch model): the model ready for inference
    """

    config = []

    config_file = saved_model_config
    if config_file is not None and os.path.isfile(config_file):
        print(f"{Fore.YELLOW}Load in config file - {config_file}{Style.RESET_ALL}")
        with open(config_file, 'rb') as f:
            config = pickle.load(f)

    if saved_model_path.endswith(".pt") or saved_model_path.endswith(".pth"):

        #status = torch.load(saved_model_path, map_location=get_device())
        status = torch.load(saved_model_path)
        config = status['config']

        if not torch.cuda.is_available():
            config.device = torch.device('cpu')

        if model_type is not None:
            config.model_type = model_type
            print(f"Use the input model type - {model_type}")

        model = create_model(config, config.model_type, total_steps=-1)

        if 'model' in status:
            print(f"{Fore.YELLOW}Load in model {Style.RESET_ALL}")
            model.load_state_dict(status['model'])
        elif 'model_state' in status:
            print(f"{Fore.YELLOW}Load in model_state {Style.RESET_ALL}")
            model.load_state_dict(status['model_state'])
        elif 'backbone_state' in status:
            print(f"{Fore.YELLOW}Load in pre/backbone/post states{Style.RESET_ALL}")
            model.pre.load_state_dict(status['pre_state'])
            model.backbone.load_state_dict(status['backbone_state'])
            model.post.load_state_dict(status['post_state'])
            model.a = status['a']
            model.b = status['b']

    elif saved_model_path.endswith(".pts"):
        model = torch.jit.load(saved_model_path, map_location=get_device())
    else:
        model, _ = load_model_onnx(model_dir="", model_file=saved_model_path, use_cpu=True)
    return model, config
