"""
Trainer for MRI denoising.
Provides the mian function to call for training:
    - trainer
"""
import cv2
import copy
import wandb
import numpy
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

from utils.utils import *
from eval_mri import eval_test
from model_base.losses import *
from utils.save_model import save_final_model
from utils.running_inference import running_inference

from model_mri import STCNNT_MRI
from data_mri import load_mri_data

from colorama import Fore, Style

# -------------------------------------------------------------------------------------------------
# trainer

def trainer(rank, config, wandb_run):
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
    c = config # shortening due to numerous uses

    train_set, val_set, test_set = load_mri_data(config=config)
    
    total_num_samples = sum([len(s) for s in train_set])
    
    total_steps = compute_total_steps(config, total_num_samples)
    logging.info(f"total_steps for this run: {total_steps}, len(train_set) {[len(s) for s in train_set]}, batch {config.batch_size}")
    
    if config.ddp:
        config.device = torch.device(f'cuda:{rank}')
    
    model = STCNNT_MRI(config=config, total_steps=total_steps)
    
    if config.ddp:
        dist.barrier()

    if rank<=0:
                        
        # model summary
        model_summary = model_info(model, config)
        logging.info(f"Configuration for this run:\n{config}")
        logging.info(f"Model Summary:\n{str(model_summary)}")
        logging.info(f"Wandb name:\n{wandb_run.name}")
                
        wandb_run.watch(model)
        
    # -----------------------------------------------
    
    if c.ddp:
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optim = model.module.optim
        sched = model.module.sched
        stype = model.module.stype
        loss_f = model.module.loss_f
        curr_epoch = model.module.curr_epoch
        samplers = [DistributedSampler(train_set_x) for train_set_x in train_set]
        shuffle = False
        
        logging.info(f"{Fore.RED}{'-'*20}Local Rank:{rank}, {c.backbone}, C {c.backbone_hrnet.C}, {c.n_head} heads, scale_ratio_in_mixer {c.scale_ratio_in_mixer}, {c.backbone_hrnet.block_str}, {'-'*20}{Style.RESET_ALL}")
    else:
        # No init required if not ddp
        device = c.device
        model = model.to(device)
        optim = model.optim
        sched = model.sched
        stype = model.stype
        loss_f = model.loss_f
        curr_epoch = model.curr_epoch
        samplers = [None for _ in train_set]
        shuffle = True
        
    # -----------------------------------------------
    
    train_loader = [DataLoader(dataset=train_set_x, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers[i],
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=True,
                                persistent_workers=c.num_workers>0) for i, train_set_x in enumerate(train_set)]

    # -----------------------------------------------
    
    if rank<=0: # main or master process
        if c.ddp: setup_logger(config) # setup master process logging

        wandb_run.watch(model)
        wandb_run.summary["trainable_params"] = c.trainable_params
        wandb_run.summary["total_params"] = c.total_params

        wandb_run.define_metric("epoch")    
        wandb_run.define_metric("train_loss_avg", step_metric='epoch')
        wandb_run.define_metric("train_mse_loss", step_metric='epoch')
        wandb_run.define_metric("train_l1_loss", step_metric='epoch')
        wandb_run.define_metric("train_ssim_loss", step_metric='epoch')
        wandb_run.define_metric("train_ssim3D_loss", step_metric='epoch')
        wandb_run.define_metric("train_psnr", step_metric='epoch')
        wandb_run.define_metric("val_loss_avg", step_metric='epoch')
        wandb_run.define_metric("val_mse_loss", step_metric='epoch')
        wandb_run.define_metric("val_l1_loss", step_metric='epoch')
        wandb_run.define_metric("val_ssim_loss", step_metric='epoch')
        wandb_run.define_metric("val_ssim3D_loss", step_metric='epoch')
        wandb_run.define_metric("val_psnr", step_metric='epoch')                            
        
        # log a few training examples
        for i, train_set_x in enumerate(train_set):            
            ind = np.random.randint(0, len(train_set_x), 8)
            x, y, gmaps_median, noise_sigmas = train_set_x[ind[0]]
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)
            for ii in range(1, len(ind)):                
                a_x, a_y, gmaps_median, noise_sigmas = train_set_x[ii]
                x = np.concatenate((x, np.expand_dims(a_x, axis=0)), axis=0)
                y = np.concatenate((y, np.expand_dims(a_y, axis=0)), axis=0)
                
            title = f"Tra_samples_{i}_Noisy_Noisy_GT_{x.shape}"
            save_image_batch_wandb(title, c.complex_i, x, np.copy(x), y)

    # -----------------------------------------------
    # save best model to be saved at the end
    best_val_loss = numpy.inf
    best_model_wts = copy.deepcopy(model.module.state_dict() if c.ddp else model.state_dict())

    train_loss = AverageMeter()

    train_mse_meter = AverageMeter()
    train_l1_meter = AverageMeter()
    train_ssim_meter = AverageMeter()
    train_ssim3D_meter = AverageMeter()
    train_psnr_meter = AverageMeter()

    mse_loss_func = MSE_Loss(complex_i=c.complex_i)
    l1_loss_func = L1_Loss(complex_i=c.complex_i)
    ssim_loss_func = SSIM_Loss(complex_i=c.complex_i, device=device)
    ssim3D_loss_func = SSIM3D_Loss(complex_i=c.complex_i, device=device)
    psnr_func = PSNR(range=1024)
        
    # -----------------------------------------------
    
    total_iters = sum([len(loader_x) for loader_x in train_loader])
    total_iters = total_iters if not c.debug else min(10, total_iters)

    # mix precision training
    scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp)
    
    optim.zero_grad(set_to_none=True)
    
    # -----------------------------------------------
    
    for epoch in range(c.num_epochs):
        logging.info(f"{Fore.GREEN}{'-'*20}Epoch:{epoch}/{c.num_epochs}, rank {rank} {'-'*20}{Style.RESET_ALL}")

        train_loss.reset()
        train_mse_meter.reset()
        train_l1_meter.reset()
        train_ssim_meter.reset()
        train_ssim3D_meter.reset()
        train_psnr_meter.reset()

        model.train()
        if c.ddp: [loader_x.sampler.set_epoch(epoch) for loader_x in train_loader]

        train_loader_iter = [iter(loader_x) for loader_x in train_loader]
        with tqdm(total=total_iters) as pbar:

            for idx in range(total_iters):

                loader_ind = idx % len(train_loader_iter)
                stuff = next(train_loader_iter[loader_ind], None)
                while stuff is None:
                    del train_loader_iter[loader_ind]
                    loader_ind = idx % len(train_loader_iter)
                    stuff = next(train_loader_iter[loader_ind], None)
                x, y, gmaps_median, noise_sigmas = stuff

                x = x.to(device)
                y = y.to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=c.use_amp):
                    output = model(x)
                    loss = loss_f(output, y)
                    loss = loss / c.iters_to_accumulate
                    
                scaler.scale(loss).backward()

                if (idx + 1) % c.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                    if(c.clip_grad_norm>0):
                        scaler.unscale_(optim)
                        nn.utils.clip_grad_norm_(model.parameters(), c.clip_grad_norm)
                    
                    scaler.step(optim)                    
                    optim.zero_grad(set_to_none=True)
                    scaler.update()
                
                    if stype == "OneCycleLR": sched.step()
                    
                curr_lr = optim.param_groups[0]['lr']
                
                if rank<=0:
                    wandb_run.log({"running_train_loss": loss.item()})
                    wandb_run.log({"lr": curr_lr})
                    
                    total=x.shape[0]
                    train_loss.update(loss.item(), n=total)

                    mse_loss = mse_loss_func(output, y).item()
                    l1_loss = l1_loss_func(output, y).item()
                    ssim_loss = ssim_loss_func(output, y).item()
                    ssim3D_loss = ssim3D_loss_func(output, y).item()
                    psnr = psnr_func(output, y).item()

                    train_mse_meter.update(mse_loss, n=total)
                    train_l1_meter.update(l1_loss, n=total)
                    train_ssim_meter.update(ssim_loss, n=total)
                    train_ssim3D_meter.update(ssim3D_loss, n=total)
                    train_psnr_meter.update(psnr, n=total)

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, tra, {x.shape}, "+
                                        f"loss {loss.item():.4f}, mse {mse_loss:.4f}, l1 {l1_loss:.4f}, "+
                                        f"ssim {ssim_loss:.4f}, ssim3D {ssim3D_loss:.4f}, psnr {psnr:.4f}, "+
                                        f"lr {curr_lr:.8f}")

            pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, tra, loss {train_loss.avg:.4f}, "+
                                    f"mse {train_mse_meter.avg:.4f}, l1 {train_l1_meter.avg:.4f}, ssim {train_ssim_meter.avg:.4f}, "+
                                    f"ssim3D {train_ssim3D_meter.avg:.4f}, psnr {train_psnr_meter.avg:.4f}, lr {curr_lr:.8f}")

        # -------------------------------------------------------
        
        if rank<=0: # main or master process
            # run eval, save and log in this process
            model_e = model.module if c.ddp else model
            val_losses = eval_val(model_e, c, val_set, epoch, device)
            if val_losses[0] < best_val_loss:
                best_val_loss = val_losses[0]
                best_model_wts = copy.deepcopy(model_e.state_dict())
                model_e.save(epoch)
                wandb_run.log({"epoch": epoch, "best_val_loss":best_val_loss})
                
            # silently log to only the file as well
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, tra, {x.shape}, {train_loss.avg:.4f}, "+
                                                f"{train_mse_meter.avg:.4f}, {train_l1_meter.avg:.4f}, {train_ssim_meter.avg:.4f}, "+
                                                f"{train_ssim3D_meter.avg:.4f}, {train_psnr_meter.avg:.4f}, lr {curr_lr:.8f}")
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, val, {x.shape}, {val_losses[0]:.4f}, "+
                                                f"{val_losses[1]:.4f}, {val_losses[2]:.4f}, {val_losses[3]:.4f}, "+
                                                f"{val_losses[4]:.4f}, {val_losses[5]:.4f}, lr {curr_lr:.8f}")
        
            wandb_run.log({"epoch": epoch,
                        "train_loss_avg": train_loss.avg,
                        "train_mse_loss": train_mse_meter.avg,
                        "train_l1_loss": train_l1_meter.avg,
                        "train_ssim_loss": train_ssim_meter.avg,
                        "train_ssim3D_loss": train_ssim3D_meter.avg,
                        "train_psnr": train_psnr_meter.avg,
                        "val_loss_avg": val_losses[0],
                        "val_mse_loss": val_losses[1],
                        "val_l1_loss": val_losses[2],
                        "val_ssim_loss": val_losses[3],
                        "val_ssim3D_loss": val_losses[4],
                        "val_psnr": val_losses[5],})

            if stype == "ReduceLROnPlateau":
                sched.step(val_losses[0])
            else: # stype == "StepLR"
                sched.step()

            if c.ddp:
                new_lr_0 = torch.zeros(1).to(rank)
                new_lr_0[0] = optim.param_groups[0]["lr"]
                dist.broadcast(new_lr_0, src=0)

                if not c.all_w_decay:
                    new_lr_1 = torch.zeros(1).to(rank)
                    new_lr_1[0] = optim.param_groups[1]["lr"]
                    dist.broadcast(new_lr_1, src=0)
        else: # child processes
            new_lr_0 = torch.zeros(1).to(rank)
            dist.broadcast(new_lr_0, src=0)
            optim.param_groups[0]["lr"] = new_lr_0.item()

            if not c.all_w_decay:
                new_lr_1 = torch.zeros(1).to(rank)
                dist.broadcast(new_lr_1, src=0)
                optim.param_groups[1]["lr"] = new_lr_1.item()

    if rank<=0: # main or master process
        # test and save model
        wandb_run.run.summary["best_val_loss"] = best_val_loss

        model = model.module if c.ddp else model
        model.save(epoch) # save the final weights
        # test last model
        eval_test(model, config, test_set=test_set, device=device, id="last")
        # test best model
        model.load_state_dict(best_model_wts)
        eval_test(model, config, test_set=test_set, device=device, id="best")
        # save both models
        save_final_model(model, config, best_model_wts)

# -------------------------------------------------------------------------------------------------
# evaluate the val set

def eval_val(model, config, val_set, epoch, device):
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
    """
    c = config # shortening due to numerous uses

    val_loader = [DataLoader(dataset=val_set_x, batch_size=1, shuffle=False, sampler=None,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0) for val_set_x in val_set]

    loss_f = model.loss_f

    val_loss_meter = AverageMeter()
    val_mse_meter = AverageMeter()
    val_l1_meter = AverageMeter()
    val_ssim_meter = AverageMeter()
    val_ssim3D_meter = AverageMeter()
    val_psnr_meter = AverageMeter()

    mse_loss_func = MSE_Loss(complex_i=c.complex_i)
    l1_loss_func = L1_Loss(complex_i=c.complex_i)
    ssim_loss_func = SSIM_Loss(complex_i=c.complex_i, device=device)
    ssim3D_loss_func = SSIM3D_Loss(complex_i=c.complex_i, device=device)
    psnr_func = PSNR(range=1024)

    model.eval()
    model.to(device)

    cutout = (c.time, c.height[-1], c.width[-1])
    overlap = (c.time//4, c.height[-1]//4, c.width[-1]//4)

    val_loader_iter = [iter(val_loader_x) for val_loader_x in val_loader]
    total_iters = sum([len(loader_x) for loader_x in val_loader])
    total_iters = total_iters if not c.debug else min(2, total_iters)

    images_logged = 0

    with torch.no_grad():
        with tqdm(total=total_iters) as pbar:

            for idx in range(total_iters):

                loader_ind = idx % len(val_loader_iter)
                batch = next(val_loader_iter[loader_ind], None)
                while batch is None:
                    del val_loader_iter[loader_ind]
                    loader_ind = idx % len(val_loader_iter)
                    batch = next(val_loader_iter[loader_ind], None)
                x, y, gmaps_median, noise_sigmas = batch

                two_D = False
                cutout_in = cutout
                overlap_in = overlap
                if x.shape[1]==1:
                    xy, og_shape, pt_shape = cut_into_patches([x,y], cutout=cutout[1:])
                    x, y = xy[0], xy[1]
                    cutout_in = (c.twoD_num_patches_cutout, *cutout[1:])
                    overlap_in = (c.twoD_num_patches_cutout//4, *overlap[1:])
                    two_D = True

                x = x.to(device)
                y = y.to(device)

                try:
                    _, output = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device=device)
                except:
                    _, output = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device="cpu")
                    y = y.to("cpu")

                if two_D:
                    xy = repatch([x,output,y], og_shape, pt_shape)
                    x, output, y = xy[0], xy[1], xy[2]

                if images_logged < 8:
                    images_logged += 1
                    title = f"Val_image_{idx}_Noisy_Pred_GT_{x.shape}"
                    save_image_batch_wandb(title, c.complex_i, x.numpy(force=True), output.numpy(force=True), y.numpy(force=True))

                loss = loss_f(output, y)

                mse_loss = mse_loss_func(output, y).item()
                l1_loss = l1_loss_func(output, y).item()
                ssim_loss = ssim_loss_func(output, y).item()
                ssim3D_loss = ssim3D_loss_func(output, y).item()
                psnr = psnr_func(output, y).item()

                total = x.shape[0]

                val_loss_meter.update(loss.item(), n=total)
                val_mse_meter.update(mse_loss, n=total)
                val_l1_meter.update(l1_loss, n=total)
                val_ssim_meter.update(ssim_loss, n=total)
                val_ssim3D_meter.update(ssim3D_loss, n=total)
                val_psnr_meter.update(psnr, n=total)

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, val, {x.shape}, "+
                                        f"{loss.item():.4f}, {mse_loss:.4f}, {l1_loss:.4f}, "+
                                        f"{ssim_loss:.4f}, {ssim3D_loss:.4f}, {psnr:.4f},")

            pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, val, {x.shape}, {val_loss_meter.avg:.4f}, "+
                                    f"{val_mse_meter.avg:.4f}, {val_l1_meter.avg:.4f}, {val_ssim_meter.avg:.4f}, "+
                                    f"{val_ssim3D_meter.avg:.4f}, {val_psnr_meter.avg:.4f}")

    return val_loss_meter.avg, val_mse_meter.avg, val_l1_meter.avg, val_ssim_meter.avg, val_ssim3D_meter.avg, val_psnr_meter.avg
