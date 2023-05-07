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

# -------------------------------------------------------------------------------------------------
# trainer

def trainer(rank, model, config, train_set, val_set, test_set):
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

    if c.ddp:
        dist.init_process_group("nccl", rank=rank, world_size=c.world_size)
        device = rank
        model = model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optim = model.module.optim
        sched = model.module.sched
        stype = model.module.stype
        samplers = [DistributedSampler(train_set_x) for train_set_x in train_set]
        shuffle = False
        loss_f = model.module.loss_f
    else:
        # No init required if not ddp
        device = c.device
        model = model.to(device)
        optim = model.optim
        sched = model.sched
        stype = model.stype
        samplers = [None for _ in train_set]
        shuffle = True
        loss_f = model.loss_f

    train_loader = [DataLoader(dataset=train_set_x, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers[i],
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0) for i, train_set_x in enumerate(train_set)]

    if rank<=0: # main or master process
        if c.ddp: setup_logger(config) # setup master process logging

        wandb.init(project=c.project, entity=c.wandb_entity, config=c,
                    name=c.run_name, notes=c.run_notes)
        wandb.watch(model)
        wandb.log({"trainable_params":c.trainable_params,
                    "total_params":c.total_params})

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
        psnr_func = PSNR()

    total_iters = sum([len(loader_x) for loader_x in train_loader])
    total_iters = total_iters if not c.debug else min(10, total_iters)

    for epoch in range(c.num_epochs):
        if rank<=0:
            logging.info(f"{'-'*20}Epoch:{epoch}/{c.num_epochs}{'-'*20}")

            train_loss.reset()
            train_mse_meter.reset()
            train_l1_meter.reset()
            train_ssim_meter.reset()
            train_ssim3D_meter.reset()
            train_psnr_meter.reset()

        model.train()
        if c.ddp: [loader_x.sampler.set_epoch(epoch) for loader_x in train_loader]

        train_loader_iter = [iter(loader_x) for loader_x in train_loader]
        with tqdm(total=total_iters, disable=rank>0) as pbar:

            for idx in range(total_iters):

                optim.zero_grad()

                loader_ind = idx % len(train_loader_iter)
                stuff = next(train_loader_iter[loader_ind], None)
                while stuff is None:
                    del train_loader_iter[loader_ind]
                    loader_ind = idx % len(train_loader_iter)
                    stuff = next(train_loader_iter[loader_ind], None)
                x, y, gmaps_median, noise_sigmas = stuff

                x = x.to(device)
                y = y.to(device)

                output = model(x)
                loss = loss_f(output, y)
                loss.backward()

                if(c.clip_grad_norm>0):
                    nn.utils.clip_grad_norm_(model.parameters(), c.clip_grad_norm)
                optim.step()

                if stype == "OneCycleLR": sched.step()
                curr_lr = optim.param_groups[0]['lr']

                if rank<=0:
                    wandb.log({"running_train_loss": loss.item()})
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
                                            f"{loss.item():.4f}, {mse_loss:.4f}, {l1_loss:.4f}, "+
                                            f"{ssim_loss:.4f}, {ssim3D_loss:.4f}, {psnr:.4f}, "+
                                            f"lr {curr_lr:.8f}")

            if rank<=0:
                pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, tra, {x.shape}, {train_loss.avg:.4f}, "+
                                        f"{train_mse_meter.avg:.4f}, {train_l1_meter.avg:.4f}, {train_ssim_meter.avg:.4f}, "+
                                        f"{train_ssim3D_meter.avg:.4f}, {train_psnr_meter.avg:.4f}, lr {curr_lr:.8f}")

        if rank<=0: # main or master process
            # run eval, save and log in this process
            model_e = model.module if c.ddp else model
            val_losses = eval_val(model_e, c, val_set, epoch, device)
            if val_losses[0] < best_val_loss:
                best_val_loss = val_losses[0]
                best_model_wts = copy.deepcopy(model_e.state_dict())

            # silently log to only the file as well
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, tra, {x.shape}, {train_loss.avg:.4f}, "+
                                                f"{train_mse_meter.avg:.4f}, {train_l1_meter.avg:.4f}, {train_ssim_meter.avg:.4f}, "+
                                                f"{train_ssim3D_meter.avg:.4f}, {train_psnr_meter.avg:.4f}, lr {curr_lr:.8f}")
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, val, {x.shape}, {val_losses[0]:.4f}, "+
                                                f"{val_losses[1]:.4f}, {val_losses[2]:.4f}, {val_losses[3]:.4f}, "+
                                                f"{val_losses[4]:.4f}, {val_losses[5]:.4f}, lr {curr_lr:.8f}")

            # save the model weights every save_cycle
            if epoch % c.save_cycle == 0:
                model_e.save(epoch)

            wandb.log({"epoch": epoch,
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
        wandb.log({"best_val_loss": best_val_loss})

        model = model.module if c.ddp else model
        model.save(epoch) # save the final weights
        # test last model
        eval_test(model, config, test_set=test_set, device=device, id="last")
        # test best model
        model.load_state_dict(best_model_wts)
        eval_test(model, config, test_set=test_set, device=device, id="best")
        # save both models
        save_final_model(model, config, best_model_wts)

    if c.ddp: # cleanup
        dist.destroy_process_group()

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
    psnr_func = PSNR()

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
                    save_image_wandb(title, c.complex_i, x.numpy(force=True), output.numpy(force=True), y.numpy(force=True))

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
