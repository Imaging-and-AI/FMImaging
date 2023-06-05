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

from model_mri import STCNNT_MRI
from data_mri import MRIDenoisingDatasetTrain, load_mri_data

from colorama import Fore, Back, Style

# -------------------------------------------------------------------------------------------------
# trainer

def create_log_str(config, epoch, rank, data_shape, loss, mse, l1, ssim, ssim3d, psnr, curr_lr, role):
    if data_shape is not None:
        data_shape_str = f"{data_shape} "
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
        
    str= f"{Fore.GREEN}Epoch {epoch}/{config.num_epochs}, {C}{role}, {Style.RESET_ALL}rank {rank}, " + data_shape_str + f"{Fore.BLUE}{Back.WHITE}{Style.BRIGHT}loss {loss:.4f},{Style.RESET_ALL} {C}mse {mse:.4f}, l1 {l1:.4f}, ssim {ssim:.4f}, ssim3D {ssim3d:.4f}, psnr {psnr:.4f}{Style.RESET_ALL}{lr_str}"
        
    return str

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
    c = config # shortening due to numerous uses

    start = time()
    train_set, val_set, test_set = load_mri_data(config=config)
    logging.info(f"load_mri_data took {time() - start} seconds ...")
    
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
        
        if wandb_run is not None:
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
        ssim_loss_f = model.module.ssim_loss_f
        curr_epoch = model.module.curr_epoch
        samplers = [DistributedSampler(train_set_x, shuffle=True) for train_set_x in train_set]
        shuffle = False        
    else:
        # No init required if not ddp
        device = c.device
        model = model.to(device)
        optim = model.optim
        sched = model.sched
        stype = model.stype
        loss_f = model.loss_f
        ssim_loss_f = model.ssim_loss_f
        curr_epoch = model.curr_epoch
        samplers = [None for _ in train_set]
        shuffle = True
        
    if c.backbone == 'hrnet':
        logging.info(f"{Fore.RED}{'-'*20}Local Rank:{rank}, global rank: {global_rank}, {c.backbone}, {c.a_type}, {c.cell_type}, snr perturb {c.snr_perturb_prob}, optim {c.optim}, {c.norm_mode}, C {c.backbone_hrnet.C}, {c.n_head} heads, scale_ratio_in_mixer {c.scale_ratio_in_mixer}, {c.backbone_hrnet.block_str}, {'-'*20}{Style.RESET_ALL}")
    elif c.backbone == 'unet':
        logging.info(f"{Fore.RED}{'-'*20}Local Rank:{rank}, global rank: {global_rank}, {c.backbone}, {c.a_type}, {c.cell_type}, snr perturb {c.snr_perturb_prob}, optim {c.optim}, {c.norm_mode}, C {c.backbone_unet.C}, {c.n_head} heads, scale_ratio_in_mixer {c.scale_ratio_in_mixer}, {c.backbone_unet.block_str}, {'-'*20}{Style.RESET_ALL}")
        
    # -----------------------------------------------
    
    train_loader = [DataLoader(dataset=train_set_x, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers[i],
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=True,
                                persistent_workers=c.num_workers>0) for i, train_set_x in enumerate(train_set)]

    # -----------------------------------------------
    
    if rank<=0: # main or master process
        if c.ddp: setup_logger(config) # setup master process logging

        if wandb_run is not None:
            wandb_run.watch(model)
            wandb_run.summary["trainable_params"] = c.trainable_params
            wandb_run.summary["total_params"] = c.total_params
            wandb_run.summary["total_mult_adds"] = c.total_mult_adds 

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
                ind = np.random.randint(0, len(train_set_x), 4)
                x, y, gmaps_median, noise_sigmas = train_set_x[ind[0]]
                x = np.expand_dims(x, axis=0)
                y = np.expand_dims(y, axis=0)
                for ii in range(1, len(ind)):                
                    a_x, a_y, gmaps_median, noise_sigmas = train_set_x[ind[ii]]
                    x = np.concatenate((x, np.expand_dims(a_x, axis=0)), axis=0)
                    y = np.concatenate((y, np.expand_dims(a_y, axis=0)), axis=0)
                    
                title = f"Tra_samples_{i}_Noisy_Noisy_GT_{x.shape}"
                vid = save_image_batch(c.complex_i, x, np.copy(x), y)
                wandb_run.log({title:wandb.Video(vid, caption=f"Tra sample {i}", fps=1, format='gif')})
                logging.info(f"{Fore.YELLOW}---> Upload tra sample - {title}")
                         
    # -----------------------------------------------
    # save best model to be saved at the end
    best_val_loss = np.inf
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
    
    for epoch in range(curr_epoch, c.num_epochs):
        logging.info(f"{Fore.GREEN}{'-'*20}Epoch:{epoch}/{c.num_epochs}, rank {rank}, global rank {global_rank} {'-'*20}{Style.RESET_ALL}")

        train_loss.reset()
        train_mse_meter.reset()
        train_l1_meter.reset()
        train_ssim_meter.reset()
        train_ssim3D_meter.reset()
        train_psnr_meter.reset()

        model.train()
        if c.ddp: [loader_x.sampler.set_epoch(epoch) for loader_x in train_loader]

        train_loader_iter = [iter(loader_x) for loader_x in train_loader]
        with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

            for idx in range(total_iters):

                loader_ind = idx % len(train_loader_iter)
                
                tm = start_timer(enable=c.with_timer)
                stuff = next(train_loader_iter[loader_ind], None)
                while stuff is None:
                    del train_loader_iter[loader_ind]
                    loader_ind = idx % len(train_loader_iter)
                    stuff = next(train_loader_iter[loader_ind], None)
                x, y, gmaps_median, noise_sigmas = stuff                
                end_timer(enable=c.with_timer, t=tm, msg="---> load batch took ")
                
                
                tm = start_timer(enable=c.with_timer)
                x = x.to(device)
                y = y.to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=c.use_amp):
                    output = model(x)
                    
                    if c.weighted_loss:
                        weights=noise_sigmas*gmaps_median                    
                        loss = loss_f(output, y, weights=weights.to(device))
                    else:
                        loss = loss_f(output, y)
                            
                    # if epoch <= 0.9*c.num_epochs:
                    #     if c.weighted_loss:
                    #         weights=noise_sigmas*gmaps_median                    
                    #         loss = loss_f(output, y, weights=weights.to(device))
                    #     else:
                    #         loss = loss_f(output, y)
                    # else:
                    #     loss = ssim_loss_f(output, y)
                        
                    #loss = ssim_loss_f(output, y)
                    
                    loss = loss / c.iters_to_accumulate
                    
                end_timer(enable=c.with_timer, t=tm, msg="---> forward pass took ")
                
                
                tm = start_timer(enable=c.with_timer)
                scaler.scale(loss).backward()
                end_timer(enable=c.with_timer, t=tm, msg="---> backward pass took ")
                
                
                tm = start_timer(enable=c.with_timer)
                if (idx + 1) % c.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                    if(c.clip_grad_norm>0):
                        scaler.unscale_(optim)
                        nn.utils.clip_grad_norm_(model.parameters(), c.clip_grad_norm)
                    
                    scaler.step(optim)                    
                    optim.zero_grad(set_to_none=True)
                    scaler.update()
                
                    if stype == "OneCycleLR": sched.step()
                end_timer(enable=c.with_timer, t=tm, msg="---> other steps took ")
                
                
                tm = start_timer(enable=c.with_timer)
                curr_lr = optim.param_groups[0]['lr']
                
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
                log_str = create_log_str(config, epoch, rank, 
                                         x.shape, 
                                         train_loss.avg, 
                                         train_mse_meter.avg, 
                                         train_l1_meter.avg, 
                                         train_ssim_meter.avg, 
                                         train_ssim3D_meter.avg, 
                                         train_psnr_meter.avg, 
                                         curr_lr, 
                                         "tra")
                
                pbar.set_description_str(log_str)                

                if wandb_run is not None:
                    wandb_run.log({"running_train_loss": loss.item()})
                    wandb_run.log({"lr": curr_lr})
                
                end_timer(enable=c.with_timer, t=tm, msg="---> logging and measuring took ")
                
            # ---------------------------------------
            log_str = create_log_str(c, epoch, rank, 
                                         None, 
                                         train_loss.avg, 
                                         train_mse_meter.avg, 
                                         train_l1_meter.avg, 
                                         train_ssim_meter.avg, 
                                         train_ssim3D_meter.avg, 
                                         train_psnr_meter.avg, 
                                         curr_lr, 
                                         "tra")
            
            pbar.set_description_str(log_str)

        # -------------------------------------------------------
       
        val_losses = eval_val(rank, model, c, val_set, epoch, device, wandb_run)
            
        # -------------------------------------------------------
        if rank<=0: # main or master process
            if val_losses[0] < best_val_loss:
                model_e = model.module if c.ddp else model
                best_val_loss = val_losses[0]
                best_model_wts = copy.deepcopy(model_e.state_dict())
                model_e.save(epoch)
                if wandb_run is not None:
                    wandb_run.log({"epoch": epoch, "best_val_loss":best_val_loss})
                
            # silently log to only the file as well
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, tra, {x.shape}, {train_loss.avg:.4f}, "+
                                                f"{train_mse_meter.avg:.4f}, {train_l1_meter.avg:.4f}, {train_ssim_meter.avg:.4f}, "+
                                                f"{train_ssim3D_meter.avg:.4f}, {train_psnr_meter.avg:.4f}, lr {curr_lr:.8f}")
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, val, {x.shape}, {val_losses[0]:.4f}, "+
                                                f"{val_losses[1]:.4f}, {val_losses[2]:.4f}, {val_losses[3]:.4f}, "+
                                                f"{val_losses[4]:.4f}, {val_losses[5]:.4f}, lr {curr_lr:.8f}")
        
            if wandb_run is not None:
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
        
            model = model.module if c.ddp else model
            model.save(epoch)
            
            # save both models
            fname_last, fname_best = save_final_model(model, config, best_model_wts, only_pt=False)

            logging.info(f"--> {Fore.YELLOW}Save last mode at {fname_last}{Style.RESET_ALL}")
            logging.info(f"--> {Fore.YELLOW}Save best mode at {fname_best}{Style.RESET_ALL}")
        
    # test best model, reload the weights
    model = STCNNT_MRI(config=config, total_steps=total_steps)
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

            wandb_run.save(fname_last+'.pt')
            #wandb_run.save(fname_last+'.pts')
            #wandb_run.save(fname_last+'.onnx')
            
            wandb_run.save(fname_best+'.pt')
            #wandb_run.save(fname_best+'.pts')
            #wandb_run.save(fname_best+'.onnx')
                            
            try:
                # test the best model, reloading the saved model
                model_jit = load_model(model_dir=None, model_file=fname_best+'.pts')
                model_onnx, _ = load_model_onnx(model_dir=None, model_file=fname_best+'.onnx', use_cpu=True)

                # pick a random case
                a_test_set = test_set[np.random.randint(0, len(test_set))]
                x, y, gmaps_median, noise_sigmas = a_test_set[np.random.randint(0, len(a_test_set))]
                    
                x = np.expand_dims(x, axis=0)
                y = np.expand_dims(y, axis=0)

                compare_model(config=config, model=model, model_jit=model_jit, model_onnx=model_onnx, device=device, x=x)
            except:
                print(f"--> ignore the extra tests ...")
    
    if c.ddp:
        dist.barrier()
    print(f"--> run finished ...")
            
# -------------------------------------------------------------------------------------------------
# evaluate the val set

def eval_val(rank, model, config, val_set, epoch, device, wandb_run, id="val"):
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

    val_loader = [DataLoader(dataset=val_set_x, batch_size=batch_size, shuffle=False, sampler=sampler[i],
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0) for i, val_set_x in enumerate(val_set)]

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

    if rank <= 0 and epoch < 2:
        logging.info(f"Eval height and width is {c.height[-1]}, {c.width[-1]}")

    cutout = (c.time, c.height[-1], c.width[-1])
    overlap = (c.time//2, c.height[-1]//4, c.width[-1]//4)

    val_loader_iter = [iter(val_loader_x) for val_loader_x in val_loader]
    total_iters = sum([len(loader_x) for loader_x in val_loader])
    total_iters = total_iters if not c.debug else min(2, total_iters)

    images_logged = 0
    
    with torch.inference_mode():
        with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

            for idx in range(total_iters):

                loader_ind = idx % len(val_loader_iter)
                batch = next(val_loader_iter[loader_ind], None)
                while batch is None:
                    del val_loader_iter[loader_ind]
                    loader_ind = idx % len(val_loader_iter)
                    batch = next(val_loader_iter[loader_ind], None)
                x, y, gmaps_median, noise_sigmas = batch

                if batch_size >1 and x.shape[-1]==c.width[-1]:
                    # run normal inference
                    x = x.to(device)
                    y = y.to(device)                    
                    output = model(x)
                else:
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
                    
                    B, T, C, H, W = x.shape
                    
                    if not config.pad_time:
                        cutout_in = (T, c.height[-1], c.width[-1])
                        overlap_in = (0, c.height[-1]//2, c.width[-1]//2)
                        
                    try:
                        _, output = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device=device)
                    except:
                        logging.info(f"{Fore.YELLOW}---> call inference on cpu ...")
                        _, output = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device="cpu")
                        y = y.to("cpu")

                    if two_D:
                        xy = repatch([x,output,y], og_shape, pt_shape)
                        x, output, y = xy[0], xy[1], xy[2]

                if rank<=0 and images_logged < config.num_uploaded and wandb_run is not None:
                    images_logged += 1
                    title = f"{id.upper()}_rank_{rank}_image_{idx}_Noisy_Pred_GT_{x.shape}"
                    vid = save_image_batch(c.complex_i, x.numpy(force=True), output.numpy(force=True), y.numpy(force=True))
                    wandb_run.log({title: wandb.Video(vid, caption=f"epoch {epoch}", fps=1, format="gif")})
                
                total = x.shape[0]    
                    
                mse_loss = mse_loss_func(output, y).item()
                l1_loss = l1_loss_func(output, y).item()
                ssim_loss = ssim_loss_func(output, y).item()
                ssim3D_loss = ssim3D_loss_func(output, y).item()
                psnr = psnr_func(output, y).item()

                if loss_f:
                    loss = loss_f(output, y)
                    val_loss_meter.update(loss.item(), n=total)

                val_mse_meter.update(mse_loss, n=total)
                val_l1_meter.update(l1_loss, n=total)
                val_ssim_meter.update(ssim_loss, n=total)
                val_ssim3D_meter.update(ssim3D_loss, n=total)
                val_psnr_meter.update(psnr, n=total)

                pbar.update(1)
                log_str = create_log_str(c, epoch, rank, 
                                         x.shape, 
                                         val_loss_meter.avg, 
                                         val_mse_meter.avg, 
                                         val_l1_meter.avg, 
                                         val_ssim_meter.avg, 
                                         val_ssim3D_meter.avg, 
                                         val_psnr_meter.avg, 
                                         -1, 
                                         id)
                
                pbar.set_description_str(log_str)
                
            # -----------------------------------
            log_str = create_log_str(c, epoch, rank, 
                                         None, 
                                         val_loss_meter.avg, 
                                         val_mse_meter.avg, 
                                         val_l1_meter.avg, 
                                         val_ssim_meter.avg, 
                                         val_ssim3D_meter.avg, 
                                         val_psnr_meter.avg, 
                                         -1, 
                                         id)
                
            pbar.set_description_str(log_str)
                
    if c.ddp:
        val_loss = torch.tensor(val_loss_meter.avg).to(device=device)
        dist.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)
        
        val_mse = torch.tensor(val_mse_meter.avg).to(device=device)
        dist.all_reduce(val_mse, op=torch.distributed.ReduceOp.AVG)
        
        val_l1 = torch.tensor(val_l1_meter.avg).to(device=device)
        dist.all_reduce(val_l1, op=torch.distributed.ReduceOp.AVG)
        
        val_ssim = torch.tensor(val_ssim_meter.avg).to(device=device)
        dist.all_reduce(val_ssim, op=torch.distributed.ReduceOp.AVG)
        
        val_ssim3D = torch.tensor(val_ssim3D_meter.avg).to(device=device)
        dist.all_reduce(val_ssim3D, op=torch.distributed.ReduceOp.AVG)
        
        val_psnr = torch.tensor(val_psnr_meter.avg).to(device=device)
        dist.all_reduce(val_psnr, op=torch.distributed.ReduceOp.AVG)
    else:
        val_loss = val_loss_meter.avg
        val_mse = val_mse_meter.avg
        val_l1 = val_l1_meter.avg
        val_ssim = val_ssim_meter.avg
        val_ssim3D = val_ssim3D_meter.avg
        val_psnr = val_psnr_meter.avg
        
    if rank<=0:
        
        log_str = create_log_str(c, epoch, rank, 
                                None, 
                                val_loss, 
                                val_mse, 
                                val_l1, 
                                val_ssim, 
                                val_ssim3D, 
                                val_psnr, 
                                -1, 
                                id)
        
        logging.info(log_str)
    
    return val_loss, val_mse, val_l1, val_ssim, val_ssim3D, val_psnr

# -------------------------------------------------------------------------------------------------

def apply_model(data, model, gmap, config, scaling_factor):
    '''
    Input 
        data : [RO E1 N SLC], remove any extra scaling
        gmap : [RO E1 SLC], no scaling added
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (0.5 is more smoothing than 1.0)
    '''

    t0 = time()

    device = get_device()

    if(data.ndim==2):
        data = data[:,:,np.newaxis,np.newaxis]

    if(data.ndim<4):
        data = np.expand_dims(data, axis=3)

    RO, E1, PHS, SLC = data.shape

    if(gmap.ndim==2):
        gmap = np.expand_dims(gmap, axis=2)

    if(gmap.shape[0]!=RO or gmap.shape[1]!=E1 or gmap.shape[2]!=SLC):
        gmap = np.ones(RO, E1, SLC)

    print(f"---> apply_model, preparation took {time()-t0} seconds ")
    print(f"---> apply_model, input array {data.shape}")
    print(f"---> apply_model, gmap array {gmap.shape}")
    print(f"---> apply_model, pad_time {config.pad_time}")
    print(f"---> apply_model, height and width {config.height, config.width}")
    print(f"---> apply_model, complex_i {config.complex_i}")
    print(f"---> apply_model, scaling_factor {scaling_factor}")
    
    c = config
    
    try:
        for k in range(SLC):
            imgslab = data[:,:,:,k]
            gmapslab = gmap[:,:,k]
            
            H, W, T = imgslab.shape
            
            x = np.transpose(imgslab, [2, 0, 1]).reshape([1, T, 1, H, W])
            g = np.repeat(gmapslab[np.newaxis, np.newaxis, np.newaxis, :, :], T, axis=1)
            
            x *= scaling_factor
            
            if config.complex_i:
                input = np.concatenate((x.real, x.imag, g), axis=2)
            else:
                input = np.concatenate((np.abs(x.real + 1j*x.imag), g), axis=2)
            
            if not c.pad_time:
                cutout = (T, c.height[-1], c.width[-1])
                overlap = (0, c.height[-1]//2, c.width[-1]//2)
            else:
                cutout = (c.time, c.height[-1], c.width[-1])
                overlap = (c.time//2, c.height[-1]//4, c.width[-1]//4)

            try:
                _, output = running_inference(model, input, cutout=cutout, overlap=overlap, batch_size=4, device=device)
            except Exception as e:
                print(e)
                print(f"{Fore.YELLOW}---> call inference on cpu ...")
                _, output = running_inference(model, input, cutout=cutout, overlap=overlap, device=torch.device('cpu'))
            
            output /= scaling_factor   
                
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()

            output = np.transpose(output, (3, 4, 2, 1, 0)).squeeze()
            
            if(k==0):
                data_filtered = np.zeros((output.shape[0], output.shape[1], PHS, SLC), dtype=data.dtype)
            
            if config.complex_i:
                data_filtered[:,:,:,k] = output[:,:,0,:] + 1j*output[:,:,1,:]
            else:
                data_filtered[:,:,:,k] = output

        t1 = time()
        print(f"---> apply_model took {t1-t0} seconds ")

    except Exception as e:
        print(e)
        data_filtered = copy.deepcopy(data)

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
