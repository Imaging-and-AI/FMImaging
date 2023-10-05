"""
Training and evaluation loops for MRI
"""

import copy
import numpy as np
from time import time

import os
import sys
import logging

from colorama import Fore, Back, Style
import nibabel as nib
import wandb

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from trainer import *
from utils.status import model_info, start_timer, end_timer, support_bfloat16
from metrics.metrics_utils import AverageMeter
from optim.optim_utils import compute_total_steps

from mri_data import MRIDenoisingDatasetTrain
from running_inference import running_inference

class MRITrainManager(TrainManager):
    """
    MRI train manager
        - support MRI double net
    """
    def __init__(self, config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager):  
        super().__init__(config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager)

    # -------------------------------------------------------------------------------------------------
            
    def _train_model(self, rank, global_rank):

        # -----------------------------------------------
        c = self.config
        config = self.config
        
        self.metric_manager.setup_wandb_and_metrics(rank)
        wandb_run = self.metric_manager.wandb_run

        rank_str = self.get_rank_str(rank)
        # -----------------------------------------------

        total_num_samples = sum([len(s) for s in self.train_set])
        total_steps = compute_total_steps(config, total_num_samples)
        logging.info(f"{rank_str}, total_steps for this run: {total_steps}, len(train_set) {[len(s) for s in self.train_set]}, batch {config.batch_size}")
                           
        # -----------------------------------------------

        print(f"{rank_str}, {Style.BRIGHT}{Fore.RED}{Back.LIGHTWHITE_EX}RUN NAME - {config.run_name}{Style.RESET_ALL}")
                
        # -----------------------------------------------
        if rank<=0:
            model_summary = model_info(self.model_manager, c)
            # logging.info(f"Configuration for this run:\n{c}") # Commenting out, prints a lot of info
            # logging.info(f"Model Summary:\n{str(model_summary)}") # Commenting out, prints a lot of info
            logging.info(f"Wandb name:\n{wandb_run.name}")
            wandb_run.watch(self.model_manager)
            wandb_run.log_code(".")
        
        # -----------------------------------------------
        if c.ddp:
            dist.barrier()
            device = torch.device(f"cuda:{rank}")
            model_manager = self.model_manager.to(device)
            model_manager = DDP(model_manager, device_ids=[rank], find_unused_parameters=True)
            if isinstance(self.train_sets,list): 
                samplers = [DistributedSampler(train_set) for train_set in self.train_sets]
            else: 
                samplers = DistributedSampler(self.train_sets)
            shuffle = False
        else:
            device = c.device
            model_manager = self.model_manager.to(device)
            if isinstance(self.train_sets,list): 
                samplers = [None] * len(self.train_sets)
            else: 
                samplers = None
            shuffle = True
        
        # -----------------------------------------------
        
        optim = self.optim_manager.optim
        sched = self.optim_manager.sched
        curr_epoch = self.optim_manager.curr_epoch
        loss_f = self.loss_f
        
        # -----------------------------------------------
        
        block_str = None
        if c.backbone == 'hrnet':
            model_str = f"heads {c.n_head}, {c.backbone_hrnet}"
            block_str = c.backbone_hrnet.block_str
        elif c.backbone == 'unet':
            model_str = f"heads {c.n_head}, {c.backbone_unet}"
            block_str = c.backbone_unet.block_str
        elif c.backbone == 'mixed_unetr':
            model_str = f"{c.backbone_mixed_unetr}"
            block_str = c.backbone_mixed_unetr.block_str

        post_block_str = None
        if c.model_type == "MRI_double_net":
            if c.post_backbone == "hrnet":
                post_block_str = c.post_hrnet.block_str
            if c.post_backbone == "mixed_unetr":
                post_block_str = c.post_mixed_unetr.block_str

        logging.info(f"{rank_str}, {Fore.RED}Local Rank:{rank}, global rank: {global_rank}, {c.backbone}, {c.a_type}, {c.cell_type}, {c.optim}, {c.global_lr}, {c.scheduler_type}, {c.losses}, {c.loss_weights}, weighted loss - snr {c.weighted_loss_snr} - temporal {c.weighted_loss_temporal} - added_noise {c.weighted_loss_added_noise}, data degrading {c.with_data_degrading}, snr perturb {c.snr_perturb_prob}, {c.norm_mode}, scale_ratio_in_mixer {c.scale_ratio_in_mixer}, amp {c.use_amp}, super resolution {c.super_resolution}, stride_s {c.stride_s}, separable_conv {c.separable_conv}, upsample method {c.upsample_method}, batch_size {c.batch_size}, {model_str}{Style.RESET_ALL}")
        logging.info(f"{rank_str}, {Fore.RED}Local Rank:{rank}, global rank: {global_rank}, block_str, {block_str}, post_block_str, {post_block_str}{Style.RESET_ALL}")
                    
        # -----------------------------------------------

        num_workers_per_loader = c.num_workers//len(self.train_sets)
        local_world_size = 1

        if c.ddp:
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            num_workers_per_loader = num_workers_per_loader // local_world_size
            num_workers_per_loader = 1 if num_workers_per_loader<1 else num_workers_per_loader
    
        logging.info(f"{rank_str}, {Fore.YELLOW}Local_world_size {local_world_size}, number of datasets {len(self.train_sets)}, cpu {os.cpu_count()}, number of workers per loader is {num_workers_per_loader}{Style.RESET_ALL}")

        if isinstance(self.train_sets,list):
            train_loaders = [DataLoader(dataset=train_set, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers[ind],
                                        num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                        persistent_workers=c.num_workers>0) for ind, train_set in enumerate(self.train_sets)]
        else:
            train_loaders = [DataLoader(dataset=self.train_sets, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers,
                                        num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                        persistent_workers=c.num_workers>0)]

        train_set_type = [train_set_x.data_type for train_set_x in self.train_sets]

        # -----------------------------------------------

        if rank<=0: # main or master process
            if c.ddp: 
                setup_logger(self.config) # setup master process logging; I don't know if this needs to be here, it is also in setup.py

            if wandb_run is not None:
                wandb_run.summary["trainable_params"] = c.trainable_params
                wandb_run.summary["total_params"] = c.total_params
                wandb_run.summary["total_mult_adds"] = c.total_mult_adds 

                wandb_run.summary["block_str"] = f"{block_str}"
                wandb_run.summary["post_block_str"] = f"{post_block_str}"

            # log a few training examples
            for i, train_set_x in enumerate(self.train_sets):
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
                vid = self.save_image_batch(c.complex_i, x, y_degraded, y, y_2x, y_degraded)
                wandb_run.log({title:wandb.Video(vid, caption=f"Tra sample {i}", fps=1, format='gif')})
                logging.info(f"{Fore.YELLOW}---> Upload tra sample - {title}, noise range {train_set_x.min_noise_level} to {train_set_x.max_noise_level}")

            logging.info(f"{Fore.YELLOW}---> noise range for validation {self.val_sets[0].min_noise_level} to {self.val_sets[0].max_noise_level}")
            
        # -----------------------------------------------

        # Handle mix precision training
        scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp)

        # Zero gradient before training
        optim.zero_grad(set_to_none=True)

        # Compute total iters
        total_iters = sum([len(train_loader) for train_loader in train_loaders])if not c.debug else 3

        # ----------------------------------------------------------------------------
        # Training loop
        
        if self.config.train_model:
            
            train_snr_meter = AverageMeter()
            
            base_snr = 0
            beta_snr = 0.9
            beta_counter = 0
            if c.weighted_loss_snr:
                # get the base_snr
                mean_signal = list()
                median_signal = list()
                for i, train_set_x in enumerate(self.train_sets):
                    stat = train_set_x.get_stat()
                    mean_signal.extend(stat['mean'])
                    median_signal.extend(stat['median'])

                base_snr = np.abs(np.median(mean_signal)) / 2

                logging.info(f"{rank_str}, {Fore.YELLOW}base_snr {base_snr:.4f}, Mean signal {np.abs(np.median(mean_signal)):.4f}, median {np.abs(np.median(median_signal)):.4f}, from {len(mean_signal)} images {Style.RESET_ALL}")

            logging.info(f"{rank_str}, {Fore.GREEN}----------> Start training loop <----------{Style.RESET_ALL}")

            if c.ddp:
                model_manager.module.check_model_learnable_status(rank_str)
            else:
                model_manager.check_model_learnable_status(rank_str)
            
            image_save_step_size = int(total_iters // config.num_saved_samples)
            if image_save_step_size == 0: image_save_step_size = 1
                
            # ----------------------------------------------------------------------------
            for epoch in range(curr_epoch, c.num_epochs):
                logging.info(f"{Fore.GREEN}{'-'*20}Epoch:{epoch}/{c.num_epochs}, rank {rank} {'-'*20}{Style.RESET_ALL}")

                model_manager.train()
                if c.ddp: [train_loader.sampler.set_epoch(epoch) for train_loader in train_loaders]
                self.metric_manager.on_train_epoch_start()
                train_loader_iters = [iter(train_loader) for train_loader in train_loaders]
                
                images_saved = 0
                
                train_snr_meter.reset()
                
                # ----------------------------------------------------------------------------
                with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:
                    for idx in range(total_iters):

                        # -------------------------------------------------------
                        tm = start_timer(enable=c.with_timer)
                        loader_ind = idx % len(train_loader_iters)
                        loader_outputs = next(train_loader_iters[loader_ind], None)
                        while loader_outputs is None:
                            del train_loader_iters[loader_ind]
                            loader_ind = idx % len(train_loader_iters)
                            loader_outputs = next(train_loader_iters[loader_ind], None)
                        data_type = train_set_type[loader_ind]
                        x, y, y_degraded, y_2x, gmaps_median, noise_sigmas = loader_outputs
                        end_timer(enable=c.with_timer, t=tm, msg="---> load batch took ")
                        
                        # -------------------------------------------------------
                        tm = start_timer(enable=c.with_timer)
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
                                model_output = self.metric_manager(x, snr, base_snr_t)
                                output, weights, output_1st_net = model_output
                                if c.weighted_loss_temporal:
                                    weights *= weights_t
                            else:
                                model_output = self.metric_manager(x)
                                output, output_1st_net = model_output
                                if c.weighted_loss_temporal:
                                    weights = weights_t

                            if torch.isnan(torch.sum(output)):
                                continue

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

                        # -------------------------------------------------------
                        if torch.isnan(loss):
                            print(f"Warning - loss is nan ... ")
                            optim.zero_grad()
                            continue

                        tm = start_timer(enable=c.with_timer)
                        scaler.scale(loss).backward()
                        end_timer(enable=c.with_timer, t=tm, msg="---> backward pass took ")

                        # -------------------------------------------------------
                        tm = start_timer(enable=c.with_timer)
                        if (idx + 1) % c.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                            if(c.clip_grad_norm>0):
                                scaler.unscale_(optim)
                                nn.utils.clip_grad_norm_(self.metric_manager.parameters(), c.clip_grad_norm)

                            scaler.step(optim)
                            optim.zero_grad()
                            scaler.update()

                            if c.scheduler_type == "OneCycleLR": sched.step()
                        end_timer(enable=c.with_timer, t=tm, msg="---> other steps took ")
                                        
                        # -------------------------------------------------------
                        tm = start_timer(enable=c.with_timer)
                        curr_lr = optim.param_groups[0]['lr']

                        train_snr_meter.update(torch.mean(snr), n=x.shape[0])
                        
                        tra_save_images = idx%image_save_step_size==0 and images_saved < config.num_saved_samples and config.save_samples
                        self.metric_manager.on_train_step_end(loss.item(), model_output, loader_outputs, rank, curr_lr, tra_save_images, epoch, images_saved)
                        images_saved += 1

                        # -------------------------------------------------------
                        pbar.update(1)
                        log_str = self.create_log_str(config, epoch, rank, 
                                         x.shape, 
                                         torch.mean(gmaps_median).cpu().item(),
                                         torch.mean(noise_sigmas).cpu().item(),
                                         train_snr_meter.avg,
                                         self.metric_manager,
                                         curr_lr, 
                                         "tra")

                        pbar.set_description_str(log_str)

                        end_timer(enable=c.with_timer, t=tm, msg="---> epoch step logging and measuring took ")
                    # ------------------------------------------------------------------------------------------------------
                    
                    # Run metric logging for each epoch 
                    tm = start_timer(enable=c.with_timer) 

                    self.metric_manager.on_train_epoch_end(epoch, rank)

                    # Print out metrics from this epoch
                    log_str = self.create_log_str(config, epoch, rank, 
                                                x.shape, 
                                                torch.mean(gmaps_median).cpu().item(),
                                                torch.mean(noise_sigmas).cpu().item(),
                                                train_snr_meter.avg,
                                                self.metric_manager,
                                                curr_lr, 
                                                "tra")

                    pbar.set_description(log_str)

                    # Write training status to log file
                    if rank<=0: 
                        logging.getLogger("file_only").info(log_str)

                    end_timer(enable=c.with_timer, t=tm, msg="---> epoch end logging and measuring took ")                
                # ------------------------------------------------------------------------------------------------------
                
                if epoch % c.eval_frequency==0 or epoch==c.num_epochs:
                    self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=epoch, device=device, optim=optim, sched=sched, id="", split="val", final_eval=False, scaling_factor=1)

                if c.scheduler_type != "OneCycleLR":
                    if c.scheduler_type == "ReduceLROnPlateau":
                        sched.step(loss.item())
                    elif c.scheduler_type == "StepLR":
                        sched.step()                        
                        
            # ----------------------------------------------------------------------------

            self.model_manager.save(os.path.join(self.config.log_dir, self.config.run_name, 'last_checkpoint'), epoch, optim, sched)
            if wandb_run is not None:
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_pre.pth'))
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_backbone.pth'))
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_post.pth'))
                
            # Load the best model from training
            if self.config.eval_train_set or self.config.eval_val_set or self.config.eval_test_set:
                logging.info(f"{Fore.CYAN}Loading the best models from training for final evaluation...{Style.RESET_ALL}")
                self.model_manager.load_pre(os.path.join(self.config.log_dir,self.config.run_name,'best_checkpoint_pre.pth'))
                self.model_manager.load_backbone(os.path.join(self.config.log_dir,self.config.run_name,'best_checkpoint_backbone.pth'))
                self.model_manager.load_post(os.path.join(self.config.log_dir,self.config.run_name,'best_checkpoint_post.pth'))
                
                if wandb_run is not None:
                    wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'best_checkpoint_pre.pth'))
                    wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'best_checkpoint_backbone.pth'))
                    wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'best_checkpoint_post.pth'))
        else: 
            epoch = 0
        
        # -----------------------------------------------
        # Evaluate models of each split
        if self.config.eval_train_set: 
            logging.info(f"{Fore.CYAN}Evaluating train set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.train_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="train", final_eval=True, scaling_factor=1)
        if self.config.eval_val_set: 
            logging.info(f"{Fore.CYAN}Evaluating val set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="val", final_eval=True, scaling_factor=1)
        if self.config.eval_test_set: 
            logging.info(f"{Fore.CYAN}Evaluating test set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.test_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="test", final_eval=True, scaling_factor=1)

        # -----------------------------------------------
        # Finish up training
        self.metric_manager.on_training_end(rank, epoch, model_manager, optim, sched, self.config.train_model)
    
        if c.ddp:
            dist.barrier()
        print(f"--> run finished ...")
    
    # -------------------------------------------------------------------------------------------------
        
    def _eval_model(self, rank, model_manager, data_sets, epoch, device, optim, sched, id, split, final_eval, scaling_factor=1):

        c = self.config
        curr_lr = optim.param_groups[0]['lr']
                
        # ------------------------------------------------------------------------
        # Determine if we will save the predictions to files for thie eval 
        if split=='train': save_samples = final_eval and self.config.save_train_samples
        elif split=='val': save_samples = final_eval and self.config.save_val_samples
        elif split=='test': save_samples = final_eval and self.config.save_test_samples
        else: raise ValueError(f"Unknown split {split} specified, should be in [train, val, test]")

        if c.ddp:
            loss_f = self.module.loss_f
            if isinstance(data_sets, list): samplers = [DistributedSampler(data_set) for data_set in data_sets]
            else: samplers = DistributedSampler(data_sets)    
        else:
            loss_f = self.loss_f
            if isinstance(data_sets, list): samplers = [None] * len(data_sets)
            else: samplers = None

        # ------------------------------------------------------------------------
        # Set upd data loader to evaluate        
        batch_size = c.batch_size if isinstance(data_sets[0], MRIDenoisingDatasetTrain) else 1
        num_workers_per_loader = c.num_workers // (2 * len(data_sets))
        
        if isinstance(data_sets, list):
            data_loaders = [DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, sampler=samplers[ind],
                                    num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                    persistent_workers=c.num_workers>0) for ind, data_set in enumerate(data_sets)]
        else:
            data_loaders = [DataLoader(dataset=data_sets, batch_size=batch_size, shuffle=False, sampler=samplers,
                                    num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                    persistent_workers=c.num_workers>0) ]
            
        # ------------------------------------------------------------------------
        self.metric_manager.on_eval_epoch_start()

        wandb_run = self.metric_manager.wandb_run
        
        # ------------------------------------------------------------------------
        model_manager.eval()

        # ------------------------------------------------------------------------
        if rank <= 0 and epoch < 1:
            logging.info(f"Eval height and width is {c.mri_height[-1]}, {c.mri_width[-1]}")

        cutout = (c.time, c.mri_height[-1], c.mri_width[-1])
        overlap = (c.time//2, c.mri_height[-1]//4, c.mri_width[-1]//4)
    
        # ------------------------------------------------------------------------
        
        data_loader_iters = [iter(data_loader) for data_loader in data_loaders]
        total_iters = sum([len(data_loader) for data_loader in data_loaders]) if not c.debug else 3
        
        # ------------------------------------------------------------------------
        # Evaluation loop
        with torch.inference_mode():
            with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

                for idx in range(total_iters):

                    loader_ind = idx % len(data_loader_iters)
                    loader_outputs = next(data_loader_iters[loader_ind], None)
                    while loader_outputs is None:
                        del data_loader_iters[loader_ind]
                        loader_ind = idx % len(data_loader_iters)
                        loader_outputs = next(data_loader_iters[loader_ind], None)
                    x, y, y_degraded, y_2x, gmaps_median, noise_sigmas = loader_outputs

                    gmaps_median = gmaps_median.to(device=device, dtype=x.dtype)
                    noise_sigmas = noise_sigmas.to(device=device, dtype=x.dtype)

                    B = x.shape[0]
                    noise_sigmas = torch.reshape(noise_sigmas, (B, 1, 1, 1, 1))

                    if self.config.super_resolution:
                        y = y_2x

                    x = x.to(device)
                    y = y.to(device)

                    if batch_size >1 and x.shape[-1]==c.width[-1]:
                        output, output_1st_net = self.model_manager(x)
                    else:
                        B, T, C, H, W = x.shape

                        cutout_in = cutout
                        overlap_in = overlap
                        if not self.config.pad_time:
                            cutout_in = (T, c.height[-1], c.width[-1])
                            overlap_in = (0, c.height[-1]//2, c.width[-1]//2)

                        try:
                            _, output = running_inference(self.model_manager, x, cutout=cutout_in, overlap=overlap_in, device=device)
                            output_1st_net = None
                        except:
                            logging.info(f"{Fore.YELLOW}---> call inference on cpu ...")
                            _, output = running_inference(self.model_manager, x, cutout=cutout_in, overlap=overlap_in, device="cpu")
                            y = y.to("cpu")

                    if scaling_factor > 0:
                        output /= scaling_factor
                        if output_1st_net is not None: output_1st_net /= scaling_factor

                    total = x.shape[0]

                    # Update evaluation metrics
                    self.metric_manager.on_eval_step_end(-1, output, loader_outputs, f"{idx}", rank, save_samples, split)

                    if rank<=0 and images_logged < self.config.num_uploaded and wandb_run is not None:
                        images_logged += 1
                        title = f"{id.upper()}_{images_logged}_{x.shape}"
                        if output_1st_net is None: 
                            output_1st_net = output
                        vid = self.save_image_batch(c.complex_i, x.numpy(force=True), output.numpy(force=True), y.numpy(force=True), y_2x.numpy(force=True), output_1st_net.numpy(force=True))
                        wandb_run.log({title: wandb.Video(vid, 
                                                        caption=f"epoch {epoch}, gmap {torch.mean(gmaps_median).item():.2f}, noise {torch.mean(noise_sigmas).item():.2f}, mse {self.metric_manager.eval_metrics['mst'].avg:.2f}, ssim {self.metric_manager.eval_metrics['ssim'].avg:.2f}, psnr {self.metric_manager.eval_metrics['psnr'].avg:.2f}", 
                                                        fps=1, format="gif")})
                    
                    # Print evaluation metrics to terminal
                    pbar.update(1)
                    
                    log_str = self.create_log_str(self.config, epoch, rank, 
                                                x.shape, 
                                                torch.mean(gmaps_median).cpu().item(),
                                                torch.mean(noise_sigmas).cpu().item(),
                                                -1,
                                                self.metric_manager,
                                                curr_lr, 
                                                "val")
                    
                    pbar.set_description(log_str)


                # Update evaluation metrics 
                self.metric_manager.on_eval_epoch_end(rank, epoch, model_manager, optim, sched, split, final_eval)

                # Print evaluation metrics to terminal
                log_str = self.create_log_str(self.config, epoch, rank, 
                                                x.shape, 
                                                torch.mean(gmaps_median).cpu().item(),
                                                torch.mean(noise_sigmas).cpu().item(),
                                                -1,
                                                self.metric_manager,
                                                curr_lr, 
                                                "val")
                
                pbar_str = f"{log_str}"
                if hasattr(self.metric_manager, 'average_eval_metrics'):
                    if isinstance(self.metric_manager.average_eval_metrics, dict):
                        for metric_name, metric_value in self.metric_manager.average_eval_metrics.items():
                            try: pbar_str += f", {Fore.CYAN} {metric_name} {metric_value:.4f}"
                            except: pass

                            # Save final evaluation metrics to a text file
                            if final_eval and rank<=0:
                                with open(os.path.join(self.config.log_dir,self.config.run_name,f'{split}_metrics.txt'), 'a') as f:
                                    try: f.write(f"{split}_{metric_name}: {metric_value:.4f}, ")
                                    except: pass

                pbar_str += f"{Style.RESET_ALL}"
                pbar.set_description(pbar_str)

                if rank<=0: 
                    logging.getLogger("file_only").info(pbar_str)                        
        return 
       
       
    def create_log_str(config, epoch, rank, data_shape, gmap_median, noise_sigma, snr, loss_meters, curr_lr, role):
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

        if role == 'tra':
            loss, mse, l1, ssim, ssim3D, ssim_loss, ssim3D_loss, psnr_loss, psnr, perp, gaussian, gaussian3D = loss_meters.get_tra_loss()
        else:
            loss, mse, l1, ssim, ssim3D, ssim_loss, ssim3D_loss, psnr_loss, psnr, perp, gaussian, gaussian3D = loss_meters.get_eval_loss()

        str= f"{Fore.GREEN}Epoch {epoch}/{config.num_epochs}, {C}{role}, {Style.RESET_ALL}{rank}, " + data_shape_str + f"{Fore.BLUE}{Back.WHITE}{Style.BRIGHT}loss {loss:.4f},{Style.RESET_ALL} {Fore.WHITE}{Back.LIGHTBLUE_EX}{Style.NORMAL}gmap {gmap_median:.2f}, sigma {noise_sigma:.2f}{snr_str}{Style.RESET_ALL} {C}mse {mse:.4f}, l1 {l1:.4f}, perp {perp:.4f}, ssim {ssim:.4f}, ssim3D {ssim3D:.4f}, gaussian {gaussian:.4f}, gaussian3D {gaussian3D:.4f}, psnr loss {psnr_loss:.4f}, psnr {psnr:.4f}{Style.RESET_ALL}{lr_str}"

        return str
  

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
        _, output = running_inference(model, input, cutout=cutout, overlap=overlap, batch_size=1, device=device)
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

# -------------------------------------------------------------------------------------------------

def tests():
    pass    

if __name__=="__main__":
    tests()
