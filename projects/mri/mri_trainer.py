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
import cv2
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

# -------------------------------------------------------------------------------------------------

class MRITrainManager(TrainManager):
    """
    MRI train manager
        - support MRI double net
    """
    def __init__(self, config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager):  
        super().__init__(config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager)

        # self.config.height = self.config.mri_height[-1]
        # self.config.width = self.config.mri_width[-1]
        self.config.height = self.config.mri_height[0]
        self.config.width = self.config.mri_width[0]

    # -------------------------------------------------------------------------------------------------
            
    def _train_model(self, rank, global_rank):

        # -----------------------------------------------
        c = self.config
        config = self.config
        
        self.metric_manager.setup_wandb_and_metrics(rank)
        if rank<=0:
            wandb_run = self.metric_manager.wandb_run
        else:
            wandb_run = None

        rank_str = get_rank_str(rank)
        # -----------------------------------------------

        total_num_samples = sum([len(s) for s in self.train_sets])
        total_steps = compute_total_steps(config, total_num_samples)
        logging.info(f"{rank_str}, total_steps for this run: {total_steps}, len(train_set) {[len(s) for s in self.train_sets]}, batch {config.batch_size}")

        # -----------------------------------------------

        print(f"{rank_str}, {Style.BRIGHT}{Fore.RED}{Back.LIGHTWHITE_EX}RUN NAME - {config.run_name}{Style.RESET_ALL}")

        # -----------------------------------------------
        if rank<=0:
            model_summary = model_info(self.model_manager, c)
            logging.info(f"Configuration for this run:\n{c}") # Commenting out, prints a lot of info
            logging.info(f"Model Summary:\n{str(model_summary)}") # Commenting out, prints a lot of info
            logging.info(f"Wandb name:\n{wandb_run.name}")
            #wandb_run.watch(self.model_manager)
            wandb_run.log_code(".")

        # -----------------------------------------------
        if c.ddp:
            dist.barrier()
            device = torch.device(f"cuda:{rank}")
            model_manager = self.model_manager.to(device)
            model_manager = DDP(model_manager, device_ids=[rank], find_unused_parameters=True)
            if isinstance(self.train_sets,list): 
                samplers = [DistributedSampler(train_set, shuffle=True) for train_set in self.train_sets]
            else: 
                samplers = DistributedSampler(self.train_sets, shuffle=True)
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

        model_str = None
        block_str = None
        if c.backbone_model == 'STCNNT_HRNET':
            model_str = f"heads {c.n_head}, {c.backbone_hrnet}"
            block_str = c.backbone_hrnet.block_str
        elif c.backbone_model == 'STCNNT_UNET':
            model_str = f"heads {c.n_head}, {c.backbone_unet}"
            block_str = c.backbone_unet.block_str
        elif c.backbone_model == 'STCNNT_mUNET':
            model_str = f"{c.backbone_mixed_unetr}"
            block_str = c.backbone_mixed_unetr.block_str

        post_block_str = None
        if c.model_type == "MRI_double_net":
            if c.post_backbone == "STCNNT_HRNET":
                post_block_str = c.post_hrnet.block_str
            if c.post_backbone == "STCNNT_mUNET":
                post_block_str = c.post_mixed_unetr.block_str

        logging.info(f"{rank_str}, {Fore.RED}Local Rank:{rank}, global rank: {global_rank}, {c.backbone_model}, {c.a_type}, {c.cell_type}, {c.optim_type}, {c.optim}, {c.scheduler_type}, {c.losses}, {c.loss_weights}, weighted loss - snr {c.weighted_loss_snr} - temporal {c.weighted_loss_temporal} - added_noise {c.weighted_loss_added_noise}, data degrading {c.with_data_degrading}, snr perturb {c.snr_perturb_prob}, {c.norm_mode}, scale_ratio_in_mixer {c.scale_ratio_in_mixer}, amp {c.use_amp}, super resolution {c.super_resolution}, stride_s {c.stride_s}, separable_conv {c.separable_conv}, upsample method {c.upsample_method}, batch_size {c.batch_size}, {model_str}{Style.RESET_ALL}")
        logging.info(f"{rank_str}, {Fore.RED}Local Rank:{rank}, global rank: {global_rank}, block_str, {block_str}, post_block_str, {post_block_str}, use_amp {c.use_amp}, cast_type {self.cast_type}{Style.RESET_ALL}")

        # -----------------------------------------------

        num_workers_per_loader = c.num_workers//len(self.train_sets)
        local_world_size = 1

        if c.ddp:
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            num_workers_per_loader = num_workers_per_loader // local_world_size
            num_workers_per_loader = 1 if num_workers_per_loader<1 else num_workers_per_loader

        logging.info(f"{rank_str}, {Fore.YELLOW}Local_world_size {local_world_size}, number of datasets {len(self.train_sets)}, cpu {os.cpu_count()}, number of workers per loader is {num_workers_per_loader}{Style.RESET_ALL}")
        if rank <=0:
            logging.info(f"{rank_str}, {Fore.YELLOW}Yaml file for this run is {self.config.yaml_file}{Style.RESET_ALL}")

        if isinstance(self.train_sets,list):
            train_loaders = [DataLoader(dataset=train_set, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers[ind],
                                        num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                        persistent_workers=c.num_workers>0, pin_memory=False) for ind, train_set in enumerate(self.train_sets)]
        else:
            train_loaders = [DataLoader(dataset=self.train_sets, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers,
                                        num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                        persistent_workers=c.num_workers>0, pin_memory=False)]

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

                wandb_run.save(self.config.yaml_file)

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

                        B, C, T, H, W = x.shape

                        if c.weighted_loss_temporal:
                            # compute temporal std
                            if C == 3:
                                std_t = torch.std(torch.abs(y[:,0,:,:,:] + 1j * y[:,1,:,:,:]), dim=1)
                            else:
                                std_t = torch.std(y(y[:,0,:,:,:], dim=1))

                            weights_t = torch.mean(std_t, dim=(-2, -1)).to(device)

                        # compute snr
                        signal = torch.mean(torch.linalg.norm(y, dim=1, keepdim=True), dim=(1, 2, 3, 4)).to(device)
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
                                model_output = self.model_manager(x, snr, base_snr_t)
                                output, weights, output_1st_net = model_output
                                if c.weighted_loss_temporal:
                                    weights *= weights_t
                            else:
                                model_output = self.model_manager(x)
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
                                nn.utils.clip_grad_norm_(self.model_manager.parameters(), c.clip_grad_norm)

                            scaler.step(optim)
                            optim.zero_grad(set_to_none=True)
                            scaler.update()

                            if c.scheduler_type == "OneCycleLR": sched.step()
                        end_timer(enable=c.with_timer, t=tm, msg="---> other steps took ")

                        # -------------------------------------------------------
                        tm = start_timer(enable=c.with_timer)
                        curr_lr = optim.param_groups[0]['lr']

                        train_snr_meter.update(torch.mean(snr), n=x.shape[0])

                        tra_save_images = idx%image_save_step_size==0 and images_saved < config.num_saved_samples and config.save_train_samples
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

                    if c.ddp:
                        self.distribute_learning_rates(rank, optim, src=0)

            # ----------------------------------------------------------------------------

            self.model_manager.save(os.path.join(self.config.log_dir, self.config.run_name, 'last_checkpoint'), epoch, optim, sched)
            if wandb_run is not None:
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_pre.pth'))
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_backbone.pth'))
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_post.pth'))

            # Load the best model from training
            if self.config.eval_train_set or self.config.eval_val_set or self.config.eval_test_set:
                logging.info(f"{Fore.CYAN}Loading the best models from training for final evaluation...{Style.RESET_ALL}")
                if self.metric_manager.best_pre_model_file is not None: self.model_manager.load_pre(self.metric_manager.best_pre_model_file)
                if self.metric_manager.best_backbone_model_file is not None: self.model_manager.load_backbone(self.metric_manager.best_backbone_model_file)
                if self.metric_manager.best_post_model_file is not None: self.model_manager.load_post(self.metric_manager.best_post_model_file)

                if wandb_run is not None:
                    if self.metric_manager.best_pre_model_file is not None: wandb_run.save(self.metric_manager.best_pre_model_file)
                    if self.metric_manager.best_backbone_model_file is not None: wandb_run.save(self.metric_manager.best_backbone_model_file)
                    if self.metric_manager.best_post_model_file is not None: wandb_run.save(self.metric_manager.best_post_model_file)
        else: 
            epoch = 0

        # -----------------------------------------------
        # Evaluate models of each split
        if self.config.eval_train_set: 
            logging.info(f"{Fore.CYAN}Evaluating the best model on the train set ... {Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.train_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="train", split="train", final_eval=True, scaling_factor=1)
        if self.config.eval_val_set: 
            logging.info(f"{Fore.CYAN}Evaluating the best model on the val set ... {Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="val", split="val", final_eval=True, scaling_factor=1)
        if self.config.eval_test_set: 
            logging.info(f"{Fore.CYAN}Evaluating the best model on the test set ... {Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.test_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="test", split="test", final_eval=True, scaling_factor=1)

        # -----------------------------------------------

        save_path, save_file_name, config_yaml_file = self.model_manager.save_entire_model(epoch=self.config.num_epochs)
        model_full_path = os.path.join(save_path, save_file_name)
        logging.info(f"{Fore.YELLOW}Entire model is saved at {model_full_path} ...{Style.RESET_ALL}")

        if wandb_run is not None:
            wandb_run.save(model_full_path)
            wandb_run.save(config_yaml_file)

        # -----------------------------------------------

        # Finish up training
        self.metric_manager.on_training_end(rank, epoch, model_manager, optim, sched, self.config.train_model)

        if c.ddp:
            dist.barrier()
        print(f"--> run finished ...")

    # =============================================================================================================================

    def _eval_model(self, rank, model_manager, data_sets, epoch, device, optim, sched, id, split, final_eval, scaling_factor=1):

        c = self.config
        curr_lr = optim.param_groups[0]['lr']

        # ------------------------------------------------------------------------
        # Determine if we will save the predictions to files for thie eval 
        if split=='train': save_samples = final_eval and self.config.save_train_samples
        elif split=='val': save_samples = final_eval and self.config.save_val_samples
        elif split=='test': save_samples = final_eval and self.config.save_test_samples
        else: raise ValueError(f"Unknown split {split} specified, should be in [train, val, test]")

        loss_f = self.loss_f

        if c.ddp:
            if isinstance(data_sets, list): samplers = [DistributedSampler(data_set) for data_set in data_sets]
            else: samplers = DistributedSampler(data_sets)
        else:
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

        if rank<=0:
            wandb_run = self.metric_manager.wandb_run
        else:
            wandb_run = None
        
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
        
        images_logged = 0
        
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

                    if batch_size >1 and x.shape[-1]==c.mri_width[-1]:
                        output, output_1st_net = self.model_manager(x)
                    else:
                        B, C, T, H, W = x.shape

                        x = torch.permute(x, (0, 2, 1, 3, 4))

                        cutout_in = cutout
                        overlap_in = overlap
                        if not self.config.pad_time:
                            cutout_in = (T, c.mri_height[-1], c.mri_width[-1])
                            overlap_in = (0, c.mri_height[-1]//2, c.mri_width[-1]//2)

                        if scaling_factor > 0:
                            x *= scaling_factor

                        try:
                            _, output = running_inference(self.model_manager, x, cutout=cutout_in, overlap=overlap_in, device=device)
                            output_1st_net = None
                        except:
                            logging.info(f"{Fore.YELLOW}---> call inference on cpu ...")
                            _, output = running_inference(self.model_manager, x, cutout=cutout_in, overlap=overlap_in, device="cpu")
                            y = y.to("cpu")

                        x = torch.permute(x, (0, 2, 1, 3, 4))
                        output = torch.permute(output, (0, 2, 1, 3, 4))

                        if scaling_factor > 0:
                            output /= scaling_factor

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=c.use_amp):
                        loss = loss_f(output, y)

                    # Update evaluation metrics
                    self.metric_manager.on_eval_step_end(loss.item(), output, loader_outputs, f"{idx}", rank, save_samples, split)

                    if rank<=0 and images_logged < self.config.num_uploaded and wandb_run is not None:
                        images_logged += 1
                        title = f"{id.upper()}_{images_logged}_{x.shape}"
                        if output_1st_net is None: 
                            output_1st_net = output
                        vid = self.save_image_batch(c.complex_i, x.numpy(force=True), output.numpy(force=True), y.numpy(force=True), y_2x.numpy(force=True), output_1st_net.numpy(force=True))
                        wandb_run.log({title: wandb.Video(vid, 
                            caption=f"epoch {epoch}, gmap {torch.mean(gmaps_median).item():.2f}, noise {torch.mean(noise_sigmas).item():.2f}, mse {self.metric_manager.eval_metrics['mse'].avg:.2f}, ssim {self.metric_manager.eval_metrics['ssim'].avg:.2f}, psnr {self.metric_manager.eval_metrics['psnr'].avg:.2f}", 
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
                                                split)
                    
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
                                                split)

                
                if hasattr(self.metric_manager, 'average_eval_metrics'):
                    #print(f"--> rank {rank}, self.metric_manager.average_eval_metrics : {self.metric_manager.average_eval_metrics}")
                    pbar_str = f"--> rank {rank}, {split}, epoch {epoch}"
                    if isinstance(self.metric_manager.average_eval_metrics, dict):
                        for metric_name, metric_value in self.metric_manager.average_eval_metrics.items():
                            try: 
                                pbar_str += f", {Fore.CYAN} {metric_name} {metric_value:.4f}"
                            except: 
                                pass

                        # Save final evaluation metrics to a text file
                        if final_eval and rank<=0:
                            metric_file = os.path.join(self.config.log_dir,self.config.run_name, f'{split}_metrics.txt')
                            with open(metric_file, 'a') as f:
                                try: f.write(f"{split}_{metric_name}: {metric_value:.4f}, ")
                                except: pass
                            wandb_run.save(metric_file)

                    pbar_str += f"{Style.RESET_ALL}"
                else:
                    pbar_str = log_str

                pbar.set_description(pbar_str)

                if rank<=0: 
                    logging.getLogger("file_only").info(pbar_str)
        return 


    def create_log_str(self, config, epoch, rank, data_shape, gmap_median, noise_sigma, snr, loss_meters, curr_lr, role):
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
            loss, mse, l1, ssim, ssim3D, ssim_loss, ssim3D_loss, psnr, psnr_loss, perp, gaussian, gaussian3D = loss_meters.get_tra_loss()
        else:
            loss, mse, l1, ssim, ssim3D, ssim_loss, ssim3D_loss, psnr, psnr_loss, perp, gaussian, gaussian3D = loss_meters.get_eval_loss()

        str= f"{Fore.GREEN}Epoch {epoch}/{config.num_epochs}, {C}{role}, {Style.RESET_ALL}{rank}, " + data_shape_str + f"{Fore.BLUE}{Back.WHITE}{Style.BRIGHT}loss {loss:.4f},{Style.RESET_ALL} {Fore.WHITE}{Back.LIGHTBLUE_EX}{Style.NORMAL}gmap {gmap_median:.2f}, sigma {noise_sigma:.2f}{snr_str}{Style.RESET_ALL} {C}mse {mse:.4f}, l1 {l1:.4f}, perp {perp:.4f}, ssim {ssim:.4f}, ssim3D {ssim3D:.4f}, gaussian {gaussian:.4f}, gaussian3D {gaussian3D:.4f}, psnr loss {psnr_loss:.4f}, psnr {psnr:.4f}{Style.RESET_ALL}{lr_str}"

        return str
  

    # -------------------------------------------------------------------------------------------------

    def distribute_learning_rates(self, rank, optim, src=0):

        N = len(optim.param_groups)
        new_lr = torch.zeros(N).to(rank)
        for ind in range(N):
            new_lr[ind] = optim.param_groups[ind]["lr"]

        dist.broadcast(new_lr, src=src)

        if rank != src:
            for ind in range(N):
                optim.param_groups[ind]["lr"] = new_lr[ind].item()

    # -------------------------------------------------------------------------------------------------

    def save_image_batch(self, complex_i, noisy, predi, clean, clean_2x, predi_1st_net):
        """
        Logs the image to wandb as a 5D gif [B,T,C,H,W]
        If complex image then save the magnitude using first 2 channels
        Else use just the first channel
        @args:
            - complex_i (bool): complex images or not
            - noisy (5D numpy array): the noisy image [B, T, C+1, H, W]
            - predi (5D numpy array): the predicted image [B, T, C, H, W]
            - clean (5D numpy array): the clean image [B, T, C, H, W]
            - clean_2x (5D numpy array): the clean image [B, T, C, 2*H, 2*W]
        """

        if noisy.ndim == 4:
            noisy = np.expand_dims(noisy, axis=0)
            predi = np.expand_dims(predi, axis=0)
            clean = np.expand_dims(clean, axis=0)
            clean_2x = np.expand_dims(clean_2x, axis=0)
            predi_1st_net = np.expand_dims(predi_1st_net, axis=0)

        noisy = np.transpose(noisy, (0, 2, 1, 3, 4))
        predi = np.transpose(predi, (0, 2, 1, 3, 4))
        clean = np.transpose(clean, (0, 2, 1, 3, 4))
        clean_2x = np.transpose(clean_2x, (0, 2, 1, 3, 4))
        predi_1st_net = np.transpose(predi_1st_net, (0, 2, 1, 3, 4))

        if complex_i:
            save_x = np.sqrt(np.square(noisy[:,:,0,:,:]) + np.square(noisy[:,:,1,:,:]))
            save_p = np.sqrt(np.square(predi[:,:,0,:,:]) + np.square(predi[:,:,1,:,:]))
            save_y = np.sqrt(np.square(clean[:,:,0,:,:]) + np.square(clean[:,:,1,:,:]))
            save_y_2x = np.sqrt(np.square(clean_2x[:,:,0,:,:]) + np.square(clean_2x[:,:,1,:,:]))
            save_p_1st_net = np.sqrt(np.square(predi_1st_net[:,:,0,:,:]) + np.square(predi_1st_net[:,:,1,:,:]))
        else:
            save_x = noisy[:,:,0,:,:]
            save_p = predi[:,:,0,:,:]
            save_y = clean[:,:,0,:,:]
            save_y_2x = clean_2x[:,:,0,:,:]
            save_p_1st_net = predi_1st_net[:,:,0,:,:]

        B, T, H, W = save_y_2x.shape

        def resize_img(im, H_2x, W_2x):
            H, W = im.shape
            if H < H_2x or W < W_2x:
                res = cv2.resize(src=im, dsize=(W_2x, H_2x), interpolation=cv2.INTER_NEAREST)
                return res
            else:
                return im

        max_col = 16
        if B>max_col:
            num_row = B//max_col
            if max_col*num_row < B: 
                num_row += 1
            composed_res = np.zeros((T, 5*H*num_row, max_col*W))
            for b in range(B):
                r = b//max_col
                c = b - r*max_col
                for t in range(T):
                    S = 5*r
                    composed_res[t, S*H:(S+1)*H, c*W:(c+1)*W] = resize_img(save_x[b,t,:,:].squeeze(), H, W)
                    composed_res[t, (S+1)*H:(S+2)*H, c*W:(c+1)*W] = resize_img(save_p[b,t,:,:].squeeze(), H, W)
                    composed_res[t, (S+2)*H:(S+3)*H, c*W:(c+1)*W] = resize_img(save_p_1st_net[b,t,:,:].squeeze(), H, W)
                    composed_res[t, (S+3)*H:(S+4)*H, c*W:(c+1)*W] = resize_img(save_y[b,t,:,:].squeeze(), H, W)
                    composed_res[t, (S+4)*H:(S+5)*H, c*W:(c+1)*W] = resize_img(save_y_2x[b,t,:,:].squeeze(), H, W)

                a_composed_res = composed_res[:,:,c*W:(c+1)*W]
                a_composed_res = np.clip(a_composed_res, a_min=0.5*np.median(a_composed_res), a_max=np.percentile(a_composed_res, 90))
                temp = np.zeros_like(a_composed_res)
                composed_res[:,:,c*W:(c+1)*W] = cv2.normalize(a_composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)
        
        elif B>2:
            composed_res = np.zeros((T, 5*H, B*W))
            for b in range(B):
                for t in range(T):
                    composed_res[t, :H, b*W:(b+1)*W] = resize_img(save_x[b,t,:,:].squeeze(), H, W)
                    composed_res[t, H:2*H, b*W:(b+1)*W] = resize_img(save_p[b,t,:,:].squeeze(), H, W)
                    composed_res[t, 2*H:3*H, b*W:(b+1)*W] = resize_img(save_p_1st_net[b,t,:,:].squeeze(), H, W)
                    composed_res[t, 3*H:4*H, b*W:(b+1)*W] = resize_img(save_y[b,t,:,:].squeeze(), H, W)
                    composed_res[t, 4*H:5*H, b*W:(b+1)*W] = resize_img(save_y_2x[b,t,:,:].squeeze(), H, W)

                a_composed_res = composed_res[:,:,b*W:(b+1)*W]
                a_composed_res = np.clip(a_composed_res, a_min=0.5*np.median(a_composed_res), a_max=np.percentile(a_composed_res, 90))
                temp = np.zeros_like(a_composed_res)
                composed_res[:,:,b*W:(b+1)*W] = cv2.normalize(a_composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)
        else:
            composed_res = np.zeros((T, B*H, 5*W))
            for b in range(B):
                for t in range(T):
                    composed_res[t, b*H:(b+1)*H, :W] = resize_img(save_x[b,t,:,:].squeeze(), H, W)
                    composed_res[t, b*H:(b+1)*H, W:2*W] = resize_img(save_p[b,t,:,:].squeeze(), H, W)
                    composed_res[t, b*H:(b+1)*H, 2*W:3*W] = resize_img(save_p_1st_net[b,t,:,:].squeeze(), H, W)
                    composed_res[t, b*H:(b+1)*H, 3*W:4*W] = resize_img(save_y[b,t,:,:].squeeze(), H, W)
                    composed_res[t, b*H:(b+1)*H, 4*W:5*W] = resize_img(save_y_2x[b,t,:,:].squeeze(), H, W)

                a_composed_res = composed_res[:,b*H:(b+1)*H,:]
                a_composed_res = np.clip(a_composed_res, a_min=0.5*np.median(a_composed_res), a_max=np.percentile(a_composed_res, 90))
                temp = np.zeros_like(a_composed_res)
                composed_res[:,b*H:(b+1)*H,:] = cv2.normalize(a_composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

        # composed_res = np.clip(composed_res, a_min=0.5*np.median(composed_res), a_max=np.percentile(composed_res, 90))
        # temp = np.zeros_like(composed_res)
        # composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

        return np.repeat(composed_res[:,np.newaxis,:,:].astype('uint8'), 3, axis=1)


# -------------------------------------------------------------------------------------------------
def tests():
    pass    

if __name__=="__main__":
    tests()
