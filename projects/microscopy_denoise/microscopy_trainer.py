"""
Training and evaluation loops for Microscopy
"""

import os
import sys
import cv2
import copy
import wandb
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from colorama import Fore, Back, Style
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

from utils import *
from trainer import *
from utils.status import model_info, start_timer, end_timer, support_bfloat16, count_parameters
from metrics.metrics_utils import AverageMeter
from optim.optim_utils import compute_total_steps

from projects.microscopy_denoise.microscopy_dataset import MicroscopyDatasetTrain
from projects.microscopy_denoise.microscopy_loss import microscopy_loss
from temp_utils.running_inference import running_inference

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

class MicroscopyTrainManager(TrainManager):
    """
    Microscopy train manager
        - support Microscopy double net
    """
    def __init__(self, config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager):
        super().__init__(config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager)

    # -------------------------------------------------------------------------------------------------

    def _train_model(self, rank, global_rank):
        """
        The training loop. Allows training on cpu/single gpu/multiple gpu (ddp)
        @args:
            - rank (int): for distributed data parallel (ddp) -1 if running on cpu or only one gpu
            - global_rank (int): for distributed data parallel (ddp)

        """
        c = self.config # shortening due to numerous uses

        # All metrics are handled by metrics.py
        self.metric_manager.setup_wandb_and_metrics(rank)

        # Freeze portions of the network, if desired
        if self.config.freeze_pre: self.model_manager.freeze_pre()
        if self.config.freeze_backbone: self.model_manager.freeze_backbone()
        if self.config.freeze_post: self.model_manager.freeze_post()

        if rank<=0:
            model_summary = model_info(self.model_manager, c)
            logging.info(f"Configuration for this run:\n{c}")
            logging.info(f"Model Summary:\n{str(model_summary)}")
            logging.info(f"Wandb name:\n{self.metric_manager.wandb_run.name}")
            #self.metric_manager.wandb_run.watch(self.model_manager)
            wandb_run = self.metric_manager.wandb_run
        else:
            wandb_run = None

        if c.ddp:
            dist.barrier()
            device = torch.device(f"cuda:{rank}")
            model_manager = self.model_manager.to(device)
            model_manager = DDP(model_manager, device_ids=[rank], find_unused_parameters=True)
            if isinstance(self.train_sets,list): samplers = [DistributedSampler(train_set) for train_set in self.train_sets]
            else: samplers = DistributedSampler(self.train_sets)
            shuffle = False
        else:
            device = c.device
            model_manager = self.model_manager.to(device)
            if isinstance(self.train_sets,list): samplers = [None] * len(self.train_sets)
            else: samplers = None
            shuffle = True

        optim = self.optim_manager.optim
        sched = self.optim_manager.sched
        curr_epoch = self.optim_manager.curr_epoch
        loss_f = self.loss_f
        logging.info(f"{Fore.RED}{'-'*20}Local Rank:{rank}, global rank {global_rank}{'-'*20}{Style.RESET_ALL}")

        if isinstance(self.train_sets,list):
            train_loaders = [DataLoader(dataset=train_set, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers[ind],
                                        num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=True,
                                        persistent_workers=c.num_workers>0) for ind, train_set in enumerate(self.train_sets)]
        else:
            train_loaders = [DataLoader(dataset=self.train_sets, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers,
                                        num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=True,
                                        persistent_workers=c.num_workers>0)]

        if rank<=0: # main or master process
            if c.ddp:
                setup_logger(self.config) # setup master process logging; I don't know if this needs to be here, it is also in setup.py

            if wandb_run is not None:
                wandb_run.summary["trainable_params"] = c.trainable_params
                wandb_run.summary["total_params"] = c.total_params
                wandb_run.summary["total_mult_adds"] = c.total_mult_adds 

                wandb_run.save(self.config.yaml_file)

                # log a few training examples
                for i, train_set_x in enumerate(self.train_sets):
                    if i > self.config.num_uploaded: break
                    ind = np.random.randint(0, len(train_set_x), 4)
                    noisy, clean, _ = train_set_x[ind[0]]
                    noisy = np.expand_dims(noisy, axis=0)
                    clean = np.expand_dims(clean, axis=0)
                    for ii in range(1, len(ind)):
                        a_x, a_y, _ = train_set_x[ind[ii]]
                        noisy = np.concatenate((noisy, np.expand_dims(a_x, axis=0)), axis=0)
                        clean = np.concatenate((clean, np.expand_dims(a_y, axis=0)), axis=0)

                    title = f"Tra_samples_{i}_Noisy_Noisy_GT_{noisy.shape}"
                    vid = self.create_image_batch(c.complex_i, noisy, noisy, clean)
                    if wandb_run is not None: wandb_run.log({title:wandb.Video(vid, caption=f"Tra sample {i}", fps=1, format='gif')})
                    logging.info(f"{Fore.YELLOW}---> Upload tra sample - {title}")

        # Handle mix precision training
        scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp)

        # Zero gradient before training
        optim.zero_grad(set_to_none=True)

        # Compute total iters
        total_iters = sum([len(train_loader) for train_loader in train_loaders])if not c.debug else 3

        # Training loop
        if self.config.train_model:

            logging.info(f"{Fore.CYAN}OPTIMIZER PARAMETERS: {optim} {Style.RESET_ALL}")

            for epoch in range(curr_epoch, c.num_epochs):
                logging.info(f"{Fore.GREEN}{'-'*20}Epoch:{epoch}/{c.num_epochs}, rank {rank} {'-'*20}{Style.RESET_ALL}")

                model_manager.train()
                if c.ddp: [train_loader.sampler.set_epoch(epoch) for train_loader in train_loaders]
                self.metric_manager.on_train_epoch_start()
                train_loader_iters = [iter(train_loader) for train_loader in train_loaders]

                images_logged = 0

                with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:
                    for idx in range(total_iters):

                        tm = start_timer(enable=c.with_timer)
                        loader_ind = idx % len(train_loader_iters)
                        loader_outputs = next(train_loader_iters[loader_ind], None)
                        while loader_outputs is None:
                            del train_loader_iters[loader_ind]
                            loader_ind = idx % len(train_loader_iters)
                            loader_outputs = next(train_loader_iters[loader_ind], None)
                        inputs, targets, ids = loader_outputs
                        end_timer(enable=c.with_timer, t=tm, msg="---> load batch took ")

                        tm = start_timer(enable=c.with_timer)
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        with torch.autocast(device_type='cuda', dtype=self.cast_type, enabled=c.use_amp):
                            output = model_manager(inputs)
                            loss = loss_f(output, targets)
                            loss = loss / c.iters_to_accumulate
                        end_timer(enable=c.with_timer, t=tm, msg="---> forward pass took ")

                        tm = start_timer(enable=c.with_timer)
                        scaler.scale(loss).backward()
                        end_timer(enable=c.with_timer, t=tm, msg="---> backward pass took ")

                        tm = start_timer(enable=c.with_timer)
                        if (idx + 1) % c.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                            if(c.clip_grad_norm>0):
                                scaler.unscale_(optim)
                                nn.utils.clip_grad_norm_(model_manager.parameters(), c.clip_grad_norm)

                            scaler.step(optim)
                            optim.zero_grad(set_to_none=True)
                            scaler.update()

                            if c.scheduler_type == "OneCycleLR": sched.step()
                        end_timer(enable=c.with_timer, t=tm, msg="---> other steps took ")

                        tm = start_timer(enable=c.with_timer)
                        curr_lr = optim.param_groups[0]['lr']

                        self.metric_manager.on_train_step_end(loss.item(), output, (inputs, targets), rank, curr_lr, self.config.save_train_samples and images_logged < self.config.num_uploaded, epoch, "tra")

                        # log some train samples as well
                        if rank<=0:
                            if images_logged < self.config.num_uploaded and wandb_run is not None:
                                images_logged += 1
                                title = f"Train_image_{images_logged}_{inputs.shape}"
                                vid = self.create_image_batch(c.complex_i, inputs.numpy(force=True), output.numpy(force=True), targets.numpy(force=True))
                                wandb_run.log({title: wandb.Video(vid, caption=f"epoch {epoch}, image {images_logged}", fps=1, format="gif")})

                        pbar.update(1)
                        pbar.set_description(f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} tra, rank {rank}, {inputs.shape}, lr {curr_lr:.8f}, loss {loss.item():.4f}{Style.RESET_ALL}")

                        end_timer(enable=c.with_timer, t=tm, msg="---> epoch step logging and measuring took ")

                    # Run metric logging for each epoch
                    tm = start_timer(enable=c.with_timer)

                    self.metric_manager.on_train_epoch_end(model_manager, optim, sched, epoch, rank)

                    # Print out metrics from this epoch
                    pbar_str = f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} tra, rank {rank},  {inputs.shape}, lr {curr_lr:.8f}"
                    if hasattr(self.metric_manager, 'average_train_metrics'):
                        if isinstance(self.metric_manager.average_train_metrics, dict):
                            for metric_name, metric_value in self.metric_manager.average_train_metrics.items():
                                try: pbar_str += f", {Fore.CYAN} {metric_name} {metric_value:.4f}"
                                except: pass
                    pbar_str += f"{Style.RESET_ALL}"
                    pbar.set_description(pbar_str)

                    # Write training status to log file
                    if rank<=0:
                        logging.getLogger("file_only").info(pbar_str)

                    end_timer(enable=c.with_timer, t=tm, msg="---> epoch end logging and measuring took ")

                if epoch % c.eval_frequency==0 or epoch==c.num_epochs:
                    self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=epoch, device=device, optim=optim, sched=sched, id="Val", split="val", final_eval=False)

                if c.scheduler_type != "OneCycleLR":
                    if c.scheduler_type == "ReduceLROnPlateau":
                        try:
                            sched.step(self.metric_manager.average_eval_metrics['loss'])
                        except:
                            warnings.warn("Average loss not available, using step loss to step scheduler.")
                            sched.step(loss.item())
                    elif c.scheduler_type == "StepLR":
                        sched.step()

                    if c.ddp:
                        self.distribute_learning_rates(rank, optim, src=0)

            if rank <= 0:
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
        else: epoch = 0

        # Evaluate models of each split
        if self.config.eval_train_set:
            logging.info(f"{Fore.CYAN}Evaluating train set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.train_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="Tra_final", split="train", final_eval=True)
        if self.config.eval_val_set:
            logging.info(f"{Fore.CYAN}Evaluating val set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="Val_final", split="val", final_eval=True)
        if self.config.eval_test_set:
            logging.info(f"{Fore.CYAN}Evaluating test set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.test_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="Tes_final", split="test", final_eval=True)

        if rank <= 0 and self.config.train_model:
            save_path, save_file_name, config_yaml_file = self.model_manager.save_entire_model(epoch=self.config.num_epochs)
            model_full_path = os.path.join(save_path, save_file_name+'.pth')
            logging.info(f"{Fore.YELLOW}Entire model is saved at {model_full_path} ...{Style.RESET_ALL}")

            if wandb_run is not None:
                wandb_run.save(model_full_path)
                wandb_run.save(config_yaml_file)

        # Finish up training
        self.metric_manager.on_training_end(rank, epoch, model_manager, optim, sched, self.config.train_model)

    def _eval_model(self, rank, model_manager, data_sets, epoch, device, optim, sched, id, split, final_eval):
        """
        Model evaluation.
        @args:
            - rank (int): used for ddp
            - model_manager (ModelManager): model to be validated
            - data_sets (torch Dataset or list of torch Datasets): the data to evaluate
            - epoch (int): the current epoch
            - device (torch.device): the device to run eval on
            - optim: optimizer for training
            - sched: scheduler for optimizer
            - id: identifier for ddp runs
            - split: one of {train, val, test}
            - final_eval: whether this is the final evaluation being run at the end of training
        @rets:
            - None; logs and checkpoints within this function
        """
        c = self.config # shortening due to numerous uses
        curr_lr = optim.param_groups[0]['lr']

        # Determine if we will save the predictions to files for thie eval
        if split=='train': save_samples = final_eval and self.config.save_train_samples
        elif split=='val': save_samples = self.config.save_val_samples
        elif split=='test': save_samples = final_eval and self.config.save_test_samples
        else: raise ValueError(f"Unknown split {split} specified, should be in [train, val, test]")

        if c.ddp:
            loss_f = self.loss_f
            if isinstance(data_sets, list): samplers = [DistributedSamplerNoDuplicate(data_set,rank=rank) for data_set in data_sets]
            else: samplers = DistributedSamplerNoDuplicate(data_sets,rank=rank)
        else:
            loss_f = self.loss_f
            if isinstance(data_sets, list): samplers = [None] * len(data_sets)
            else: samplers = None
        
        if split=="test":
            config_copy = copy.deepcopy(self.config)
            config_copy.losses = [loss for loss in config_copy.losses if loss != "perceptual"]
            loss_f = microscopy_loss(config=config_copy)

        batch_size = 1 if final_eval else c.batch_size

        # Set up data loader to evaluate
        if isinstance(data_sets, list):
            data_loaders = [DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, sampler=samplers[ind],
                                    num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=False,
                                    persistent_workers=c.num_workers>0) for ind, data_set in enumerate(data_sets)]
        else:
            data_loaders = [DataLoader(dataset=data_sets, batch_size=batch_size, shuffle=False, sampler=samplers,
                                    num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=False,
                                    persistent_workers=c.num_workers>0) ]

        self.metric_manager.on_eval_epoch_start()

        model_manager.eval()

        cutout = (c.micro_time, c.micro_height[-1], c.micro_width[-1])
        overlap = (c.micro_time//2, c.micro_height[-1]//4, c.micro_width[-1]//4)

        data_loader_iters = [iter(data_loader) for data_loader in data_loaders]
        total_iters = sum([len(data_loader) for data_loader in data_loaders]) 
        total_iters = total_iters if not c.debug else min(total_iters, 3)

        if rank<=0:
            wandb_run = self.metric_manager.wandb_run
        else:
            wandb_run = None
        images_logged = 0

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
                    inputs, targets, ids = loader_outputs

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    B, C, T, H, W = inputs.shape

                    if (H==c.micro_height[0] or H==c.micro_height[-1]) and (W==c.micro_width[0] or W==c.micro_width[-1]):
                        output = model_manager(inputs)
                        if isinstance(output, tuple):
                            output_1st_net = output[1]
                            output = output[0]
                        else:
                            output_1st_net = None
                    else:

                        inputs = torch.permute(inputs, (0, 2, 1, 3, 4))

                        cutout_in = cutout
                        overlap_in = overlap
                        if not self.config.pad_time:
                            cutout_in = (T, c.micro_height[-1], c.micro_width[-1])
                            overlap_in = (0, c.micro_height[-1]//2, c.micro_width[-1]//2)

                        _, output = running_inference(model_manager, inputs, cutout=cutout_in, overlap=overlap_in, device=device, batch_size=c.batch_size)
                        output_1st_net = None

                        inputs = torch.permute(inputs, (0, 2, 1, 3, 4))
                        output = torch.permute(output, (0, 2, 1, 3, 4))

                    with torch.autocast(device_type='cuda', dtype=self.cast_type, enabled=c.use_amp):
                        loss = loss_f(output, targets)

                    ids = ids[0] if B==1 else f"epoch_{epoch}_{split}"

                    # Update evaluation metrics
                    caption, _ = self.metric_manager.on_eval_step_end(loss.item(), output, (inputs,targets), ids, rank, save_samples, split)

                    if rank<=0:
                        if images_logged < self.config.num_uploaded and wandb_run is not None:
                            images_logged += 1
                            title = f"{id.upper()}_{images_logged}_{inputs.shape}"
                            if output_1st_net is None: 
                                output_1st_net = output
                            vid = self.create_image_batch(c.complex_i, inputs.numpy(force=True), output.numpy(force=True), targets.numpy(force=True))
                            wandb_run.log({title: wandb.Video(vid, caption=f"epoch {epoch}, {caption}", fps=1, format="gif")})

                    # Print evaluation metrics to terminal
                    pbar.update(1)
                    pbar.set_description(f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} {split}, rank {rank}, {id} {inputs.shape}, lr {curr_lr:.8f}, loss {loss.item():.4f}{Style.RESET_ALL}")

                # TODO: test what happens to avg when one node has no data
                # Update evaluation metrics
                self.metric_manager.on_eval_epoch_end(rank, epoch, model_manager, optim, sched, split, final_eval)

                # Print evaluation metrics to terminal
                shape = "no inputs" if total_iters == 0 else inputs.shape
                pbar_str = f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} {split}, rank {rank}, {id} {shape}, lr {curr_lr:.8f}"
                if hasattr(self.metric_manager, 'average_eval_metrics'):
                    if isinstance(self.metric_manager.average_eval_metrics, dict):
                        for metric_name, metric_value in self.metric_manager.average_eval_metrics.items():
                            try: pbar_str += f", {Fore.MAGENTA} {metric_name} {metric_value:.4f}"
                            except: pass

                        # Save final evaluation metrics to a text file
                        if final_eval and rank<=0:
                            metric_file = os.path.join(self.config.log_dir,self.config.run_name,f'{split}_metrics.txt')
                            with open(metric_file, 'w') as f:
                                for metric_name, metric_value in self.metric_manager.average_eval_metrics.items():
                                    try: f.write(f"{split}_{metric_name}: {metric_value:.4f}, ")
                                    except: pass

                pbar_str += f"{Style.RESET_ALL}"
                pbar.set_description(pbar_str)

                if rank<=0:
                    logging.getLogger("file_only").info(pbar_str)

        return
    
    # -------------------------------------------------------------------------------------------------
    
    def create_image_batch(self, complex_i, noisy, predi, clean):
        """
        Logs the image to wandb as a 5D gif [B,T,C,H,W]
        If complex image then save the magnitude using first 2 channels
        Else use just the first channel
        @args:
            - complex_i (bool): complex images or not
            - noisy (5D numpy array): the noisy image [B, C+1, T, H, W]
            - predi (5D numpy array): the predicted image [B, C, T, H, W]
            - clean (5D numpy array): the clean image [B, C, T, H, W]
        @rets:
            - video (4D numpy array): [T, C, H, W] The standard video format. (C=1)
        """

        if noisy.ndim == 4:
            noisy = np.expand_dims(noisy, axis=0)
            predi = np.expand_dims(predi, axis=0)
            clean = np.expand_dims(clean, axis=0)

        if complex_i:
            save_x = np.sqrt(np.square(noisy[:,0,:,:,:]) + np.square(noisy[:,1,:,:,:]))
            save_p = np.sqrt(np.square(predi[:,0,:,:,:]) + np.square(predi[:,1,:,:,:]))
            save_y = np.sqrt(np.square(clean[:,0,:,:,:]) + np.square(clean[:,1,:,:,:]))
        else:
            save_x = noisy[:,0,:,:,:]
            save_p = predi[:,0,:,:,:]
            save_y = clean[:,0,:,:,:]

        B, T, H, W = save_y.shape

        def resize_img(im, H_2x, W_2x):
            H, W = im.shape
            if H != H_2x or W != W_2x:
                res = cv2.resize(src=im, dsize=(W_2x, H_2x), interpolation=cv2.INTER_NEAREST)
                return res
            else:
                return im

        max_col = 16
        if B>max_col:
            num_row = B//max_col
            if max_col*num_row < B: 
                num_row += 1
            composed_res = np.zeros((T, 3*H*num_row, max_col*W))
            for b in range(B):
                r = b//max_col
                c = b - r*max_col
                for t in range(T):
                    S = 3*r
                    composed_res[t, S*H:(S+1)*H, c*W:(c+1)*W] = normalize_image(save_x[b,t], percentiles=(2,99))
                    composed_res[t, (S+1)*H:(S+2)*H, c*W:(c+1)*W] = normalize_image(save_p[b,t], percentiles=(2,99))
                    composed_res[t, (S+2)*H:(S+3)*H, c*W:(c+1)*W] = normalize_image(save_y[b,t], percentiles=(2,99))

                a_composed_res = composed_res[:,:,c*W:(c+1)*W]
                temp = np.zeros_like(a_composed_res)
                composed_res[:,:,c*W:(c+1)*W] = cv2.normalize(a_composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)
        
        elif B>2:
            composed_res = np.zeros((T, 3*H, B*W))
            for b in range(B):
                for t in range(T):
                    composed_res[t, :H, b*W:(b+1)*W] = normalize_image(save_x[b,t], percentiles=(2,99))
                    composed_res[t, H:2*H, b*W:(b+1)*W] = normalize_image(save_p[b,t], percentiles=(2,99))
                    composed_res[t, 2*H:3*H, b*W:(b+1)*W] = normalize_image(save_y[b,t], percentiles=(2,99))

                a_composed_res = composed_res[:,:,b*W:(b+1)*W]
                temp = np.zeros_like(a_composed_res)
                composed_res[:,:,b*W:(b+1)*W] = cv2.normalize(a_composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)
        else:
            composed_res = np.zeros((T, B*H, 3*W))
            for b in range(B):
                for t in range(T):
                    composed_res[t, b*H:(b+1)*H, :W] = normalize_image(save_x[b,t], percentiles=(2,99))
                    composed_res[t, b*H:(b+1)*H, W:2*W] = normalize_image(save_p[b,t], percentiles=(2,99))
                    composed_res[t, b*H:(b+1)*H, 2*W:3*W] = normalize_image(save_y[b,t], percentiles=(2,99))

                a_composed_res = composed_res[:,b*H:(b+1)*H,:]
                temp = np.zeros_like(a_composed_res)
                composed_res[:,b*H:(b+1)*H,:] = cv2.normalize(a_composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

        # composed_res = np.clip(composed_res, a_min=0.5*np.median(composed_res), a_max=np.percentile(composed_res, 90))
        # temp = np.zeros_like(composed_res)
        # composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

        return np.repeat(composed_res[:,np.newaxis,:,:].astype('uint8'), 3, axis=1)
