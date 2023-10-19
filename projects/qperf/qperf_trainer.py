"""
Training and evaluation loops for QPerf
"""

import copy
import numpy as np
from time import time

import os
import sys
import logging
import gc

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

from torchinfo import summary

from trainer import *
from utils.status import start_timer, end_timer, support_bfloat16
from metrics.metrics_utils import AverageMeter
from optim.optim_utils import compute_total_steps

from qperf_data import QPerfDataSet, normalize_data, denormalize_data
from projects.mri.LSUV import LSUVinit

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

def qpref_model_info(model, config):
    c = config
    input_size = (c.batch_size, c.qperf_T, 2)
    col_names=("num_params", "params_percent", "mult_adds", "input_size", "output_size", "trainable")
    row_settings=["var_names", "depth"]
    dtypes=[torch.float32]

    model_summary = summary(model, verbose=0, mode="train", depth=c.summary_depth,\
                            input_size=input_size, col_names=col_names,\
                            row_settings=row_settings, dtypes=dtypes,\
                            device=config.device)

    c.trainable_params = model_summary.trainable_params
    c.total_params = model_summary.total_params
    c.total_mult_adds = model_summary.total_mult_adds

    torch.cuda.empty_cache()

    return model_summary

# -------------------------------------------------------------------------------------------------

class QPerfTrainManager(TrainManager):
    def __init__(self, config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager):  
        super().__init__(config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager)

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
            model_summary = qpref_model_info(self.model_manager, c)
            logging.info(f"Configuration for this run:\n{c}") # Commenting out, prints a lot of info
            logging.info(f"Model Summary:\n{str(model_summary)}") # Commenting out, prints a lot of info
            logging.info(f"Wandb name:\n{wandb_run.name}")
            # try:
            #     wandb_run.watch(self.model_manager)
            # except:
            #     pass
            wandb_run.log_code(".")

        # -----------------------------------------------

        if c.ddp:
            dist.barrier()
            device = torch.device(f"cuda:{rank}")
            model_manager = self.model_manager.to(device)
        else:
            device = c.device
            model_manager = self.model_manager.to(device)

        if not config.disable_LSUV:
            if config.pre_model_load_path is None and config.backbone_model_load_path is None and config.post_model_load_path is None:
                t0 = time()
                num_samples = len(self.train_sets[-1])
                sampled_picked = np.random.randint(0, num_samples, size=1024)
                input_data  = torch.stack([torch.from_numpy(self.train_sets[-1][i][0]) for i in sampled_picked])
                print(f"{rank_str}, prepared data {input_data.shape}, LSUV prep data took {time()-t0 : .2f} seconds ...")

                t0 = time()
                LSUVinit(model_manager, input_data.to(device=device, dtype=torch.float32), verbose=True, cuda=True)
                print(f"{rank_str}, LSUVinit took {time()-t0 : .2f} seconds ...")

        # -----------------------------------------------
        if c.ddp:
            model_manager = DDP(model_manager, device_ids=[rank], find_unused_parameters=True)
            if isinstance(self.train_sets,list): 
                samplers = [DistributedSampler(train_set, shuffle=True) for train_set in self.train_sets]
            else: 
                samplers = DistributedSampler(self.train_sets, shuffle=True)
            shuffle = False
        else:
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

        logging.info(f"{rank_str}, {Fore.RED}Local Rank:{rank}, global rank: {global_rank}, {c.n_layer}, {c.qperf_T}, {c.use_pos_embedding}, {c.optim_type}, {c.optim}, {c.scheduler_type}, {c.losses}, {c.loss_weights}{Style.RESET_ALL}")

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

        # -----------------------------------------------

        #torch.autograd.set_detect_anomaly(True)

        if rank<=0: # main or master process
            if c.ddp: 
                setup_logger(self.config) 

            if wandb_run is not None:
                wandb_run.summary["trainable_params"] = c.trainable_params
                wandb_run.summary["total_params"] = c.total_params
                wandb_run.summary["total_mult_adds"] = c.total_mult_adds 

                wandb_run.save(self.config.yaml_file)

            # log a few training examples
            for i, train_set_x in enumerate(self.train_sets):
                ind = np.random.randint(0, len(train_set_x), 16)
                for ii in ind:
                    x, y, p = train_set_x[ii]

                    Fp = p[0]
                    Vp = p[1]
                    Visf = p[2]
                    PS = p[3]
                    Delay = int(p[4])
                    foot = int(p[5])
                    peak = int(p[6])
                    valley = int(p[7])
                    used_n = int(p[8])

                    x, y, p = denormalize_data(x, y, p)

                    N = x.shape[0]

                    print(f"--> upload tra {ii} ...")
                    wandb.log({f"tra {ii}" : wandb.plot.line_series(
                                xs=list(np.arange(N)),
                                ys=[list(x[:,0]), list(x[:,1]), list(y.flatten())],
                                keys=["aif", "myo", "myo_clean"],
                                title=f"Tra, Fp={Fp:.2f},Vp={Vp:.2f},Visf={Visf:.2f},PS={PS:.2f},Delay={Delay},foot={foot},peak={peak},valley={valley},used_n={used_n}",
                                xname="T")})

        # -----------------------------------------------

        # Handle mix precision training
        scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp)

        # Zero gradient before training
        optim.zero_grad(set_to_none=True)

        # Compute total iters
        total_iters = sum([len(train_loader) for train_loader in train_loaders])if not c.debug else 3

        dtype = torch.float32
        if config.use_amp:
            dtype=torch.bfloat16

        # ----------------------------------------------------------------------------
        # Training loop

        if self.config.train_model:

            logging.info(f"{rank_str}, {Fore.GREEN}----------> Start training loop <----------{Style.RESET_ALL}")
            logging.info(f"{rank_str}, {Fore.YELLOW}len(train_loader) is {len(train_loaders[0])}, train_set size is {len(self.train_sets[0])}, batch_size {c.batch_size} {Style.RESET_ALL}")

            if c.ddp:
                model_manager.module.check_model_learnable_status(rank_str)
            else:
                model_manager.check_model_learnable_status(rank_str)

            # ----------------------------------------------------------------------------
            epoch = curr_epoch
            for epoch in range(curr_epoch, c.num_epochs):
                logging.info(f"{Fore.GREEN}{'-'*20}Epoch:{epoch}/{c.num_epochs}, rank {rank} {'-'*20}{Style.RESET_ALL}")

                model_manager.train()
                if c.ddp: [train_loader.sampler.set_epoch(epoch) for train_loader in train_loaders]
                self.metric_manager.on_train_epoch_start()
                train_loader_iters = [iter(train_loader) for train_loader in train_loaders]

                if epoch > 0:
                    for tra in self.train_sets:
                        tra.generate_picked_samples()

                # ----------------------------------------------------------------------------
                with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:
                    for idx in range(total_iters):

                        # if idx > 20:
                        #     break

                        # -------------------------------------------------------
                        tm = start_timer(enable=c.with_timer)

                        loader_ind = idx % len(train_loader_iters)
                        loader_outputs = next(train_loader_iters[loader_ind], None)

                        while loader_outputs is None:
                            del train_loader_iters[loader_ind]
                            #self.train_sets[loader_ind].generate_picked_samples()
                            loader_ind = idx % len(train_loader_iters)
                            loader_outputs = next(train_loader_iters[loader_ind], None)

                        x, y, p = loader_outputs

                        x = x.to(device=device, dtype=dtype)
                        y = y.to(device, dtype=dtype)
                        p = p.to(device, dtype=dtype)

                        end_timer(enable=c.with_timer, t=tm, msg="---> load batch took ")

                        # del loader_outputs
                        # if idx % 100 == 0:
                        #     gc.collect()
                        # pbar.update(1)

                        # continue

                        # -------------------------------------------------------
                        tm = start_timer(enable=c.with_timer)

                        B, T, D = x.shape

                        with torch.autocast(device_type='cuda', dtype=dtype, enabled=c.use_amp):
                            model_output = model_manager(x.to(dtype=dtype))
                            y_hat, p_estimated = model_output

                            if torch.isnan(torch.sum(y_hat)):
                                continue

                            loss = loss_f(model_output, (y, p))

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
                                nn.utils.clip_grad_norm_(model_manager.parameters(), c.clip_grad_norm)

                            scaler.step(optim)
                            optim.zero_grad(set_to_none=True)
                            scaler.update()

                            if c.scheduler_type == "OneCycleLR": 
                                sched.step()

                        end_timer(enable=c.with_timer, t=tm, msg="---> other steps took ")

                        # -------------------------------------------------------
                        tm = start_timer(enable=c.with_timer)
                        curr_lr = optim.param_groups[0]['lr']

                        loss_value = loss.detach().item()

                        self.metric_manager.on_train_step_end(loss_value, model_output, loader_outputs, rank, curr_lr, False, epoch, False)

                        # -------------------------------------------------------

                        pbar.update(1)
                        log_str = self.create_log_str(config, epoch, rank, 
                                         None, 
                                         self.metric_manager,
                                         curr_lr, 
                                         "tra")

                        pbar.set_description_str(log_str)

                        end_timer(enable=c.with_timer, t=tm, msg="---> epoch step logging and measuring took ")

                        if idx % 100 == 0:
                            del x, y, p, loss, model_output, loader_outputs, y_hat, p_estimated
                            gc.collect()
                            torch.cuda.empty_cache()
                    # ------------------------------------------------------------------------------------------------------

                    # Run metric logging for each epoch 
                    tm = start_timer(enable=c.with_timer) 

                    self.metric_manager.on_train_epoch_end(epoch, rank)

                    # Print out metrics from this epoch
                    log_str = self.create_log_str(config, epoch, rank, 
                                                None, 
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
                    self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=epoch, device=device, optim=optim, sched=sched, id="val_in_training", split="val", final_eval=False, scaling_factor=1)

                if c.scheduler_type != "OneCycleLR":
                    if c.scheduler_type == "ReduceLROnPlateau":
                        sched.step(loss_value)
                    elif c.scheduler_type == "StepLR":
                        sched.step()

                    if c.ddp:
                        self.distribute_learning_rates(rank, optim, src=0)

            # ----------------------------------------------------------------------------

            model_manager.save(os.path.join(self.config.log_dir, self.config.run_name, 'last_checkpoint'), epoch, optim, sched)
            if wandb_run is not None:
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_pre.pth'))
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_backbone.pth'))
                wandb_run.save(os.path.join(self.config.log_dir,self.config.run_name,'last_checkpoint_post.pth'))

            # Load the best model from training
            if self.config.eval_train_set or self.config.eval_val_set or self.config.eval_test_set:
                logging.info(f"{Fore.CYAN}Loading the best models from training for final evaluation...{Style.RESET_ALL}")
                if self.metric_manager.best_pre_model_file is not None: model_manager.load_pre(self.metric_manager.best_pre_model_file)
                if self.metric_manager.best_backbone_model_file is not None: model_manager.load_backbone(self.metric_manager.best_backbone_model_file)
                if self.metric_manager.best_post_model_file is not None: model_manager.load_post(self.metric_manager.best_post_model_file)

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

            logging.info(f"{Fore.CYAN}Evaluating the best model on the val set without noise ... {Style.RESET_ALL}")
            self.val_sets[0].add_noise = [False, False]
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="val-clean", split="val", final_eval=True, scaling_factor=1)

        if self.config.eval_test_set: 
            logging.info(f"{Fore.CYAN}Evaluating the best model on the test set ... {Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.test_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="test", split="test", final_eval=True, scaling_factor=1)

        # -----------------------------------------------

        save_path, save_file_name, config_yaml_file = model_manager.save_entire_model(epoch=self.config.num_epochs)
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
            if isinstance(data_sets, list): samplers = [DistributedSampler(data_set, shuffle=True) for data_set in data_sets]
            else: samplers = DistributedSampler(data_sets, shuffle=True)
        else:
            if isinstance(data_sets, list): samplers = [None] * len(data_sets)
            else: samplers = None

        # ------------------------------------------------------------------------
        # Set up data loader to evaluate
        batch_size = c.batch_size
        num_workers_per_loader = c.num_workers // (2 * len(data_sets))

        print(f"{Fore.YELLOW}--> num_workers_per_loader for eval i {num_workers_per_loader} ... {Style.RESET_ALL}")

        if isinstance(data_sets, list):
            data_loaders = [DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, sampler=samplers[ind],
                                    num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                    persistent_workers=False, pin_memory=False) for ind, data_set in enumerate(data_sets)]
        else:
            data_loaders = [DataLoader(dataset=data_sets, batch_size=batch_size, shuffle=False, sampler=samplers,
                                    num_workers=num_workers_per_loader, prefetch_factor=c.prefetch_factor, drop_last=True,
                                    persistent_workers=False, pin_memory=False) ]

        # ------------------------------------------------------------------------
        self.metric_manager.on_eval_epoch_start()

        if rank<=0:
            wandb_run = self.metric_manager.wandb_run
        else:
            wandb_run = None

        # ------------------------------------------------------------------------
        model_manager.eval()
        # ------------------------------------------------------------------------

        data_loader_iters = [iter(data_loader) for data_loader in data_loaders]
        total_iters = sum([len(data_loader) for data_loader in data_loaders]) if not c.debug else 3

        dtype=torch.float32
        if c.use_amp:
            dtype=torch.bfloat16

        # ------------------------------------------------------------------------
        # Evaluation loop
        with torch.inference_mode():
            with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

                samples_logged = 0

                for idx in range(total_iters):

                    loader_ind = idx % len(data_loader_iters)
                    loader_outputs = next(data_loader_iters[loader_ind], None)
                    while loader_outputs is None:
                        del data_loader_iters[loader_ind]
                        loader_ind = idx % len(data_loader_iters)
                        loader_outputs = next(data_loader_iters[loader_ind], None)
                    x, y, p = loader_outputs

                    # del x, y, p, loader_outputs
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    # pbar.update(1)
                    # continue

                    # ----------------------------------------------------------------------

                    B = x.shape[0]

                    x = x.to(device=device, dtype=dtype)
                    y = y.to(device=device, dtype=dtype)
                    p = p.to(device=device, dtype=dtype)

                    # ----------------------------------------------------------------------

                    with torch.autocast(device_type='cuda', dtype=dtype, enabled=c.use_amp):
                        output = model_manager(x)
                        loss = loss_f(output, (y, p))

                    # Update evaluation metrics
                    self.metric_manager.on_eval_step_end(loss.detach().item(), output, (x, y, p), f"epoch_{epoch}_{idx}", rank, save_samples, split)

                    # ----------------------------------------------------------------------
                    # if required, upload samples
                    if rank<=0 and samples_logged < self.config.num_uploaded and wandb_run is not None:
                        samples_logged += 1
                        title = f"{id.upper()}_{samples_logged}_{x.shape}"

                        N = x.shape[0]
                        y_hat, p_estimated = output

                        x = x.to(dtype=torch.float32).detach().cpu().numpy()
                        y = y.to(dtype=torch.float32).detach().cpu().numpy()
                        y_hat = y_hat.to(dtype=torch.float32).detach().cpu().numpy()

                        x, y, p = denormalize_data(x, y, p)
                        x, y_hat, p_estimated = denormalize_data(x, y_hat, p_estimated)

                        Fp = p[idx, 0]
                        Vp = p[idx, 1]
                        Visf = p[idx, 2]
                        PS = p[idx, 3]
                        Delay = p[idx, 4]

                        Fp_est = p_estimated[idx, 0]
                        Vp_est = p_estimated[idx, 1]
                        Visf_est = p_estimated[idx, 2]
                        PS_est = p_estimated[idx, 3]
                        Delay_est = p_estimated[idx, 4]

                        wandb.log({title : wandb.plot.line_series(
                                    xs=list(np.arange(N)),
                                    #ys=[list(x[idx,:,0]), list(x[idx,:,1]), list(y[idx,:].flatten()), list(y_hat[idx,:].flatten())],
                                    #keys=["aif", "myo", "myo_clean", "myo_model"],
                                    # ys=[list(x[idx,:,1]), list(y[idx,:].flatten()), list(y_hat[idx,:].flatten())],
                                    # keys=["myo", "myo_clean", "myo_model"],
                                    ys=[list(y[idx,:].flatten()), list(y_hat[idx,:].flatten())],
                                    keys=["myo_clean", "myo_model"],
                                    title=f"{id}, Fp={Fp:.2f}, Vp={Vp:.2f}, Visf={Visf:.2f}, PS={PS:.2f}, Delay={Delay} - Fp={Fp_est:.2f}, Vp={Vp_est:.2f}, Visf={Visf_est:.2f}, PS={PS_est:.2f}, Delay={Delay_est}",
                                    xname="T")})

                    # ----------------------------------------------------------------------
                    # Print evaluation metrics to terminal
                    pbar.update(1)

                    log_str = self.create_log_str(self.config, epoch, rank, 
                                                x.shape, 
                                                self.metric_manager,
                                                curr_lr, 
                                                split)

                    pbar.set_description(log_str)

                    # ----------------------------------------------------------------------

                    if idx % 100 == 0:
                        del x, y, p, loss, output, loader_outputs
                        gc.collect()
                        torch.cuda.empty_cache()

                # -----------------------------------------------------------------------------------------------------------

                # Update evaluation metrics 
                print(f"--> self.metric_manager.on_eval_epoch_end ... ")
                self.metric_manager.on_eval_epoch_end(rank, epoch, model_manager, optim, sched, split, final_eval)

                # Print evaluation metrics to terminal
                log_str = self.create_log_str(self.config, epoch, rank, 
                                                x.shape, 
                                                self.metric_manager,
                                                curr_lr, 
                                                split)

                if hasattr(self.metric_manager, 'average_eval_metrics'):
                    pbar_str = f"--> rank {rank}, {split}, epoch {epoch}"
                    if isinstance(self.metric_manager.average_eval_metrics, dict):
                        for metric_name, metric_value in self.metric_manager.average_eval_metrics.items():
                            try: pbar_str += f", {Fore.CYAN} {metric_name} {metric_value:.8f}"
                            except: pass

                            # Save final evaluation metrics to a text file
                            if final_eval and rank<=0:
                                metric_file = os.path.join(self.config.log_dir,self.config.run_name, f'{split}_metrics.txt')
                                with open(metric_file, 'a') as f:
                                    try: f.write(f"{split}_{metric_name}: {metric_value:.8f}, ")
                                    except: pass
                                wandb_run.save(metric_file)

                    pbar_str += f"{Style.RESET_ALL}"
                else:
                    pbar_str = log_str

                pbar.set_description(pbar_str)

                if rank<=0: 
                    logging.getLogger("file_only").info(pbar_str)
        return 


    def create_log_str(self, config, epoch, rank, data_shape, loss_meters, curr_lr, role):
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

        if role == 'tra':
            loss, mse, l1, gauss, mae, Fp, Vp, Visf, PS, Delay = loss_meters.get_tra_loss()
        else:
            loss, mse, l1, gauss, mae, Fp, Vp, Visf, PS, Delay = loss_meters.get_eval_loss()

        str= f"{Fore.GREEN}Epoch {epoch}/{config.num_epochs}, {C}{role}, {Style.RESET_ALL}{rank}, " + data_shape_str + f"{Fore.BLUE}{Back.WHITE}{Style.BRIGHT}loss {loss:.8f},{Style.RESET_ALL} {C}mse {mse:.8f}, l1 {l1:.8f}, gauss {gauss:.8f}, mae {mae:.8f}, Fp {Fp:.8f}, Vp {Vp:.8f}, Visf {Visf:.8f}, PS {PS:.8f}, Delay {Delay:.8f}{Style.RESET_ALL}{lr_str}"

        return str

# -------------------------------------------------------------------------------------------------
def tests():
    pass    

if __name__=="__main__":
    tests()
