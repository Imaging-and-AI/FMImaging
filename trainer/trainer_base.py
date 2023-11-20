"""
A base model for training, supporting multi-node, multi-gpu training
"""

import os
import sys
import logging
import warnings

from colorama import Fore, Style

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

from trainer_utils import *
from utils.status import model_info, start_timer, end_timer, support_bfloat16
from setup.setup_utils import setup_logger

class TrainManager(object):
    """
    Base Runtime model for training. This class supports:
        - single node, single process, single gpu training
        - single node, multiple process, multiple gpu training
        - multiple nodes, multiple processes, multiple gpu training
    """
    def __init__(self, config, train_sets, val_sets, test_sets, loss_f, model_manager, optim_manager, metric_manager):
    
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - train_sets (List[torch Dataset] or torch Dataset): train split dataset(s)
            - val_sets (List[torch Dataset] or torch Dataset): val split dataset(s)
            - test_sets (List[torch Dataset] or torch Dataset): test split dataset(s)
            - loss_f (function): loss function we aim to minimize
            - model_manager (ModelManager): ModelManager object that contains pre/backbone/post model and forward function
            - optim_manager (OptimManager): OptimManager object that contains optimizer and scheduler
            - metric_maanger (MetricManager): MetricManager object that tracks metrics and checkpoints models during training
        """
        super().__init__()
        self.config = config
        self.train_sets = train_sets
        self.val_sets = val_sets
        self.test_sets = test_sets
        self.loss_f = loss_f
        self.model_manager = model_manager
        self.optim_manager = optim_manager
        self.metric_manager = metric_manager

        if self.config.use_amp:
            if support_bfloat16(self.config.device):
                self.cast_type = torch.bfloat16
            else:
                self.cast_type = torch.float16
        else:
            self.cast_type = torch.float32

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
            # model_summary = model_info(self.model_manager, c)
            # logging.info(f"Configuration for this run:\n{c}") # Commenting out, prints a lot of info
            # logging.info(f"Model Summary:\n{str(model_summary)}") # Commenting out, prints a lot of info
            logging.info(f"Wandb name:\n{self.metric_manager.wandb_run.name}")
            self.metric_manager.wandb_run.watch(self.model_manager)

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

                with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:
                    for idx in range(total_iters):

                        tm = start_timer(enable=c.with_timer)
                        loader_ind = idx % len(train_loader_iters)
                        loader_outputs = next(train_loader_iters[loader_ind], None)
                        while loader_outputs is None:
                            del train_loader_iters[loader_ind]
                            loader_ind = idx % len(train_loader_iters)
                            loader_outputs = next(train_loader_iters[loader_ind], None)
                        inputs, labels, ids = loader_outputs
                        end_timer(enable=c.with_timer, t=tm, msg="---> load batch took ")

                        tm = start_timer(enable=c.with_timer)
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        with torch.autocast(device_type='cuda', dtype=self.cast_type, enabled=c.use_amp):
                            output = model_manager(inputs)
                            loss = loss_f(output, labels)
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

                        self.metric_manager.on_train_step_end(loss.item(), output, labels, rank, curr_lr)

                        pbar.update(1)
                        pbar.set_description(f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} tra, rank {rank}, {inputs.shape}, lr {curr_lr:.8f}, loss {loss.item():.4f}{Style.RESET_ALL}")

                        end_timer(enable=c.with_timer, t=tm, msg="---> epoch step logging and measuring took ")
                        
                    # Run metric logging for each epoch 
                    tm = start_timer(enable=c.with_timer) 

                    self.metric_manager.on_train_epoch_end(epoch, rank)

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
                    self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=epoch, device=device, optim=optim, sched=sched, id="", split="val", final_eval=False)

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
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.train_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="train", final_eval=True)
        if self.config.eval_val_set: 
            logging.info(f"{Fore.CYAN}Evaluating val set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.val_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="val", final_eval=True)
        if self.config.eval_test_set: 
            logging.info(f"{Fore.CYAN}Evaluating test set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=model_manager, data_sets=self.test_sets, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="test", final_eval=True)

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

        # Set upd data loader to evaluate
        if isinstance(data_sets, list):
            data_loaders = [DataLoader(dataset=data_set, batch_size=c.batch_size, shuffle=False, sampler=samplers[ind],
                                    num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=True,
                                    persistent_workers=c.num_workers>0) for ind, data_set in enumerate(data_sets)]
        else:
            data_loaders = [DataLoader(dataset=data_sets, batch_size=c.batch_size, shuffle=False, sampler=samplers,
                                    num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=True,
                                    persistent_workers=c.num_workers>0) ]
            
        self.metric_manager.on_eval_epoch_start()

        model_manager.eval()

        data_loader_iters = [iter(data_loader) for data_loader in data_loaders]
        total_iters = sum([len(data_loader) for data_loader in data_loaders]) if not c.debug else 3
        
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
                    inputs, labels, ids = loader_outputs

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.autocast(device_type='cuda', dtype=self.cast_type, enabled=c.use_amp):
                        output = model_manager(inputs)
                        loss = loss_f(output, labels)

                    # Update evaluation metrics
                    self.metric_manager.on_eval_step_end(loss.item(), output, labels, ids, rank, save_samples, split)

                    # Print evaluation metrics to terminal
                    pbar.update(1)
                    pbar.set_description(f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} {split}, rank {rank}, {id} {inputs.shape}, lr {curr_lr:.8f}, loss {loss.item():.4f}{Style.RESET_ALL}")


                # Update evaluation metrics 
                self.metric_manager.on_eval_epoch_end(rank, epoch, model_manager, optim, sched, split, final_eval)

                # Print evaluation metrics to terminal
                pbar_str = f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} {split}, rank {rank}, {id} {inputs.shape}, lr {curr_lr:.8f}"
                if hasattr(self.metric_manager, 'average_eval_metrics'):
                    if isinstance(self.metric_manager.average_eval_metrics, dict):
                        for metric_name, metric_value in self.metric_manager.average_eval_metrics.items():
                            try: pbar_str += f", {Fore.CYAN} {metric_name} {metric_value:.4f}"
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
       
    def train(self):

        # -------------------------------------------------------
        # get the rank and runtime info
        if self.config.ddp:
            rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        else:
            rank = -1
            global_rank = -1
            print(f"---> ddp is off <---", flush=True)

        print(f"--------> run training on local rank {rank}", flush=True)

        # -------------------------------------------------------
        # initialize wandb

        if global_rank<=0:
            self.metric_manager.init_wandb()
            
        # -------------------------------------------------------
        # if ddp is used, broadcast the parameters from rank0 to all other ranks (originally used for sweep, kept to make sure unified params)

        if self.config.ddp:

            # if rank<=0:
            #     c_list = [self.config]
            #     print(f"{Fore.RED}--->before, on local rank {rank}, {c_list[0].run_name}{Style.RESET_ALL}", flush=True)
            # else:
            #     c_list = [None]
            #     print(f"{Fore.RED}--->before, on local rank {rank}, {self.config.run_name}{Style.RESET_ALL}", flush=True)

            # if world_size > 1:
            #     torch.distributed.broadcast_object_list(c_list, src=0, group=None, device=rank)

            # print(f"{Fore.RED}--->after, on local rank {rank}, {c_list[0].run_name}{Style.RESET_ALL}", flush=True)
            # if rank>0:
            #     self.config = c_list[0]

            # print(f"---> config synced for the local rank {rank}")
            # if world_size > 1: dist.barrier()

            print(f"{Fore.RED}---> Ready to run on local rank {rank}, {self.config.run_name}{Style.RESET_ALL}", flush=True)

            self.config.device = torch.device(f'cuda:{rank}')

        # -------------------------------------------------------
        # run the training for current rank and wandb run
        try: 
            self._train_model(rank=rank, global_rank=global_rank)

            print(f"{Fore.RED}---> Run finished on local rank {rank} <---{Style.RESET_ALL}", flush=True)

        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}Interrupted from the keyboard ...{Style.RESET_ALL}", flush=True)

            if self.config.ddp:
                torch.distributed.destroy_process_group()

            # make sure the runtime is cleaned, by brutally removing processes
            clean_after_training()

            if self.metric_manager.wandb_run is not None: 
                print(f"{Fore.YELLOW}Remove {self.metric_manager.wandb_run.name} ...{Style.RESET_ALL}", flush=True)

        # -------------------------------------------------------
        # after the run, release the process groups
        if self.config.ddp:
            if dist.is_initialized():
                print(f"---> dist.destory_process_group on local rank {rank}", flush=True)
                dist.destroy_process_group()

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

def tests():
    pass    

if __name__=="__main__":
    tests()
