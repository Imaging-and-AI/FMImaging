"""
Base infrastructure for training, supporting multi-node and multi-gpu training
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
from training_schemes import *
from utils.status import model_info, start_timer, end_timer, support_bfloat16
from setup.setup_utils import setup_logger

class TrainManager(object):
    """
    Base Runtime model for training. This class supports:
        - single node, single process, single gpu training
        - single node, multiple process, multiple gpu training
        - multiple nodes, multiple processes, multiple gpu training
    """
    def __init__(self, config, model_manager, optim_manager, metric_manager):
    
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - model_manager (ModelManager): ModelManager object that contains pre/backbone/post model and forward function
            - optim_manager (OptimManager): OptimManager object that contains optimizer and scheduler
            - metric_manager (MetricManager): MetricManager object that tracks metrics and checkpoints models during training
        """
        super().__init__()
        self.config = config
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

    def _train_and_eval_model(self, rank, global_rank):
        """
        The training loop. Allows training on cpu/single gpu/multiple gpu (ddp)
        @args:
            - rank (int): for distributed data parallel (ddp) -1 if running on cpu or only one gpu
            - global_rank (int): for distributed data parallel (ddp)
            
        """
        c = self.config # shortening due to numerous uses     

        # All metrics are handled by the metric manager
        self.metric_manager.setup_wandb_and_metrics(rank)

        # Freeze portions of the network, if desired
        for task_ind, task_name in enumerate(self.config.tasks):
            if self.config.freeze_pre[task_ind]: 
                self.model_manager.tasks[task_name].pre_component.freeze()
            else:
                self.model_manager.tasks[task_name].pre_component.unfreeze()
            if self.config.freeze_post[task_ind]: 
                self.model_manager.tasks[task_name].post_component.freeze()
            else:
                self.model_manager.tasks[task_name].post_component.unfreeze()
        if self.config.freeze_backbone: 
            self.model_manager.backbone_component.freeze()
        else:
            self.model_manager.backbone_component.unfreeze()

        # Send models to device
        if c.ddp:
            dist.barrier()
            device = torch.device(f"cuda:{rank}")
            self.model_manager = self.model_manager.to(device)
            for task in self.model_manager.tasks.values(): task.to(device)
            self.model_manager = DDP(self.model_manager, device_ids=[rank], find_unused_parameters=False)
        else:
            device = c.device
            self.model_manager = self.model_manager.to(device)
            for task in self.model_manager.tasks.values(): task.to(device)

        # Print out model summary
        if rank<=0:
            logging.info(f"Configuration for this run:\n{c}")
            model_info(self.model_manager, c)
            logging.info(f"Wandb name:\n{self.metric_manager.wandb_run.name}")
            self.metric_manager.wandb_run.watch(self.model_manager)
            if c.ddp: 
                setup_logger(self.config) # setup master process logging; I don't know if this needs to be here, it is also in setup.py
        
        # Extracting optim and sched for convenience
        optim = self.optim_manager.optim
        sched = self.optim_manager.sched
        curr_epoch = self.optim_manager.curr_epoch
        logging.info(f"{Fore.RED}{'-'*20}Local Rank:{rank}, global rank {global_rank}{'-'*20}{Style.RESET_ALL}")

        # Zero gradient before training
        optim.zero_grad(set_to_none=True)

        # Training loop
        if self.config.train_model:

            # Create train dataloaders
            train_dataloaders = {}
            model_module = self.model_manager.module if self.config.ddp else self.model_manager 
            for task_ind, task in enumerate(model_module.tasks.values()):
                task_train_sets = task.train_set
                if c.ddp:
                    shuffle = False
                    if isinstance(task_train_sets,list): samplers = [DistributedSampler(train_set) for train_set in task_train_sets]
                    else: 
                        samplers = [DistributedSampler(task_train_sets)]
                        task_train_sets = [task_train_sets]
                else:
                    shuffle = True
                    if isinstance(task_train_sets,list): samplers = [None] * len(task_train_sets)
                    else: 
                        samplers = [None]
                        task_train_sets = [task_train_sets]
                train_dataloaders[task.task_name] = [DataLoader(dataset=train_set, batch_size=c.batch_size[task_ind], shuffle=shuffle, 
                                                                sampler=samplers[ind], num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, 
                                                                drop_last=True, persistent_workers=self.config.num_workers>0) for ind, train_set in enumerate(task_train_sets)]
                
            # Set up training scheme
            if self.config.training_scheme == "single_task":
                training_scheme = SingleTaskTrainingScheme(self.config, self.cast_type)
            else:
                raise ValueError(f"Unknown training scheme {self.config.training_scheme} specified.")
                
            # Compute total iters
            total_iters = training_scheme.compute_total_iters(train_dataloaders) if not c.debug else 3

            logging.info(f"{Fore.CYAN}OPTIMIZER PARAMETERS: {optim} {Style.RESET_ALL}")

            for epoch in range(curr_epoch, c.num_epochs):
                logging.info(f"{Fore.GREEN}{'-'*20}Epoch:{epoch}/{c.num_epochs}, rank {rank} {'-'*20}{Style.RESET_ALL}")

                self.model_manager.train()

                self.metric_manager.on_train_epoch_start()

                with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:
                    for idx in range(total_iters):

                        tm = start_timer(enable=c.with_timer)
                        loss, model_outputs, inputs, gt_outputs, ids, task_name = training_scheme(idx, total_iters, train_dataloaders, epoch, self.model_manager, optim)
                        end_timer(enable=c.with_timer, t=tm, msg="---> full training scheme took ")

                        if (idx + 1) % c.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                            if c.scheduler_type == "OneCycleLR": 
                                sched.step()
                        
                        tm = start_timer(enable=c.with_timer)
                        curr_lr = optim.param_groups[0]['lr']

                        self.metric_manager.on_train_step_end(task_name, loss.item(), model_outputs, gt_outputs, rank, curr_lr)

                        pbar.update(1)
                        pbar.set_description(f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} tra, rank {rank}, {inputs.shape}, lr {curr_lr:.8f}, task {task_name}, loss {loss.item():.4f}{Style.RESET_ALL}")

                        end_timer(enable=c.with_timer, t=tm, msg="---> epoch step logging and measuring took ")
                        
                    # Run metric logging for each epoch 
                    tm = start_timer(enable=c.with_timer) 

                    self.metric_manager.on_train_epoch_end(self.model_manager, optim, sched, epoch, rank)

                    # Print out metrics from this epoch
                    pbar_str = f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} tra, rank {rank},  {inputs.shape}, lr {curr_lr:.8f}"
                    if hasattr(self.metric_manager, 'average_train_metrics'):
                        if isinstance(self.metric_manager.average_train_metrics, dict):
                            for task_name in self.metric_manager.train_metrics.keys():
                                for metric_name, metric_value in self.metric_manager.average_train_metrics[task_name].items():
                                    try: pbar_str += f", {Fore.CYAN} {task_name}_{metric_name} {metric_value:.4f}"
                                    except: pass
                    pbar_str += f"{Style.RESET_ALL}"
                    pbar.set_description(pbar_str)

                    # Write training status to log file
                    if rank<=0: 
                        logging.getLogger("file_only").info(pbar_str)

                    end_timer(enable=c.with_timer, t=tm, msg="---> epoch end logging and measuring took ")

                if epoch % c.eval_frequency==0 or epoch==c.num_epochs:
                    self._eval_model(rank=rank, model_manager=self.model_manager, epoch=epoch, device=device, optim=optim, sched=sched, id="", split="val", final_eval=False)

                if c.scheduler_type != "OneCycleLR":
                    if c.scheduler_type == "ReduceLROnPlateau":
                        try: 
                            sched.step(self.metric_manager.average_eval_metrics['total_loss'])
                        except:
                            warnings.warn("Average loss not available, using step loss to step scheduler.")
                            sched.step(loss.item())
                    elif c.scheduler_type == "StepLR":
                        sched.step()

                    if c.ddp:
                        self.distribute_learning_rates(rank, optim, src=0)

            # Load the best model from training
            dist.barrier()
            if self.config.eval_train_set or self.config.eval_val_set or self.config.eval_test_set:
                logging.info(f"{Fore.CYAN}Loading the best models from training for final evaluation...{Style.RESET_ALL}")
                if self.config.ddp: 
                    self.model_manager.module.load_entire_model(os.path.join(self.config.log_dir,self.config.run_name,'entire_models','entire_model_best_checkpoint.pth'), device=device)
                else: 
                    self.model_manager.load_entire_model(os.path.join(self.config.log_dir,self.config.run_name,'entire_models','entire_model_best_checkpoint.pth'))
       
        else: # Not training
            epoch = 0

        # Evaluate models of each split
        if self.config.eval_train_set: 
            logging.info(f"{Fore.CYAN}Evaluating train set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=self.model_manager, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="train", final_eval=True)
        if self.config.eval_val_set: 
            logging.info(f"{Fore.CYAN}Evaluating val set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=self.model_manager, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="val", final_eval=True)
        if self.config.eval_test_set: 
            logging.info(f"{Fore.CYAN}Evaluating test set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model_manager=self.model_manager, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="test", final_eval=True)

        # Finish up training
        self.metric_manager.on_training_end(rank, epoch, self.model_manager, optim, sched, self.config.train_model)
        
    def _eval_model(self, rank, model_manager, epoch, device, optim, sched, id, split, final_eval):
        """
        Model evaluation.
        @args:
            - rank (int): used for ddp
            - model_manager (ModelManager): model to be validated
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

        # Set up eval data loaders
        eval_dataloaders = {}
        data_loader_iters = []
        data_loader_iters_tasks_names = []
        model_module = self.model_manager.module if self.config.ddp else self.model_manager 
        for task_ind, task in enumerate(model_module.tasks.values()):
            if split=='train': task_datasets = task.train_set
            elif split=='val': task_datasets = task.val_set
            elif split=='test': task_datasets = task.test_set
            if c.ddp:
                if isinstance(task_datasets, list): 
                    samplers = [DistributedSamplerNoDuplicate(task_dataset,rank=rank) for task_dataset in task_datasets]
                else: 
                    samplers = [DistributedSamplerNoDuplicate(task_datasets,rank=rank)]
                    task_datasets = [task_datasets]
            else:
                if isinstance(task_datasets,list): 
                    samplers = [None] * len(task_datasets)
                else: 
                    samplers = [None]
                    task_datasets = [task_datasets]
            
            eval_dataloaders[task.task_name] = [DataLoader(dataset=data_set, batch_size=c.batch_size[task_ind], shuffle=False, 
                                                           sampler=samplers[ind],num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, 
                                                           drop_last=False, persistent_workers=c.num_workers>0) for ind, data_set in enumerate(task_datasets)]

            data_loader_iters += [iter(data_loader) for data_loader in eval_dataloaders[task.task_name]]
            data_loader_iters_tasks_names += [task.task_name for _ in range(len(eval_dataloaders[task.task_name]))]

        # Set up a few things before starting eval loop
        self.metric_manager.on_eval_epoch_start()

        model_manager.eval()

        total_iters = 3
        if not c.debug:
            total_iters = 0
            for task in model_module.tasks.values():
                total_iters += sum([len(data_loader) for data_loader in eval_dataloaders[task.task_name]])

        # Evaluation loop
        with torch.inference_mode():
            with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

                for idx in range(total_iters):

                    # Sample data from the dataloaders
                    loader_ind = idx % len(data_loader_iters)
                    loader_outputs = next(data_loader_iters[loader_ind], None)
                    loader_task = data_loader_iters_tasks_names[loader_ind]
                    while loader_outputs is None:
                        del data_loader_iters[loader_ind]
                        loader_ind = idx % len(data_loader_iters)
                        loader_outputs = next(data_loader_iters[loader_ind], None)
                        loader_task = data_loader_iters_tasks_names[loader_ind]
                    inputs, labels, ids = loader_outputs

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Run inference
                    with torch.autocast(device_type='cuda', dtype=self.cast_type, enabled=c.use_amp):
                        adjusted_batch = False
                        if inputs.shape[0] == 1: 
                            inputs = inputs.repeat(2,1,1,1,1) # Take care of batch size = 1 case so batch norm doesn't throw an error
                            adjusted_batch = True
                        output = model_manager(inputs, loader_task)
                        if adjusted_batch:
                            output = output[0:1]
                            inputs = inputs[0:1]
                        loss = model_module.tasks[loader_task].loss_f(output, labels)

                    # Update evaluation metrics
                    self.metric_manager.on_eval_step_end(loader_task, loss.item(), output, labels, ids, rank, save_samples, split)

                    # Print evaluation metrics to terminal
                    pbar.update(1)
                    pbar.set_description(f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} {split}, rank {rank}, {id} {inputs.shape}, lr {curr_lr:.8f}, loss {loss.item():.4f}{Style.RESET_ALL}")

                # Update evaluation metrics 
                self.metric_manager.on_eval_epoch_end(rank, epoch, model_manager, optim, sched, split, final_eval)

                # Print evaluation metrics to terminal
                pbar_str = f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} {split}, rank {rank}, {id} {inputs.shape}, lr {curr_lr:.8f}"
                if hasattr(self.metric_manager, 'average_eval_metrics'):
                    if isinstance(self.metric_manager.average_eval_metrics, dict):
                        for task_name in self.metric_manager.eval_metrics.keys():
                            for metric_name, metric_value in self.metric_manager.average_eval_metrics[task_name].items():
                                try: pbar_str += f", {Fore.MAGENTA} {task_name}_{metric_name} {metric_value:.4f}"
                                except: pass

                        # Save final evaluation metrics to a text file
                        if final_eval and rank<=0:
                            metric_file = os.path.join(self.config.log_dir,self.config.run_name,f'{split}_metrics.txt')
                            with open(metric_file, 'w') as f:
                                for task_name in self.metric_manager.eval_metrics.keys():
                                    for metric_name, metric_value in self.metric_manager.average_eval_metrics[task_name].items():
                                        try: f.write(f"{split}_{task_name}_{metric_name}: {metric_value:.4f}, ")
                                        except: pass

                pbar_str += f"{Style.RESET_ALL}"
                pbar.set_description(pbar_str)

                if rank<=0: 
                    logging.getLogger("file_only").info(pbar_str)
                        
        return 
       
    def run(self):

        # -------------------------------------------------------
        # Get the rank and runtime info
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
        # Initialize wandb

        if global_rank<=0:
            self.metric_manager.init_wandb()
            
        # -------------------------------------------------------
        # If ddp is used, broadcast the parameters from rank0 to all other ranks (originally used for sweep, commented out for now)

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

            # self.config.device = torch.device(f'cuda:{rank}')

        # -------------------------------------------------------
        # Run the training and evaluation loops for each rank
        try: 
            self._train_and_eval_model(rank=rank, global_rank=global_rank)
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
        # After the run, release the process groups
        if self.config.ddp:
            if dist.is_initialized():
                print(f"---> dist.destory_process_group on local rank {rank}", flush=True)
                dist.destroy_process_group()


# -------------------------------------------------------------------------------------------------

def tests():
    pass    

if __name__=="__main__":
    tests()
