import torch
from torch import nn
import os, sys

from pathlib import Path

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

from utils.status import start_timer, end_timer

class SingleTaskTrainingScheme(nn.Module):
    """
    Implements training scheme logic for simple single task training
    Training schemes manage the data sampling and loss calculation
    @args:
        - config (namespace): nested Namespace containing all args for run
        - tasks (list): list of TaskManagers 
        - loaders (list): list of dataloaders
        - model_manager (ModelManager): model manager object
    """

    def __init__(self, config, cast_type):
        
        super().__init__()

        self.config = config
        self.cast_type = cast_type
        self.device = config.device

        # Handle mix precision training
        self.scaler = torch.GradScaler('cuda', enabled=self.config.use_amp)

    def compute_total_iters(self, all_train_loaders):
        """
        Compute total number of iterations per epoch for this training scheme
        """
        total_iters = 0
        for task_name, task_loaders in all_train_loaders.items():
            total_iters += sum([len(task_loader) for task_loader in task_loaders]) 
        return total_iters        
        
    def _sampler(self, idx, epoch, train_dataloaders, model_manager):
        """
        Sample from dataloaders
        """

        if idx==0:
            self.train_loader_iters = []
            self.train_loader_iters_tasks_names = []
            for task_name, task_loaders in train_dataloaders.items():
                if self.config.ddp: [task_loader.sampler.set_epoch(epoch) for task_loader in task_loaders]
                self.train_loader_iters += [iter(task_loader) for task_loader in task_loaders]
                self.train_loader_iters_tasks_names += [task_name for _ in range(len(task_loaders))]

        tm = start_timer(enable=self.config.with_timer)

        loader_ind = idx % len(self.train_loader_iters)
        loader_outputs = next(self.train_loader_iters[loader_ind], None)
        loader_task = self.train_loader_iters_tasks_names[loader_ind]
        while loader_outputs is None:
            del self.train_loader_iters[loader_ind]
            loader_ind = idx % len(self.train_loader_iters)
            loader_outputs = next(self.train_loader_iters[loader_ind], None)
            loader_task = self.train_loader_iters_tasks_names[loader_ind]

        inputs, outputs, ids = loader_outputs
        inputs = inputs.to(self.device)
        outputs = outputs.to(self.device)

        end_timer(enable=self.config.with_timer, t=tm, msg="---> training scheme: load batch took ")        

        return inputs, outputs, ids, loader_task

    def _model_forward_pass(self, task_name, inputs, outputs, model_manager, model_module):
        """
        Forward pass through model and compute loss
        """
        tm = start_timer(enable=self.config.with_timer)

        with torch.autocast(device_type='cuda', dtype=self.cast_type, enabled=self.config.use_amp):
            model_outputs = model_manager(inputs, task_name)
            loss = model_module.tasks[task_name].loss_f(model_outputs,outputs)
            loss = loss / self.config.iters_to_accumulate

        end_timer(enable=self.config.with_timer, t=tm, msg="---> training scheme: forward pass and loss calc took ")

        return loss, model_outputs
    
    def _model_update(self, loss, idx, optim, model_manager, total_iters): 
        """
        Update model weights
        """

        tm = start_timer(enable=self.config.with_timer)  

        self.scaler.scale(loss).backward()

        end_timer(enable=self.config.with_timer, t=tm, msg="---> training scheme: backward pass took ")

        tm = start_timer(enable=self.config.with_timer)

        if (idx + 1) % self.config.iters_to_accumulate == 0 or (idx + 1 == total_iters):
            if(self.config.clip_grad_norm>0):
                self.scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model_manager.parameters(), self.config.clip_grad_norm)

            self.scaler.step(optim)
            optim.zero_grad(set_to_none=True)
            self.scaler.update()

        end_timer(enable=self.config.with_timer, t=tm, msg="---> training scheme: other steps took ")

    def forward(self, idx, total_iters, train_dataloaders, epoch, model_manager, optim):
        """
        Forward pass of the training scheme
        """
        # Extract module from ddp 
        model_module = model_manager.module if self.config.ddp else model_manager 

        # Sample from dataloaders
        inputs, outputs, ids, task_name = self._sampler(idx, epoch, train_dataloaders, model_manager)

        # Forward pass through model
        loss, model_outputs = self._model_forward_pass(task_name, inputs, outputs, model_manager, model_module)

        # Model updates 
        self._model_update(loss, idx, optim, model_manager, total_iters)

        return loss, model_outputs, inputs, outputs, ids, task_name
