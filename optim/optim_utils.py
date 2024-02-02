"""
Defines helper functions for optimizer
"""

import torch 
import numpy as np
import torch.distributed as dist

#-------------------------------------------------------------------------------------------
def compute_total_updates(config, tasks):

    total_num_samples = 0
    total_num_updates = 0

    for task_ind, task in enumerate(tasks.values()):
        if isinstance(task.train_set, list):
            num_samples = sum([len(a_train_set) for a_train_set in task.train_set])
        else:
            num_samples = len(task.train_set)

        total_num_samples += num_samples

        total_num_updates += int(np.ceil(num_samples/(config.batch_size[task_ind]*config.iters_to_accumulate))*config.num_epochs)
    
    return total_num_samples, total_num_updates

# -------------------------------------------------------------------------------------------------
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

# -------------------------------------------------------------------------------------------------    
