"""
Defines helper functions for optimizer
"""

import torch 
import numpy as np
import torch.distributed as dist

#-------------------------------------------------------------------------------------------
def compute_total_steps(config, num_samples):
    if config.ddp: 
        num_samples /= dist.get_world_size()

    total_steps = int(np.ceil(num_samples/(config.batch_size*config.iters_to_accumulate))*config.num_epochs)
    
    return total_steps

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
def divide_optim_into_groups(optim, group, all_w_decay):

    optim_param_groups = optim.state_dict()['param_groups']
    optim_state = optim.state_dict()['state']
    
    if all_w_decay:
        pre_optim_param_groups = optim_param_groups[:2]
    else:
        pre_optim_param_groups = optim_param_groups[:1]
    num_pre_optim_params = sum([len(param_group['params']) for param_group in pre_optim_param_groups])
    pre_optim_state = {}
    for k in np.arange(0,num_pre_optim_params):
        if k in optim_state.keys(): pre_optim_state[k] = optim_state[k]
    if group=='pre': return {'state':pre_optim_state, 'param_groups':pre_optim_param_groups, 'curr_epoch':optim.curr_epoch}
    
    if all_w_decay:
        backbone_optim_param_groups = optim_param_groups[2:4]
    else:
        backbone_optim_param_groups = optim_param_groups[1:2]
    num_backbone_optim_params = sum([len(param_group['params']) for param_group in backbone_optim_param_groups])
    backbone_optim_state = {}
    for k in np.arange(num_pre_optim_params,num_pre_optim_params+num_backbone_optim_params):
        if k in optim_state.keys(): backbone_optim_state[k] = optim_state[k]
    if group=='backbone': return {'state':backbone_optim_state, 'param_groups':backbone_optim_param_groups, 'curr_epoch':optim.curr_epoch}

    if all_w_decay:
        post_optim_param_groups = optim_param_groups[4:]
    else:
        post_optim_param_groups = optim_param_groups[2:]
    num_post_optim_params = sum([len(param_group['params']) for param_group in post_optim_param_groups])
    post_optim_state = {}
    for k in np.arange(num_pre_optim_params+num_backbone_optim_params,num_pre_optim_params+num_backbone_optim_params+num_post_optim_params):
        if k in optim_state.keys(): post_optim_state[k] = optim_state[k]
    if group=='post': return {'state':post_optim_state, 'param_groups':post_optim_param_groups, 'curr_epoch':optim.curr_epoch}

    raise ValueError(f"Unknown group specified for optim split: {group}")

