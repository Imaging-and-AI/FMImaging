"""
Helper functions for train manager
"""

import os
import torch
import torch.utils.data

# -------------------------------------------------------------------------------------------------         
class DistributedSamplerNoDuplicate(torch.utils.data.DistributedSampler):
    """ A distributed sampler that doesn't add duplicates. Arguments are the same as DistributedSampler """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # some ranks may have fewer samples, that's fine
            if self.rank >= len(self.dataset) % self.num_replicas:
                self.num_samples -= 1
            self.total_size = len(self.dataset)

# -------------------------------------------------------------------------------------------------         
def clean_after_training():
    """Clean after the training
    """
    #os.system("kill -9 $(ps aux | grep torchrun | grep -v grep | awk '{print $2}') ")
    #os.system("kill -9 $(ps aux | grep wandb | grep -v grep | awk '{print $2}') ")
    #os.system("kill -9 $(ps aux | grep mri | grep -v grep | awk '{print $2}') ")
    pass

# -------------------------------------------------------------------------------------------------
def get_bar_format():
    """Get the default bar format
    """
    return '{desc}{percentage:3.0f}%|{bar:10}{r_bar}'

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