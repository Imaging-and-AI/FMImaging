"""
Trainer for cifar 10.
Provides the mian function to call for training:
    - trainer
"""
import copy
import wandb
import numpy
import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from eval_cifar import eval_test
from utils.save_model import save_final_model

# -------------------------------------------------------------------------------------------------
# trainer

def trainer(rank, model, config, train_set, val_set):
    """
    The trainer cycle. Allows training on cpu/single gpu/multiple gpu(ddp)
    @args:
        - rank (int): for distributed data parallel (ddp)
            -1 if running on cpu or only one gpu
        - model (torch model): model to be trained
        - config (Namespace): config of the run
        - train_set (torch Dataset): the data to train on
        - val_set (torch Dataset): the data to validate each epoch
    """
    c = config # shortening due to numerous uses

    if c.ddp:
        dist.init_process_group("gloo", rank=rank, world_size=c.world_size)
        device = rank
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
        optim = model.module.optim
        sched = model.module.sched
        stype = model.module.stype
        sampler = DistributedSampler(train_set)
    else:
        # No init required if not ddp
        device = c.device
        model = model.to(device)
        optim = model.optim
        sched = model.sched
        stype = model.stype
        sampler = None

    train_loader = DataLoader(dataset=train_set, batch_size=c.batch_size, shuffle=True, sampler=sampler,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0)

    wandb.watch(model)

    # save best model to be saved at the end
    best_val_loss = numpy.inf
    best_model_wts = copy.deepcopy(model.state_dict())

    # general cross entropy loss
    loss_f = nn.CrossEntropyLoss()
    train_loss = AverageMeter()

    for epoch in range(c.num_epochs):
        logging.info(f"{'-'*30}Epoch:{epoch}/{c.num_epochs}{'-'*30}")

        model.train()
        if c.ddp: train_loader.sampler.set_epoch(epoch)
        train_loss.reset()

        train_loader_iter = iter(train_loader)
        total_iters = len(train_loader) if not c.debug else 10
        with tqdm(total=total_iters) as pbar:

            for idx in range(total_iters):

                optim.zero_grad()

                inputs, labels = next(train_loader_iter)
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                loss = loss_f(output, labels)
                loss.backward()

                if(c.clip_grad_norm>0):
                    nn.utils.clip_grad_norm_(model.parameters(), c.clip_grad_norm)
                optim.step()

                if stype == "OneCycleLR": sched.step()
                curr_lr = optim.param_groups[0]['lr']

                train_loss.update(loss.item(), n=c.batch_size)
                wandb.log({"running_train_loss": loss.item()})

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, {train_loss.avg:.4f}, lr {curr_lr:.8f}")

        pbar.set_postfix_str(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, {train_loss.avg:.4f}, lr {curr_lr:.8f}")
        # silently log to only the file as well
        logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, {train_loss.avg:.4f}, lr {curr_lr:.8f}")

        if rank<=0: # main or master process
            # run eval, save and log in this process
            val_loss_avg = eval_val(model, c, val_set, epoch, device)
            if(val_loss_avg<best_val_loss):
                best_val_loss = val_loss_avg
                best_model_wts = copy.deepcopy(model.state_dict())

            # save the model weights every save_cycle
            if epoch % c.save_cycle == 0:
                model.module.save(epoch) if c.ddp else model.save(epoch)

            wandb.log({"epoch": epoch,
                        "train_loss_avg": train_loss.avg,
                        "val_loss_avg": val_loss_avg})

            if stype == "ReduceLROnPlateau":
                sched.step(val_loss_avg)
            else: # stype == "StepLR"
                sched.step()

            if c.ddp:
                # share new lr across devices
                new_lr_0 = optim.param_groups[0]['lr']
                dist.broadcast(torch.tensor(new_lr_0), src=0)
                if c.no_w_decay:
                    new_lr_1 = optim.param_groups[1]['lr']
                    dist.broadcast(torch.tensor(new_lr_1), src=0)
        else: # child processes
            # update the lr from master process
            new_lr_0 = torch.zeros(1)
            dist.broadcast(torch.tensor(new_lr_0), src=0)
            optim.param_groups[0]['lr'] = new_lr_0

            if c.no_w_decay:
                new_lr_1 = torch.zeros(1)
                dist.broadcast(torch.tensor(new_lr_1), src=0)
                optim.param_groups[1]['lr'] = new_lr_1

    if rank<=0: # main or master process
        # test and save model
        wandb.log({"best_val_loss": best_val_loss})

        model = model.module if c.ddp else model
        model.save(epoch) # save the final weights
        eval_test(model, config, test_set=None, device=device)
        save_final_model(model, config, best_model_wts)

    if c.ddp: # cleanup
        dist.destroy_process_group()

# -------------------------------------------------------------------------------------------------
# evaluate the val set

def eval_val(model, config, val_set, epoch, device):
    """
    The validation evaluation.
    @args:
        - model (torch model): model to be validated
        - config (Namespace): config of the run
        - val_set (torch Dataset): the data to validate on
        - epoch (int): the current epoch
        - device (torch.device): the device to run eval on
    @rets:
        - val_loss_avg (float): the average val loss
    """
    c = config # shortening due to numerous uses

    val_loader = DataLoader(dataset=val_set, batch_size=c.batch_size, shuffle=True, sampler=None,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0)

    loss_f = nn.CrossEntropyLoss()
    val_loss = AverageMeter()

    model.eval()
    model.to(device)

    val_loader_iter = iter(val_loader)
    total_iters = len(val_loader) if not c.debug else 10
    with tqdm(total=total_iters) as pbar:

        for idx in range(total_iters):

            inputs, labels = next(val_loader_iter)
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = loss_f(output, labels)

            val_loss.update(loss.item(), n=c.batch_size)
            wandb.log({"running_val_loss": loss.item()})

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, val, {inputs.shape}, {val_loss.avg:.4f}")

    pbar.set_postfix_str(f"Epoch {epoch}/{c.num_epochs}, val, {inputs.shape}, {val_loss.avg:.4f}")
    logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, val, {inputs.shape}, {val_loss.avg:.4f}")

    return val_loss.avg
