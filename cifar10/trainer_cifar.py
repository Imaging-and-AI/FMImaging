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
        dist.init_process_group("nccl", rank=rank, world_size=c.world_size)
        device = rank
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
        optim = model.module.optim
        sched = model.module.sched
        stype = model.module.stype
        sampler = DistributedSampler(train_set)
        shuffle = False
    else:
        # No init required if not ddp
        device = c.device
        model = model.to(device)
        optim = model.optim
        sched = model.sched
        stype = model.stype
        sampler = None
        shuffle = True

    train_loader = DataLoader(dataset=train_set, batch_size=c.batch_size, shuffle=shuffle, sampler=sampler,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0)

    if rank<=0: # main or master process
        setup_logger(config)
        logging.info(f"Configuration for this run:\n{config}")

        trainable_params, total_params = get_number_of_params(model)
        logging.info(f"Trainable parameters: {trainable_params:,}, Total parameters: {total_params:,}")

        wandb.init(project=c.project, entity=c.wandb_entity, config=c,
                    name=c.run_name, notes=c.run_notes)
        wandb.watch(model)
        wandb.log({"trainable_params":trainable_params,
                    "total_params": total_params})

        # save best model to be saved at the end
        best_val_loss = numpy.inf
        best_val_acc = 0
        best_model_wts = copy.deepcopy(model.module.state_dict() if c.ddp else model.state_dict())

    # general cross entropy loss
    loss_f = nn.CrossEntropyLoss()
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for epoch in range(c.num_epochs):
        if rank<=0: logging.info(f"{'-'*20}Epoch:{epoch}/{c.num_epochs}{'-'*20}")

        model.train()
        if c.ddp: train_loader.sampler.set_epoch(epoch)
        train_loss.reset()

        train_loader_iter = iter(train_loader)
        total_iters = len(train_loader) if not c.debug else 10
        with tqdm(total=total_iters, disable=rank>0) as pbar:

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

                total=inputs.shape[0]
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == labels).sum().item()
                train_acc.update(correct/total, n=total)

                train_loss.update(loss.item(), n=total)
                if rank<=0: wandb.log({"running_train_loss": loss.item()})

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, {loss.item():.4f}, lr {curr_lr:.8f}")

            pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, {train_loss.avg:.4f}, {train_acc.avg:.4f}, lr {curr_lr:.8f}")

        if rank<=0: # main or master process
            # run eval, save and log in this process
            model_e = model.module if c.ddp else model
            val_loss_avg, val_acc = eval_val(model_e, c, val_set, epoch, device)
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_wts = copy.deepcopy(model_e.state_dict())
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # silently log to only the file as well
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, {train_loss.avg:.4f}, lr {curr_lr:.8f}, val, {val_loss_avg:.4f}, {val_acc:.4f}")

            # save the model weights every save_cycle
            if epoch % c.save_cycle == 0:
                model_e.save(epoch)

            wandb.log({"epoch": epoch,
                        "train_loss_avg": train_loss.avg,
                        "train_acc": train_acc.avg})

            if stype == "ReduceLROnPlateau":
                sched.step(val_loss_avg)
            else: # stype == "StepLR"
                sched.step()

            if c.ddp:
                new_lr_0 = torch.zeros(1).to(rank)
                new_lr_0[0] = optim.param_groups[0]["lr"]
                dist.broadcast(new_lr_0, src=0)

                if c.no_w_decay:
                    new_lr_1 = torch.zeros(1).to(rank)
                    new_lr_1[0] = optim.param_groups[1]["lr"]
                    dist.broadcast(new_lr_1, src=0)
        else: # child processes
            new_lr_0 = torch.zeros(1).to(rank)
            dist.broadcast(new_lr_0, src=0)
            optim.param_groups[0]["lr"] = new_lr_0.item()

            if c.no_w_decay:
                new_lr_1 = torch.zeros(1).to(rank)
                dist.broadcast(new_lr_1, src=0)
                optim.param_groups[1]["lr"] = new_lr_1.item()

    if rank<=0: # main or master process
        # test and save model
        wandb.log({"best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc})

        model = model.module if c.ddp else model
        model.save(epoch) # save the final weights
        # test last model
        eval_test(model, config, test_set=None, device=device, id="last")
        # test best model
        model.load_state_dict(best_model_wts)
        eval_test(model, config, test_set=None, device=device, id="best")
        # save both models
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
        - val_acc_avg (float): the average val loss
    """
    c = config # shortening due to numerous uses

    val_loader = DataLoader(dataset=val_set, batch_size=c.batch_size, shuffle=False, sampler=None,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0)

    loss_f = nn.CrossEntropyLoss()
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    model.eval()
    model.to(device)

    val_loader_iter = iter(val_loader)
    total_iters = len(val_loader) if not c.debug else 10
    with tqdm(total=total_iters) as pbar:

        for idx in range(total_iters):

            inputs, labels = next(val_loader_iter)

            inputs = inputs.to(device)
            labels = labels.to(device)
            total = labels.size(0)

            output = model(inputs)
            loss = loss_f(output, labels)
            val_loss.update(loss.item(), n=total)

            _, predicted = torch.max(output.data, 1)
            correct = (predicted == labels).sum().item()
            val_acc.update(correct/total, n=total)

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, val, {inputs.shape}, {loss.item():.4f}, {correct/total:.4f}")

        pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, val, {inputs.shape}, {val_loss.avg:.4f}, {val_acc.avg:.4f}")
    wandb.log({f"val_loss_avg":val_loss.avg,
                f"val_acc":val_acc.avg})

    return val_loss.avg, val_acc.avg
