"""
Trainer for cifar 10.
Provides the mian function to call for training:
    - trainer
"""
import copy
import wandb
import numpy
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchmetrics

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from eval_cifar import create_base_test_set, save_results
from utils.save_model import save_final_model

from colorama import Fore, Style

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
        dist.init_process_group("nccl", rank=rank, world_size=c.world_size, timeout=timedelta(seconds=1800))
        device = rank
        model = model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optim = model.module.optim
        sched = model.module.sched
        stype = model.module.stype
        loss_f = model.module.loss_f
        sampler = DistributedSampler(train_set)
        shuffle = False
    else:
        # No init required if not ddp
        device = c.device
        model = model.to(device)
        optim = model.optim
        sched = model.sched
        stype = model.stype
        loss_f = model.loss_f
        sampler = None
        shuffle = True

    train_loader = DataLoader(dataset=train_set, batch_size=c.batch_size, shuffle=shuffle, sampler=sampler,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=True,
                                persistent_workers=c.num_workers>0)

    if rank<=0: # main or master process
        if c.ddp: setup_logger(config) # setup master process logging

        wandb.init(project=c.project, entity=c.wandb_entity, config=c, name=c.run_name, notes=c.run_notes)
        wandb.watch(model)
        wandb.run.summary["trainable_params"] = c.trainable_params
        wandb.run.summary["total_params"] = c.total_params

        # save best model to be saved at the end
        best_val_loss = numpy.inf
        best_val_acc = 0
        best_model_wts = copy.deepcopy(model.module.state_dict() if c.ddp else model.state_dict())

        wandb.define_metric("epoch")    
        wandb.define_metric("train_loss", step_metric='epoch')
        wandb.define_metric("train_acc", step_metric='epoch')
        wandb.define_metric("val_loss", step_metric='epoch')
        wandb.define_metric("val_acc", step_metric='epoch')
            
    # general cross entropy loss    
    train_loss = AverageMeter()
    train_acc_1 = AverageMeter()
    train_acc_5 = AverageMeter()

    Accuracy_1 = torchmetrics.Accuracy(task="multiclass", num_classes=config.num_classes, top_k=1).to(device=device)
    Accuracy_5 = torchmetrics.Accuracy(task="multiclass", num_classes=config.num_classes, top_k=5).to(device=device)

    # mix precision training
    scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp)

    optim.zero_grad(set_to_none=True)

    for epoch in range(c.num_epochs):
        if rank<=0: logging.info(f"{Fore.GREEN}{'-'*20}Epoch:{epoch}/{c.num_epochs}{'-'*20}{Style.RESET_ALL}")

        model.train()
        if c.ddp: train_loader.sampler.set_epoch(epoch)
        train_loss.reset()

        train_loader_iter = iter(train_loader)
        total_iters = len(train_loader) if not c.debug else 10
        with tqdm(total=total_iters, disable=rank>0) as pbar:

            for idx in range(total_iters):
                
                inputs, labels = next(train_loader_iter)
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=c.use_amp):
                    output = model(inputs)
                    loss = loss_f(output, labels)
                    loss = loss / c.iters_to_accumulate
                    
                scaler.scale(loss).backward()

                if (idx + 1) % c.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                    if(c.clip_grad_norm>0):
                        scaler.unscale_(optim)
                        nn.utils.clip_grad_norm_(model.parameters(), c.clip_grad_norm)
                    
                    scaler.step(optim)                    
                    optim.zero_grad(set_to_none=True)
                    scaler.update()
                
                    if stype == "OneCycleLR": sched.step()
                    
                curr_lr = optim.param_groups[0]['lr']

                acc_1 = Accuracy_1(output, labels).item()
                acc_5 = Accuracy_5(output, labels).item()
                
                train_acc_1.update(acc_1, n=output.shape[0])
                train_acc_5.update(acc_5, n=output.shape[0])
                train_loss.update(loss.item(), n=output.shape[0])
                
                if rank<=0: 
                    wandb.log({"running_train_loss": loss.item()})
                    wandb.log({"lr": curr_lr})

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, loss {train_loss.avg:.4f}, lr {curr_lr:.8f}")

            pbar.set_description(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, loss {train_loss.avg:.4f}, {Fore.YELLOW}acc-1 {train_acc_1.avg:.4f}{Style.RESET_ALL}, {Fore.RED}acc-5 {train_acc_5.avg:.4f}{Style.RESET_ALL}, lr {curr_lr:.8f}")

        if rank<=0: # main or master process
            # run eval, save and log in this process
            model_e = model.module if c.ddp else model
            val_loss_avg, val_acc_1, val_acc_5 = eval_val(model_e, c, val_set, epoch, device)
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_wts = copy.deepcopy(model_e.state_dict())
            if val_acc_1 > best_val_acc:
                best_val_acc = val_acc_1
                model_e.save(epoch)

            # silently log to only the file as well
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, tra, {inputs.shape}, loss {train_loss.avg:.4f}, acc 1 {train_acc_1.avg:.4f}, acc 5 {train_acc_5.avg:.4f}, lr {curr_lr:.8f}")
            logging.getLogger("file_only").info(f"Epoch {epoch}/{c.num_epochs}, val, loss {val_loss_avg:.4f}, acc 1 {val_acc_1:.4f}, acc 5 {val_acc_5:.4f}")

            # save the model weights every save_cycle
            # if epoch % c.save_cycle == 0:
            #     model_e.save(epoch)

            wandb.log({"epoch": epoch,
                        "train_loss": train_loss.avg,
                        "train_acc_1": train_acc_1.avg,
                        "train_acc_5": train_acc_5.avg,
                        "val_loss":val_loss_avg,
                        "val_acc_1":val_acc_1,
                        "val_acc_5":val_acc_5})

            if stype == "ReduceLROnPlateau":
                sched.step(1.0 - val_acc_1)
            else: # stype == "StepLR"
                sched.step()

            if c.ddp:
                new_lr_0 = torch.zeros(1).to(rank)
                new_lr_0[0] = optim.param_groups[0]["lr"]
                dist.broadcast(new_lr_0, src=0)

                if not c.all_w_decay:
                    new_lr_1 = torch.zeros(1).to(rank)
                    new_lr_1[0] = optim.param_groups[1]["lr"]
                    dist.broadcast(new_lr_1, src=0)
        else: # child processes
            new_lr_0 = torch.zeros(1).to(rank)
            dist.broadcast(new_lr_0, src=0)
            optim.param_groups[0]["lr"] = new_lr_0.item()

            if not c.all_w_decay:
                new_lr_1 = torch.zeros(1).to(rank)
                dist.broadcast(new_lr_1, src=0)
                optim.param_groups[1]["lr"] = new_lr_1.item()

    if rank<=0: # main or master process
        # test and save model
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_val_acc"] = best_val_acc

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

def eval(model, config, data_set, epoch, device, id="", run_mode="val"):
    """
    The validation evaluation.
    @args:
        - model (torch model): model to be validated
        - config (Namespace): config of the run
        - data_set (torch Dataset): the data to validate on
        - epoch (int): the current epoch
        - device (torch.device): the device to run eval on
    @rets:
        - data_loss_avg (float): the average val loss
        - data_acc_avg (float): the average val loss
    """
    c = config # shortening due to numerous uses

    data_loader = DataLoader(dataset=data_set, batch_size=c.batch_size, shuffle=False, sampler=None,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0)

    loss_f = model.loss_f
    data_loss = AverageMeter()
    data_acc_1 = AverageMeter()
    data_acc_5 = AverageMeter()

    Accuracy_1 = torchmetrics.Accuracy(task="multiclass", num_classes=config.num_classes, top_k=1).to(device=device)
    Accuracy_5 = torchmetrics.Accuracy(task="multiclass", num_classes=config.num_classes, top_k=5).to(device=device)
    
    model.eval()
    model.to(device)

    data_loader_iter = iter(data_loader)
    total_iters = len(data_loader) if not c.debug else 10
    
    with torch.no_grad():
        with tqdm(total=total_iters) as pbar:

            for idx in range(total_iters):

                inputs, labels = next(data_loader_iter)

                inputs = inputs.to(device)
                labels = labels.to(device)
                total = labels.size(0)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=c.use_amp):
                    output = model(inputs)
                    loss = loss_f(output, labels)
                    
                data_loss.update(loss.item(), n=total)

                acc_1 = Accuracy_1(output, labels).item()
                acc_5 = Accuracy_5(output, labels).item()
                
                data_acc_1.update(acc_1, n=output.shape[0])
                data_acc_5.update(acc_5, n=output.shape[0])

                pbar.update(1)
                pbar.set_description(f"{run_mode}, epoch {epoch}/{c.num_epochs}, {inputs.shape}, loss {loss.item():.4f}, acc {data_acc_1.avg:.4f}")

                if run_mode == "test":
                    wandb.log({f"running_test_loss_{id}": loss.item(), f"running_test_acc_1_{id}": data_acc_1.avg})
                
            pbar.set_description(f"{run_mode} {id}, epoch {epoch}/{c.num_epochs}, {inputs.shape}, loss {data_loss.avg:.4f}, acc-1 {data_acc_1.avg:.4f}, acc-5 {data_acc_5.avg:.4f}")

    return data_loss.avg, data_acc_1.avg, data_acc_5.avg

# -------------------------------------------------------------------------------------------------
def eval_val(model, config, val_set, epoch, device):
    loss, acc_1, acc_5 = eval(model=model, config=config, data_set=val_set, epoch=epoch, device=device, id="", run_mode="val")
    return loss, acc_1, acc_5
    
def eval_test(model, config, test_set=None, device="cpu", id=""):
    # if no test_set given then load the base set
    if test_set is None: test_set = create_base_test_set(config)

    loss, acc_1, acc_5 = eval(model=model, config=config, data_set=test_set, epoch=config.num_epochs, device=device, id=id, run_mode="test")
    
    wandb.run.summary[f"test_loss_avg_{id}"] = loss
    wandb.run.summary[f"test_acc_1_{id}"] = acc_1
    wandb.run.summary[f"test_acc_5_{id}"] = acc_5
    
    save_results(config, loss, acc_1, id)

    return loss, acc_1, acc_5