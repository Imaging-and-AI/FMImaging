"""
Set up the metrics for qperf
"""
import copy
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
from colorama import Fore, Back, Style
import nibabel as nib

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from utils import *
from metrics import MetricManager, get_metric_function, AverageMeter

# -------------------------------------------------------------------------------------------------
class QPerfMetricManager(MetricManager):
    
    def __init__(self, config):
        super().__init__(config)  

        self.best_pre_model_file = None
        self.best_backbone_model_file = None
        self.best_post_model_file = None

    # ---------------------------------------------------------------------------------------
    def setup_wandb_and_metrics(self, rank):

        self.best_val_loss = np.inf
        
        device = self.config.device

        self.mse_loss_func = nn.MSELoss()
        self.l1_loss_func = nn.L1Loss()

        self.train_metrics = {'loss': AverageMeter(),
                              'mse':AverageMeter(),
                              'l1':AverageMeter(),
                              'Fp':AverageMeter(),
                              'Vp':AverageMeter(),
                              'Visf':AverageMeter(),
                              'PS':AverageMeter(),
                              'Delay':AverageMeter(),
                            }

        self.eval_metrics = {'loss': AverageMeter(),
                              'mse':AverageMeter(),
                              'l1':AverageMeter(),
                              'Fp':AverageMeter(),
                              'Vp':AverageMeter(),
                              'Visf':AverageMeter(),
                              'PS':AverageMeter(),
                              'Delay':AverageMeter(),
                            }
            
        self.train_metric_functions = {
                              'mse':self.mse_loss_func,
                              'l1':self.l1_loss_func,
                              'Fp':self.l1_loss_func,
                              'Vp':self.l1_loss_func,
                              'Visf':self.l1_loss_func,
                              'PS':self.l1_loss_func,
                              'Delay':self.l1_loss_func
                            }

        self.eval_metric_functions = copy.deepcopy(self.train_metric_functions)

        if rank<=0:
            self.wandb_run.define_metric("epoch")    
            for metric_name in self.train_metrics.keys():
                self.wandb_run.define_metric('train_'+metric_name, step_metric='epoch')
            for metric_name in self.eval_metrics.keys():
                self.wandb_run.define_metric('val_'+metric_name, step_metric='epoch')

            self.best_val = -np.inf

    def get_tra_loss(self):
        return self.train_metrics['loss'].avg, \
                self.train_metrics['mse'].avg, \
                self.train_metrics['l1'].avg, \
                self.train_metrics['Fp'].avg, \
                self.train_metrics['Vp'].avg, \
                self.train_metrics['Visf'].avg, \
                self.train_metrics['PS'].avg, \
                self.train_metrics['Delay'].avg
                
    def get_eval_loss(self):
        return self.eval_metrics['loss'].avg, \
                self.eval_metrics['mse'].avg, \
                self.eval_metrics['l1'].avg, \
                self.eval_metrics['Fp'].avg, \
                self.eval_metrics['Vp'].avg, \
                self.eval_metrics['Visf'].avg, \
                self.eval_metrics['PS'].avg, \
                self.eval_metrics['Delay'].avg

    # ---------------------------------------------------------------------------------------
    def on_train_step_end(self, loss, output, labels, rank, curr_lr, save_samples, epoch, ids):

        x, y, params = labels
        y_hat, params_est = output

        B = y.shape[0]

        y = y.to(y_hat.device)
        params = params.to(params_est.device)

        self.train_metrics['loss'].update(loss, n=B)

        for metric_name in ['mse', 'l1']:
            metric_value = self.train_metric_functions[metric_name](y_hat.flatten(), y.flatten())
            self.train_metrics[metric_name].update(metric_value.item(), n=B)

        for ii, metric_name in enumerate(['Fp', 'Vp', 'PS', 'Visf', 'Delay']):
            metric_value = self.train_metric_functions[metric_name](params[:, ii], params_est[:, ii])
            self.train_metrics[metric_name].update(metric_value.item(), n=B)

        if rank<=0: 
            self.wandb_run.log({"lr": curr_lr})
            for metric_name in self.train_metrics.keys():
                if metric_name=='loss':
                    self.wandb_run.log({"running_train_loss": loss})
                else:
                    self.wandb_run.log({f"running_train_{metric_name}": self.train_metrics[metric_name].avg})

    # ---------------------------------------------------------------------------------------
    def on_eval_step_end(self, loss, output, labels, ids, rank, save_samples, split):

        x, y, params = labels
        y_hat, params_est = output

        B = y.shape[0]

        y = y.to(y_hat.device)
        params = params.to(params_est.device)

        self.eval_metrics['loss'].update(loss, n=B)

        for metric_name in ['mse', 'l1']:
            metric_value = self.eval_metric_functions[metric_name](y_hat.flatten(), y.flatten())
            self.eval_metrics[metric_name].update(metric_value.item(), n=B)

        for ii, metric_name in enumerate(['Fp', 'Vp', 'PS', 'Visf', 'Delay']):
            metric_value = self.eval_metric_functions[metric_name](params[:, ii], params_est[:, ii])
            self.eval_metrics[metric_name].update(metric_value.item(), n=B)

    # ---------------------------------------------------------------------------------------        
    def on_eval_epoch_end(self, rank, epoch, model_manager, optim, sched, split, final_eval):
        """
        Runs at the end of the evaluation loop
        """

        # Otherwise aggregate the measurements over the steps
        average_metrics = dict()
        if self.config.ddp:
            for metric_name in self.eval_metrics.keys():
                v = torch.tensor(self.eval_metrics[metric_name].avg).to(device=self.device)
                dist.all_reduce(v, op=torch.distributed.ReduceOp.AVG)
                average_metrics[metric_name] = v
        else:
            average_metrics = {metric_name: self.eval_metrics[metric_name].avg for metric_name in self.eval_metrics.keys()}

        if rank<=0: 
            for metric_name in average_metrics.keys():
                self.wandb_run.log({f"val_{metric_name}": average_metrics[metric_name]})

        # Checkpoint best models during training
        if rank<=0: 

            if not final_eval:

                model_epoch = model_manager.module if self.config.ddp else model_manager 
                checkpoint_model = False
                if average_metrics['loss'] is not None and average_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = average_metrics['loss']
                    checkpoint_model = True

                if checkpoint_model:
                    self.best_pre_model_file, self.best_backbone_model_file, self.best_post_model_file = model_epoch.save(f"best_checkpoint_epoch_{epoch}", epoch, optim, sched)   
                    self.wandb_run.log({"epoch":epoch, "best_val_loss":self.best_val_loss})

                # Update wandb with eval metrics
                for metric_name, avg_metric_eval in average_metrics.items():
                    self.wandb_run.log({"epoch":epoch, f"{split}_{metric_name}": avg_metric_eval})

            # Save the average metrics for this epoch into self.average_eval_metrics
            self.average_eval_metrics = average_metrics

    # ---------------------------------------------------------------------------------------
    def on_training_end(self, rank, epoch, model_manager, optim, sched, ran_training):
        """
        Runs once when training finishes
        """
        if rank<=0: # main or master process
            
            if ran_training:
                # Log the best loss and metrics from the run and save final model
                self.wandb_run.summary["best_val_loss"] = self.best_val_loss
                
                model_epoch = model_manager.module if self.config.ddp else model_manager 
                model_epoch.save('final_epoch', epoch, optim, sched)

            # Finish the wandb run
            self.wandb_run.finish() 
        

# ---------------------------------------------------------------------------------------

def tests():
    print('Passed all tests')

# ---------------------------------------------------------------------------------------
if __name__=="__main__":
    tests()