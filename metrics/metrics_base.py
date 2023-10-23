"""
Set up the metrics 
"""
import os
import sys
import numpy as np
import wandb
import torch
import torch.distributed as dist

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from metrics_utils import get_metric_function, AverageMeter

# -------------------------------------------------------------------------------------------------
class MetricManager(object):
    """
    Manages metrics and logging
    """
    
    def __init__(self, config):
        """
        @args:
            - config (Namespace): nested namespace containing all args
        """
        super().__init__()
        self.config = config
        self.device = config.device
        self.wandb_run = None

    def init_wandb(self):
        """
        Runs once at beginning of training if global_rank<=0 to initialize wandb object
        """
        self.wandb_run = wandb.init(project=self.config.project, 
                                    entity=self.config.wandb_entity, 
                                    config=self.config, 
                                    name=self.config.run_name, 
                                    notes=self.config.run_notes,
                                    dir=self.config.wandb_dir)

    def setup_wandb_and_metrics(self, rank):
        """
        Runs once at beginning of training for all processes to setup metrics 
        """

        # Set up common metrics depending on the task type
        if self.config.task_type=='class': 
            # Set up metric dicts, which we'll use during training to track metrics
            self.train_metrics = {'loss': AverageMeter(),
                                  'auroc':AverageMeter()}
            self.eval_metrics = {'loss': AverageMeter(),
                                 'acc_1':AverageMeter(),
                                 'auroc':AverageMeter(),
                                 'f1':AverageMeter()}
            
            # Define vars used by the metric functions
            if self.config.no_out_channel==1 or self.config.no_out_channel==2: # Assumes no multilabel problems
                self.metric_task = 'binary' 
            else: 
                self.metric_task = 'multiclass'
            self.multidim_average = 'global'

            # Set up dictionary of functions mapped to each metric name
            self.train_metric_functions = {metric_name: get_metric_function(metric_name, self.config, self.metric_task, self.multidim_average).to(device=self.device) for metric_name in self.train_metrics if metric_name!='loss'}
            self.eval_metric_functions = {metric_name: get_metric_function(metric_name, self.config, self.metric_task, self.multidim_average).to(device=self.device) for metric_name in self.eval_metrics if metric_name!='loss'}

        elif self.config.task_type=='seg': 
            # Set up metric dicts, which we'll use during training to track metrics
            self.train_metrics = {'loss': AverageMeter(),
                                  'f1':AverageMeter()}
            self.eval_metrics = {'loss': AverageMeter(),
                                 'f1':AverageMeter()}
            
            # Define vars used by the metric functions
            if self.config.no_out_channel==1 or self.config.no_out_channel==2: # Assumes no multilabel problems
                self.metric_task = 'binary' 
            else: 
                self.metric_task = 'multiclass'
            self.multidim_average = 'samplewise'

            # Set up dictionary of functions mapped to each metric name
            self.train_metric_functions = {metric_name: get_metric_function(metric_name, self.config, self.metric_task, self.multidim_average).to(device=self.device) for metric_name in self.train_metrics if metric_name!='loss'}
            self.eval_metric_functions = {metric_name: get_metric_function(metric_name, self.config, self.metric_task, self.multidim_average).to(device=self.device) for metric_name in self.eval_metrics if metric_name!='loss'}
        
        elif self.config.task_type=='enhance': 
            # Set up metric dicts, which we'll use during training to track metrics
            self.train_metrics = {'loss': AverageMeter(),
                                  'ssim':AverageMeter(),
                                  'psnr':AverageMeter()}
            self.eval_metrics = {'loss': AverageMeter(),
                                  'ssim':AverageMeter(),
                                  'psnr':AverageMeter()}
            
            # Define vars used by the metric functions 
            self.metric_task = 'multiclass' # Keep as multiclass for enhance applications
            self.multidim_average = 'global' # Keep as global for enhance applications

            # Set up dictionary of functions mapped to each metric name
            self.train_metric_functions = {metric_name: get_metric_function(metric_name, self.config, self.metric_task, self.multidim_average).to(device=self.device) for metric_name in self.train_metrics if metric_name!='loss'}
            self.eval_metric_functions = {metric_name: get_metric_function(metric_name, self.config, self.metric_task, self.multidim_average).to(device=self.device) for metric_name in self.eval_metrics if metric_name!='loss'}

        else:
            raise NotImplementedError(f"No metrics implemented for task type {self.config.task_type}.")

        if rank<=0:

            # Initialize metrics to track in wandb      
            self.wandb_run.define_metric("epoch")    
            for metric_name in self.train_metrics.keys():
                self.wandb_run.define_metric('train_'+metric_name, step_metric='epoch')
            for metric_name in self.eval_metrics.keys():
                self.wandb_run.define_metric('val_'+metric_name, step_metric='epoch')
            
            # Initialize metrics to track for checkpointing best-performing model
            self.best_val_loss = np.inf
            if self.config.task_type=='class':
                self.best_val_auroc = -1
            elif self.config.task_type=='seg':
                self.best_val_f1 = -1
            elif self.config.task_type=='enhance':
                self.best_val_psnr = -np.inf

    def on_train_epoch_start(self):
        """
        Runs on the start of each training epoch
        """

        # Reset metric values in AverageMeter
        for metric_name in self.train_metrics.keys():
            self.train_metrics[metric_name].reset()

    def on_train_step_end(self, loss, output, labels, rank, curr_lr):
        """
        Runs on the end of each training step
        """
        # Adjust outputs to correct format for computing metrics
        if self.config.task_type=='class':
            output = torch.nn.functional.softmax(output, dim=1)
            if self.metric_task=='binary': 
                output = output[:,-1]
        
        elif self.config.task_type=='seg':
            output = torch.argmax(output,1)
            output = output.reshape(output.shape[0],-1)
            labels = labels.reshape(labels.shape[0],-1)
            
        # Update train metrics based on the predictions this step
        for metric_name in self.train_metrics.keys():
            if metric_name=='loss':
                self.train_metrics[metric_name].update(loss, n=output.shape[0])
            else:
                metric_value = self.train_metric_functions[metric_name](output, labels)
                if self.multidim_average=='samplewise':
                    metric_value = torch.mean(metric_value)
                self.train_metrics[metric_name].update(metric_value.item(), n=output.shape[0])

        if rank<=0: 
            self.wandb_run.log({"lr": curr_lr})
            
    def on_train_epoch_end(self, epoch, rank):
        """
        Runs at the end of each training epoch
        """

        # Aggregate the measurements taken over each step
        if self.config.ddp:
            # Note to self: # original code did not do this on train split, just logged the .avg for each metric name
            average_metrics = {metric_name: torch.tensor(self.train_metrics[metric_name].avg).to(device=self.device) for metric_name in self.train_metrics.keys()}
            average_metrics = {avg_metric_name: dist.all_reduce(avg_metric_val, op=torch.distributed.ReduceOp.AVG) for avg_metric_name, avg_metric_val in average_metrics.items()}

        else:
            average_metrics = {metric_name: self.train_metrics[metric_name].avg for metric_name in self.train_metrics.keys()}

        # Log the metrics for this epoch to wandb
        if rank<=0: # main or master process
            for metric_name, avg_metric_val in average_metrics.items():
                self.wandb_run.log({"epoch": epoch, "train_"+metric_name: avg_metric_val})

        # Save the average metrics for this epoch into self.average_train_metrics
        self.average_train_metrics = average_metrics

    def on_eval_epoch_start(self):
        """
        Runs at the start of each evaluation loop
        """
        self.all_preds = []
        self.all_labels = []
        for metric_name in self.eval_metrics:
            self.eval_metrics[metric_name].reset()

    def on_eval_step_end(self, loss, output, labels, ids, rank, save_samples, split):
        """
        Runs at the end of each evaluation step
        """
        # Adjust outputs to correct format for computing metrics
        if self.config.task_type=='class':
            output = torch.nn.functional.softmax(output, dim=1)
            if self.metric_task=='binary': 
                output = output[:,-1]
        
        elif self.config.task_type=='seg':
            output = torch.nn.functional.softmax(output, dim=1)
            output = torch.argmax(output,1)
            og_shape = output.shape[1:]
            output = output.reshape(output.shape[0],-1)
            labels = labels.reshape(labels.shape[0],-1)

        # If exact_metrics was specified in the config, we'll save all the predictions so that we are computing exactly correct metrics over the entire eval set
        # If exact_metrics was not specified, then we'll average the metric over each eval step. Sometimes this produces the same result (e.g., average of losses over steps = average of loss over epoch), sometimes it does not (e.g., for auroc)
        if self.config.exact_metrics:
            if self.config.task_type=='class':
                self.all_preds += [output]
                self.all_labels += [labels]

            elif self.config.task_type in ['seg','enhance']:
                raise NotImplementedError('Exact metric computation not implemented for segmentation or enhancement; not needed for average Dice or average loss, etc.')
            
        # Update each metric with the outputs from this step 
        for metric_name in self.eval_metrics.keys():
            if metric_name=='loss':
                self.eval_metrics[metric_name].update(loss, n=output.shape[0])
            else:
                if not self.config.exact_metrics:
                    metric_value = self.eval_metric_functions[metric_name](output, labels)
                    if self.multidim_average=='samplewise':
                        metric_value = torch.mean(metric_value)
                    self.eval_metrics[metric_name].update(metric_value.item(), n=output.shape[0])

        # Save outputs if desired
        if save_samples and rank<=0:
            save_path = os.path.join(self.config.log_dir,self.config.run_name,'saved_samples',split)
            os.makedirs(save_path, exist_ok=True)
            for b_output, b_id in zip(output, ids):
                b_output = b_output.detach().cpu().numpy().astype('float32')
                if self.config.task_type=='seg':
                    b_output = b_output.reshape(og_shape)
                b_save_path = os.path.join(save_path,b_id+'_output.npy')
                np.save(b_save_path,b_output)

    def on_eval_epoch_end(self, rank, epoch, model_manager, optim, sched, split, final_eval):
        """
        Runs at the end of the evaluation loop
        """

        # Directly compute metrics from saved predictions if using exact metrics
        if self.config.exact_metrics:
            self.all_preds = torch.concatenate(self.all_preds)
            self.all_labels = torch.concatenate(self.all_labels)
            for metric_name in self.eval_metrics.keys():
                if metric_name!='loss':
                    metric_value = self.eval_metric_functions[metric_name](self.all_preds, self.all_labels).item()
                    if self.multidim_average=='samplewise':
                        metric_value = torch.mean(metric_value)
                    self.eval_metrics[metric_name].update(metric_value, n=self.all_preds.shape[0])

        # Otherwise aggregate the measurements over the steps
        if self.config.ddp:
            average_metrics = {metric_name: torch.tensor(self.val_metrics[metric_name].avg).to(device=self.device) for metric_name in self.eval_metrics.keys()}
            average_metrics = {avg_metric_name: dist.all_reduce(avg_metric_val, op=torch.distributed.ReduceOp.AVG) for avg_metric_name, avg_metric_val in average_metrics.items()}

        else:
            average_metrics = {metric_name: self.eval_metrics[metric_name].avg for metric_name in self.eval_metrics.keys()}

        # Checkpoint best models during training
        if rank<=0: 

            if not final_eval:

                # Determine whether to checkpoint this model
                model_epoch = model_manager.module if self.config.ddp else model_manager 
                checkpoint_model = False    
                if self.config.task_type=='class':
                    checkpoint_model = average_metrics['auroc'] > self.best_val_auroc
                elif self.config.task_type=='seg':
                    checkpoint_model = average_metrics['f1'] > self.best_val_f1
                elif self.config.task_type=='enhance':
                    checkpoint_model = average_metrics['psnr'] > self.best_val_psnr

                # Save model and update best metrics
                if checkpoint_model:
                    model_epoch.save('best_checkpoint', epoch, optim, sched)   
                    if self.config.task_type=='class':
                        self.best_val_auroc = average_metrics['auroc']
                        self.wandb_run.log({"epoch":epoch, "best_val_auroc":self.best_val_auroc})
                    elif self.config.task_type=='seg':
                        self.best_val_f1 = average_metrics['f1']
                        self.wandb_run.log({"epoch":epoch, "best_val_f1":self.best_val_f1})
                    elif self.config.task_type=='enhance':
                        self.best_val_psnr = average_metrics['psnr']
                        self.wandb_run.log({"epoch":epoch, "best_val_psnr":self.best_val_psnr})

                if average_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = average_metrics['loss']

                # Update wandb with eval metrics
                for metric_name, avg_metric_eval in average_metrics.items():
                    self.wandb_run.log({"epoch":epoch, f"{split}_{metric_name}": avg_metric_eval})

            # Save the average metrics for this epoch into self.average_eval_metrics
            self.average_eval_metrics = average_metrics
        
    def on_training_end(self, rank, epoch, model_manager, optim, sched, ran_training):
        """
        Runs once when training finishes
        """
        if rank<=0: # main or master process
            
            if ran_training:
                # Log the best loss and metrics from the run and save final model
                self.wandb_run.summary["best_val_loss"] = self.best_val_loss
                if self.config.task_type=='class':
                    self.wandb_run.summary["best_val_auroc"] = self.best_val_auroc
                elif self.config.task_type=='seg':
                    self.wandb_run.summary["best_val_f1"] = self.best_val_f1
                elif self.config.task_type=='enhance':
                    self.wandb_run.summary["best_val_psnr"] = self.best_val_psnr
                
                model_epoch = model_manager.module if self.config.ddp else model_manager 
                model_epoch.save('final_epoch', epoch, optim, sched)   

            # Finish the wandb run
            self.wandb_run.finish() 
        

def tests():
    print('Passed all tests')

    
if __name__=="__main__":
    tests()