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
                                    group=self.config.group,
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
        self.train_metrics = {}
        self.train_metric_functions = {}
        self.eval_metrics = {}
        self.eval_metric_functions = {}
        self.metric_task = {}
        self.multidim_average = {}

        # Loop through tasks, add metrics for each
        for task_ind, task_name in enumerate(self.config.tasks):
            task_type = self.config.task_type[task_ind]
            task_out_channel = self.config.no_out_channel[task_ind]

            if task_type =='class': 
                # Set up metric dicts, which we'll use during training to track metrics
                task_train_metrics = {'loss': AverageMeter(),
                                    'auroc': AverageMeter()}
                task_eval_metrics = {'loss': AverageMeter(),
                                    'acc_1': AverageMeter(),
                                    'auroc': AverageMeter(),
                                    'f1': AverageMeter()}
                
                # Define vars used by the metric functions
                if task_out_channel==1 or task_out_channel==2: # Assumes no multilabel problems
                    task_metric = 'binary' 
                else: 
                    task_metric = 'multiclass'
                multidim_average = 'global'

                # Set up dictionary of functions mapped to each metric name
                task_train_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, task_metric, multidim_average).to(device=self.device) for metric_name in task_train_metrics if metric_name!='loss'}
                task_eval_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, task_metric, multidim_average).to(device=self.device) for metric_name in task_eval_metrics if metric_name!='loss'}

                # Add task metrics to the overall metrics
                self.train_metrics[task_name] = task_train_metrics
                self.train_metric_functions[task_name] = task_train_metric_functions
                self.eval_metrics[task_name] = task_eval_metrics
                self.eval_metric_functions[task_name] = task_eval_metric_functions
                self.metric_task[task_name] = task_metric
                self.multidim_average[task_name] = multidim_average

            elif task_type=='seg': 
                # Set up metric dicts, which we'll use during training to track metrics
                task_train_metrics = {'loss': AverageMeter(),
                                    'f1': AverageMeter()}
                task_eval_metrics = {'loss': AverageMeter(),
                                    'f1': AverageMeter()}
                
                # Define vars used by the metric functions
                if task_out_channel==1 or task_out_channel==2: # Assumes no multilabel problems
                    task_metric = 'binary' 
                else: 
                    task_metric = 'multiclass'
                multidim_average = 'samplewise'

                # Set up dictionary of functions mapped to each metric name
                task_train_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, task_metric, multidim_average).to(device=self.device) for metric_name in task_train_metrics if metric_name!='loss'}
                task_eval_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, task_metric, multidim_average).to(device=self.device) for metric_name in task_eval_metrics if metric_name!='loss'}

                # Add task metrics to the overall metrics
                self.train_metrics[task_name] = task_train_metrics
                self.train_metric_functions[task_name] = task_train_metric_functions
                self.eval_metrics[task_name] = task_eval_metrics
                self.eval_metric_functions[task_name] = task_eval_metric_functions
                self.metric_task[task_name] = task_metric
                self.multidim_average[task_name] = multidim_average
            
            elif task_type=='enhance': 
                # Set up metric dicts, which we'll use during training to track metrics
                task_train_metrics = {'loss': AverageMeter(),
                                    'ssim': AverageMeter(),
                                    'psnr': AverageMeter()}
                task_eval_metrics = {'loss': AverageMeter(),
                                    'ssim': AverageMeter(),
                                    'psnr': AverageMeter()}
                
                # Define vars used by the metric functions 
                task_metric = 'multiclass' # Keep as multiclass for enhance applications
                multidim_average = 'global' # Keep as global for enhance applications

                # Set up dictionary of functions mapped to each metric name
                task_train_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, task_metric, multidim_average).to(device=self.device) for metric_name in task_train_metrics if metric_name!='loss'}
                task_eval_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, task_metric, multidim_average).to(device=self.device) for metric_name in task_eval_metrics if metric_name!='loss'}

                # Add task metrics to the overall metrics
                self.train_metrics[task_name] = task_train_metrics
                self.train_metric_functions[task_name] = task_train_metric_functions
                self.eval_metrics[task_name] = task_eval_metrics
                self.eval_metric_functions[task_name] = task_eval_metric_functions
                self.metric_task[task_name] = task_metric
                self.multidim_average[task_name] = multidim_average

            else:
                raise NotImplementedError(f"No metrics implemented for task type {task_type}.")

        if rank<=0:

            if self.wandb_run is not None:
                # Initialize metrics to track in wandb      
                self.wandb_run.define_metric("epoch")    
                for task_ind, task_name in enumerate(self.config.tasks):
                    for metric_name in self.train_metrics[task_name].keys():
                        self.wandb_run.define_metric(f'{task_name}/train_{metric_name}', step_metric='epoch')
                    for metric_name in self.eval_metrics[task_name].keys():
                        self.wandb_run.define_metric(f'{task_name}/val_{metric_name}', step_metric='epoch')
            
            # Initialize metrics to track for checkpointing best-performing model
            self.best_val_loss = np.inf
            self.bast_val_losses_by_task = {task_name: np.inf for task_name in self.config.tasks}

    def on_train_epoch_start(self):
        """
        Runs on the start of each training epoch
        """

        # Reset metric values in AverageMeter
        for task_name in self.train_metrics:
            for metric_name in self.train_metrics[task_name].keys():
                self.train_metrics[task_name][metric_name].reset()

    def on_train_step_end(self, task_name, loss, output, labels, rank, curr_lr):
        """
        Runs on the end of each training step
        """
        task_ind = self.config.tasks.index(task_name)
        task_type = self.config.task_type[task_ind]

        # Adjust outputs to correct format for computing metrics
        if task_type=='class':
            output = torch.nn.functional.softmax(output, dim=1)
            if self.metric_task[task_name]=='binary': 
                output = output[:,-1]
        
        elif task_type=='seg':
            output = torch.argmax(output,1)
            output = output.reshape(output.shape[0],-1)
            labels = labels.reshape(labels.shape[0],-1)
            
        elif task_type=='enhance':
            if labels.shape[2]==1: # 2D
                output = output[:,:,0,:,:]
                labels = labels[:,:,0,:,:]

        # Update train metrics based on the predictions this step
        for metric_name in self.train_metrics[task_name].keys():
            if metric_name=='loss':
                self.train_metrics[task_name][metric_name].update(loss, n=output.shape[0])
            else:
                metric_value = self.train_metric_functions[task_name][metric_name](output, labels)
                if self.multidim_average[task_name]=='samplewise':
                    metric_value = torch.mean(metric_value)
                self.train_metrics[task_name][metric_name].update(metric_value.item(), n=output.shape[0])

        if rank<=0: 
            if self.wandb_run is not None: self.wandb_run.log({"lr": curr_lr})
            
    def on_train_epoch_end(self, model_manager, optim, sched, epoch, rank):
        """
        Runs at the end of each training epoch
        """

        # Aggregate the measurements taken over each step
        if self.config.ddp:
            
            average_metrics = {task_name: {} for task_name in self.train_metrics.keys()}

            for task_name in self.train_metrics.keys():
                for metric_name in self.train_metrics[task_name].keys():

                    batch_vals = torch.tensor(self.train_metrics[task_name][metric_name].vals).to(device=self.device)
                    batch_counts = torch.tensor(self.train_metrics[task_name][metric_name].counts).to(device=self.device)
                    batch_products = batch_vals * batch_counts

                    dist.all_reduce(batch_products, op=torch.distributed.ReduceOp.SUM)
                    dist.all_reduce(batch_counts, op=torch.distributed.ReduceOp.SUM)

                    total_products = sum(batch_products)
                    total_counts = sum(batch_counts)
                    average_metrics[task_name][metric_name] = total_products.item() / total_counts.item()

        else:

            average_metrics = {task_name: {} for task_name in self.train_metrics.keys()}

            for task_name in self.train_metrics.keys():
                for metric_name in self.train_metrics[task_name].keys():
                    average_metrics[task_name][metric_name] = self.train_metrics[task_name][metric_name].avg


        if rank<=0: # main or master process

            # Log the metrics for this epoch to wandb
            average_metrics['total_loss'] = sum([average_metrics[task_name]['loss'] for task_name in self.config.tasks])
            if self.wandb_run is not None: 
                self.wandb_run.log({"epoch": epoch, f"train/total_loss": average_metrics['total_loss']}, commit=False)
                for task_name in self.train_metrics.keys():
                    for metric_name, avg_metric_val in average_metrics[task_name].items():
                        self.wandb_run.log({"epoch": epoch, f"train/{task_name}/{metric_name}": avg_metric_val}, commit=False)

            # Checkpoint the most recent model
            model_epoch = model_manager.module if self.config.ddp else model_manager 
            model_epoch.save_entire_model(save_filename='entire_model_last_epoch', epoch=epoch, optim=optim, sched=sched)
            if self.config.save_model_components: model_manager.save_model_components(save_filename='last_epoch')

            # Checkpoint every epoch, if desired
            if epoch % self.config.checkpoint_frequency==0:
                model_epoch = model_manager.module if self.config.ddp else model_manager 
                model_epoch.save_entire_model(epoch=epoch, optim=optim, sched=sched)

        # Save the average metrics for this epoch into self.average_train_metrics
        self.average_train_metrics = average_metrics

    def on_eval_epoch_start(self):
        """
        Runs at the start of each evaluation loop
        """
        self.all_preds = {task_name: [] for task_name in self.eval_metrics.keys()}
        self.all_labels = {task_name: [] for task_name in self.eval_metrics.keys()}
        for task_name in self.eval_metrics.keys():
            for metric_name in self.eval_metrics[task_name]:
                self.eval_metrics[task_name][metric_name].reset()

    def on_eval_step_end(self, task_name, loss, output, labels, ids, rank, save_samples, split):
        """
        Runs at the end of each evaluation step
        """

        task_ind = self.config.tasks.index(task_name)
        task_type = self.config.task_type[task_ind]

        # Adjust outputs to correct format for computing metrics
        if task_type=='class':
            output = torch.nn.functional.softmax(output, dim=1)
            if self.metric_task[task_name]=='binary': 
                output = output[:,-1]
        
        elif task_type=='seg':
            output = torch.nn.functional.softmax(output, dim=1)
            output = torch.argmax(output,1)
            og_shape = output.shape[1:]
            output = output.reshape(output.shape[0],-1)
            labels = labels.reshape(labels.shape[0],-1)

        elif task_type=='enhance':
            if labels.shape[2]==1: # 2D
                output = output[:,:,0,:,:]
                labels = labels[:,:,0,:,:]

        # If exact_metrics was specified in the config, we'll save all the predictions so that we are computing exactly correct metrics over the entire eval set
        # If exact_metrics was not specified, then we'll average the metric over each eval step. Sometimes this produces the same result (e.g., average of losses over steps = average of loss over epoch), sometimes it does not (e.g., for auroc)
        if self.config.exact_metrics:
            if self.config.task_type[task_ind]=='class':
                self.all_preds[task_name] += [output]
                self.all_labels[task_name] += [labels]

            else:
                raise NotImplementedError('Exact metric computation not implemented for segmentation or enhancement; not needed for average Dice or average loss.')
            
        # Update each metric with the outputs from this step 
        for metric_name in self.eval_metrics[task_name].keys():
            if metric_name=='loss':
                self.eval_metrics[task_name][metric_name].update(loss, n=output.shape[0])
            else:
                if not self.config.exact_metrics:
                    metric_value = self.eval_metric_functions[task_name][metric_name](output, labels)
                    if self.multidim_average[task_name]=='samplewise':
                        metric_value = torch.mean(metric_value)
                    self.eval_metrics[task_name][metric_name].update(metric_value.item(), n=output.shape[0])

        # Save outputs if desired
        if save_samples:
            save_path = os.path.join(self.config.log_dir,self.config.run_name,'tasks',task_name,'saved_samples',split)
            os.makedirs(save_path, exist_ok=True)
            for b_output, b_id in zip(output, ids):
                b_output = b_output.detach().cpu().numpy().astype('float32')
                if task_type=='seg':
                    b_output = b_output.reshape(og_shape)
                b_save_path = os.path.join(save_path,b_id+'_output.npy')
                np.save(b_save_path,b_output)

    def on_eval_epoch_end(self, rank, epoch, model_manager, optim, sched, split, final_eval):
        """
        Runs at the end of the evaluation loop
        """

        # Directly compute metrics from saved predictions if using exact metrics
        if self.config.exact_metrics:
            for task_name in self.config.tasks:
                self.all_preds[task_name] = torch.concatenate(self.all_preds[task_name])
                self.all_labels[task_name] = torch.concatenate(self.all_labels[task_name])
                for metric_name in self.eval_metrics[task_name].keys():
                    if metric_name!='loss':
                        metric_value = self.eval_metric_functions[task_name][metric_name](self.all_preds[task_name], self.all_labels[task_name]).item()
                        if self.multidim_average[task_name]=='samplewise':
                            metric_value = torch.mean(metric_value)
                        self.eval_metrics[task_name][metric_name].update(metric_value, n=self.all_preds[task_name].shape[0])

        # Aggregate the measurements over the steps
        if self.config.ddp:
            average_metrics = {task_name:{} for task_name in self.config.tasks}
            for task_name in self.config.tasks:
                for metric_name in self.eval_metrics[task_name].keys():

                    batch_vals = torch.tensor(self.eval_metrics[task_name][metric_name].vals).to(device=self.device)
                    batch_counts = torch.tensor(self.eval_metrics[task_name][metric_name].counts).to(device=self.device)
                    batch_products = batch_vals * batch_counts

                    dist.all_reduce(batch_products, op=torch.distributed.ReduceOp.SUM)
                    dist.all_reduce(batch_counts, op=torch.distributed.ReduceOp.SUM)

                    total_products = sum(batch_products)
                    total_counts = sum(batch_counts)
                    average_metrics[task_name][metric_name] = total_products.item() / total_counts.item()

        else:
            average_metrics = {task_name:{} for task_name in self.config.tasks}
            for task_name in self.config.tasks:
                average_metrics[task_name] = {metric_name: self.eval_metrics[task_name][metric_name].avg for metric_name in self.eval_metrics[task_name].keys()}

        # Checkpoint best models during training
        if rank<=0: 
            
            average_metrics['total_loss'] = sum([average_metrics[task_name]['loss'] for task_name in self.config.tasks])

            if not final_eval:

                # Update losses and determine whether to checkpoint this model
                checkpoint_model = False
                for task_name in self.config.tasks:
                    if average_metrics[task_name]['loss'] < self.bast_val_losses_by_task[task_name]:
                        self.bast_val_losses_by_task[task_name] = average_metrics[task_name]['loss']
                if average_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = average_metrics['total_loss']
                    checkpoint_model = True

                # Save model and update best metrics
                if checkpoint_model:
                    model_epoch = model_manager.module if self.config.ddp else model_manager 
                    model_epoch.save_entire_model(save_filename='entire_model_best_checkpoint', epoch=epoch, optim=optim, sched=sched)
                    if self.config.save_model_components: model_manager.save_model_components(save_filename='best_checkpoint')

                # Update wandb with eval metrics
                self.wandb_run.log({"epoch":epoch, "best_loss": self.best_val_loss}, commit=False)
                self.wandb_run.log({"epoch": epoch, f"{split}/total_loss": average_metrics['total_loss']}, commit=False)
                for task_name in self.config.tasks:
                    for metric_name, avg_metric_eval in average_metrics[task_name].items():
                        self.wandb_run.log({"epoch":epoch, f"{split}/{task_name}/{metric_name}": avg_metric_eval}, commit=False)

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

                # Final saves of backbone and tasks - commenting out, redundant with last_epoch saves
                # model_manager.save_entire_model(save_filename='final_model', epoch=epoch, optim=optim, sched=sched)
                # if self.config.save_model_components: model_manager.save_model_components(save_filename='final_model')
            
            # Finish the wandb run
            self.wandb_run.finish() 
        

def tests():
    pass

    
if __name__=="__main__":
    tests()