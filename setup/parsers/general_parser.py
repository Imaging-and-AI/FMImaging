import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

Setup_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Setup_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

from config_utils import *

class general_parser(object):
    """
    General parser that contains args used by all projects
    @args:
        no args
    @rets:
        no rets; self.parser contains args
    """

    def __init__(self):
        
        self.parser = argparse.ArgumentParser("")

        # Path args
        self.parser.add_argument("--run_name", type=str, default='project_'+str(datetime.now().strftime("%H-%M-%S-%Y%m%d")), help='Name to identify this run (in logs and wandb)')
        self.parser.add_argument("--log_dir", type=str, default=os.path.join(Project_DIR, 'logs'), help='Directory to store log files')
        self.parser.add_argument("--data_dir", type=str, nargs='+', default=[os.path.join(Project_DIR,'data')], help='Directory where data is stored in order of tasks; will be passed to dataloader')
        self.parser.add_argument("--split_csv_path", type=none_or_str, nargs='+', default=[None], help='Path to csv that specifies data splits in order of tasks; if not specified, data will be split into 60% train, 20% val, 20% test randomly (used with default dataloader only)')
        self.parser.add_argument("--pre_component_load_path", type=none_or_str, nargs='+', default=[None], help='Path to load pre model(s) from in order of tasks; set to None if not loading a model')
        self.parser.add_argument("--backbone_component_load_path", type=none_or_str, default=None, help='Path to load backbone model from; set to None if not loading a model')
        self.parser.add_argument("--post_component_load_path", type=none_or_str, nargs='+', default=[None], help='Path to load post model(s) from in order of tasks; set to None if not loading a model')
        self.parser.add_argument("--full_model_load_path", type=none_or_str, default=None, help='Path to load full model (containing pre/backbone/post) from; set to None if not loading a model')
        self.parser.add_argument("--yaml_load_path", type=none_or_str, default=None, help='Path to load yaml config from; set to None if not loading a config. Note that this config will overwrite user args.')
        self.parser.add_argument("--override", action="store_true", help="Whether to override files already saved in log_dir/run_name")
        
        # Train/eval args 
        self.parser.add_argument("--train_model", type=str_to_bool, default=True, help="Whether to run training; if False, only eval will run")
        self.parser.add_argument("--training_scheme", type=str, default="single_task", choices=["single_task"],help='Which trainings scheme to use')
        self.parser.add_argument("--continued_training", type=str_to_bool, default=False, help="Whether to continue training; if True, will load the optimizer and scheduler states along with the model weights; used only if load_paths are specified")
        self.parser.add_argument("--eval_train_set", type=str_to_bool, default=False, help="Whether to run inference on the train set at the end of training")
        self.parser.add_argument("--eval_val_set", type=str_to_bool, default=True, help="Whether to run inference on the val set at the end of training")
        self.parser.add_argument("--eval_test_set", type=str_to_bool, default=True, help="Whether to run inference on the test set at the end of training")
        self.parser.add_argument("--save_train_samples", type=str_to_bool, default=False, help="Whether to save output samples if running inference on the train set at the end of training")
        self.parser.add_argument("--save_val_samples", type=str_to_bool, default=False, help="Whether to save output samples if running inference on the val set at the end of training")
        self.parser.add_argument("--save_test_samples", type=str_to_bool, default=True, help="Whether to save output samples if running inference on the test set at the end of training")

        # Wandb args
        self.parser.add_argument("--project", type=str, default='FMImaging', help='Project name for wandb')
        self.parser.add_argument("--run_notes", type=str, default='Default project notes', help='Notes for the current run for wandb')
        self.parser.add_argument("--wandb_entity", type=str, default="gadgetron", help='Wandb entity to link with')
        self.parser.add_argument("--wandb_dir", type=str, default=os.path.join(Project_DIR, 'wandb'), help='directory for saving wandb')
        
        # Task args
        self.parser.add_argument('--tasks', type=str, nargs='+', default=["task_name"], help="Name of each task")
        self.parser.add_argument('--task_type', type=str, nargs='+', default=["class"], choices=['class','seg','enhance'], help="Task type for each task")
        self.parser.add_argument("--loss_func", type=str, nargs='+', default=['CrossEntropy'], choices=['CrossEntropy','MSE'], help='Which loss function to use in order of tasks')
        self.parser.add_argument("--height", type=int, default=[256],  nargs='+', help='Height (number of rows) of input in order of tasks; will interpolate to this (used with default dataloader only)')
        self.parser.add_argument("--width", type=int, default=[256],  nargs='+', help='Width (number of columns) of input in order of tasks; will interpolate to this (used with default dataloader only)')
        self.parser.add_argument("--time", type=int, default=[1],  nargs='+', help='Temporal/depth dimension of input in order of tasks; will crop/pad to this (used with default dataloader only)')
        self.parser.add_argument("--no_in_channel", type=int,  nargs='+', default=[1], help='Number of input channels in order of tasks')
        self.parser.add_argument("--no_out_channel", type=int,  nargs='+', default=[2], help='Number of output channels or classes in order of tasks')
        self.parser.add_argument("--use_patches", type=str_to_bool,  nargs='+', default=[False], help='Whether to train on patches, specified in order of tasks (used with default dataloader only)')
        self.parser.add_argument("--patch_height", type=int, nargs='+', default=[32], help='Height (number of rows) of patch in order of tasks; will crop to this (used with default dataloader only)')
        self.parser.add_argument("--patch_width", type=int, nargs='+', default=[32], help='Width (number of columns) of patch in order of tasks; will crop to this (used with default dataloader only)')
        self.parser.add_argument("--patch_time", type=int, nargs='+', default=[1], help='Temporal/depth dimension of patch in order of tasks; will crop to this (used with default dataloader only)')
        
        # Augmentation args
        self.parser.add_argument("--affine_aug", type=str_to_bool, nargs='+', default=[True], help="Whether to apply affine transforms, specified in order of tasks (used with default dataloader only)")
        self.parser.add_argument("--brightness_aug", type=str_to_bool, nargs='+', default=[True], help="Whether to apply brightness jitter transforms, specified in order of tasks (used with default dataloader only)")
        self.parser.add_argument("--gaussian_blur_aug", type=str_to_bool, nargs='+', default=[True], help="Whether to apply gaussian blur transforms, specified in order of tasks (used with default dataloader only)")

        # Model args
        self.parser.add_argument('--pre_component', type=str, nargs='+', default=["Identity"], choices=['Identity'], help="Which pre model to use in order of tasks")
        self.parser.add_argument('--backbone_component', type=str, default="STCNNT_HRNET", choices=['Identity',
                                                                                                 'omnivore',
                                                                                                 'STCNNT_HRNET',
                                                                                                 'STCNNT_UNET',
                                                                                                 'STCNNT_mUNET'], 
                                                                                                help="Which backbone model to use")
        self.parser.add_argument('--post_component', type=str, nargs='+', default=["NormPoolLinear"], choices=['Identity',
                                                                                      'NormPoolLinear',
                                                                                      'ConvPoolLinear',
                                                                                      'UperNet2D',
                                                                                      'UperNet3D',
                                                                                      'SimpleConv',
                                                                                      'SimpleMultidepthConv',
                                                                                      'UNETR2D',
                                                                                      'UNETR3D'], help="Which task head to use in order of tasks")
        self.parser.add_argument('--freeze_pre', type=str_to_bool, nargs='+', default=[False], help="Whether to freeze the pre model in order of tasks")
        self.parser.add_argument('--freeze_backbone', type=str_to_bool, default=False, help="Whether to freeze the backbone model")
        self.parser.add_argument('--freeze_post', type=str_to_bool, nargs='+', default=[False], help="Whether to freeze the post model in order of tasks")
        
        # Optimizer args
        self.parser.add_argument("--optim_type", type=str, default="adamw", choices=["adam", "adamw", "nadam", "sgd", "sophia", "lbfgs"],help='Which optimizer to use')
        self.parser.add_argument("--scheduler_type", type=none_or_str, default="ReduceLROnPlateau", choices=["ReduceLROnPlateau", "StepLR", "OneCycleLR", None], help='Which LR scheduler to use')
        
        # General training args
        self.parser.add_argument("--device", type=str, default='cuda', choices=['cpu','cuda'], help='Device to train on')
        self.parser.add_argument("--debug", "-D", action="store_true", help='Option to run in debug mode')
        self.parser.add_argument("--summary_depth", type=int, default=6, help='Depth to print the model summary through')
        self.parser.add_argument("--num_workers", type=int, default=-1, help='Number of total workers for data loading; if <=0, use os.cpu_count()')
        self.parser.add_argument("--prefetch_factor", type=int, default=8, help='Number of batches loaded in advance by each worker')
        self.parser.add_argument("--use_amp", action="store_true", help='Whether to train with mixed precision')
        self.parser.add_argument("--with_timer", action="store_true", help='Whether to train with timing')
        self.parser.add_argument("--seed", type=int, default=None, help='Seed for randomization')
        self.parser.add_argument("--eval_frequency", type=int, default=1, help="How often (in epochs) to evaluate val set")
        self.parser.add_argument("--checkpoint_frequency", type=int, default=10, help="How often (in epochs) to save the model")
        self.parser.add_argument("--save_model_components", type=str_to_bool, default=False, help="Whether to save pre, post, and backbone model components in addition to saving entire model")
        self.parser.add_argument("--exact_metrics", type=str_to_bool, default=False, help="Whether to store all validation preds and gt labels to compute exact metrics, or use approximate metrics via averaging over batch")
        self.parser.add_argument("--ddp", action="store_true", help='Whether training with ddp; if so, call torchrun from command line')
        
        # Training parameters
        self.parser.add_argument("--num_epochs", type=int, default=50, help='Number of epochs to train for')
        self.parser.add_argument("--batch_size", type=int, nargs='+', default=[64], help='Size of each batch in order of tasks')
        self.parser.add_argument("--clip_grad_norm", type=float, default=0, help='Gradient norm clip, if <=0, no clipping')
        self.parser.add_argument("--iters_to_accumulate", type=int, default=1, help='Number of iterations to accumulate gradients; if >1, gradient accumulation')
