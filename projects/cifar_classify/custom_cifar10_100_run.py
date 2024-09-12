"""
Custom cifar run file 
"""

import os
import sys
import copy
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from colorama import Fore, Back, Style

# Default functions
from setup.setup_base import parse_config_and_setup_run
from setup.config_utils import config_to_yaml
from model.model_base import ModelComponent, ModelManager
from optim.optim_base import OptimManager
from trainer.trainer_base import TrainManager
from metrics.metrics_base import MetricManager
from task.task_base import TaskManager

# Custom functions
from custom_cifar_dataset import cifar_dataset

class CifarTaskManager(TaskManager):
    def __init__(self, config, task_name="cifar"):
        super().__init__(config=config, task_name=task_name, task_ind=0)

    def create_datasets(self):
        self.train_set = cifar_dataset(config=self.config, split='train') 
        self.val_set = cifar_dataset(config=self.config, split='val') 
        self.test_set = cifar_dataset(config=self.config, split='test') 

# -------------------------------------------------------------------------------------------------
def main():

    # -----------------------------
    # Parse args to config (no customization)
    config = parse_config_and_setup_run() 

    # -----------------------------
    # Define a single task - contains the pre and post heads, loss functions, and datasets
    config.tasks = ["cifar"]

    task_ind = 0
    task_name = config.tasks[task_ind]

    tasks = dict()
    tasks[task_name] = CifarTaskManager(config, task_name)

    a = tasks[task_name].train_set[12]

    # -----------------------------
    # Set up the model
    # let the pre_component be "identity"
    # the backbone be a Vit model
    # the post component be ViTLinear
    tasks[task_name].create_pre_component() 
    backbone_component = ModelComponent(config=config,
                                        component_name=config.backbone_component,
                                        input_feature_channels=tasks[task_name].pre_component.output_feature_channels)
    tasks[task_name].create_post_component(backbone_component.output_feature_channels)

    # -----------------------------
    # Create a ModelManager, which defines the forward pass and connects pre->backbone->post 
    model_manager = ModelManager(config, tasks, backbone_component)

    # -----------------------------
    # Create OptimManager, which defines optimizers and schedulers
    optim_manager = OptimManager(config, model_manager, tasks)

    # -----------------------------
    # Create MetricManager, which tracks metrics and checkpoints models during training
    metric_manager = MetricManager(config)

    # -----------------------------
    # Create TrainManager, which will control model training
    train_manager = TrainManager(config,
                                model_manager,
                                optim_manager,
                                metric_manager)

    # -----------------------------
    # Save config to yaml file
    yaml_file = config_to_yaml(config,os.path.join(config.log_dir,config.run_name))
    config.yaml_file = yaml_file 

    # -----------------------------
    # Execute training and evaluation
    train_manager.run()

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":    
    main()
