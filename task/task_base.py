import os
import sys
import yaml
import logging
from colorama import Fore, Style

from torch import nn
from torch.utils.data.dataloader import DataLoader

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

from data.data_base import NumpyDataset
from loss.loss_base import get_loss_func 
from model.model_base import ModelComponent

class TaskManager(nn.Module):
    """
    Defines each task, including the task's loss function, datasets, and pre/post model componenets
    Contains generic task save/load functionality
    """

    def __init__(self, config, task_name=None, task_ind=None):
        """
        @args:
            - config (Namespace): nested namespace containing all args
            - task_name (str): name of task
            - task_ind (int): index of task
        """

        super().__init__()

        self.config = config
        self.task_name = task_name
        self.task_ind = task_ind
        self.task_type = config.task_type[task_ind]
        self.task_in_channels = config.no_in_channel[task_ind]
        self.task_out_channels = config.no_out_channel[task_ind]
        self.pre_name = config.pre_component[task_ind]
        self.post_name = config.post_component[task_ind]
        self.loss_f_name = config.loss_func[task_ind]

        # Create loss function and datasets
        logging.info(f"{Fore.MAGENTA}{'-'*10}Creating task {self.task_name}{'-'*10}{Style.RESET_ALL}")
        self.create_loss_function()
        self.create_datasets()

        # We will create the pre and post components from run.py

    def create_loss_function(self):
        """
        Creates loss function for this task
        By default, the loss function self.loss_f will be passed (model_outputs, datalodaer_labels) and should return a float
        """
        self.loss_f = get_loss_func(self.loss_f_name) 

    def create_datasets(self):
        """
        Creates datasets for this task
        Each can be a list of or individual torch datasets
        """
        self.train_set = NumpyDataset(config=self.config, task_name=self.task_name, task_ind=self.task_ind, split='train')
        self.val_set = NumpyDataset(config=self.config, task_name=self.task_name, task_ind=self.task_ind, split='val')
        self.test_set = NumpyDataset(config=self.config, task_name=self.task_name, task_ind=self.task_ind, split='test')

    def create_pre_component(self):
        """
        Creates the pre model component for this task
        Note: this needs to be called before the backbone componenet is initialized, so this function is typically called from run.py
        """
        self.pre_component = ModelComponent(config=self.config, # Full config
                                            component_name=self.pre_name, # Name of the pre model component for this task
                                            input_feature_channels=self.task_in_channels, # Pass in the number of input feature channels for this task
                                            output_feature_channels=None,
                                            task_ind=self.task_ind) # Pass in None by default; assuming the pre component's output feature channels are set in the config

    def create_post_component(self, component_input_feature_channels):
        """
        Creates the post model component for this task
        Note: this needs to be called after the backbone componenet is initialized, so this function is typically called from run.py
        @args:
            - component_input_feature_channels (List[int]): the number of input feature channels for the post model component; will depend on the backbone being used
        """
        self.post_component = ModelComponent(config=self.config, # Full config
                                             component_name=self.post_name, # Name of the post model component for this task
                                             input_feature_channels=component_input_feature_channels, # Pass in the number of input feature channels for the post head, will depend on the backbone
                                             output_feature_channels=self.task_out_channels,
                                             task_ind=self.task_ind) # Set the number of output feature channels for this task

    def save(self, save_dir=None, save_filename=None):
        """
        Save this task, including the pre and post model states and task configs
        @args:
            - save_dir (str): optional directory to save task
            - save_filename (str): optional filename to save task (do not include file extension)
        @saves:
            - yaml file of task config 
            - torch state dicts of pre and post components
        """
        if save_dir is None: save_dir = os.path.join(self.config.log_dir, self.config.run_name, "tasks", self.task_name)
        if save_filename is None: save_filename = self.task_name

        os.makedirs(save_dir, exist_ok=True)

        pre_save_path = self.pre_component.save(save_dir, f"pre_component_{save_filename}")
        post_save_path = self.post_component.save(save_dir, f"post_component_{save_filename}")

        task_config = {
            "task name": self.task_name,
            "task ind": self.task_ind,
            "task type": self.task_type,
            "task in channels": self.task_in_channels,
            "task out channels": self.task_out_channels,
            "pre name": self.pre_name,
            "post name": self.post_name,
            "loss f name": self.loss_f_name,
            "len train set": len(self.train_set),
            "len val set": len(self.val_set),
            "len test set": len(self.test_set),
            "config": self.config,
        }
        
        with open(os.path.join(save_dir, f"{save_filename}_config.yaml"), 'w') as file:
            yaml_config = yaml.dump(task_config, file)

    def load(self, pre_load_path=None, post_load_path=None):
        """
        Load the pre and post model states
        @args:
            - pre_load_path (str): path to pre model component state file
            - post_load_path (str): path to post model component state file
        """
        if pre_load_path not in ["None","none", None]:
            self.pre_component.load(pre_load_path)
        if post_load_path not in ["None","none", None]:
            self.post_component.load(post_load_path)