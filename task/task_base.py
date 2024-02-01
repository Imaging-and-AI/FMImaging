import os
import sys
import yaml
import logging
from colorama import Fore, Style

from torch import nn
from torch.utils.data.dataloader import DataLoader

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
                                            output_feature_channels=None) # Pass in None by default; assuming the pre component's output feature channels are set in the config

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
                                             output_feature_channels=self.task_out_channels) # Set the number of output feature channels for this task

    def save(self, save_name_modifier=None):
        """
        Save this task, including the pre and post model states and task configs
        @args:
            - save_name_modifier (str): optional string to append to the end of the task name for saving (e.g. epoch number)
        @saves:
            - yaml file of task config 
            - torch state dicts of pre and post components
        """

        save_name = self.task_name
        if save_name_modifier is not None:
            save_name += '_'+save_name_modifier
        save_path = os.path.join(self.config.log_dir, self.config.run_name.replace(' ','_'), 'tasks', self.task_name)
        os.makedirs(save_path, exist_ok=True)

        task_config = {
            "task name": self.task_name,
            "task ind": self.task_ind,
            "task type": self.task_type,
            "task in channels": self.task_in_channels,
            "task out channels": self.task_out_channels,
            "pre name": self.pre_name,
            "post name": self.post_name,
            "loss f name": self.loss_f_name,
            "config": self.config,
            "len train set": len(self.train_set),
            "len val set": len(self.val_set),
            "len test set": len(self.test_set),
        }

        config_save_path = os.path.join(save_path, save_name+'_config.yaml')
        logging.info(f"{Fore.YELLOW}Saving task config at {config_save_path}{Style.RESET_ALL}")
        with open(config_save_path, 'w') as file:
            yaml_config = yaml.dump(task_config, file)

        pre_save_path = self.pre_component.save(f"/tasks/{self.task_name}/{save_name}_pre_component")
        post_save_path = self.post_component.save(f"/tasks/{self.task_name}/{save_name}_post_component")

    def load(self, pre_load_path=None, post_load_path=None):
        """
        Load the pre and post model states
        """
        if pre_load_path not in ["None","none", None]:
            self.pre_component.load(pre_load_path)
        if post_load_path not in ["None","none", None]:
            self.post_component.load(post_load_path)