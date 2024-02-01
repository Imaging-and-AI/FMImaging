
"""
Define a model component, which is a component of the full model architecture (e.g., pre, post, or backbone components) with basic save/load/freeze functionality
Define a model manager, which establishes the forward function and connects the pre/backbone/post components; also provides full model save/load functionality
"""

import os
import sys
import logging
from colorama import Fore, Style

import torch
import torch.nn as nn

from backbone import *
from task_heads import *
from model_utils import *
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

# -------------------------------------------------------------------------------------------------

class ModelComponent(nn.Module):
    """
    Define a model component, which is some part of the full model architecture (e.g., the pre, backbone, or post head)
    Provides generic save, freeze, load functionality
    """

    def __init__(self, config, component_name, input_feature_channels=None, output_feature_channels=None):
        """
        @args:
            - config (Namespace): nested namespace containing all args
            - component_name (str): name of the model component to create
            - input_feature_channels (List[int]): list of ints specifying the channel dimensions of the inputs to the model
            - output_feature_channels (List[int]): list of ints specifying the channel dimensions of the outputs of the model; can be None if output_feature_channels is computed within the component       
        """
        
        super().__init__()

        self.config = config
        self.component_name = component_name
        self.input_feature_channels = input_feature_channels
        self.output_feature_channels = output_feature_channels

        if not isinstance(self.input_feature_channels, list):
            self.input_feature_channels = [self.input_feature_channels]
        if not isinstance(self.output_feature_channels, list):
            self.output_feature_channels = [self.output_feature_channels]

        self.create_model_component()

    @property
    def device(self):
        return next(self.parameters()).device
    
    def create_model_component(self): 
        """
        Rules these models abide by: 
            model init returns: 
                - model component 
                - output_feature_channels (List[int]): List of ints containing the number of features in each tensor returned from the model component
            model forward pass returns: 
                - model outputs (List[tensor]): outputs from the model component, each tensor can have varying shape in the form B C* D* H* W*, where C* is specified in output_feature_channels
        """
        
        # Pre heads
        if self.component_name=='Identity':
            self.model, self.output_feature_channels = identity_model(self.config, self.input_feature_channels)
        # Backbone models
        elif self.component_name=='omnivore':
            self.model, self.output_feature_channels = omnivore(self.config, self.input_feature_channels)
        elif self.component_name=='STCNNT_HRNET':
            self.model, self.output_feature_channels = STCNNT_HRnet_model(self.config, self.input_feature_channels)
        elif self.component_name=='STCNNT_UNET':
            self.model, self.output_feature_channels = STCNNT_Unet_model(self.config, self.input_feature_channels)
        elif self.component_name=='STCNNT_mUNET':
            self.model, self.output_feature_channels = STCNNT_Mixed_Unetr_model(self.config, self.input_feature_channels)
        # Post heads
        elif self.component_name=='UperNet2D': # 2D seg
            self.model = UperNet2D(self.config, self.input_feature_channels, self.output_feature_channels)
        elif self.component_name=='UperNet3D': # 3D seg
            self.model = UperNet3D(self.config, self.input_feature_channels, self.output_feature_channels)
        elif self.component_name=='SimpleConv': # 2D or 3D seg
            self.model = SimpleConv(self.config, self.input_feature_channels, self.output_feature_channels)
        elif self.component_name=='NormPoolLinear': # 2D or 3D class
            self.model = NormPoolLinear(self.config, self.input_feature_channels, self.output_feature_channels)
        elif self.component_name=='ConvPoolLinear': # 2D or 3D class
            self.model = ConvPoolLinear(self.config, self.input_feature_channels, self.output_feature_channels)
        elif self.component_name=='SimpleMultidepthConv': # 2D or 3D enhancement
            self.model = SimpleMultidepthConv(self.config, self.input_feature_channels, self.output_feature_channels)
        elif self.component_name=='UNETR2D': # 2D enhancement
            self.model = UNETR2D(self.config, self.input_feature_channels, self.output_feature_channels)
        elif self.component_name=='UNETR3D': # 2D or 3d enhancement, works with both
            self.model = UNETR3D(self.config, self.input_feature_channels, self.output_feature_channels)
        else:
            raise NotImplementedError(f"Model not implemented: {self.model_name}")

    def save(self, model_save_name): 
        """
        Save weights of model component
        @args:
            - model_save_name (str): what to name this saved model
        @rets: 
            - save_path (str): location of saved model
        """
        save_path = os.path.join(self.config.log_dir, self.config.run_name.replace(' ','_'), f"{model_save_name}.pth")
        logging.info(f"{Fore.YELLOW}Saving model component at {save_path}{Style.RESET_ALL}")

        save_dict = {
            "state_dict": self.model.state_dict(), 
            "config": self.config,
            "model_name": self.model_name,
            "input_feature_channels": self.input_feature_channels,
            "output_feature_channels": self.output_feature_channels
        }

        torch.save(save_dict, save_path)

        return save_path

    def load(self, load_path, device=None):
        """
        Load weights of a model component checkpoint
        @args:
            - load_path (str): path to load the weights from
            - device (torch.device): device to setup the model on
        """

        assert os.path.isfile(load_path), f"{Fore.YELLOW} Specific load path {load_path} does not exist {Style.RESET_ALL}"
        logging.info(f"{Fore.YELLOW}Loading model from {load_path}{Style.RESET_ALL}")

        status = torch.load(load_path, map_location=self.config.device)
        assert 'state_dict' in status, f"{Fore.YELLOW} Model weights in specified load path {load_path} are not available {Style.RESET_ALL}"

        self.model.load_state_dict(status['state_dict'])
        logging.info(f"{Fore.GREEN} Model loading {load_path.split('/')[-1]} successful {Style.RESET_ALL}")

    def freeze(self):
        "Freeze component parameters"

        self.model.requires_grad_(False)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Define the forward pass through the model component
        """
        return self.model(x)

# -------------------------------------------------------------------------------------------------

class ModelManager(nn.Module):
    """
    Manager to connect the pre, backbone, and post model components
    Provides generic save and load functionality
    """
    def __init__(self, config, tasks, backbone_component):
        """
        @args:
            - config (Namespace): nested namespace containing all args
            - tasks (dict): dictionary of {task_name: TaskManager}, which includes the tasks' pre/post model components
            - backbone_component (ModelComponent): backbone model component
        """
        super().__init__()

        self.config = config
        self.tasks = tasks
        self.backbone_component = backbone_component
        
        # Check that all pre/post components have the same output/input feature channels, which is required for connecting them to the same backbone
        assert len(set([tuple(task.pre_component.output_feature_channels) for task in tasks.values()])) == 1, "All pre components must have the same output feature channels to connect to the same backbone" 
        assert len(set([tuple(task.post_component.input_feature_channels) for task in tasks.values()])) == 1, "All post components must have the same input feature channels to connect to the same backbone"
        
    @property
    def device(self):
        return next(self.parameters()).device

    def check_model_learnable_status(self, rank_str=""):
        """ 
        Count how many parameters are learnable
        """

        num = 0
        num_learnable = 0
        for task in self.tasks.values():
            for param in task.pre_component.parameters():
                num += 1
                if param.requires_grad:
                    num_learnable += 1

        print(f"{rank_str} model, pre components, learnable tensors {num_learnable} out of {num} ...")

        num = 0
        num_learnable = 0
        for param in self.backbone_component.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        print(f"{rank_str} model, backbone component, learnable tensors {num_learnable} out of {num} ...")

        num = 0
        num_learnable = 0
        for task in self.tasks.values():
            for param in task.post_component.parameters():
                num += 1
                if param.requires_grad:
                    num_learnable += 1

        print(f"{rank_str} model, post components, learnable tensors {num_learnable} out of {num} ...")

    def save_entire_model(self, epoch, optim=None, sched=None, save_file_name=None):
        """
        Save entire model, including pre/backbone/post components
        @args:
            - epoch (int): current epoch of the training cycle
            - optim (torch.optim): optimizer to save
            - sched (torch.optim.lr_scheduler): scheduler to save
            - save_file_name (str): name to save file
        """
        run_name = self.config.run_name.replace(" ", "_")
        if save_file_name is None:
            save_file_name = f"{run_name}_epoch-{epoch}"
        save_path = os.path.join(self.config.log_dir, run_name, 'entire_model')
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, f"{save_file_name}.pth")
        logging.info(f"{Fore.YELLOW}Saving model status at {model_file}{Style.RESET_ALL}")
        
        save_dict = {
            "epoch":epoch,
            "config": self.config,
        }
        if optim is not None: 
            save_dict["optim_state"] = optim.state_dict()
        if sched is not None:
            save_dict["sched_state"] = sched.state_dict()
        for task_name, task in self.tasks.items():
            save_dict[task_name+"_pre_state"] = task.pre_component.state_dict()
            save_dict[task_name+"_post_state"] = task.post_component.state_dict()
        save_dict["backbone_state"] = self.backbone_component.state_dict()
        torch.save(save_dict, model_file)

        return os.path.join(save_path, save_file_name)
    
    def load_entire_model(self, load_path, device=torch.device('cpu')):
        """
        Load an entire model's weights, including pre/backbone/post components
        @args:
            - load_path (str): path to load model
            - device (torch.device): device to setup the model on
        """

        assert os.path.exists(load_path), f"{Fore.YELLOW} Specified load path {load_path} does not exist {Style.RESET_ALL}"
        logging.info(f"{Fore.YELLOW}Loading model weights from {load_path}{Style.RESET_ALL}")

        status = torch.load(load_path, map_location=device)

        assert 'backbone_state' in status, f"{Fore.YELLOW} Backbone weights in specified load_path are not available {Style.RESET_ALL}"
        self.backbone_component.load_state_dict(status['backbone_state'])

        for task_name, task in self.tasks.items():
            assert task_name+"_pre_state" in status, f"{Fore.YELLOW} Pre component weights for task {task_name} in specified load_path are not available {Style.RESET_ALL}"
            assert task_name+"_post_state" in status, f"{Fore.YELLOW} Post component weights for task {task_name} in specified load_path are not available {Style.RESET_ALL}"
            task.pre_component.load_state_dict(status[task_name+"_pre_state"])
            task.post_component.load_state_dict(status[task_name+"_post_state"])
                
    def forward(self, x, task_name=None):
        """
        Define the forward pass through the model
        @args:
            - x (5D torch.Tensor): input image, B C D/T H W
            - task_name (str): name of the task to run
        @rets:
            - output (tensor): final output from model for this task
        """
        if task_name is None: task_name = self.config.tasks[0]
        
        pre_output = self.tasks[task_name].pre_component(x)
        backbone_output = self.backbone_component(pre_output)
        post_output = self.tasks[task_name].post_component(backbone_output)
        return post_output[-1]


# -------------------------------------------------------------------------------------------------

def tests():
    pass

    
if __name__=="__main__":
    tests()
