
"""
Define a model, which includes the pre/backbone/post heads with basic save and load functionality and the forward function
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

from optim.optim_utils import divide_optim_into_groups
from setup import config_to_yaml

# -------------------------------------------------------------------------------------------------

class ModelManager(nn.Module):
    """
    Manager to set up a model, including pre, backbone, and post components
    Provides generic save and load functionality
    """
    def __init__(self, config):
        """
        @args:
            - config (Namespace): nested namespace containing all args
        """
        super().__init__()

        self.config = config

        # Create models
        self.create_pre()
        self.create_backbone()
        self.create_post()

    @property
    def device(self):
        return next(self.parameters()).device
    
    def create_pre(self): 
        """
        Sets up the pre model architecture
        Rules these models should abide by: 
            inputs: config file
            init returns: model and pre_feature_channels (List[int])
            forward pass returns: List[tensor] (assumed length 1), each tensor is shape B C* D* H* W*, where C* is specified in pre_feature_channels
        @args:
            - None; uses values from self.config
        @outputs:
            - self.pre: model for processing inputs prior to being input into the backbone
            - self.pre_feature_channels: list of ints specifying the channel dimensions returned from the pre model
        """
        if self.config.pre_model=='Identity':
            self.pre, self.pre_feature_channels = identity_model(self.config)
        else:
            raise NotImplementedError(f"Pre model not implemented: {self.config.pre_model}")

    def save_pre(self, model_save_name, epoch, optim, sched):
        """
        Save pre model checkpoint
        @args:
            - model_save_name (str): what to name this saved model
            - epoch (int): current epoch of the training cycle
            - optim: optimizer to save along with this model
            - sched: schedule to save along with this model
        @rets: 
            - None; saves into self.config.log_dir
        """
        save_path = os.path.join(self.config.log_dir, self.config.run_name, f"{model_save_name}.pth")
        logging.info(f"{Fore.YELLOW}Saving model status at {save_path}{Style.RESET_ALL}")

        # Get the pre model's optimizer group
        pre_optim_state_dict = divide_optim_into_groups(optim, "pre", self.config.optim.all_w_decay)

        # Save all info
        save_dict = {
            "epoch":epoch,
            "pre_model_state": self.pre.state_dict(), 
            "pre_optimizer_state": pre_optim_state_dict, 
            "config": self.config,
        }

        if sched is not None: save_dict["scheduler_state"] = sched.state_dict()
        torch.save(save_dict, save_path)
        return save_path

    def load_pre(self, load_path, device=None):
        """
        Load a pre module checkpoint from the pre load path in config
        @args:
            - load_path (str): path to load the weights from
            - device (torch.device): device to setup the model on
        """

        logging.info(f"{Fore.YELLOW}Loading model from {load_path}{Style.RESET_ALL}")

        if os.path.isfile(load_path):
            status = torch.load(load_path, map_location=self.config.device)

            if 'pre_model_state' in status:
                self.pre.load_state_dict(status['pre_model_state'])
                logging.info(f"{Fore.GREEN} Pre model loading successful {Style.RESET_ALL}")
            else: 
                logging.warning(f"{Fore.YELLOW} Model weights in specified load_path are not available {Style.RESET_ALL}")

        else:
            logging.warning(f"{Fore.YELLOW}{load_path} does not exist .... {Style.RESET_ALL}")

    def freeze_pre(self):
        "Freeze pre model parameters"
        self.pre.requires_grad_(False)
        for param in self.pre.parameters():
            param.requires_grad = False

    def create_backbone(self): 
        """
        Sets up the backbone model architecture
        Rules these models should abide by: 
            inputs: config file and pre_feature_channels
            init returns: model and feature_channels (List[int])
            forward pass returns: List[tensor], each tensor can have varying shape in the form B C* D* H* W*, where C* is specified in pre_feature_channels
        @args:
            - None; uses values from self.config
        @outputs:
            - self.backbone: trunk of the model
            - self.feature_channels: list of ints specifying number of channels returned from the backbone
        """

        if self.config.backbone_model=='Identity':
            self.backbone, self.feature_channels = identity_model(self.config, self.pre_feature_channels)
        elif self.config.backbone_model=='omnivore_tiny':
            self.omnivore = omnivore_tiny(self.config, self.pre_feature_channels)
            self.backbone = self.omnivore
            self.feature_channels = self.omnivore.feature_channels
        elif self.config.backbone_model=='omnivore_small':
            self.omnivore = omnivore_small(self.config, self.pre_feature_channels)
            self.backbone = self.omnivore
            self.feature_channels = self.omnivore.feature_channels
        elif self.config.backbone_model=='omnivore_base':
            self.omnivore = omnivore_base(self.config, self.pre_feature_channels)
            self.backbone = self.omnivore
            self.feature_channels = self.omnivore.feature_channels
        elif self.config.backbone_model=='omnivore_large':
            self.omnivore = omnivore_large(self.config, self.pre_feature_channels)
            self.backbone = self.omnivore
            self.feature_channels = self.omnivore.feature_channels
        elif self.config.backbone_model=='STCNNT_HRNET':
            self.backbone, self.feature_channels = STCNNT_HRnet_model(self.config, self.pre_feature_channels)
        elif self.config.backbone_model=='STCNNT_UNET':
            self.backbone, self.feature_channels = STCNNT_Unet_model(self.config, self.pre_feature_channels)
        elif self.config.backbone_model=='STCNNT_mUNET':
            self.backbone, self.feature_channels = STCNNT_Mixed_Unetr_model(self.config, self.pre_feature_channels)
        else:
            raise NotImplementedError(f"Backbone model not implemented: {self.config.backbone_model}")

    def save_backbone(self, model_save_name, epoch, optim, sched):
        """
        Save backbone model checkpoint
        @args:
            - model_save_name (str): what to name this saved model
            - epoch (int): current epoch of the training cycle
            - optim: optimizer to save along with this model
            - sched: schedule to save along with this model
        @rets: 
            - None; saves into self.config.log_dir
        """
        save_path = os.path.join(self.config.log_dir, self.config.run_name, f"{model_save_name}.pth")
        logging.info(f"{Fore.YELLOW}Saving model status at {save_path}{Style.RESET_ALL}")

        backbone_optim_state_dict = divide_optim_into_groups(optim, "backbone", self.config.optim.all_w_decay)

        save_dict = {
            "epoch":epoch,
            "backbone_model_state": self.backbone.state_dict(), 
            "backbone_optimizer_state": backbone_optim_state_dict, 
            "config": self.config,
        }
        if sched is not None: save_dict["scheduler_state"] = sched.state_dict()
        torch.save(save_dict, save_path)
        return save_path

    def load_backbone(self, load_path, device=None):
        """
        Load a backbone checkpoint from the backbone load path in config + load optimizer, config, and scheduler
        @args:
            - load_path (str): path to load the weights from
            - device (torch.device): device to setup the model on
        """
        logging.info(f"{Fore.YELLOW}Loading model from {load_path}{Style.RESET_ALL}")

        if os.path.isfile(load_path):
            status = torch.load(load_path, map_location=self.config.device)

            if 'backbone_model_state' in status:
                self.backbone.load_state_dict(status['backbone_model_state'])
                logging.info(f"{Fore.GREEN} Backbone model loading successful {Style.RESET_ALL}")
            else: 
                logging.warning(f"{Fore.YELLOW} Model weights in specified load_path are not available {Style.RESET_ALL}")

        else:
            logging.warning(f"{Fore.YELLOW}{load_path} does not exist .... {Style.RESET_ALL}")

    def freeze_backbone(self):
        "Freeze backbone model parameters"
        self.backbone.requires_grad_(False)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def create_post(self): 

        """
        Sets up the post model architecture
        Rules these models should abide by: 
            inputs: config file and feature_channels
            init returns: model
            forward pass returns: List[tensor] (assumed length 1), each tensor is shape of expected output for task
        @args:
            - None; uses values from self.config
        @outputs:
            - self.post: task-specific head to process trunk outputs
        """

        if self.config.post_model=='Identity':
            self.post = nn.Identity()
        elif self.config.post_model=='UperNet2D': # 2D seg
            self.post = UperNet2D(self.config, self.feature_channels)
        elif self.config.post_model=='UperNet3D': # 3D seg
            self.post = UperNet3D(self.config, self.feature_channels)
        elif self.config.post_model=='SimpleConv': # 2D or 3D seg
            self.post = SimpleConv(self.config, self.feature_channels)
        elif self.config.post_model=='NormPoolLinear': # 2D or 3D class
            self.post = NormPoolLinear(self.config, self.feature_channels)
        elif self.config.post_model=='ConvPoolLinear': # 2D or 3D class
            self.post = ConvPoolLinear(self.config, self.feature_channels)
        elif self.config.post_model=='SimpleMultidepthConv': # 2D or 3D enhancement
            self.post = SimpleMultidepthConv(self.config, self.feature_channels)
        else:
            raise NotImplementedError(f"Post model not implemented: {self.config.post_model}")

    def save_post(self, model_save_name, epoch, optim, sched):
        """
        Save post model checkpoint
        @args:
            - model_save_name (str): what to name this saved model
            - epoch (int): current epoch of the training cycle
            - optim: optimizer to save along with this model
            - sched: schedule to save along with this model
        @rets: 
            - None; saves into self.config.log_dir
        """
        save_path = os.path.join(self.config.log_dir, self.config.run_name, f"{model_save_name}.pth")
        logging.info(f"{Fore.YELLOW}Saving model status at {save_path}{Style.RESET_ALL}")

        post_optim_state_dict = divide_optim_into_groups(optim, "post", self.config.optim.all_w_decay)

        save_dict = {
            "epoch":epoch,
            "post_model_state": self.post.state_dict(), 
            "post_optimizer_state": post_optim_state_dict, 
            "config": self.config,
        }
        if sched is not None: save_dict["scheduler_state"] = sched.state_dict()
        torch.save(save_dict, save_path)
        return save_path

    def load_post(self, load_path, device=None):
        """
        Load a post module checkpoint from the post load path in config
        @args:
            - load_path (str): path to load the weights from
            - device (torch.device): device to setup the model on
        """

        logging.info(f"{Fore.YELLOW}Loading model from {load_path}{Style.RESET_ALL}")

        if os.path.isfile(load_path):
            status = torch.load(load_path, map_location=self.config.device)

            if 'post_model_state' in status:
                self.post.load_state_dict(status['post_model_state'])
                logging.info(f"{Fore.GREEN} Post model loading successful {Style.RESET_ALL}")
            else: 
                logging.warning(f"{Fore.YELLOW} Model weights in specified load_path are not available {Style.RESET_ALL}")

        else:
            logging.warning(f"{Fore.YELLOW}{load_path} does not exist .... {Style.RESET_ALL}")

    def freeze_post(self):
        "Freeze post model parameters"
        self.post.requires_grad_(False)
        for param in self.post.parameters():
            param.requires_grad = False

    def check_model_learnable_status(self, rank_str=""):
        num = 0
        num_learnable = 0
        for param in self.pre.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        print(f"{rank_str} model, pre, learnable tensors {num_learnable} out of {num} ...")

        num = 0
        num_learnable = 0
        for param in self.backbone.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        print(f"{rank_str} model, backbone, learnable tensors {num_learnable} out of {num} ...")

        num = 0
        num_learnable = 0
        for param in self.post.parameters():
            num += 1
            if param.requires_grad:
                num_learnable += 1

        print(f"{rank_str} model, post, learnable tensors {num_learnable} out of {num} ...")

    def load(self):
        # Load models if paths specified
        if self.config.pre_model_load_path is not None: self.load_pre(self.config.pre_model_load_path)
        if self.config.backbone_model_load_path is not None: self.load_backbone(self.config.backbone_model_load_path)
        if self.config.post_model_load_path is not None: self.load_post(self.config.post_model_load_path)
        
    def load(self, model_save_name):
        self.load_pre(model_save_name+"_pre.pth")
        self.load_backbone(model_save_name+"_backbone.pth")
        self.load_post(model_save_name+"_post.pth")

    def save(self, model_save_name, epoch, optim, sched):
        pre_model_file = self.save_pre(model_save_name+"_pre", epoch, optim, sched)
        backbone_model_file = self.save_backbone(model_save_name+"_backbone", epoch, optim, sched)
        post_model_file = self.save_post(model_save_name+"_post", epoch, optim, sched)
        return pre_model_file, backbone_model_file, post_model_file

    def save_entire_model(self, epoch, save_file_name=None):
        """
        Save entire model
        @args:
            - epoch (int): current epoch of the training cycle
        @args (from config):
            - save_path (str): saved model full path and name
            - save_file_name (str): saved model file name
        """
        run_name = self.config.run_name.replace(" ", "_")

        if save_file_name is None:
            save_file_name = f"{run_name}_epoch-{epoch}"
            
        save_path = os.path.join(self.config.log_dir, run_name)
        os.makedirs(save_path, exist_ok=True)

        model_file = os.path.join(save_path, f"{save_file_name}.pth")
        logging.info(f"{Fore.YELLOW}Saving model status at {model_file}{Style.RESET_ALL}")
        
        torch.save({
            "epoch":epoch,
            "model_state": self.state_dict(), 
            "config": self.config,
        }, model_file)

        yaml_file = config_to_yaml(self.config, save_path, save_name=save_file_name)
        logging.info(f"{Fore.YELLOW}Saving model config at {yaml_file}{Style.RESET_ALL}")

        return save_path, save_file_name, yaml_file

    def load_entire_model(self, save_path, save_file_name, device=torch.device('cpu')):
        """
        Load a saved model
        @args:
            - save_path (str): path to load model
            - save_file_name (str): model file name
            - device (torch.device): device to setup the model on
        """
        
        model_full_path = os.path.join(save_path, save_file_name)
        logging.info(f"{Fore.YELLOW}Loading model from {model_full_path}{Style.RESET_ALL}")

        if os.path.isfile(model_full_path):
            status = torch.load(model_full_path, map_location=device)
            self.config = status['config']
            self.config.device = device
            self.load_state_dict(status['model_state'])
        else:
            logging.warning(f"{Fore.YELLOW}{model_full_path} does not exist .... {Style.RESET_ALL}")
            
    def forward(self, x):
        """
        @args:
            - x (5D torch.Tensor): input image, B C D/T H W
        @rets:
            - output: final output from model for this task
        """
        pre_output = self.pre(x)
        backbone_output = self.backbone(pre_output[-1])
        post_output = self.post(backbone_output)
        return post_output


# -------------------------------------------------------------------------------------------------

from setup.setup_base import parse_config_and_setup_run
def tests():

    # When you write tests for real, set all these to eval, add asserts

    test_config = parse_config_and_setup_run()

    test_input = torch.ones((test_config.batch_size,test_config.no_in_channel,test_config.time,test_config.height,test_config.width))
    seg_output = torch.ones((test_config.batch_size,test_config.no_out_channel,test_config.time,test_config.height,test_config.width))
    class_output = torch.ones((test_config.batch_size,test_config.no_out_channel))

    print('\n\nInput shape:',test_input.shape)
    print('Intended seg output shape:',seg_output.shape)
    print('Intended class output shape:',class_output.shape)

    for pre_model in ['Identity']:
        for backbone_model in ['STCNNT_HRNET','STCNNT_UNET','STCNNT_mUNET','omnivore_tiny','omnivore_small','omnivore_base','omnivore_large']:
            if 'omnivore' in backbone_model:
                post_model_choices = ['Identity','NormPoolLinear','ConvPoolLinear','UperNet2D','UperNet3D']
            else: 
                post_model_choices = ['Identity','NormPoolLinear','ConvPoolLinear','SimpleConv']
            for post_model in post_model_choices:

                print(f"\n\nConfig: {pre_model}->{backbone_model}->{post_model}")

                test_config.pre_model = pre_model
                test_config.backbone_model = backbone_model
                test_config.post_model = post_model

                test_model = ModelManager(test_config)
                test_output = test_model(test_input)

                if isinstance(test_output, list):
                    print(f'\tOutput is LIST shapes {[to.shape for to in test_output]}')
                else:
                    print(f'\tOutput is {type(test_output)} shape {test_output.shape}')
                
                if post_model=='Identity':
                    print(f'\t\tFeatures:',test_model.feature_channels)



    print('Passed all tests')

    
if __name__=="__main__":
    tests()