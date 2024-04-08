"""
Support functions for arg parsing to config
"""

import argparse
import os
import yaml

class Nestedspace(argparse.Namespace):
    """
    Define a nested namespace for dealing with nested args.
    from https://stackoverflow.com/questions/18668227/argparse-subcommands-with-nested-namespaces
    """
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

def nestedspace_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Nestedspace:
  """
  Construct a nestedspace.
  adapted from https://matthewpburruss.com/post/yaml/
  """
  return Nestedspace(**loader.construct_mapping(node))

def get_nestedspace_loader():
  """
  Add nestedspace constructors to PyYAML loader.
  adapted from https://matthewpburruss.com/post/yaml/
  """
  loader = yaml.SafeLoader
  loader.add_constructor("tag:yaml.org,2002:python/object:config_utils.Nestedspace", nestedspace_constructor)
  return loader

def config_to_yaml(config, save_path, save_name=None):
    """Save nestedspace config to yaml file."""
    if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        yaml_file_name = os.path.join(save_path,'config.yaml')
    else:
        yaml_file_name = os.path.join(save_path,f"{save_name}.yaml")
    print(f"--> save config as yaml at {yaml_file_name}")
    with open(yaml_file_name, "w", encoding = "utf-8") as yaml_file:
        dump = yaml.dump(config, default_flow_style = False, allow_unicode = True, encoding = None)
        yaml_file.write(dump)
        
    return yaml_file_name

def yaml_to_config(yaml_path, new_log_dir=None, new_run_name=None):
    """Load yaml into nestedspace config"""
    with open(yaml_path, "r") as yaml_file:
        saved_config = yaml.load(yaml_file, Loader=get_nestedspace_loader())
        if new_log_dir is not None: saved_config.log_dir = new_log_dir # Modify the log path to a new directory so we don't overwrite anything
        if new_run_name is not None: saved_config.run_name = new_run_name # Modify the log path to a new directory so we don't overwrite anything
    return saved_config

def check_for_unknown_args(config,args_to_check):
    """Check if user input any unknown args after completing all parsing"""
    unknown_args = []
    for arg in args_to_check:
        if arg not in config: 
            if '.' in arg:
                config_check = config
                for arg_element in arg.split('.'):
                    if arg_element in config_check:
                        config_check = getattr(config_check, arg_element)
                    else:
                        unknown_args += [arg]
                        break
            else: 
                unknown_args += [arg]
    return unknown_args
    
def none_or_str(value):
    """Convert arg from a string to None"""
    if value in ['None', 'none', None]:
        return None
    return value

def str_to_bool(value):
    """Convert arg from a string to bool"""
    if str(value) in ['1','True','true','T','t']: return True
    else: return False

def check_args(config):
    """Check arguments user input"""
    config.run_name = config.run_name.replace(' ','_')
    if os.path.exists(os.path.join(config.log_dir,config.run_name)) and not config.override:
        raise RuntimeError(f"User specified a log_dir ({config.log_dir}) and run_name ({config.run_name}) that already exist; either specify a different run name or override current results with --override")
    for data_dir in config.data_dir:
        if not os.path.exists(data_dir):
            raise RuntimeError(f"User specified a data_dir ({data_dir}) that does not exist")
    for csv_path in config.split_csv_path:
        if csv_path is not None:
            if not os.path.exists(csv_path):
                raise RuntimeError(f"User specified a split_csv_path ({csv_path}) that does not exist")
    for pre_component_load_path in config.pre_component_load_path:
        if pre_component_load_path not in [None,"None"]:
            if not os.path.exists(pre_component_load_path):
                raise RuntimeError(f"User specified a pre_component_load_path ({pre_component_load_path}) that does not exist")
    if config.backbone_component_load_path is not None:
        if not os.path.exists(config.backbone_component_load_path):
            raise RuntimeError(f"User specified a backbone_model_load_path ({config.backbone_component_load_path}) that does not exist")
    for post_component_load_path in config.post_component_load_path:
        if post_component_load_path not in [None,"None"]:
            if not os.path.exists(post_component_load_path):
                raise RuntimeError(f"User specified a post_component_load_path ({post_component_load_path}) that does not exist")
    if config.yaml_load_path is not None:
        if not os.path.exists(config.yaml_load_path):
            raise RuntimeError(f"User specified a yaml_load_path ({config.yaml_load_path}) that does not exist")
    for height in config.height:
        if height <= 0:
            raise RuntimeError("Height should be greater than or equal to 1")
    for width in config.width:
        if width <= 0:
            raise RuntimeError("Width should be greater than or equal to 1")
    for time in config.time:
        if time <= 0:
            raise RuntimeError("Time should be greater than or equal to 1")
    for no_in_channel in config.no_in_channel:
        if no_in_channel <= 0:
            raise RuntimeError("no_in_channel should be greater than or equal to 1")
    for no_out_channel in config.no_out_channel:
        if no_out_channel <= 0:
            raise RuntimeError("no_out_channel should be greater than or equal to 1")
    if config.num_workers<=0: 
        config.num_workers = os.cpu_count()
    if config.prefetch_factor <= 0:
       config.prefetch_factor = 2
    if len(config.optim.lr)==1:
        config.optim.lr = [config.optim.lr[0]]*3
    if "LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_WORLD_SIZE" in os.environ:
        config.ddp = True
    if config.ddp and not ("LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_WORLD_SIZE" in os.environ):
        raise RuntimeError("--ddp specified but ddp environmental variables not available; remember to run with torchrun if using ddp.")
    num_tasks = len(config.tasks)
    if len(config.height) != num_tasks:
        raise ValueError(f"Number of heights ({len(config.height)}) does not match number of tasks ({num_tasks})")
    if len(config.width) != num_tasks:
        raise ValueError(f"Number of widths ({len(config.width)}) does not match number of tasks ({num_tasks})")
    if len(config.time) != num_tasks:
        raise ValueError(f"Number of times ({len(config.time)}) does not match number of tasks ({num_tasks})")
    if len(config.no_in_channel) != num_tasks:
        raise ValueError(f"Number of no_in_channels ({len(config.no_in_channel)}) does not match number of tasks ({num_tasks})")
    if len(config.no_out_channel) != num_tasks:
        raise ValueError(f"Number of no_out_channels ({len(config.no_out_channel)}) does not match number of tasks ({num_tasks})")
    if len(config.split_csv_path) != num_tasks:
        if config.split_csv_path[0] is None: config.split_csv_path = [None]*num_tasks
        else: raise ValueError(f"Number of split_csv_paths ({len(config.split_csv_path)}) does not match number of tasks ({num_tasks})")
    if len(config.pre_component_load_path) != num_tasks:
        if config.pre_component_load_path[0] is None: config.pre_component_load_path = [None]*num_tasks
        else: raise ValueError(f"Number of pre_component_load_paths ({len(config.pre_component_load_path)}) does not match number of tasks ({num_tasks})")
    if len(config.post_component_load_path) != num_tasks:
        if config.post_component_load_path[0] is None: config.post_component_load_path = [None]*num_tasks
        else: raise ValueError(f"Number of post_component_load_paths ({len(config.post_component_load_path)}) does not match number of tasks ({num_tasks})")
    if len(config.use_patches) != num_tasks:
        if config.use_patches[0] == False: config.use_patches = [False]*num_tasks
        else: raise ValueError(f"Number of use_patches ({len(config.use_patches)}) does not match number of tasks ({num_tasks})")
    print(config.patch_height)
    if len(config.patch_height) != num_tasks:
        config.patch_height = [config.patch_height[0]]*num_tasks
    if len(config.patch_width) != num_tasks:
        config.patch_width = [config.patch_width[0]]*num_tasks
    if len(config.patch_time) != num_tasks:
        config.patch_time = [config.patch_time[0]]*num_tasks
    if len(config.batch_size) != num_tasks:
        config.batch_size = [config.batch_size[0]]*num_tasks
    if len(config.data_dir) != num_tasks:
        raise ValueError(f"Number of data_dirs ({len(config.data_dir)}) does not match number of tasks ({num_tasks})")
    if len(config.task_type)!=num_tasks:
        raise ValueError(f"Number of task_types ({len(config.task_type)}) does not match number of tasks ({num_tasks})")
    if len(config.loss_func)!=num_tasks:
        raise ValueError(f"Number of loss_func ({len(config.loss_func)}) does not match number of tasks ({num_tasks})")
    if len(config.affine_aug)!=num_tasks:
        config.affine_aug = [config.affine_aug[0]]*num_tasks
    if len(config.brightness_aug)!=num_tasks:
        config.brightness_aug = [config.brightness_aug[0]]*num_tasks
    if len(config.gaussian_blur_aug)!=num_tasks:
        config.gaussian_blur_aug = [config.gaussian_blur_aug[0]]*num_tasks
    if len(config.pre_component)!=num_tasks:
        raise ValueError(f"Number of pre_components ({len(config.pre_component)}) does not match number of tasks ({num_tasks})")
    if len(config.post_component)!=num_tasks:
        raise ValueError(f"Number of post_components ({len(config.post_component)}) does not match number of tasks ({num_tasks})")
    if len(config.freeze_pre)!=num_tasks:
        config.freeze_pre = [config.freeze_pre[0]]*num_tasks
    if len(config.freeze_post)!=num_tasks:
        config.freeze_post = [config.freeze_post[0]]*num_tasks
    if not os.path.exists(config.wandb_dir): 
        os.makedirs(config.wandb_dir, exist_ok=True)
    if 'ViT' in config.backbone_component:
        assert len(config.ViT.patch_size) in [1, 3], "ViT patch size must have 1 or 3 elements"
        if len(config.ViT.patch_size)==1:
            config.ViT.patch_size = [config.ViT.patch_size[0]]*3
    if 'SWIN' in config.backbone_component:
        assert len(config.SWIN.patch_size) in [1, 3], "SWIN patch size must have 1 or 3 elements"
        if len(config.SWIN.patch_size)==1:
            config.SWIN.patch_size = [config.SWIN.patch_size[0]]*3
        assert len(config.SWIN.window_size) in [1, 3], "SWIN window size must have 1 or 3 elements"
        if len(config.SWIN.window_size)==1:
            config.SWIN.window_size = [config.SWIN.window_size[0]]*3
    if 'omnivore' in config.backbone_component:
        assert len(config.omnivore.patch_size) in [1, 3], "omnivore patch size must have 1 or 3 elements"
        if len(config.omnivore.patch_size)==1:
            config.omnivore.patch_size = [config.omnivore.patch_size[0]]*3
        assert len(config.omnivore.window_size) in [1, 3], "omnivore window size must have 1 or 3 elements"
        if len(config.omnivore.window_size)==1:
            config.omnivore.window_size = [config.omnivore.window_size[0]]*3
    if config.inference_only:
        assert config.inference_dir not in [None, "None", "none"], "If inference_only is specified, user must provide an inference_dir"
        assert os.path.exists(config.inference_dir), f"inference_dir ({config.inference_dir}) does not exist"




