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
  loader.add_constructor("tag:yaml.org,2002:python/object:config.config.Nestedspace", nestedspace_constructor)
  return loader

def config_to_yaml(config, save_path):
    """Save nestedspace config to yaml file."""
    if not os.path.exists(save_path): os.makedirs(save_path)
    yaml_file = os.path.join(save_path,'config.yaml')
    print(f"--> save config as yaml at {yaml_file}")
    with open(yaml_file, "w", encoding = "utf-8") as yaml_file:
        dump = yaml.dump(config, default_flow_style = False, allow_unicode = True, encoding = None)
        yaml_file.write(dump)
        
    return yaml_file

def yaml_to_config(yaml_path, new_log_dir, new_run_name):
    """Load yaml into nestedspace config"""
    with open(yaml_path, "r") as yaml_file:
        saved_config = yaml.load(yaml_file, Loader=get_nestedspace_loader())
        saved_config.log_dir = new_log_dir # Modify the log path to a new directory so we don't overwrite anything
        saved_config.new_run_name = new_run_name # Modify the log path to a new directory so we don't overwrite anything
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
    save_path = os.path.join(config.log_dir, config.run_name)
    if os.path.exists(save_path) and not config.override:
        raise RuntimeError(f"User specified a log_dir ({config.log_dir}) and run_name ({config.run_name}) that already exist; either specify a different run name or override current results with --override")
    if not os.path.exists(config.data_dir):
        raise RuntimeError(f"User specified a data_dir ({config.data_dir}) that does not exist")
    if config.split_csv_path is not None:
        if not os.path.exists(config.split_csv_path):
            raise RuntimeError(f"User specified a split_csv_path ({config.split_csv_path}) that does not exist")
    if config.pre_model_load_path is not None:
        if not os.path.exists(config.pre_model_load_path):
            raise RuntimeError(f"User specified a pre_model_load_path ({config.pre_model_load_path}) that does not exist")
    if config.backbone_model_load_path is not None:
        if not os.path.exists(config.backbone_model_load_path):
            raise RuntimeError(f"User specified a backbone_model_load_path ({config.backbone_model_load_path}) that does not exist")
    if config.post_model_load_path is not None:
        if not os.path.exists(config.post_model_load_path):
            raise RuntimeError(f"User specified a post_model_load_path ({config.post_model_load_path}) that does not exist")
    if config.yaml_load_path is not None:
        if not os.path.exists(config.yaml_load_path):
            raise RuntimeError(f"User specified a yaml_load_path ({config.yaml_load_path}) that does not exist")
    if config.height <= 0:
        raise RuntimeError("Height should be greater than or equal to 1")
    if config.width <= 0:
        raise RuntimeError("Width should be greater than or equal to 1")
    if config.time <= 0:
        raise RuntimeError("Time should be greater than or equal to 1")
    if config.no_in_channel <= 0:
        raise RuntimeError("no_in_channel should be greater than or equal to 1")
    if config.no_out_channel <= 0:
        raise RuntimeError("no_out_channel should be greater than or equal to 1")
    if config.num_workers<=0: 
        config.num_workers = os.cpu_count()
    if config.prefetch_factor <= 0:
       config.prefetch_factor = 2


