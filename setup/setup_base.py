"""
Setup file, which parses the args to a config and does initial setup of logger, seed, etc.
"""

import os
import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from parsers import *
from config_utils import *
from setup_utils import *

def parse_config(custom_parser=None):
    """
    Parse config 

    @args:
        - custom_parser (parser): contains parser for any project-specific, custom args
    @rets:
        - full_config (Nested Namespace): the config created from parsing input args
    """
    ############################### PARSE ARGS TO CONFIG ###############################
    # Parse general args 
    general_config, unknown_args_main = general_parser().parser.parse_known_args(namespace=Nestedspace())

    # Parse task args
    task_config, unknown_args_task = task_parser(task_types=general_config.task_type).parser.parse_known_args(namespace=Nestedspace())

    # Parse optimizer args
    optim_config, unknown_args_optim = optim_parser(optim_type=general_config.optim_type).parser.parse_known_args(namespace=Nestedspace())

    # Parse scheduler args
    sched_config, unknown_args_sched = sched_parser(scheduler_type=general_config.scheduler_type).parser.parse_known_args(namespace=Nestedspace())

    # Parse model args
    model_config, unknown_args_model = model_parser(model_type=general_config.backbone_component).parser.parse_known_args(namespace=Nestedspace())

    # Parse project-specific args from the path in general_config.custom_parser, if specified
    if custom_parser is not None:
        project_config, unknown_args_project = custom_parser().parser.parse_known_args(namespace=Nestedspace())
    else: 
        project_config, unknown_args_project = Nestedspace(), []

    # Combine all parsed args
    full_config = Nestedspace(**vars(general_config), 
                              **vars(task_config),
                              **vars(optim_config),
                              **vars(sched_config),
                              **vars(model_config),
                              **vars(project_config))
    
    # Raise error if there are args that are not recognized by parsers
    unknown_args = set([v.split("--")[-1].split("=")[0] for v in unknown_args_main + unknown_args_task + unknown_args_optim + unknown_args_sched + unknown_args_model + unknown_args_project if "--" in v])
    unknown_args = check_for_unknown_args(full_config, unknown_args)
    if len(unknown_args)>0: 
        raise NameError(f'User input arguments that are not recognized: {str(unknown_args)}')
    
    return full_config

def parse_config_and_setup_run(custom_parser=None):
    """
    Parse args to config and setup the run
    @args:
        - custom_parser (parser): contains parser for any project-specific, custom args
    @rets:
        - full_config (Nested Namespace): the config created from parsing input args
    """

    ############################### PARSE ARGS TO CONFIG ###############################
    full_config = parse_config(custom_parser=custom_parser)
    
    # Replace config with values from an existing yaml file, if yaml file is specified
    if full_config.yaml_load_path is not None:
        full_config = yaml_to_config(full_config.yaml_load_path,full_config.log_dir,full_config.run_name)

    # Replace config with values from an existing yaml file, if inference_dir is specified, and adjust relevant config params for inference only
    if full_config.inference_only:
        assert full_config.inference_dir not in [None, "None", "none"], "If inference_only is True, inference_dir must be specified"
        assert os.path.exists(full_config.inference_dir), f"inference_dir {full_config.inference_dir} does not exist"

        inference_dir = full_config.inference_dir
        inference_only = full_config.inference_only

        full_config = yaml_to_config(os.path.join(full_config.inference_dir,'config.yaml'),new_log_dir=full_config.inference_log_dir,new_run_name=full_config.inference_run_name)
        full_config.inference_dir = inference_dir
        full_config.inference_only = inference_only
        full_config.train_model = False
        full_config.pre_component_load_path = [None]*len(full_config.tasks)
        full_config.post_component_load_path = [None]*len(full_config.tasks)
        full_config.backbone_component_load_path = None
        full_config.entire_model_load_path = os.path.join(full_config.inference_dir, 'entire_models', 'entire_model_best_checkpoint.pth')
        full_config.eval_train_set = False
        full_config.eval_val_set = True
        full_config.eval_test_set = True
        full_config.save_train_samples = False
        full_config.save_val_samples = False
        full_config.save_test_samples = True

    # Check args 
    check_args(full_config)

    ############################### RUN SETUP FUNCTIONS ###############################
    # Run a few setup functions that are used throughout the remainder of training...
    setup_run(full_config)

    return full_config
