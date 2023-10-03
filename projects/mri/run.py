"""
Custom run file 
"""

import sys
from pathlib import Path
Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))
Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

# Default functions
from setup.setup_base import parse_config_and_setup_run
from data.data_base import NumpyDataset
from loss.loss_base import get_loss_func
from model.model_base import ModelManager
from optim.optim_base import OptimManager
from trainer.trainer_base import TrainManager
from metrics.metrics_base import MetricManager

# Custom functions
from custom_parser import custom_parser
from custom_dataset import custom_dataset
from custom_loss import custom_loss 
from custom_model import custom_ModelManager


# -------------------------------------------------------------------------------------------------
def main():
    
    """
    Parse input arguments to config
    To customize args: define a custom parser class and pass it to parse_to_config()
        ex code: config = parse_to_config(custom_parser)
    To use default args: just call parse_to_config with no args 
        ex code: config = parse_to_config()
    Rules for custom parser in custom_parser.py
    """
    config = parse_config_and_setup_run(custom_parser) 


    """
    Define train, val, and test datasets
    To customize datasets: define a function with a custom dataset and initialize it here
        ex code:
        train_set = custom_dataset(config=config, split='train') 
        val_set = custom_dataset(config=config, split='val') 
        test_set = custom_dataset(config=config, split='test') 
        Note: these can be a list of datasets if desired, e.g., train_set = [custom_dataset1(), custom_dataset2()]
    To use default datasets: 
        ex code:
        train_set = NumpyDataset(config=config, split='train')
        val_set = NumpyDataset(config=config, split='val')
        test_set = NumpyDataset(config=config, split='test')
    Rules for custom dataset in custom_dataset.py
    """
    train_set = custom_dataset(config=config, split='train') 
    val_set = custom_dataset(config=config, split='val') 
    test_set = custom_dataset(config=config, split='test') 
    

    """
    Define loss function
    To customize loss function: define a function with a custom loss and initialize it here
        ex code: loss_f = custom_loss(config=config)
    To use default loss: 
        ex code: loss_f = get_loss_func(config=config) 
    Rules for custom loss in custom_loss.py
    """
    loss_f = custom_loss(config=config)


    """
    Create a ModelManager
    To customize ModelManager (e.g., to test a new architecture or change forward function): inheret the ModelManager class and make customizations
        ex code: model = custom_ModelManager(config=config)
    To use default ModelManager: 
        ex code: model = ModelManager(config=config)
    Rules for custom model in custom_model.py
    """
    model_manager = custom_ModelManager(config=config) 


    """
    Create a OptimManager, MetricManager, and TrainManager.
    MetricManager can be customized by inhereting MetricManager class, similar to ModelManager.
    OptimManager and TrainManager shouldn't need customizations, but if desired they can be inhereted and customized similarly to ModelManager.
    """
    # Create optimizer and scheduler
    optim_manager = OptimManager(config=config, model_manager=model_manager, train_set=train_set)

    # Create MetricManager, which tracks metrics and checkpoints models during training
    metric_manager = MetricManager(config=config)

    # Create trainer, which will manage model training
    trainer = TrainManager(config=config,
                           train_sets=train_set,
                           val_sets=val_set,
                           test_sets=test_set,
                           loss_f=loss_f,
                           model_manager=model_manager,
                           optim_manager=optim_manager,
                           metric_manager=metric_manager)

    # Execute training on task
    trainer.train()
    
# -------------------------------------------------------------------------------------------------
if __name__=="__main__":    
    main()
