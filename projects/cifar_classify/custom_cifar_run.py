"""
Custom cifar run file 
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
from custom_cifar_dataset import cifar_dataset


# -------------------------------------------------------------------------------------------------
def main():

    # Parse args to config (no customization)
    config = parse_config_and_setup_run() 

    # Define datasets using custom cifar dataset
    train_set = cifar_dataset(config=config, split='train') 
    val_set = cifar_dataset(config=config, split='val') 
    test_set = cifar_dataset(config=config, split='test') 
    
    # Get loss function (no customization)
    loss_f = get_loss_func(config=config) 

    # Define model (no customization)
    model_manager = ModelManager(config=config) 

    # Create optimizer and scheduler (no customization)
    optim_manager = OptimManager(config=config, model_manager=model_manager, train_set=train_set)

    # Create MetricManager, which tracks metrics and checkpoints models during training (no customization)
    metric_manager = MetricManager(config=config)

    # Create trainer, which will manage model training (no customization)
    trainer = TrainManager(config=config,
                           train_sets=train_set,
                           val_sets=val_set,
                           test_sets=test_set,
                           loss_f=loss_f,
                           model_manager=model_manager,
                           optim_manager=optim_manager,
                           metric_manager=metric_manager)

    # Execute training on task (no customization)
    trainer.train()
    
# -------------------------------------------------------------------------------------------------
if __name__=="__main__":    
    main()
