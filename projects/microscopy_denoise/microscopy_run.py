"""
Microscopy run file 
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

# Custom Microscopy functions
from projects.microscopy_denoise.microscopy_parser import microscopy_parser
from projects.microscopy_denoise.microscopy_dataset import load_microscopy_data_all
from projects.microscopy_denoise.microscopy_loss import microscopy_loss 
from projects.microscopy_denoise.microscopy_model import microscopy_ModelManager
from projects.microscopy_denoise.microscopy_metrics import MicroscopyMetricManager
from projects.microscopy_denoise.microscopy_trainer import MicroscopyTrainManager


# -------------------------------------------------------------------------------------------------
def main():

    config = parse_config_and_setup_run(microscopy_parser) 

    train_set, val_set, test_set = load_microscopy_data_all(config=config)
    
    loss_f = microscopy_loss(config=config)

    model_manager = microscopy_ModelManager(config=config) 
    # Load model if specified
    model_manager.load()

    # Create optimizer and scheduler
    optim_manager = OptimManager(config=config, model_manager=model_manager, train_set=train_set)

    # Create MetricManager, which tracks metrics and checkpoints models during training
    metric_manager = MicroscopyMetricManager(config=config)

    # Create trainer, which will manage model training
    trainer = MicroscopyTrainManager(config=config,
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
