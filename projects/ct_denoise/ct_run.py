"""
CT run file 
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

# Custom CT functions
from ct_parser import ct_parser
from ct_dataset import load_ct_data_all
from ct_loss import ct_loss 
from ct_model import ct_ModelManager
from ct_metrics import CTMetricManager
from ct_trainer import CTTrainManager


# -------------------------------------------------------------------------------------------------
def main():

    config = parse_config_and_setup_run(ct_parser) 

    train_set, val_set, test_set = load_ct_data_all(config=config)
    
    loss_f = ct_loss(config=config)

    model_manager = ct_ModelManager(config=config) 
    # Load model if specified
    model_manager.load()

    # Create optimizer and scheduler
    optim_manager = OptimManager(config=config, model_manager=model_manager, train_set=train_set)

    # Create MetricManager, which tracks metrics and checkpoints models during training
    metric_manager = CTMetricManager(config=config)

    # Create trainer, which will manage model training
    trainer = CTTrainManager(config=config,
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
