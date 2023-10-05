"""
Standard run file 
"""

from setup.setup_base import parse_config_and_setup_run
from data.data_base import NumpyDataset
from loss.loss_base import get_loss_func
from model.model_base import ModelManager
from optim.optim_base import OptimManager
from trainer.trainer_base import TrainManager
from metrics.metrics_base import MetricManager

# -------------------------------------------------------------------------------------------------
def main():
    
    # Parse input arguments to config
    config = parse_config_and_setup_run()

    # Define train, val, and test datasets
    train_set = NumpyDataset(config=config, split='train')
    val_set = NumpyDataset(config=config, split='val')
    test_set = NumpyDataset(config=config, split='test')
    
    # Define loss function
    loss_f = get_loss_func(config=config) 

    # Create a ModelManager
    model_manager = ModelManager(config=config)
    
    # load model if needed
    model_manager.load()
    
    # Create OptimManager, which defines optimizers and schedulers
    optim_manager = OptimManager(config=config, model_manager=model_manager, train_set=train_set)

    # Create MetricManager, which tracks metrics and checkpoints models during training
    metric_manager = MetricManager(config=config)

    # Create TrainManager, which will control model training
    trainer = TrainManager(config=config,
                           train_sets=train_set,
                           val_sets=val_set,
                           test_sets=test_set,
                           loss_f=loss_f,
                           model_manager=model_manager,
                           optim_manager=optim_manager,
                           metric_manager=metric_manager)
    
    # Execute training
    trainer.train()
    
# -------------------------------------------------------------------------------------------------
if __name__=="__main__":    
    main()
