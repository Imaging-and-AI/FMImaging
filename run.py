"""
Standard run file 
"""

from setup.setup_base import parse_config_and_setup_run
from model.model_base import ModelComponent, ModelManager
from optim.optim_base import OptimManager
from trainer.trainer_base import TrainManager
from metrics.metrics_base import MetricManager
from task.task_base import TaskManager

# -------------------------------------------------------------------------------------------------
def main():
    
    # -----------------------------
    # Parse input arguments to config
    config = parse_config_and_setup_run()
    
    # -----------------------------
    # Define tasks - contains the pre and post heads, loss functions, and datasets
    tasks = {}
    for task_ind, task_name in enumerate(config.tasks):
        tasks[task_name] = TaskManager(config, task_name, task_ind)
    
    # -----------------------------
    # Instatiate the model - must do this in sequence by creating the pre components first, then the backbone, then the post components
    for task in tasks.values(): task.create_pre_component() 
    backbone_component = ModelComponent(config=config,
                                        component_name=config.backbone_component,
                                        input_feature_channels=task.pre_component.output_feature_channels)
    for task in tasks.values(): task.create_post_component(backbone_component.output_feature_channels)
        
    # -----------------------------
    # Create a ModelManager, which defines the forward pass and connects pre->backbone->post 
    model_manager = ModelManager(config, tasks, backbone_component)
    
    # -----------------------------
    # Load model weights, if a load path is specified
    if config.full_model_load_path is not None:
        model_manager.load_entire_model(config.full_model_load_path, device=config.device)
    for task_ind, task in enumerate(tasks.values()): 
        task.load(pre_load_path=config.pre_component_load_path[task_ind], post_load_path=config.post_component_load_path[task_ind])
    if config.backbone_component_load_path is not None:
        backbone_component.load(config.backbone_component_load_path)

    # -----------------------------
    # Create OptimManager, which defines optimizers and schedulers
    optim_manager = OptimManager(config, model_manager, tasks)

    # -----------------------------
    # Create MetricManager, which tracks metrics and checkpoints models during training
    metric_manager = MetricManager(config)

    # -----------------------------
    # Create TrainManager, which will control model training
    train_manager = TrainManager(config,
                                model_manager,
                                optim_manager,
                                metric_manager)
    
    # -----------------------------
    # Execute training and evaluation
    train_manager.run()

    # -----------------------------
    # Final saves of backbone and tasks
    for task_name, task in tasks.items():
        task.save(save_name_modifier='final')
    backbone_component.save(model_save_name='backbone/final_backbone_component')
    model_manager.save_entire_model(optim_manager.curr_epoch, optim_manager.epoch, optim_manager.sched, 'final_model')


# -------------------------------------------------------------------------------------------------
if __name__=="__main__":    
    main()


"""
Log directory
/path/to/log/dir 
    entire model
        entire_model.pth
    backbone
        backbone_ckpt.pth
    tasks
        task1
            pre.pth
            post.pth
            task_config.yaml
        task2
            pre.pth
            post.pth
            task_config.yaml
        task3
            pre.pth
            post.pth
            task_config.yaml
    config.yaml
    log.txt
    metrics.txt
    saved_samples
        train
        val
        test

Load functionality
Load 1: give /path/to/log/dir, load entire model
Load 2: give path for each task and backbone
Potential future Load 3: you could give a /path/to/log/dir, glob all of the task/backbones, and load instead of using entire model

"""


