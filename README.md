# Refactor of FMImaging, v1

### Purpose
This is the v1 refactor of the FMImaging codebase, which contains basic utilities for single-task segmentation, classification, and image enhancement with DDP. The purpose of the refactor was to:
  * Reduce the amount of new code needed for new projects, with zero-code solutions for basic applications and lightweight customizations for others.
  * Reduce the amount of rewritten code per project.
  * Make organization clearer, for example by consolidating configs, metrics, optimizers, losses, etc., to keep codebase clean as we continue to add complexity.
  * Prepare consolidated codebase for FM experiments, including:
    * Build in pre/post/backbone structure.
    * Include utils for segmentation, classification, and enhancement tasks.

### Overview
The codebase organizes directories by utility. The ```run.py``` file shows how the codebase progresses:
  * In the ```setup``` dir, args are parsed into a config and initial setup functions are run.
  * In the ```data``` dir, torch datasets are created.
  * In the ```loss``` dir, the loss function is defined.
  * In the ```model``` dir, the Model Manager is defined, which contains the pre/backbone/post modules and utils for saving and loading the model.
  * In the ```optim``` dir, the Optim Manager is defined, which contains the optimizer and scheduler.
  * In the ```metrics``` dir, the Metric Manager is defined, which tracks all performance metrics during training.
  * In the ```trainer``` dir, the Train Manager is defined, which controls DDP and the train/eval loops.

Each project can be stored in the ```projects``` dir. 

These dirs are intended to be modular, so one utility can be customized without needing to rewrite code in other dirs.
We can use this codebase with no customizations, just specifying hyperparameters and variables via the command line and using the default functions. Alternatively, we can supply customizations to each of the above components when needed. Both solutions are described below.

### Warnings
A final few notes on the refactor:
  * There will be bugs. I can help debug, just let me know.
  * I tested the omnivore model most extensively; STCNNT runs but I have not trained it to completion.
  * There are files and functions I took out when I couldn't tell if/how they were used or how to make them general solutions (e.g., running_inference.py, utilities.py, wrapper around ddp cmd). We can put old functions back in, but try to keep the organization of the refactor.
  * There are additional utils we can build out (e.g., adding more args and optimizers, adding more augmentation functions, adding more losses). I put basic utils in. New utils should be relatively easy to add in the config + in the current organization.
  * I have not tested DDP on multiple nodes.

### Using the codebase with no customizations

### Using the codebase with customizations
