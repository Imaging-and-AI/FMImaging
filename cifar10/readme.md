# Cifar and ImageNet training

This project provides model and training code for cifar and imagenet datasets. This servers as a testing bed for backbone models.

### wandb sweep

To run the parameter sweep, a sweep configuration file is needed (e.g. [sweep_conf](./sweep_conf.py)). 

```
# first, generate the sweep and record the sweep in
python3 ./sweep_conf.py

# second, run the sweep

sh ./run_sweep.sh -g 0,1 -n 2 -d cifar10 -s $sweep_id

sh ./run_sweep.sh -g 0,1,2,3 -n 4 -d imagenet -s $sweep_id

```

The [run_sweep](./run_sweep.sh) script will start a sweeping run on selected gpus (-g) for a node, by calling `torchrun --standalone --nproc-per-node $nproc_per_node ./main_cifar.py ...`.
