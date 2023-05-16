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

sh ./run_sweep.sh -g 0 -n 1 -d cifar10 -s $sweep_id -r 100 -p 9001 # single GPU, single process training

```

The [run_sweep](./run_sweep.sh) script will start a sweeping run on selected gpus (-g) for a node, by calling `torchrun --standalone --nproc-per-node $nproc_per_node ./main_cifar.py ...`.

**Warning** Current wandb sweep does not work well with toruchrun for multi-gpu training. So current solution is good for one gpu and one process usecage (e.g. -g 0 -n 1).

### multi-node training

The ```torchrun``` or ```python -m torch.distributed.launch --use-env ...``` can be called to perform the multi-node training. To run the multi-node training on a cluster of 4 nodes, 

```
# on master node (fsi1)
python3 ./cifar10/run_imagenet.py --nproc-per-node 4 --nnodes 4 --node_rank 0 --rdzv_id 100 --rdzv_endpoint 172.16.0.4:9001 

# on other nodes
python3 ./cifar10/run_imagenet.py --nproc-per-node 4 --nnodes 4 --node_rank 1 --rdzv_id 100 --rdzv_endpoint 172.16.0.4:9001
python3 ./cifar10/run_imagenet.py --nproc-per-node 4 --nnodes 4 --node_rank 2 --rdzv_id 100 --rdzv_endpoint 172.16.0.4:9001
python3 ./cifar10/run_imagenet.py --nproc-per-node 4 --nnodes 4 --node_rank 3 --rdzv_id 100 --rdzv_endpoint 172.16.0.4:9001
```