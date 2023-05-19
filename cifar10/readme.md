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

The [run_sweep](./run_sweep.sh) script will start a sweeping run on selected gpus (-g) for a node, by calling `torchrun --standalone --nproc_per_node $nproc_per_node ./main_cifar.py ...`.

**Warning** Current wandb sweep does not work well with toruchrun for multi-gpu training. So current solution is good for one gpu and one process usecage (e.g. -g 0 -n 1).

### multi-node training

The ```torchrun``` or ```python -m torch.distributed.launch --use-env ...``` can be called to perform the multi-node training. To run the multi-node training on a cluster of 4 nodes, 

on the local set up

- single node, single gpu training
```
torchrun --nproc_per_node 1 --standalone cifar10/main_cifar.py --ddp

python3 ./cifar10/run_cifar.py --nproc_per_node 1 --standalone

python3 ./cifar10/run_imagenet.py --nproc_per_node 1 --standalone

```

- single node, multiple gpu training
```
torchrun --nproc_per_node 2 --standalone cifar10/main_cifar.py --ddp

python3 ./cifar10/run_cifar.py --nproc_per_node 2 --standalone

python3 ./cifar10/run_imagenet.py --nproc_per_node 2 --standalone

```

- two nodes, multiple gpu training
```
# every node has two processes, two nodes are trained together
# gt7
torchrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 cifar10/main_cifar.py --ddp
# gt3
torchrun --nproc_per_node 2 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 cifar10/main_cifar.py --ddp

# gt7
python3 ./cifar10/run_cifar.py --nproc_per_node 2 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001
# gt3
python3 ./cifar10/run_cifar.py --nproc_per_node 2 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001

# gt7
python3 ./cifar10/run_imagenet.py --nproc_per_node 2 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001
# gt3
python3 ./cifar10/run_imagenet.py --nproc_per_node 2 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001

```

- four nodes, on cloud
```
# imagenet
python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 4 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 4 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 4 --node_rank 2 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 4 --node_rank 3 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

# imagenet
python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 8 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 8 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 8 --node_rank 2 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 8 --node_rank 3 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 8 --node_rank 4 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 8 --node_rank 5 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 8 --node_rank 6 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_imagenet.py --nproc_per_node 4 --nnodes 8 --node_rank 7 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

# cifar
python3 ./cifar10/run_cifar.py --nproc_per_node 4 --nnodes 4 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_cifar.py --nproc_per_node 4 --nnodes 4 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_cifar.py --nproc_per_node 4 --nnodes 4 --node_rank 2 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 ./cifar10/run_cifar.py --nproc_per_node 4 --nnodes 4 --node_rank 3 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001


```

## Start/stop VMs

Install az cli:
```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login --use-device-code

```

```
rg=xueh2-a100-eastus2

for n in node1 node2 node3 node4 node5 node6 node7 node8
do
    echo "stop node $n ..."
    az vm stop --name $n -g $rg
    az vm deallocate --name $n -g $rg
done

for n in node1 node2 node3 node4 node5 node6 node7 node8
do
    echo "start node $n ..."
    az vm start --name $n -g $rg
done

for n in fsi{1..8}
do
    echo "update node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "git clone git@github.com:AzR919/STCNNT.git /home/gtuser/mrprogs/STCNNT.git"
done

for n in fsi{1..8}
do
    echo "update node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "cd /home/gtuser/mrprogs/STCNNT.git && git pull"
done



```
