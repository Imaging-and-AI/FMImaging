# MRI image enhancement training

This project provides model and training code for MRI image enhancement training. 

### wandb sweep

To run the parameter sweep, a sweep configuration file is needed (e.g. [sweep_conf](./sweep_conf.py)). 

```
# first, generate the sweep and record the sweep in
python3 ./sweep_conf.py

# second, run the sweep

# run on 2 gpus, with 2 processes
sh ./run_sweep.sh -g 0,1 -n 2 -s $sweep_id  -r 100 -p 9001

# run on 4 gpus, with 4 processes
sh ./run_sweep.sh -g 0,1,2,3 -n 4 -s $sweep_id  -r 100 -p 9001

# run on one gpu
sh ./run_sweep.sh -g 0 -n 1 -s $sweep_id -r 100 -p 9001

```

**Warning** Current wandb sweep does not work well with toruchrun for multi-gpu training. So current solution is good for one gpu and one process usecage (e.g. -g 0 -n 1).

### multi-node training

on the local set up

- single node, single gpu training
```
torchrun --nproc_per_node 1 --standalone $HOME/mrprogs/STCNNT.git/mri/main_mri.py --ddp

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 1 --standalone

```

- single node, multiple gpu training
```
torchrun --nproc_per_node 2 --standalone $HOME/mrprogs/STCNNT.git/mri/main_mri.py --ddp

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 2 --standalone

```

- two nodes, multiple gpu training
```
# every node has two processes, two nodes are trained together
# gt7
torchrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 mri/main_mri.py --ddp
# gt3
torchrun --nproc_per_node 2 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 mri/main_mri.py --ddp

# gt7
python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 2 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001
# gt3
python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 2 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001


```

- four nodes, on cloud
```

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 4 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 4 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 4 --node_rank 2 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 4 --node_rank 3 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

- eight nodes, on cloud

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 2 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 3 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 4 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 5 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 6 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 7 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

```

## Use the run shell on the VM cluster
```
sh /home/gtuser/mrprogs/STCNNT.git/mri/run_cloud.sh -d 16 -e 172.16.0.4 -n 4 -p 9001 -r 100
```