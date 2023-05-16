# Examples

This folder archives useful example code for model training.

## Multi-node training with model parallel

To perform the multi-node, multi-gpu training, the first task is to determine how many GPUS a copy of the model will use. For example, if 1 GPU is sufficient to hold the model, for a 4x A100 node, nproc_per_node can be 4. This corresponds to starting 4 processes for one node. Every process will hold a copy of the model and run on one GPU. In this case, the local_rank will be [0, 1, 2, 3]. If we have 4 nodes, the world-size will be 16 (number of total number of processes). The local world_size is 4.

If a model requires two GPUs to hold, we have to set nproc_per_node to be 2. This corresponds to starting 2 processes for one node. Every process will hold a copy of the model and run on two GPU. In this case, the local_rank will be [0, 1]. If we have 4 nodes, the world-size will be 8 (number of total number of processes). The local world_size is 2. The key difference is for the local rank 0, this process will use two GPUs (e.g. gpu_id=0,1). For the local rank 1, gpu_id will be 2,3. 

Only the local_rank is used to compute the gpu_id and used to set the `device` parameters for all layers.

One example can be seen at `example.py`

This file demos how to perform the multi-node training with model parallel:

```
# on gt7
torchrun --nproc_per_node 1 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 multinode.py

# on gt3
torchrun --nproc_per_node 1 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 multinode.py
```

```
# fsi 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 1 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:8905 multinode.py --local_world_size 1
# fsi 4
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 1 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:8905 multinode.py --local_world_size 1

```

## Multi-GPU DDP training with wandb sweep

DDP training creates multiple processes. Each process holds a copy of the model. Wandb sweep needs to request a set of tuneable parameters from the wandb service and changes the model and training configuration. 

For a ddp training to create N processes, only one process should request testing parameters from wandb and syncs this parameter set to all other processes:

```
if config_default.ddp:      
    # get the local rank
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    # use rank=0 process as the master
    is_master = rank<=0
    dist.init_process_group("nccl")   

    # create a key-value store to sync processes     
    store=dist.TCPStore("localhost", 9001, dist.get_world_size(), is_master=is_master, timeout=timedelta(seconds=30), wait_for_workers=True)
else:
    rank = -1
    
print(f"Start training on rank {rank}.")

wandb_run = None

if(config_default.sweep_id != 'none'):
    if rank<=0:
        # only request new parameter for process 0
        print(f"---> get the config from wandb on local rank {rank}")
        wandb_run = wandb.init(entity=config_default.wandb_entity)

        # update parameter configuration with requested para set
        config = set_up_config_for_sweep(wandb.config, config_default)   
        
        # at the process 0, add parameter to the store
        config_str = pickle.dumps(config)     
        store.set("config", config_str)
    else:
        # for all other processes, request the parameter set
        print(f"---> get the config from key store on local rank {rank}")
        config_str = store.get("config")
        config = pickle.loads(config_str)
else:
    # if we are not running sweep, no need to sync parameters
    if rank<=0:
        # Config is a variable that holds and saves hyperparameters and inputs
        config = config_default
        wandb_run = wandb.init(project=config.project, 
                entity=config.wandb_entity, 
                config=config, 
                name=config.run_name, 
                notes=config.run_notes)

# wait for all processes get the config
if config.ddp:                        
    dist.barrier()

print(f"---> config synced for the local rank {rank}")
```
