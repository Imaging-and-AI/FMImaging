# Examples

This folder archives useful example code for model training.

## Multi-node training with model parallel

example.py

This file demos how to perform the multi-node training with model parallel:

```
# on gt7
torchrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 multinode.py

# on gt3
torchrun --nproc_per_node 2 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 multinode.py
```
