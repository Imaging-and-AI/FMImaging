
# Train with torchrun

```
torchrun --standalone --nproc_per_node 8 ./projects/qperf/run.py --batch_size 512 --data_dir /data/qperf/mat --log_dir /data/qperf/cache --n_layer 16 --use_pos_embedding --project QPerf --clip_grad_norm 0.1 --override --ddp --use_amp --optim.weight_decay 0.1
```

```
torchrun --standalone --nproc_per_node 8 ./projects/qperf/run.py --batch_size 1024 --data_dir /data/qperf/mat --log_dir /data/qperf/cache --n_layer 8 --use_pos_embedding --project QPerf --clip_grad_norm 0.1 --override --ddp --optim.weight_decay 0.1 --num_epochs 10 --use_amp
```