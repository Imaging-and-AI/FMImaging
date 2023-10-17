
# Train with torchrun


```
torchrun --standalone --nproc_per_node 8 ./projects/qperf/run.py --batch_size 4096 --data_dir /data/qperf/new_data --log_dir /data/qperf/new_data/cache --n_layer 8 --use_pos_embedding --project QPerf --clip_grad_norm 0.1 --override --ddp --optim_type sophia --optim.weight_decay 0.1 --num_epochs 10 --prefetch_factor 64 --use_amp --foot_to_end --run_name qperf_new_data --losses mse l1 gauss --loss_weights 1.0 10.0 10.0 --loss_weights_params 1.0 0.1 0.1 0.1 5.0 --max_samples -1

torchrun --standalone --nproc_per_node 8 ./projects/qperf/run.py --batch_size 1024 --data_dir /data/qperf/new_data --log_dir /data/qperf/new_data/cache --n_layer 8 --use_pos_embedding --project QPerf --clip_grad_norm 0.1 --override --ddp --optim.weight_decay 0.1 --num_epochs 10 --prefetch_factor 64 --pre_model_load_path /data/qperf/cache/project_18-30-41-20231013/best_checkpoint_epoch_2_pre.pth --backbone_model_load_path /data/qperf/cache/project_18-30-41-20231013/best_checkpoint_epoch_2_backbone.pth --post_model_load_path /data/qperf/cache/project_18-30-41-20231013/best_checkpoint_epoch_2_post.pth --load_optim_and_sched True --foot_to_end
```