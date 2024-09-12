#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

export run_dir=/tmp/cifar
mkdir -p $run_dir
mkdir -p $run_dir/data

python3 ./projects/cifar_classify/custom_cifar_run.py --project cifar_demo --run_name cifar_demo --log_dir ${run_dir}/logs --data_dir ${run_dir}/data --backbone_component ViT --ViT.patch_size 1 8 8 --post_component ViTLinear --height 128 --width 128 --time 1 --no_in_channel 3 --no_out_channel 10 --time 1 --task_type=class --optim_type=adam --scheduler_type=OneCycleLR --loss_func=CrossEntropy --num_workers=8 --num_epochs=16 --batch_size=32 --clip_grad_norm=1.0 --optim.lr=1e-4 --optim.weight_decay=0 --override