#!/bin/bash

# script to run batch training

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name all_temporal --att_types temporal temporal temporal --run_notes 3_temporal_stack

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name all_local --att_types local local local --run_notes 3_local_stack

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name all_global --att_types global global global --run_notes 3_global_stack

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name lgt --att_types local global temporal --run_notes local_global_temporal

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name ltg --att_types local temporal global --run_notes local_temporal_global

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name glt --att_types global local temporal --run_notes global_local_temporal

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name gtl --att_types global temporal local --run_notes global_temporal_local

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name tlg --att_types temporal local global --run_notes temporal_local_global

python3 cifar10/main_cifar.py --data_root /home/rehmana2/projects/STCNNT_2/cifar10 \
    --check_path /home/rehmana2/projects/STCNNT_2/checkpoints --model_path /home/rehmana2/projects/STCNNT_2/models \
    --log_path /home/rehmana2/projects/STCNNT_2/logs --results_path /home/rehmana2/projects/STCNNT_2/results \
    --time 1 --height 32 --width 32 --num_epoch 100 --batch_size 128 --device cuda:0 \
    --run_name tgl --att_types temporal global local --run_notes temporal_global_local
