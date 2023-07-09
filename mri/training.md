# MRI image enhancement training

# run the second training

```

export BASE_DIR=/data

model=$BASE_DIR/mri/test/complex_model/mri-HRNET-20230621_132139_784364_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_13-22-06-20230621_last.pt

model=$BASE_DIR/mri/models/mri-HRNET-20230621_132139_784364_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_13-22-06-20230621_last.pt

model=$BASE_DIR/mri/models/mri-HRNET-20230702_013521_019623_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_01-35-34-20230702_best.pt


export BASE_DIR=/export/Lab-Xue/projects/

model=$BASE_DIR/mri/models/mri-HRNET-20230702_013521_019623_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_01-35-34-20230702_best.pt


for n in fsi{1..16}
do
    echo "copy to $n ..."
    VM_name=$n.eastus2.cloudapp.azure.com
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name 'mkdir -p /export/Lab-Xue/projects/mri/models'
    scp -i ~/.ssh/xueh2-a100.pem $model gtuser@$VM_name:$BASE_DIR/mri/models/
done

ulimit -n 65536

python3 ./mri/main_mri.py --use_amp --project mri --data_root $BASE_DIR/mri/data --check_path $BASE_DIR/mri/checkpoints --model_path $BASE_DIR/mri/models --log_path $BASE_DIR/mri/logs --results_path $BASE_DIR/mri/results --batch_size 16 --weight_decay 1 --weighted_loss --losses perpendicular gaussian gaussian3D l1 --loss_weights 1.0 5.0 5.0 1.0 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2020.h5 --train_data_types 2dt 2dt 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test.h5 train_3D_3T_retro_cine_2020_small_2DT_test.h5 train_3D_3T_retro_cine_2020_small_2D_test.h5 train_3D_3T_retro_cine_2020_500_samples.h5 --test_data_types 3d 2dt 2d 2dt --ratio 95 5 100 --complex_i --residual --load_path $model --run_name continued_training --optim sophia --global_lr 0.000025 --num_epochs 60 --scheduler_type ReduceLROnPlateau --height 32 64 --width 32 64 --time 12


torchrun --node_rank 0 --nnodes 8 --nproc_per_node 4 --master_port 9987 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4 ./mri/main_mri.py --ddp --use_amp --project mri --data_root $BASE_DIR/mri/data --check_path $BASE_DIR/mri/checkpoints --model_path $BASE_DIR/mri/models --log_path $BASE_DIR/mri/logs --results_path $BASE_DIR/mri/results --batch_size 16 --weight_decay 1 --weighted_loss --losses perpendicular ssim l1 --loss_weights 1.0 1.0 1.0 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2020.h5 --train_data_types 2dt 2dt 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test.h5 train_3D_3T_retro_cine_2020_small_2DT_test.h5 train_3D_3T_retro_cine_2020_small_2D_test.h5 train_3D_3T_retro_cine_2020_500_samples.h5 --test_data_types 3d 2dt 2d 2dt --ratio 95 5 100 --complex_i --residual --load_path $model --run_name continued_training_azure --optim adamw --global_lr 0.00001 --num_epochs 60 --scheduler_type ReduceLROnPlateau --height 32 64 --width 32 64 --time 12 --with_data_degrading

torchrun --standalone --nproc_per_node 8 ./mri/main_mri.py --ddp --use_amp --project mri --data_root $BASE_DIR/mri/data --check_path $BASE_DIR/mri/checkpoints --model_path $BASE_DIR/mri/models --log_path $BASE_DIR/mri/logs --results_path $BASE_DIR/mri/results --batch_size 16 --weight_decay 1 --weighted_loss --losses mse perpendicular gaussian gaussian3D l1 --loss_weights 1.0 1.0 5.0 5.0 1.0 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2020.h5 --train_data_types 2dt 2dt 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test.h5 train_3D_3T_retro_cine_2020_small_2DT_test.h5 train_3D_3T_retro_cine_2020_small_2D_test.h5 train_3D_3T_retro_cine_2020_500_samples.h5 --test_data_types 3d 2dt 2d 2dt --ratio 95 5 100 --complex_i --residual --load_path $model --run_name continued_training --optim sophia --global_lr 0.0001 --num_epochs 100 --scheduler_type ReduceLROnPlateau --height 32 64 --width 32 64 --time 12 --save_samples --num_workers 48 --prefetch_factor 32 --min_noise_level 12 --max_noise_level 24 --with_data_degrading

```