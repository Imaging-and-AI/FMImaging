# MRI image enhancement training

# run the second training

```

export BASE_DIR=/data

model=$BASE_DIR/mri/test/complex_model/mri-HRNET-20230621_132139_784364_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_13-22-06-20230621_last.pt

model=$BASE_DIR/mri/models/mri-HRNET-20230621_132139_784364_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_13-22-06-20230621_last.pt

model=$BASE_DIR/mri/models/mri-HRNET-20230702_013521_019623_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_01-35-34-20230702_best.pt


export BASE_DIR=/export/Lab-Xue/projects/

model=$BASE_DIR/mri/models/mri-HRNET-20230702_013521_019623_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_01-35-34-20230702_best.pt

model=$BASE_DIR/mri/models/mri-HRNET-20230708_034122_305779_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_03-41-31-20230708_best.pt

for n in fsi{1..16}
do
    echo "copy to $n ..."
    VM_name=$n.eastus2.cloudapp.azure.com
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name 'mkdir -p /export/Lab-Xue/projects/mri/models'
    scp -i ~/.ssh/xueh2-a100.pem $model gtuser@$VM_name:$BASE_DIR/mri/models/
done

ulimit -n 65536

python3 ./mri/main_mri.py --use_amp --project mri --data_root $BASE_DIR/mri/data --check_path $BASE_DIR/mri/checkpoints --model_path $BASE_DIR/mri/models --log_path $BASE_DIR/mri/logs --results_path $BASE_DIR/mri/results --batch_size 16 --weight_decay 1 --weighted_loss --losses perpendicular gaussian gaussian3D l1 --loss_weights 1.0 5.0 5.0 1.0 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2020.h5 --train_data_types 2dt 2dt 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test.h5 train_3D_3T_retro_cine_2020_small_2DT_test.h5 train_3D_3T_retro_cine_2020_small_2D_test.h5 train_3D_3T_retro_cine_2020_500_samples.h5 --test_data_types 3d 2dt 2d 2dt --ratio 95 5 100 --complex_i --residual --load_path $model --run_name continued_training --optim sophia --global_lr 0.000025 --num_epochs 60 --scheduler_type ReduceLROnPlateau --height 32 64 --width 32 64 --time 12


torchrun --node_rank 0 --nnodes 8 --nproc_per_node 4 --master_port 9987 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4 ./mri/main_mri.py --ddp --use_amp --project mri --data_root $BASE_DIR/mri/data --check_path $BASE_DIR/mri/checkpoints --model_path $BASE_DIR/mri/models --log_path $BASE_DIR/mri/logs --results_path $BASE_DIR/mri/results --batch_size 16 --weight_decay 1 --weighted_loss --losses mse perpendicular gaussian gaussian3D l1 --loss_weights 1.0 1.0 5.0 5.0 1.0 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2020.h5 --train_data_types 2dt 2dt 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test.h5 train_3D_3T_retro_cine_2020_small_2DT_test.h5 train_3D_3T_retro_cine_2020_small_2D_test.h5 train_3D_3T_retro_cine_2020_500_samples.h5 --test_data_types 3d 2dt 2d 2dt --ratio 95 5 100 --complex_i --residual --load_path $model --run_name continued_training --optim sophia --global_lr 0.0001 --num_epochs 100 --scheduler_type ReduceLROnPlateau --height 32 64 --width 32 64 --time 12 --save_samples --num_workers 48 --prefetch_factor 32 --min_noise_level 12 --max_noise_level 24 --with_data_degrading

torchrun --standalone --nproc_per_node 8 ./mri/main_mri.py --ddp --use_amp --project mri --data_root $BASE_DIR/mri/data --check_path $BASE_DIR/mri/checkpoints --model_path $BASE_DIR/mri/models --log_path $BASE_DIR/mri/logs --results_path $BASE_DIR/mri/results --batch_size 16 --weight_decay 1 --weighted_loss --losses mse perpendicular gaussian gaussian3D l1 --loss_weights 1.0 1.0 5.0 5.0 1.0 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2020.h5 --train_data_types 2dt 2dt 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test.h5 train_3D_3T_retro_cine_2020_small_2DT_test.h5 train_3D_3T_retro_cine_2020_small_2D_test.h5 train_3D_3T_retro_cine_2020_500_samples.h5 --test_data_types 3d 2dt 2d 2dt --ratio 95 5 100 --complex_i --residual --load_path $model --run_name continued_training --optim sophia --global_lr 0.0001 --num_epochs 100 --scheduler_type ReduceLROnPlateau --height 32 64 --width 32 64 --time 12 --num_workers 48 --prefetch_factor 32 --min_noise_level 12 --max_noise_level 24 # --with_data_degrading --save_samples

# ---------------------------------
# first stage training

python3 ./mri/run_mri.py --standalone --node_rank 0 --nproc_per_node 4 --use_amp --tra_ratio 25 --val_ratio 10 --min_noise_level 1.0 --max_noise_level 24.0 --losses mse perpendicular psnr l1 --loss_weights 1.0 1.0 1.0 1.0 --model_type STCNNT_MRI

python3 ./mri/run_mri.py --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4 --nnodes 2 --node_rank 0 --nproc_per_node 4 --use_amp --tra_ratio 90 --val_ratio 10 --min_noise_level 1.0 --max_noise_level 24.0 --losses mse perpendicular psnr l1 --loss_weights 1.0 1.0 1.0 1.0 --model_type STCNNT_MRI

# second stage training

torchrun --node_rank 0 --nproc_per_node 4 --nnodes 8 --master_port 9987 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4 ./mri/main_mri.py --ddp --use_amp --project mri --data_root $BASE_DIR/mri/data --check_path $BASE_DIR/mri/checkpoints --model_path $BASE_DIR/mri/models --log_path $BASE_DIR/mri/logs --results_path $BASE_DIR/mri/results --batch_size 16 --weight_decay 1 --weighted_loss --losses perpendicular l1 gaussian gaussian3D --loss_weights 1.0 1.0 5.0 5.0 1.0 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2020.h5 BARTS_RetroCine_3T_2023.h5 BARTS_RetroCine_1p5T_2023.h5 MINNESOTA_UHVC_RetroCine_1p5T_2023.h5 MINNESOTA_UHVC_RetroCine_1p5T_2022.h5 --train_data_types 2dt 2dt 2dt 2dt 2dt 2dt 2dt 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test.h5 train_3D_3T_retro_cine_2020_small_2DT_test.h5 train_3D_3T_retro_cine_2020_small_2D_test.h5 train_3D_3T_retro_cine_2020_500_samples.h5 --test_data_types 3d 2dt 2d 2dt --ratio 95 5 100 --complex_i --residual --run_name continued_training_MRI_hrnet --optim sophia --global_lr 0.0001 --num_epochs 100 --scheduler_type ReduceLROnPlateau --height 32 64 --width 32 64 --time 12 --num_workers 48 --prefetch_factor 32 --min_noise_level 12 --max_noise_level 24 --lr_backbone 0.00001 --lr_post 0.0001 --load_path $model --model_type MRI_hrnet --post_hrnet.block_str T1L1G1T1L1G1 --disable_backbone


 torchrun --nproc_per_node 6 --max_restarts 6 --master_port 9050 ./mri/main_mri.py --data_root /data/mri/data/ --check_path /export/Lab-Xue/projects/mri/checkpoints --model_path /export/Lab-Xue/projects/mri/models --log_path /export/Lab-Xue/projects/mri/logs --results_path /export/Lab-Xue/projects/mri/results --summary_depth 6 --save_cycle 200 --device cuda --ddp --project mri --backbone_hrnet.use_interpolation 1 --backbone_unet.use_unet_attention 1 --backbone_unet.use_interpolation 1 --backbone_unet.with_conv 1 --backbone_LLM.add_skip_connections 1 --num_epochs 75 --batch_size 16 --window_size 8 8 --patch_size 2 2 --global_lr 0.0001 --clip_grad_norm 1.0 --weight_decay 1 --iters_to_accumulate 1 --num_workers 64 --prefetch_factor 4 --scheduler_type ReduceLROnPlateau --scheduler.ReduceLROnPlateau.patience 0 --scheduler.ReduceLROnPlateau.cooldown 0 --scheduler.ReduceLROnPlateau.factor 0.85 --scheduler.OneCycleLR.pct_start 0.2 --backbone_hrnet.num_resolution_levels 2 --backbone_unet.num_resolution_levels 2 --backbone_LLM.num_stages 3 --backbone_small_unet.channels 16 32 64 --backbone_small_unet.block_str T1L1G1 T1L1G1 T1L1G1 --min_noise_level 2.0 --max_noise_level 24.0 --height 32 64 --width 32 64 --time 12 --num_uploaded 12 --snr_perturb 10.0 --post_hrnet.block_str T1L1G1T1L1G1 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2020.h5 BARTS_RetroCine_3T_2023.h5 BARTS_RetroCine_1p5T_2023.h5 MINNESOTA_UHVC_RetroCine_1p5T_2023.h5 MINNESOTA_UHVC_RetroCine_1p5T_2022.h5 --train_data_types 2dt 2dt 2dt 2dt 2dt 2dt 2dt 2dt 3d --test_files train_3D_3T_retro_cine_2020_small_3D_test.h5 train_3D_3T_retro_cine_2020_small_2DT_test.h5 train_3D_3T_retro_cine_2020_small_2D_test.h5 train_3D_3T_retro_cine_2020_500_samples.h5 --test_data_types 3d 2dt 2d 2dt --ratio 25 10 100 --max_load -1 --lr_pre -1 --lr_backbone -1 --lr_post -1 --model_type STCNNT_MRI --optim sophia --backbone hrnet --a_type conv --cell_type parallel --cosine_att 1 --att_with_relative_postion_bias 0 --backbone_hrnet.C 32 --backbone_unet.C 32 --backbone_LLM.C 32 --block_dense_connection 1 --norm_mode batch2d --mixer_type conv --shuffle_in_window 0 --scale_ratio_in_mixer 1.0 --mixer_kernel_size 3 --mixer_padding 1 --mixer_stride 1 --normalize_Q_K --backbone_hrnet.block_str T1L1G1 T1L1G1T1L1G1 T1L1G1T1L1G1 T1L1G1T1L1G1 --load_path /export/Lab-Xue/projects/mri/checkpoints/ --continued_training --use_amp --complex_i --residual --weighted_loss --with_data_degrading --not_add_noise --losses mse perpendicular psnr l1 --loss_weights 1.0 1.0 1.0 1.0 1.0 --run_name mri-HRNET-20230716_160729_056467_C-32-1_amp-True_no_add_noise_complex_residual_weighted_loss_with_data_degrading_no_noise-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1 --run_notes mri-HRNET-20230716_160729_056467_C-32-1_amp-True_no_add_noise_complex_residual_weighted_loss_with_data_degrading_no_noise-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1 --snr_perturb_prob 0.0 --n_head 32

python3 ./mri/run_mri.py --nproc_per_node 6 --standalone --data_root /data/mri/data --tra_ratio 25 --val_ratio 10 --run_list 0 --model_type STCNNT_hrnet --run_extra_note first_stage

python3 ./mri/run_mri.py --nproc_per_node 6 --standalone --data_root /data/mri/data --tra_ratio 25 --val_ratio 10 --run_list 0 --load_path /export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230716_190117_960318_C-32-1_amp-False_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-70.pth --run_extra_note double_net --model_type MRI_double_net --lr_pre 1e-6 --lr_backbone 1e-6 --lr_post 1e-4 --not_load_post --disable_pre --disable_backbone --save_samples --min_noise_level 1.0 --losses mse perpendicular l1 gaussian gaussian3D ssim --loss_weights 0.1 0.1 0.1 10.0 10.0 10.0 # --use_amp 


python3 ./mri/run_mri.py --nproc_per_node 4 --node_rank 0 --rdzv_endpoint 172.16.0.4 --data_root /data/mri/data --tra_ratio 25 --val_ratio 10 --run_list 0 --load_path /export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230716_190117_960318_C-32-1_amp-False_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-70.pth --run_extra_note second_stage --model_type MRI_hrnet --lr_pre 1e-6 --lr_backbone 1e-6 --lr_post 1e-4 --not_load_post --disable_pre --disable_backbone --use_amp --save_samples --min_noise_level 1.0 --losses mse perpendicular psnr l1 gaussian gaussian3D ssim --loss_weights 0.1 1.0 1.0 1.0 20.0 20.0 2.0

python3 ./mri/run_mri.py --nproc_per_node 4 --nnodes 2 --node_rank 0 --rdzv_endpoint 172.16.0.4 --tra_ratio 25 --val_ratio 10 --run_list 0 --load_path /export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230716_190117_960318_C-32-1_amp-False_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-70.pth --model_type MRI_double_net --lr_pre 1e-6 --lr_backbone 1e-6 --lr_post 1e-4 --not_load_post --disable_pre --disable_backbone --min_noise_level 1.0 --max_noise_level 8.0 --losses mse perpendicular gaussian gaussian3D ssim --loss_weights 0.01 1.0 100.0 100.0 10.0 --run_extra_note 2nd_stage_noise_1_8

```