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

# ---------------------------------------------
# baseline, for validation
cd ~/mrprogs/FMImaging_for_paper
python3 ./mri/run_mri.py --standalone --nproc_per_node 4 --use_amp --num_epochs 10 --batch_size 16 --data_root /data1/mri --run_extra_note 1st --num_workers 32 --model_backbone hrnet --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1T1L1G1 --run_list 0 --tra_ratio 10 --val_ratio 5

cd ~/mrprogs/FMImaging

#--nnodes 2 --rdzv_endpoint 172.16.0.192:9050 --node_rank 0 --nproc_per_node 2

python3 ./projects/mri/inference/run_mri.py --standalone --nproc_per_node 4 --num_epochs 30 --batch_size 8 --run_extra_note 1st --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-4 --lr_pre 1e-4 --lr_post 1e-4 --lr_backbone 1e-4 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 100 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/Lab-Xue/projects/data/logs --add_salt_pepper --add_possion --weighted_loss_snr



python3 ./projects/mri/inference/run_mri.py --standalone  --nproc_per_node 8 --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231126_141320_915089_STCNNT_MRI_C-64-1_amp-False_complex_residual_with_data_degrading-T1L1G1T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_47 --model_type STCNNT_MRI --losses perpendicular  perceptual charbonnier  --loss_weights 1 1 1 1.0 1.0 --min_noise_level 0.1 --max_noise_level 100 --lr_pre 1e-5 --lr_backbone 1e-5 --lr_post 1e-5 --global_lr 1e-5 --run_extra_note 1st_more_epochs_perf_cine_NN80_perp1  --data_root /data/FM_data_repo/mri --num_epochs 40 --batch_size 4 --model_backbone STCNNT_HRNET --model_block_str T1L1G1T1L1G1 T1L1G1T1L1G1 --scheduler_factor 0.5 --disable_LSUV --log_root /export/Lab-Xue/projects/data/logs --continued_training --scheduler_type ReduceLROnPlateau --train_files train_3D_3T_retro_cine_2020.h5 BARTS_Perfusion_3T_2023.h5 --add_salt_pepper --add_possion

python3 ./projects/mri/inference/run_mri.py --nnodes 2 --rdzv_endpoint 172.16.0.192:9050 --node_rank 0 --nproc_per_node 2 --num_epochs 30 --batch_size 8 --run_extra_note 1st --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-4 --lr_pre 1e-4 --lr_post 1e-4 --lr_backbone 1e-4 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 100 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/Lab-Xue/projects/data/logs --add_salt_pepper --add_possion --weighted_loss_snr


python3 ./projects/mri/inference/run_mri.py --standalone --nproc_per_node 4 --num_epochs 30 --batch_size 8 --run_extra_note 1st --num_workers 32 --model_backbone omnivore --model_type omnivore_MRI --model_block_str T1L1G1 T1L1G1 --mri_height 64 --mri_width 64 --global_lr 1e-3 --lr_pre 1e-3 --lr_post 1e-3 --lr_backbone 1e-3 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 100 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/Lab-Xue/projects/data/logs --add_salt_pepper --add_possion --weighted_loss_snr --scheduler_type ReduceLROnPlateau --scheduler_factor 0.5

# test for overfitting

python3 ./projects/mri/inference/run_mri.py --standalone --nproc_per_node 4 --num_epochs 30 --batch_size 8 --run_extra_note 1st_tra20_val10 --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-4 --lr_pre 1e-4 --lr_post 1e-4 --lr_backbone 1e-4 --run_list 0 --tra_ratio 20 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 100 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/Lab-Xue/projects/data/logs --add_salt_pepper --add_possion --weighted_loss_snr


# ---------------------------------
# second stage training

# base on 1st net
python3 ./projects/mri/inference/run_mri.py --nproc_per_node 4 --standalone --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/data/logs/mri-main-1st_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231118_040652_735337_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_29 --model_type MRI_double_net --losses mse perpendicular perceptual charbonnier gaussian3D  --loss_weights 1 1 1.0 1.0 1.0 1.0 --min_noise_level 2 --max_noise_level 80.0 --lr_pre 1e-6 --lr_backbone 1e-6 --lr_post 1e-6 --global_lr 1e-6 --freeze_pre --freeze_backbone  --run_extra_note 2nd --data_root /data1/mri/data --num_epochs 20 --batch_size 8 --model_backbone STCNNT_HRNET --model_block_str T1L1G1 T1L1G1 --scheduler_factor 0.5 --not_load_post --disable_LSUV --post_model_of_1st_net /export/Lab-Xue/projects/data/logs/mri-main-1st_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231118_040652_735337_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_29_post.pth

# continue with the 2nd net
python3 ./projects/mri/inference/run_mri.py --nproc_per_node 4 --standalone --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/mri-main/logs/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231110_205214_795958_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_10 --model_type MRI_double_net --losses mse l1 perpendicular dwt gaussian gaussian3D  --loss_weights 1 1 1.0 2.0 2.0 2.0 --min_noise_level 0.1 --max_noise_level 16.0 --lr_pre 0.00001 --lr_backbone 0.00001 --lr_post 0.00001  --freeze_pre --freeze_backbone  --run_extra_note 2nd --data_root /data/FM_repo_data/mri/ --num_epochs 20 --batch_size 8 --model_backbone STCNNT_HRNET --model_block_str T1L1G1 T1L1G1 --scheduler_factor 0.8 --disable_LSUV --continued_training

python3 ./projects/mri/inference/run_mri.py --nnodes 2 --rdzv_endpoint 172.16.0.6 --node_rank 0 --nproc_per_node 4 --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231126_141320_915089_STCNNT_MRI_C-64-1_amp-False_complex_residual_with_data_degrading-T1L1G1T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_47 --model_type STCNNT_MRI --losses mse perpendicular perceptual charbonnier gaussian3D dwt  --loss_weights 1 1 1.0 1.0 1.0 1.0 1.0 --min_noise_level 0.1 --max_noise_level 100.0 --lr_pre 1e-6 --lr_backbone 1e-6 --lr_post 1e-6 --global_lr 1e-6 --run_extra_note 1st_more_epochs --data_root /export/Lab-Xue/projects/mri/data/ --num_epochs 50 --batch_size 8 --model_backbone STCNNT_HRNET --model_block_str T1L1G1T1L1G1 T1L1G1T1L1G1 --scheduler_factor 0.5 --disable_LSUV --log_root /export/Lab-Xue/projects/data/logs --continued_training

python3 ./projects/mri/inference/run_mri.py --nnodes 4 --rdzv_endpoint 172.16.0.4 --node_rank 0 --nproc_per_node 4 --tra_ratio 90 --val_ratio 10 --model_type STCNNT_MRI --losses mse perpendicular perceptual charbonnier gaussian3D  --loss_weights 1 1 1.0 1.0 1.0 1.0 1.0 --min_noise_level 0.1 --max_noise_level 200.0 --lr_pre 1e-4 --lr_backbone 1e-4 --lr_post 1e-4 --global_lr 1e-4 --run_extra_note 1st --data_root /export/Lab-Xue/projects/mri/data/ --num_epochs 80 --batch_size 8 --model_backbone STCNNT_HRNET --model_block_str T1L1G1T1L1G1 T1L1G1T1L1G1 --backbone_C 128 --scheduler_factor 0.5 --disable_LSUV --log_root /export/Lab-Xue/projects/data/logs


# ---------------------------------

# super-resolution model

python3 ./mri/run_mri.py --nproc_per_node 4 --standalone --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/mri/checkpoints/mri-validation-STCNNT_MRI_20230827_210539_792328_C-32-1_amp-False_weighted_loss_OFF_complex_residual-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-40.pth --model_type MRI_double_net --losses mse perpendicular l1 gaussian gaussian3D  --loss_weights 0.1 1.0 1.0 10.0 10.0 10.0 --min_noise_level 1.0 --max_noise_level 12.0 --lr_pre 0.00001 --lr_backbone 0.00001 --lr_post 0.0001 --not_load_post --disable_pre --disable_backbone  --run_extra_note 2nd_stage_super_resolution --super_resolution --disable_LSUV  --data_root /data/mri/data


python3 ./mri/run_mri.py --standalone --node_rank 0 --nproc_per_node 4 --use_amp --tra_ratio 90 --val_ratio 10 --not_add_noise --with_data_degrading --losses mse perpendicular psnr l1 gaussian gaussian3D --loss_weights 1.0 1.0 1.0 1.0 10.0 10.0 --model_type MRI_hrnet --separable_conv --super_resolution --run_extra_note with_separable_conv_super_resolution --data_root /data/mri/data

torchrun --standalone --nproc_per_node 2 --nnodes 1  ./mri/main_mri.py --data_root /data/mri/data --check_path /export/Lab-Xue/projects/mri/checkpoints --model_path /export/Lab-Xue/projects/mri/models --log_path /export/Lab-Xue/projects/mri/logs --results_path /export/Lab-Xue/projects/mri/results --summary_depth 6 --save_cycle 200 --device cuda --project mri --num_epochs 100 --batch_size 32 --window_size 8 8 --patch_size 2 2 --global_lr 0.0001 --weight_decay 1 --use_amp --iters_to_accumulate 1 --prefetch_factor 4 --scheduler_type ReduceLROnPlateau --scheduler.ReduceLROnPlateau.patience 0 --scheduler.ReduceLROnPlateau.cooldown 0 --scheduler.ReduceLROnPlateau.factor 0.95 --scheduler.OneCycleLR.pct_start 0.2 --min_noise_level 2.0 --max_noise_level 14.0 --height 32 64 --width 32 64 --time 12 --num_uploaded 12 --snr_perturb 0.15 --train_files MINNESOTA_UHVC_RetroCine_1p5T_2023_with_2x_resized.h5 --train_data_types 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test_with_2x_resized.h5 train_3D_3T_retro_cine_2020_small_2DT_test_with_2x_resized.h5 train_3D_3T_retro_cine_2020_small_2D_test_with_2x_resized.h5 train_3D_3T_retro_cine_2020_500_samples_with_2x_resized.h5 --test_data_types 3d 2dt 2d 2dt --ratio 25 5 100 --optim sophia --backbone hrnet --a_type conv --cell_type parallel --cosine_att 1 --att_with_relative_postion_bias 0 --norm_mode batch2d --mixer_type conv --shuffle_in_window 0 --scale_ratio_in_mixer 1.0 --mixer_kernel_size 3 --mixer_padding 1 --mixer_stride 1 --normalize_Q_K --backbone_hrnet.block_str T1L1G1 T1L1G1 T1L1G1 T1L1G1 --complex_i --losses mse perpendicular psnr l1 gaussian gaussian3D --loss_weights 1.0 1.0 1.0 1.0 10.0 10.0 --run_name test_more_losses --run_notes test_more_losses --snr_perturb_prob 0.0 --n_head 32 --weighted_loss_snr --weighted_loss_temporal --weighted_loss_added_noise --with_data_degrading --save_samples --model_type MRI_hrnet --residual --not_add_noise --disable_LSUV --readout_resolution_ratio 0.85 0.7 0.65 0.55 --phase_resolution_ratio 0.85 0.7 0.65 0.55 --kspace_filter_sigma 1.5 2.0 2.5 3.0 --kspace_T_filter_sigma 1.0 1.25 1.5 --separable_conv

# working fine
 python3 ./mri/run_mri.py --standalone --node_rank 0 --nproc_per_node 4 --use_amp --tra_ratio 10 --val_ratio 5 --not_add_noise --with_data_degrading --losses mse perpendicular l1 gaussian gaussian3D --loss_weights 1.0 1.0 1.0 10.0 10.0 --model_type MRI_hrnet --separable_conv  --run_extra_note with_separable_conv_super_resolution0

# new double net
torchrun --standalone --nproc_per_node 6 ./mri/main_mri.py --ddp --data_root /data/mri/data --check_path /export/Lab-Xue/projects/mri/checkpoints --model_path /export/Lab-Xue/projects/mri/models --log_path /export/Lab-Xue/projects/mri/logs --results_path /export/Lab-Xue/projects/mri/results --summary_depth 6 --save_cycle 200 --device cuda --project mri --num_epochs 100 --window_size 8 8 --patch_size 2 2 --global_lr 0.0001 --weight_decay 1 --iters_to_accumulate 1 --num_workers 48 --prefetch_factor 4 --scheduler_type ReduceLROnPlateau --scheduler.ReduceLROnPlateau.patience 0 --scheduler.ReduceLROnPlateau.cooldown 0 --scheduler.ReduceLROnPlateau.factor 0.95 --scheduler.OneCycleLR.pct_start 0.2 --min_noise_level 2.0 --max_noise_level 14.0 --height 32 64 --width 32 64 --time 12 --num_uploaded 12 --snr_perturb 0.15 --train_files MINNESOTA_UHVC_RetroCine_1p5T_2023_with_2x_resized.h5 --train_data_types 2dt --test_files train_3D_3T_retro_cine_2020_small_3D_test_with_2x_resized.h5 train_3D_3T_retro_cine_2020_small_2DT_test_with_2x_resized.h5 train_3D_3T_retro_cine_2020_small_2D_test_with_2x_resized.h5 train_3D_3T_retro_cine_2020_500_samples_with_2x_resized.h5 --test_data_types 3d 2dt 2d 2dt --ratio 90 10 100 --optim sophia --a_type conv --cell_type parallel --cosine_att 1 --att_with_relative_postion_bias 0 --norm_mode instance2d --mixer_type conv --shuffle_in_window 0 --scale_ratio_in_mixer 1.0 --mixer_kernel_size 3 --mixer_padding 1 --mixer_stride 1 --normalize_Q_K --backbone_hrnet.block_str T1L1G1 T1L1G1T1L1G1 T1L1G1T1L1G1 T1L1G1 --complex_i --run_name test_mixed_unetr --run_notes test_mixed_unetr --snr_perturb_prob 0.0 --n_head 32 --weighted_loss_snr --weighted_loss_temporal --weighted_loss_added_noise --residual --readout_resolution_ratio 0.85 0.7 0.65 --phase_resolution_ratio 0.85 0.7 0.65 --kspace_filter_sigma 1.5 2.0 2.5 3.0 --kspace_T_filter_sigma 0.0 --losses mse perpendicular l1 gaussian gaussian3D ssim --loss_weights 0.01 0.01 0.01 10.0 10.0 10.0  --post_hrnet.separable_conv --backbone mixed_unetr --model_type MRI_double_net --backbone_mixed_unetr.C 32 --backbone_mixed_unetr.num_resolution_levels 2 --backbone_mixed_unetr.block_str T1L1G1 T1L1G1T1L1G1 T1L1G1 T1L1G1 --backbone_mixed_unetr.use_unet_attention 1 --backbone_mixed_unetr.use_interpolation 1 --backbone_mixed_unetr.with_conv 0 --backbone_mixed_unetr.min_T 16 --backbone_mixed_unetr.encoder_on_skip_connection 1 --backbone_mixed_unetr.encoder_on_input 1 --backbone_mixed_unetr.transformer_for_upsampling 0 --backbone_mixed_unetr.n_heads 32 32 32 --backbone_mixed_unetr.use_conv_3d 1 --post_mixed_unetr.block_str T1L1G1 T1L1G1 --post_mixed_unetr.n_heads 32 32 --post_mixed_unetr.use_window_partition 0 --post_mixed_unetr.use_conv_3d 1 --separable_conv --super_resolution --post_backbone mixed_unetr --not_add_noise  --batch_size 2 --load_path /export/Lab-Xue/projects/mri/checkpoints/test_mixed_unetr_epoch-6.pth --lr_pre 1e-5 --lr_post 1e-5 --lr_backbone 1e-5 --disable_LSUV

# ---------------------------------
