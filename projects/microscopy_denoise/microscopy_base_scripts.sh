# General base scripts for microscopy
export CUDA_VISIBLE_DEVICES=1

# Training from scratch
python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log --train_files Base_All_train.h5 Alex_bActin-NM2A_train.h5 Alex_timed_camera_100_train.h5  Chris_zebra_timed_liver_train.h5 Ryo_tile_train.h5 Alex_timed_camera_020_train.h5 Alex_wide_field_train.h5 Chris_zebra_timed_pancs_train.h5 Alex_timed_camera_040_train.h5 Chris_zebra_train.h5 --test_files Base_All_test.h5  --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 100 --batch_size 2 --backbone_C 64 --standalone --losses mse l1 ssim perceptual --loss_weights 10.0 1.0 2.0 1.0 --max_load 20000000 --num_workers 4 --save_samples --no_clip_data --scaling_vals 0 1024 --micro_height 64 128 --micro_width 64 128 --scheduler_type OneCycleLR

# Finetuning
python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log \
    --train_files Base_Actin_train.h5 --test_files Base_Actin_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 30 --batch_size 8 --backbone_C 64 --standalone \
    --losses mse l1 ssim perceptual --loss_weights 1.0 1.0 5.0 5.0 --max_load 200 --num_workers 4 --save_samples \
    --load_path /home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 5

model=/home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch
model=/home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240120_235433_496767_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log \
    --train_files Alex_wide_field_train.h5 --test_files Alex_wide_field_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 30 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim --loss_weights 10.0 1.0 1.0 --micro_height 64 128 --micro_width 64 128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note Alex_wide_test --no_clip_data

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log \
    --train_files Chris_zebra_train.h5 --test_files Chris_zebra_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 30 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim --loss_weights 10.0 1.0 1.0 --micro_height 64 128 --micro_width 64 128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note FineTuning_Chris_zebra --no_clip_data

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy     --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log     --train_files Alex_wide_field_train.h5 --test_files Alex_wide_field_test.h5     --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone     --losses mse l1 ssim --loss_weights 10.0 1.0 1.0 --max_load 200 --num_workers 4 --save_samples     --load_path $model     --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025     --train_samples 200000 --run_extra_note FT_Alex_wide --micro_height 64 128 --micro_width 64 128 --scheduler_type OneCycleLR --no_clip_data

# Inference
python3 ./projects/microscopy_denoise/microscopy_inference.py \
    --input_dir /home/gtuser/rehmana2/projects/stcnnt/data/ --input_file_s Base_Actin_test.h5 \
    --output_dir /home/gtuser/rehmana2/projects/stcnnt/infer_results/ \
    --saved_model_path /home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-100.pth \
    --pad_time --image_order THW --device cuda --batch_size 16

# snr test

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /export/Lab-Xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy_rois --output_dir /export/Lab-Xue/projects/microscopy/snr/Alex_wide_field --saved_model_path /export/Lab-Xue/projects/data/logs/microscopy-Alex_wide_test_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240121_200048_643087_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch --image_order HWT --added_noise_sd 0.1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /export/Lab-Xue/publications/CNNT_paper/data/micro_datasets_rcan/Chris_zebra/test/noisy_rois/ --output_dir /export/Lab-Xue/projects/microscopy/snr/Chris_zebra --saved_model_path /export/Lab-Xue/projects/data/logs/microscopy-Alex_wide_test_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240121_200048_643087_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch --image_order HWT --added_noise_sd 0.1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64

# run a inference
python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /export/Lab-Xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy_rois --input_file_s Image_001.npy --output_dir /export/Lab-Xue/projects/microscopy/snr/Alex_wide_field/res --saved_model_path /export/Lab-Xue/projects/data/logs/microscopy-Alex_wide_test_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240121_200048_643087_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch --image_order HWT --no_clip_data --batch_size 2 --patch_size_inference 64