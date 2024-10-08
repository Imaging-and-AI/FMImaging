# General base scripts for microscopy
export CUDA_VISIBLE_DEVICES=1

data_dir=/home/gtuser/rehmana2/projects/stcnnt/data/
log_dir=/home/gtuser/log

data_dir=/data/FM_data_repo/light_microscopy/
log_dir=/data/log

data_dir=/isilon/lab-xue/projects/fm/data
log_dir=/isilon/lab-xue/projects/fm/log

data_dir=/home/gtuser/rehmana2/projects/stcnnt/data/ 
log_dir=/home/gtuser/rehmana2/projects/stcnnt/log

# Training from scratch
python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log --train_files Base_All_train.h5 Alex_bActin-NM2A_train.h5 Alex_wide_field_train.h5 Chris_zebra_train.h5 Light_Sheet_noisy_35_clean_train.h5 Light_Sheet_noisy_clean_train.h5 --test_files Base_All_test.h5  --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 100 --batch_size 2 --backbone_C 64 --standalone --losses mse l1 ssim perceptual --loss_weights 10.0 1.0 2.0 1.0 --max_load 20000000 --num_workers 32 --save_samples --no_clip_data --scaling_vals 0 1024 --micro_height 64 128 --micro_width 64 128 --scheduler_type OneCycleLR --model_block_str T1L1G1 T1L1G1T1L1G1

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy  --data_root $data_dir --log_root $log_dir --train_files Base_All_train.h5 --test_files Base_All_test.h5  --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 100 --batch_size 8 --backbone_C 64 --standalone --losses mse l1 ssim perceptual --loss_weights 10.0 1.0 2.0 1.0 --max_load 20000000 --num_workers 32 --no_clip_data --scaling_vals 0 1024 --micro_height 32 64 --micro_width 32 64 --scheduler_type OneCycleLR --model_block_str C3C3C3C3C3C3 C3C3C3C3C3C3

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

data_dir=/isilon/lab-xue/projects/fm/data/
log_dir=/home/gtuser/log
model=/isilon/lab-xue/projects/data/logs/microscopy-20240120_235433_496767_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch

data_dir=/home/gtuser/rehmana2/projects/stcnnt/data/
log_dir=/home/gtuser/log

data_dir=/data/FM_data_repo/light_microscopy/
log_dir=/data/log
model=/isilon/lab-xue/projects/data/logs/microscopy-20240120_235433_496767_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/microscopy-20240208_035121_464287_STCNNT_Microscopy_C-64-1_amp-False_residual-C3C3C3C3C3C3_C3C3C3C3C3C3/final_epoch

# -----------------------

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root $data_dir --log_root $log_dir \
    --train_files iSim_021324_AF555_Low_High_train.h5 iSim_021324_AF555_Mid_High_train.h5 --tra_ratio 80 --val_ratio 10 --test_ratio 10 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim perceptual --loss_weights 1.0 1.0 5.0 1.0 --micro_height 64 128 --micro_width 64 128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note FineTuning_iSim_021324_AF555 --no_clip_data --scheduler_type OneCycleLR --scaling_vals 0 1024

# -----------------------

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root $data_dir --log_root $log_dir \
    --train_files Alex_wide_field_train.h5 --test_files Alex_wide_field_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim perceptual --loss_weights 1.0 1.0 5.0 1.0 --micro_height 64 128 --micro_width 64 128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note FineTuning_Alex_wide --no_clip_data --scheduler_type OneCycleLR --scaling_vals 0 1024

# -----------------------

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root $data_dir --log_root $log_dir \
    --train_files Chris_zebra_train.h5 --test_files Chris_zebra_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim --loss_weights 10.0 10.0 5.0 --micro_height 64 128 --micro_width 64 128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note FineTuning_Chris_zebra --no_clip_data --scheduler_type OneCycleLR --scaling_vals 0 256 

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root $data_dir --log_root $log_dir \
    --train_files Chris_zebra_train.h5 --test_files Chris_zebra_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim --loss_weights 10.0 10.0 5.0 --micro_height 64 128 --micro_width 64 128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note FineTuning_Chris_zebra --no_clip_data --scheduler_type OneCycleLR --scaling_vals 0 256 --model_block_str C3C3C3C3C3C3 C3C3C3C3C3C3

# -----------------------

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy --data_root $data_dir --log_root $log_dir --train_files Chris_zebra_train.h5 --test_files Chris_zebra_test.h5     --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 40 --batch_size 1 --backbone_C 64 --standalone --losses mse l1 ssim --loss_weights 1.0 1.0 10.0 --micro_height 64 128 --micro_width 64 128 --max_load 20000000 --num_workers 4 --load_path $model --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025     --train_samples 20000000 --run_extra_note FineTuning_Chris_zebra_mse_l1_ssim --no_clip_data --scheduler_type OneCycleLR --scaling_vals 0 256

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root $data_dir --log_root $log_dir \
    --train_files Light_Sheet_noisy_clean_train.h5 --test_files Light_Sheet_noisy_clean_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim --loss_weights 10.0 10.0 5.0 --micro_height 64  128 --micro_width 64  128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note FineTuning_Light_Sheet --no_clip_data --scheduler_type OneCycleLR --scaling_vals 0 256

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root $data_dir --log_root $log_dir \
    --train_files Alex_bActin-NM2A_train.h5 --test_files Alex_bActin-NM2A_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim perceptual --loss_weights 10.0 10.0 5.0 1.0 --micro_height 64  128 --micro_width 64 128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note FineTuning_Alex_bActin --no_clip_data --scheduler_type OneCycleLR --scaling_vals 0 1024

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root $data_dir --log_root $log_dir \
    --train_files Ryo_tile_train.h5 --test_files Ryo_tile_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone \
    --losses mse l1 ssim --loss_weights 10.0 10.0 5.0 --micro_height 64  128 --micro_width 64 128 --max_load 20000000 --num_workers 4 \
    --load_path $model \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 20000000 --run_extra_note FineTuning_Ryo_tile --no_clip_data --scheduler_type OneCycleLR --scaling_vals 0 256

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy     --data_root $data_dir --log_root $log_dir     --train_files Alex_wide_field_train.h5 --test_files Alex_wide_field_test.h5     --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone     --losses mse l1 ssim --loss_weights 10.0 1.0 1.0 --max_load 200 --num_workers 4 --save_samples     --load_path $model     --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025     --train_samples 200000 --run_extra_note FT_Alex_wide --micro_height 64 128 --micro_width 64 128 --scheduler_type OneCycleLR --no_clip_data

# Inference
python3 ./projects/microscopy_denoise/microscopy_inference.py \
    --input_dir $data_dir --input_file_s Base_Actin_test.h5 \
    --output_dir /home/gtuser/rehmana2/projects/stcnnt/infer_results/ \
    --saved_model_path /home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-100.pth \
    --pad_time --image_order THW --device cuda --batch_size 16

# snr test

# ===================================================

export CUDA_VISIBLE_DEVICES=0
model=/isilon/lab-xue/projects/data/logs/microscopy-Alex_wide_test_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240121_200048_643087_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch

model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Alex_wide_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240204_145153_068481_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_wide_field --saved_model_path $model --image_order HWT --added_noise_sd 1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 1024

python3 ./projects/microscopy_denoise/microscopy_inference_for_uncertainty_PCA.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_wide_field --saved_model_path $model --image_order HWT --no_clip_data --patch_size_inference 64 --scaling_vals 0 1024 --batch_size 1 --low_acc

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_wide_field_all --saved_model_path $model --image_order HWT --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 1024

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_wide_field/res_rois --saved_model_path $model --image_order HWT --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 1024 --cuda_devices 0 1 2 3

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_wide_field/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 1024 --cuda_devices 0 1 2 3

# ===================================================

export CUDA_VISIBLE_DEVICES=1
model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Chris_zebra_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240123_104605_092663_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch
model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Chris_zebra_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240204_001419_854169_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch
model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Chris_zebra_mse_l1_ssim_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240204_135020_046475_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch

model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Chris_zebra_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240413_163334_642791_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/checkpoint_epoch_29

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Chris_zebra/test/noisy_rois/ --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra --saved_model_path $model --image_order HWT --added_noise_sd 1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 256

python3 ./projects/microscopy_denoise/microscopy_inference_for_uncertainty_PCA.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Chris_zebra/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra --saved_model_path $model --image_order HWT --no_clip_data --patch_size_inference 64 --scaling_vals 0 256 --batch_size 2 --low_acc

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Chris_zebra/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra/res_rois --saved_model_path $model --image_order HWT --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256 --cuda_devices 0 1 2 3

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Chris_zebra/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256 --cuda_devices 0 1 2 3 

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/ai-data/060723_Zebrafish_fixed_and_live_timecourses/060723_Zebrafish_CNNT_expts_live_liver_timecourse/train/noisy/ --output_dir /isilon/ai-data/060723_Zebrafish_fixed_and_live_timecourses/060723_Zebrafish_CNNT_expts_live_liver_timecourse/train/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256 --cuda_devices 0 1 2 3 

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/ai-data/060723_Zebrafish_fixed_and_live_timecourses/060723_Zebrafish_CNNT_expts_live_liver_timecourse_Z_axis/train/noisy/ --output_dir /isilon/ai-data/060723_Zebrafish_fixed_and_live_timecourses/060723_Zebrafish_CNNT_expts_live_liver_timecourse_Z_axis/train/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256 --cuda_devices 0 1 2 3 

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/ai-data/060723_Zebrafish_fixed_and_live_timecourses/060723_Zebrafish_CNNT_expts_live_pancreas_timecourse_Z_axis/train/noisy/ --output_dir /isilon/ai-data/060723_Zebrafish_fixed_and_live_timecourses/060723_Zebrafish_CNNT_expts_live_pancreas_timecourse_Z_axis/train/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256 --cuda_devices 0 1 2 3 4 5 6 7

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Chris_zebra_timed_liver/test/noisy --output_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Chris_zebra_timed_liver/test/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256 --cuda_devices 0 1 2 3 4 5 6 7

# ===================================================

model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Chris_zebra_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240204_001419_854169_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch

model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Chris_zebra_C3_STCNNT_HRNET_C3C3C3C3C3C3_C3C3C3C3C3C3_20240208_170049_615428_STCNNT_Microscopy_C-64-1_amp-False_residual-C3C3C3C3C3C3_C3C3C3C3C3C3/last_epoch

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/micro_tiffs/Chris_zebra_timed_liver/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra_timed_liver/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256 --cuda_devices 0 1 2 3

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/micro_tiffs/Chris_zebra_timed_liver/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra_timed_liver/res_snr --saved_model_path $model --image_order THW --added_noise_sd 1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 256 --cuda_devices 0 1 2 3 

python3 ./projects/microscopy_denoise/microscopy_inference_for_uncertainty_PCA.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/micro_tiffs/Chris_zebra_timed_liver/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra_timed_liver/res_pca --saved_model_path $model --image_order THW --no_clip_data --patch_size_inference 64 --scaling_vals 0 256 --batch_size 2 --low_acc --frame 14 --cuda_devices 0 1 2 3 


# ---------

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/micro_tiffs/Chris_zebra_timed_pancs/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra_timed_pancs/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256 --cuda_devices 0 1 2 3

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/micro_tiffs/Chris_zebra_timed_pancs/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra_timed_pancs/res_snr --saved_model_path $model --image_order THW --added_noise_sd 1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 256 --cuda_devices 0 1 2 3 

python3 ./projects/microscopy_denoise/microscopy_inference_for_uncertainty_PCA.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/micro_tiffs/Chris_zebra_timed_pancs/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Chris_zebra_timed_pancs/res_pca --saved_model_path $model --image_order THW --no_clip_data --patch_size_inference 64 --scaling_vals 0 256 --batch_size 2 --low_acc --frame 14 --cuda_devices 0 1 2 3 

# ===================================================

export CUDA_VISIBLE_DEVICES=2
model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Alex_bActin_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240125_081046_666737_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/checkpoint_epoch_19

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_bActin-NM2A/test/noisy_rois/ --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_bActin-NM2A --saved_model_path $model --image_order HWT --added_noise_sd 1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 1024

python3 ./projects/microscopy_denoise/microscopy_inference_for_uncertainty_PCA.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_bActin-NM2A/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_bActin-NM2A --saved_model_path $model --image_order HWT --no_clip_data --patch_size_inference 64 --scaling_vals 0 1024 --batch_size 2 --low_acc

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_bActin-NM2A/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_bActin-NM2A_tile/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 1024 --cuda_devices 0 1 2 3 

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_bActin-NM2A/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_bActin-NM2A_tile/res_rois --saved_model_path $model --image_order HWT --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 1024 --cuda_devices 0 1 2 3 

# ===================================================

export CUDA_VISIBLE_DEVICES=3
model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Light_Sheet_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240128_221146_911789_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Light_Sheet_noisy_35_clean/test/noisy_rois/ --output_dir /isilon/lab-xue/projects/microscopy/snr/Light_Sheet_noisy_35_clean --saved_model_path $model --image_order HWT --added_noise_sd 1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 256

python3 ./projects/microscopy_denoise/microscopy_inference_for_uncertainty_PCA.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Light_Sheet_noisy_35_clean/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Light_Sheet_noisy_35_clean --saved_model_path $model --image_order HWT --no_clip_data --patch_size_inference 64 --scaling_vals 0 256 --batch_size 2 --low_acc

# ===================================================

export CUDA_VISIBLE_DEVICES=4
model=/isilon/lab-xue/projects/data/logs/microscopy-FineTuning_Ryo_tile_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240129_141121_596371_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Ryo_tile/test/noisy_rois/ --output_dir /isilon/lab-xue/projects/microscopy/snr/Ryo_tile --saved_model_path $model --image_order HWT --added_noise_sd 1 --rep 32 --no_clip_data --batch_size 2 --patch_size_inference 64 --scaling_vals 0 256

python3 ./projects/microscopy_denoise/microscopy_inference_for_uncertainty_PCA.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Ryo_tile/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Ryo_tile --saved_model_path $model --image_order HWT --no_clip_data --patch_size_inference 64 --scaling_vals 0 256 --batch_size 2 --frame -1 --low_acc

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Ryo_tile/test/noisy_rois --output_dir /isilon/lab-xue/projects/microscopy/snr/Ryo_tile/res_rois --saved_model_path $model --image_order HWT --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256  --cuda_devices 0 1 2 3 

python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Ryo_tile/test/noisy --output_dir /isilon/lab-xue/projects/microscopy/snr/Ryo_tile/res --saved_model_path $model --image_order THW --no_clip_data --batch_size 2 --patch_size_inference 64  --scaling_vals 0 256  --cuda_devices 0 1 2 3 

# ===================================================
# run a inference
python3 ./projects/microscopy_denoise/microscopy_inference.py --input_dir /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy_rois --input_file_s Image_001.npy --output_dir /isilon/lab-xue/projects/microscopy/snr/Alex_wide_field/res --saved_model_path /isilon/lab-xue/projects/data/logs/microscopy-Alex_wide_test_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240121_200048_643087_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch --image_order HWT --no_clip_data --batch_size 2 --patch_size_inference 64

