# General base scripts for microscopy

# Training from scratch
python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log \
    --train_files Base_All_train.h5 --test_files Base_Actin_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 100 --batch_size 8 --backbone_C 64 --standalone \
    --losses mse l1 ssim perceptual --loss_weights 1.0 1.0 5.0 5.0 --max_load 200 --num_workers 4 --save_samples

# Finetuning
python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log \
    --train_files Base_Actin_train.h5 --test_files Base_Actin_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 30 --batch_size 8 --backbone_C 64 --standalone \
    --losses mse l1 ssim perceptual --loss_weights 1.0 1.0 5.0 5.0 --max_load 200 --num_workers 4 --save_samples \
    --load_path /home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 5

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy \
    --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log \
    --train_files Alex_wide_field_train.h5 --test_files Alex_wide_field_test.h5 \
    --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 30 --batch_size 8 --backbone_C 64 --standalone \
    --losses mse l1 ssim perceptual --loss_weights 1.0 1.0 5.0 5.0 --max_load 200 --num_workers 4 --save_samples \
    --load_path /home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch \
    --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025 \
    --train_samples 5 --run_extra_note Alex_wide_test --no_clip_data

python3 ./projects/microscopy_denoise/microscopy_run_ddp.py --project microscopy     --data_root /home/gtuser/rehmana2/projects/stcnnt/data/ --log_root /home/gtuser/rehmana2/projects/stcnnt/log     --train_files Alex_wide_field_train.h5 --test_files Alex_wide_field_test.h5     --cuda_device 0,1,2,3 --nproc_per_node 4 --num_epochs 20 --batch_size 1 --backbone_C 64 --standalone     --losses mse l1 ssim --loss_weights 10.0 1.0 1.0 --max_load 200 --num_workers 4 --save_samples     --load_path /home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch     --global_lr 0.000025 --lr_pre  0.000025 --lr_backbone  0.000025 --lr_post  0.000025     --train_samples 200000 --run_extra_note Alex_wide_test --micro_height 64 128 --micro_width 64 128 --scheduler_type OneCycleLR

# Inference
python3 ./projects/microscopy_denoise/microscopy_inference.py \
    --input_dir /home/gtuser/rehmana2/projects/stcnnt/data/ --input_file_s Base_Actin_test.h5 \
    --output_dir /home/gtuser/rehmana2/projects/stcnnt/infer_results/ \
    --saved_model_path /home/gtuser/rehmana2/projects/stcnnt/log/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/microscopy-20240115_083657_781264_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-100.pth \
    --pad_time --image_order THW --device cuda --batch_size 16

# snr test

python3 ./projects/microscopy_denoise/microscopy_inference_pseudo_replica.py --input_dir /export/Lab-Xue/publications/CNNT_paper/data/micro_datasets_rcan/Alex_wide_field/test/noisy_rois --output_dir /export/Lab-Xue/projects/microscopy/snr/Alex_wide_field --saved_model_path /export/Lab-Xue/projects/data/logs/microscopy-Alex_wide_test_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20240120_205339_366857_STCNNT_Microscopy_C-64-1_amp-False_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch --image_order HWT --added_noise_sd 0.1 --rep 32 --no_clip_data --batch_size 2