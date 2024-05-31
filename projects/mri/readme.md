# MRI image enhancement with SNR unit training

This project provides model and training code for MRI image enhancement training. 

### Data

```
# training data
train_3D_3T_retro_cine_2018.h5
train_3D_3T_retro_cine_2019.h5
train_3D_3T_retro_cine_2020.h5

# test data
test_2DT_sig_1_120_2000.h5
```

### Training

```
export data_root=/data/mri
export log_root=/data/log/mri
export wandb_dir=/data/wandb
export NGPU=8

export num_epochs=75
export batch_size=16

# base training
python3 ./projects/mri/inference/run_mri.py --standalone --nproc_per_node ${NGPU} --num_epochs ${num_epochs} --batch_size ${batch_size} --run_extra_note 1st --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-4 --lr_pre 1e-4 --lr_post 1e-4 --lr_backbone 1e-4 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 120 --min_noise_level 0.1 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root ${data_root} --log_root ${log_root} --wandb_dir ${wandb_dir} --add_salt_pepper --add_possion
```