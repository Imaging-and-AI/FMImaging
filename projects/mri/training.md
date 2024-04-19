# MRI image enhancement training

# run the second training

```

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

# base training
python3 ./projects/mri/inference/run_mri.py --standalone --nproc_per_node 8 --num_epochs 120 --batch_size 8 --run_extra_note 1st --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-4 --lr_pre 1e-4 --lr_post 1e-4 --lr_backbone 1e-4 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 120 --min_noise_level 0.1 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /data/logs --add_salt_pepper --add_possion

python3 ./projects/mri/inference/run_mri.py --nnodes 6 --rdzv_endpoint 172.16.0.5:9050 --node_rank 0 --nproc_per_node 4 --num_epochs 30 --batch_size 4 --run_extra_note 1st_vgg10_30_more_epochs --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1T1L1G1 T1L1G1T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-5 --lr_pre 1e-5 --lr_post 1e-5 --lr_backbone 1e-5 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.75 --losses perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 10.0 1.0 10.0 1.0 --min_noise_level 0.01 --max_noise_level 100 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /export/Lab-Xue/projects/mri/data --log_root /export/Lab-Xue/projects/data/logs --add_salt_pepper --add_possion --weighted_loss_snr --load_path /export/Lab-Xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231207_032002_166088_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_24 --scheduler_type OneCycleLR


python3 ./projects/mri/inference/run_mri.py --standalone  --nproc_per_node 8 --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231126_141320_915089_STCNNT_MRI_C-64-1_amp-False_complex_residual_with_data_degrading-T1L1G1T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_47 --model_type STCNNT_MRI --losses perpendicular  perceptual charbonnier  --loss_weights 1 1 1 1.0 1.0 --min_noise_level 0.1 --max_noise_level 100 --lr_pre 1e-5 --lr_backbone 1e-5 --lr_post 1e-5 --global_lr 1e-5 --run_extra_note 1st_more_epochs_perf_cine_NN80_perp1  --data_root /data/FM_data_repo/mri --num_epochs 40 --batch_size 4 --model_backbone STCNNT_HRNET --model_block_str T1L1G1T1L1G1 T1L1G1T1L1G1 --scheduler_factor 0.5 --disable_LSUV --log_root /export/Lab-Xue/projects/data/logs --continued_training --scheduler_type ReduceLROnPlateau --train_files train_3D_3T_retro_cine_2020.h5 BARTS_Perfusion_3T_2023.h5 --add_salt_pepper --add_possion --weighted_loss_snr

python3 ./projects/mri/inference/run_mri.py --nnodes 2 --rdzv_endpoint 172.16.2.25:9050 --node_rank 0 --nproc_per_node 8 --num_epochs 60 --batch_size 8 --run_extra_note 1st --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-4 --lr_pre 1e-4 --lr_post 1e-4 --lr_backbone 1e-4 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 100 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/Lab-Xue/projects/data/logs --add_salt_pepper --add_possion --weighted_loss_snr


python3 ./projects/mri/inference/run_mri.py --standalone --nproc_per_node 4 --num_epochs 30 --batch_size 8 --run_extra_note 1st --num_workers 32 --model_backbone omnivore --model_type omnivore_MRI --model_block_str T1L1G1 T1L1G1 --mri_height 64 --mri_width 64 --global_lr 1e-3 --lr_pre 1e-3 --lr_post 1e-3 --lr_backbone 1e-3 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 100 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/Lab-Xue/projects/data/logs --add_salt_pepper --add_possion --weighted_loss_snr --scheduler_type ReduceLROnPlateau --scheduler_factor 0.5

# ---------------
# 1st
python3 ./projects/mri/inference/run_mri.py --nnodes 2 --rdzv_endpoint 10.180.91.25:9050 --node_rank 0 --nproc_per_node 8 --num_epochs 20 --batch_size 8 --run_extra_note 1st --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-4 --lr_pre 1e-4 --lr_post 1e-4 --lr_backbone 1e-4 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --min_noise_level 0.1 --max_noise_level 60 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/lab-xue/projects/data/logs --add_salt_pepper --add_possion

# continue
python3 ./projects/mri/inference/run_mri.py --nnodes 2 --rdzv_endpoint 10.180.91.25:9050 --node_rank 0 --nproc_per_node 8 --num_epochs 20 --batch_size 8 --run_extra_note 1st --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-5 --lr_pre 1e-5 --lr_post 1e-5 --lr_backbone 1e-5 --run_list 0 --tra_ratio 90 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --min_noise_level 0.1 --max_noise_level 120 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/lab-xue/projects/data/logs --add_salt_pepper --add_possion  --load_path /export/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20240305_163602_949329_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_59 --continued_training --scheduler_type ReduceLROnPlateau

# ------------------------------------------------------------
# test for overfitting

python3 ./projects/mri/inference/run_mri.py --standalone --nproc_per_node 4 --num_epochs 30 --batch_size 8 --run_extra_note 1st_tra20_val10 --num_workers 32 --model_backbone STCNNT_HRNET --model_type STCNNT_MRI --model_block_str T1L1G1 T1L1G1 --mri_height 32 64 --mri_width 32 64 --global_lr 1e-4 --lr_pre 1e-4 --lr_post 1e-4 --lr_backbone 1e-4 --run_list 0 --tra_ratio 20 --val_ratio 10 --scheduler_factor 0.5 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --max_noise_level 100 --norm_mode instance2d --backbone_C 64 --disable_LSUV --data_root /data/FM_data_repo/mri --log_root /export/Lab-Xue/projects/data/logs --add_salt_pepper --add_possion --weighted_loss_snr

# test for PCA uncertainty
python3 ./projects/mri/inference/run_inference_for_uncertainty_PCA.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/res_test --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im_c --gmap_fname gmap_c --saved_model_path /export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-30.pth --model_type STCNNT_MRI --data_root /data/FM_data_repo/mri --train_files train_3D_3T_retro_cine_2020.h5 --ratio 10 5 10 --data_x_y_mode True --complex_i 

model=/export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-30.pth
res_dir=res

model=/export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-30.pth
res_dir=res

model=/export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-30.pth
res_dir=res

model=/export/Lab-Xue/projects/data/logs/mri-1st_lr1e-4_omnivore_T1L1G1_T1L1G1_20231210_160530_129063_omnivore_MRI_NN_100.0_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1/mri-1st_lr1e-4_omnivore_T1L1G1_T1L1G1_20231210_160530_129063_omnivore_MRI_NN_100.0_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1_epoch-30.pth
res_dir=res_omnivore

model=/export/Lab-Xue/projects/data/logs/mri-1st_STCNNT_UNET_C3C3C3_C3C3C3_20231215_212141_363603_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-C3C3C3_C3C3C3/mri-1st_STCNNT_UNET_C3C3C3_C3C3C3_20231215_212141_363603_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-C3C3C3_C3C3C3_epoch-30.pth
res_dir=res_unet_C3C3C3_C3C3C3

case=Retro_Lin_Cine_2DT_LAX_GLS_66016_026197138_026197147_246_20230522-132310_slc_1

case=Retro_Lin_Cine_2DT_LAX_GLS_66016_029244964_029244973_1044_20230525-093212_slc_5

python3 ./projects/mri/inference/run_inference_for_uncertainty_PCA.py --input_dir /export/Lab-Xue/projects/mri/data/mri_test/${case} --output_dir /export/Lab-Xue/projects/mri/data/mri_test/${case}/${res_dir} --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname noisy_c --gmap_fname gmap_c --saved_model_path ${model} --model_type STCNNT_MRI --data_root /data/FM_data_repo/mri --train_files train_3D_3T_retro_cine_2020.h5 --ratio 10 5 10 --data_x_y_mode True --complex_i --low_acc 

cases=(
        Retro_Lin_Cine_2DT_LAX_GLS_66016_006537389_006537398_271_20230515-124045_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_696272449_696272458_103_20230116-095714_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_010403878_010403887_51_20230516-083553_slc_4     Retro_Lin_Cine_2DT_LAX_GLS_66016_696272503_696272512_215_20230116-120330_slc_2
        Retro_Lin_Cine_2DT_LAX_GLS_66016_010403959_010403968_190_20230516-104136_slc_2    Retro_Lin_Cine_2DT_LAX_GLS_66016_701089318_701089327_45_20230117-083455_slc_10
        Retro_Lin_Cine_2DT_LAX_GLS_66016_010404040_010404049_370_20230516-143342_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_701089345_701089354_79_20230117-091837_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_010404067_010404076_409_20230516-150558_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_701089426_701089435_274_20230117-122202_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_018168903_018168912_115_20230518-091331_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_701089534_701089543_475_20230117-155707_slc_12
        Retro_Lin_Cine_2DT_LAX_GLS_66016_019479704_019479713_337_20230518-165716_slc_12   Retro_Lin_Cine_2DT_LAX_GLS_66016_701089534_701089543_475_20230117-155707_slc_6
        Retro_Lin_Cine_2DT_LAX_GLS_66016_022167752_022167761_410_20230519-154215_slc_5    Retro_Lin_Cine_2DT_LAX_GLS_66016_707659447_707659456_305_20230119-083227_slc_10
        Retro_Lin_Cine_2DT_LAX_GLS_66016_026197138_026197147_231_20230522-130828_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_713786918_713786927_55_20230120-084153_slc_13
        Retro_Lin_Cine_2DT_LAX_GLS_66016_026197219_026197228_388_20230522-164718_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_713786972_713786981_161_20230120-111910_slc_10
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029244535_029244544_190_20230523-121058_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_713786972_713786981_174_20230120-113307_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029244685_029244694_536_20230524-084138_slc_6    Retro_Lin_Cine_2DT_LAX_GLS_66016_713786999_713787008_210_20230120-122225_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029244730_029244739_596_20230524-094026_slc_2    Retro_Lin_Cine_2DT_LAX_GLS_66016_713787080_713787089_366_20230120-154119_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029244784_029244793_704_20230524-113731_slc_4    Retro_Lin_Cine_2DT_LAX_GLS_66016_713787080_713787089_389_20230120-161506_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029244937_029244946_992_20230525-083019_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_713787107_713787116_425_20230120-164424_slc_5
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029244964_029244973_1044_20230525-093212_slc_10  Retro_Lin_Cine_2DT_LAX_GLS_66016_912968311_912968320_190_20230403-113357_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029244964_029244973_1044_20230525-093212_slc_12  Retro_Lin_Cine_2DT_LAX_GLS_66016_912968311_912968320_191_20230403-113420_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029245153_029245162_1394_20230525-164809_slc_13  Retro_Lin_Cine_2DT_LAX_GLS_66016_916653770_916653779_316_20230404-144815_slc_10
        Retro_Lin_Cine_2DT_LAX_GLS_66016_029245153_029245162_1394_20230525-164809_slc_9   Retro_Lin_Cine_2DT_LAX_GLS_66016_920405569_920405578_108_20230405-093829_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_038583334_038583343_38_20230526-082216_slc_6     Retro_Lin_Cine_2DT_LAX_GLS_66016_920405731_920405740_406_20230405-145431_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_038583385_038583394_131_20230526-104034_slc_5    Retro_Lin_Cine_2DT_LAX_GLS_66016_924436060_924436069_125_20230406-113255_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_038583574_038583583_471_20230526-161939_slc_5    Retro_Lin_Cine_2DT_LAX_GLS_66016_927025756_927025765_131_20230411-103309_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_042237101_042237110_318_20230530-135151_slc_12   Retro_Lin_Cine_2DT_LAX_GLS_66016_927025756_927025765_139_20230411-103737_slc_13
        Retro_Lin_Cine_2DT_LAX_GLS_66016_042237182_042237191_447_20230530-161916_slc_2    Retro_Lin_Cine_2DT_LAX_GLS_66016_927025810_927025819_254_20230411-122204_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_046185591_046185600_218_20230531-124114_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_927025891_927025900_360_20230411-144025_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_050658369_050658378_101_20230601-094822_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_927025918_927025927_401_20230411-152547_slc_7
        Retro_Lin_Cine_2DT_LAX_GLS_66016_050658479_050658488_323_20230601-161128_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_927026057_927026066_639_20230412-102212_slc_4
        Retro_Lin_Cine_2DT_LAX_GLS_66016_053329985_053329994_97_20230602-103640_slc_3     Retro_Lin_Cine_2DT_LAX_GLS_66016_927026057_927026066_639_20230412-102212_slc_5
        Retro_Lin_Cine_2DT_LAX_GLS_66016_059440301_059440310_289_20230606-142431_slc_4    Retro_Lin_Cine_2DT_LAX_GLS_66016_927026057_927026066_644_20230412-103118_slc_3
        Retro_Lin_Cine_2DT_LAX_GLS_66016_059440301_059440310_304_20230606-143949_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_934594233_934594242_277_20230413-132714_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_063913305_063913314_658_20230608-092612_slc_10   Retro_Lin_Cine_2DT_LAX_GLS_66016_934594374_934594383_642_20230414-093219_slc_8
        Retro_Lin_Cine_2DT_LAX_GLS_66016_063913437_063913446_932_20230608-152921_slc_2    Retro_Lin_Cine_2DT_LAX_GLS_66016_934594455_934594464_819_20230414-123010_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_075234342_075234351_60_20230612-084147_slc_1     Retro_Lin_Cine_2DT_LAX_GLS_66016_934594513_934594522_909_20230414-143420_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_075234396_075234405_133_20230612-095926_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_941869606_941869615_37_20230417-080850_slc_9
        Retro_Lin_Cine_2DT_LAX_GLS_66016_078855260_078855269_140_20230613-114253_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_941869633_941869642_102_20230417-094408_slc_2
        Retro_Lin_Cine_2DT_LAX_GLS_66016_078855314_078855323_209_20230613-124845_slc_8    Retro_Lin_Cine_2DT_LAX_GLS_66016_946046610_946046619_237_20230418-123851_slc_2
        Retro_Lin_Cine_2DT_LAX_GLS_66016_078855341_078855350_261_20230613-134112_slc_3    Retro_Lin_Cine_2DT_LAX_GLS_66016_953894519_953894528_172_20230421-111216_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_663651905_663651914_84_20230103-090942_slc_4     Retro_Lin_Cine_2DT_LAX_GLS_66016_953894710_953894719_512_20230421-171041_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_663651932_663651941_126_20230103-100702_slc_11   Retro_Lin_Cine_2DT_LAX_GLS_66016_953894743_953894752_570_20230424-093205_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_663651959_663651968_188_20230103-115452_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_953894851_953894860_762_20230424-141430_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_663651986_663651995_230_20230103-125428_slc_5    Retro_Lin_Cine_2DT_LAX_GLS_66016_961332807_961332816_139_20230425-104848_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_663652013_663652022_283_20230103-134829_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_961332946_961332955_404_20230425-151141_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_663652125_663652134_503_20230104-091802_slc_6    Retro_Lin_Cine_2DT_LAX_GLS_66016_961332946_961332955_407_20230425-151507_slc_2
        Retro_Lin_Cine_2DT_LAX_GLS_66016_663652287_663652296_871_20230104-161620_slc_11   Retro_Lin_Cine_2DT_LAX_GLS_66016_961332973_961332982_456_20230425-161052_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_670647846_670647855_38_20230105-081644_slc_7     Retro_Lin_Cine_2DT_LAX_GLS_66016_961332973_961332982_466_20230425-162308_slc_11
        Retro_Lin_Cine_2DT_LAX_GLS_66016_670647873_670647882_95_20230105-093224_slc_1     Retro_Lin_Cine_2DT_LAX_GLS_66016_961333027_961333036_564_20230426-085001_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_670647927_670647936_208_20230105-113846_slc_2    Retro_Lin_Cine_2DT_LAX_GLS_66016_961333137_961333146_766_20230426-132739_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_670647981_670647990_289_20230105-134628_slc_3    Retro_Lin_Cine_2DT_LAX_GLS_66016_961333137_961333146_766_20230426-132739_slc_3
        Retro_Lin_Cine_2DT_LAX_GLS_66016_670648008_670648017_336_20230105-153253_slc_9    Retro_Lin_Cine_2DT_LAX_GLS_66016_974505563_974505572_131_20230428-111927_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_670648008_670648017_343_20230105-154126_slc_10   Retro_Lin_Cine_2DT_LAX_GLS_66016_974505725_974505734_417_20230428-174714_slc_10
        Retro_Lin_Cine_2DT_LAX_GLS_66016_676791873_676791882_97_20230109-100642_slc_2     Retro_Lin_Cine_2DT_LAX_GLS_66016_974505725_974505734_417_20230428-174714_slc_11
        Retro_Lin_Cine_2DT_LAX_GLS_66016_676791873_676791882_97_20230109-100642_slc_6     Retro_Lin_Cine_2DT_LAX_GLS_66016_977913409_977913418_84_20230502-090658_slc_12
        Retro_Lin_Cine_2DT_LAX_GLS_66016_676791900_676791909_152_20230109-110451_slc_5    Retro_Lin_Cine_2DT_LAX_GLS_66016_981272102_981272111_51_20230503-083258_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_676791927_676791936_194_20230109-132328_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_981272129_981272138_91_20230503-092328_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_676792271_676792280_887_20230110-160235_slc_11   Retro_Lin_Cine_2DT_LAX_GLS_66016_984794716_984794725_146_20230504-101718_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_676792381_676792390_1131_20230111-104830_slc_1   Retro_Lin_Cine_2DT_LAX_GLS_66016_984794743_984794752_188_20230504-105226_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_676792574_676792583_1534_20230112-084259_slc_12  Retro_Lin_Cine_2DT_LAX_GLS_66016_984794851_984794860_415_20230504-151515_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_676792601_676792610_1562_20230112-091742_slc_1   Retro_Lin_Cine_2DT_LAX_GLS_66016_988695213_988695222_301_20230505-123745_slc_7
        Retro_Lin_Cine_2DT_LAX_GLS_66016_691587622_691587631_38_20230113-083506_slc_4     Retro_Lin_Cine_2DT_LAX_GLS_66016_988695240_988695249_372_20230505-140215_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_691587676_691587685_162_20230113-114343_slc_1    Retro_Lin_Cine_2DT_LAX_GLS_66016_988695240_988695249_372_20230505-140215_slc_3
        Retro_Lin_Cine_2DT_LAX_GLS_66016_691587784_691587793_367_20230113-155416_slc_7    Retro_Lin_Cine_2DT_LAX_GLS_66016_988695240_988695249_372_20230505-140215_slc_5
        Retro_Lin_Cine_2DT_LAX_GLS_66016_691587811_691587820_415_20230113-165942_slc_6    Retro_Lin_Cine_2DT_LAX_GLS_66016_993085128_993085137_265_20230509-124912_slc_1
        Retro_Lin_Cine_2DT_LAX_GLS_66016_691587811_691587820_421_20230113-170745_slc_2    Retro_Lin_Cine_2DT_LAX_GLS_66016_993085182_993085191_356_20230509-143917_slc_8
        Retro_Lin_Cine_2DT_LAX_GLS_66016_696272422_696272431_64_20230116-092017_slc_1     Retro_Lin_Cine_2DT_LAX_GLS_66016_996656923_996656932_503_20230511-083402_slc_1
    )

export CUDA_VISIBLE_DEVICES=6
model=/export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-30.pth
res_dir=res

for index in ${!cases[*]}; do 
    echo "${cases[$index]}"
    python3 ./projects/mri/inference/run_inference_for_uncertainty_PCA.py --input_dir /export/Lab-Xue/projects/mri/data/model_uncertainty/${cases[$index]} --output_dir /export/Lab-Xue/projects/mri/data/model_uncertainty/${cases[$index]}/${res_dir} --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname noisy_c --gmap_fname gmap_c --saved_model_path ${model} --model_type STCNNT_MRI --data_root /data/FM_data_repo/mri --train_files train_3D_3T_retro_cine_2020.h5 --ratio 10 5 10 --data_x_y_mode True --complex_i --low_acc
done

export CUDA_VISIBLE_DEVICES=7
model=/export/Lab-Xue/projects/data/logs/mri-1st_STCNNT_UNET_C3C3C3_C3C3C3_20231215_212141_363603_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-C3C3C3_C3C3C3/mri-1st_STCNNT_UNET_C3C3C3_C3C3C3_20231215_212141_363603_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-C3C3C3_C3C3C3_epoch-30.pth
res_dir=res_unet_C3C3C3_C3C3C3

for index in ${!cases[*]}; do 
    echo "${cases[$index]}"
    python3 ./projects/mri/inference/run_inference_for_uncertainty_PCA.py --input_dir /export/Lab-Xue/projects/mri/data/model_uncertainty/${cases[$index]} --output_dir /export/Lab-Xue/projects/mri/data/model_uncertainty/${cases[$index]}/${res_dir} --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname noisy_c --gmap_fname gmap_c --saved_model_path ${model} --model_type STCNNT_MRI --data_root /data/FM_data_repo/mri --train_files train_3D_3T_retro_cine_2020.h5 --ratio 10 5 10 --data_x_y_mode True --complex_i --low_acc
done

model=/export/Lab-Xue/projects/data/logs/mri-1st_only_white_noise_STCNNT_UNET_C3C3C3_C3C3C3_20231216_030110_851970_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_only_white_noise-C3C3C3_C3C3C3/mri-1st_only_white_noise_STCNNT_UNET_C3C3C3_C3C3C3_20231216_030110_851970_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_only_white_noise-C3C3C3_C3C3C3_epoch-30.pth
res_dir=res_unet_C3C3C3_C3C3C3_only_white_noise

export CUDA_VISIBLE_DEVICES=5

for index in ${!cases[*]}; do 
    echo "${cases[$index]}"
    python3 ./projects/mri/inference/run_inference_for_uncertainty_PCA.py --input_dir /export/Lab-Xue/projects/mri/data/model_uncertainty/${cases[$index]} --output_dir /export/Lab-Xue/projects/mri/data/model_uncertainty/${cases[$index]}/${res_dir} --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname noisy_c --gmap_fname gmap_c --saved_model_path ${model} --model_type STCNNT_MRI --data_root /data/FM_data_repo/mri --train_files train_3D_3T_retro_cine_2020.h5 --ratio 10 5 10 --data_x_y_mode True --complex_i --low_acc
done

# for LGE

export CUDA_VISIBLE_DEVICES=6
model=/export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-30.pth
res_dir=res

python3 ./projects/mri/inference/run_inference_for_uncertainty_PCA.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_WB_LGE_comparison_2023/20230703/WB_LGE_MOCO_AVE_OnTheFly_41837_1837798573_1837798582_338_20230703-134952/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_WB_LGE_comparison_2023/20230703/WB_LGE_MOCO_AVE_OnTheFly_41837_1837798573_1837798582_338_20230703-134952/${res_dir} --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path ${model} --model_type STCNNT_MRI --data_root /data/FM_data_repo/mri --train_files train_3D_3T_retro_cine_2020.h5 --ratio 10 5 10 --data_x_y_mode True --complex_i --low_acc --frame 0

python3 ./projects/mri/inference/run_inference_for_uncertainty_PCA.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00188_FID22498_G33_SAX5_FB_de_tpat3_res256_Ave16_BW610_PHASres84/res/DebugOutput --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00188_FID22498_G33_SAX5_FB_de_tpat3_res256_Ave16_BW610_PHASres84/${res_dir} --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path ${model} --model_type STCNNT_MRI --data_root /data/FM_data_repo/mri --train_files train_3D_3T_retro_cine_2020.h5 --ratio 10 5 10 --data_x_y_mode True --complex_i --low_acc --frame 0


model=/export/Lab-Xue/projects/data/logs/mri-1st_lr1e-4_omnivore_T1L1G1_T1L1G1_20231210_160530_129063_omnivore_MRI_NN_100.0_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1/mri-1st_lr1e-4_omnivore_T1L1G1_T1L1G1_20231210_160530_129063_omnivore_MRI_NN_100.0_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1_epoch-30.pth
res_dir=res_omnivore

for index in ${!cases[*]}; do 
    echo "${cases[$index]}"
    python3 ./projects/mri/inference/run_inference_for_uncertainty_PCA.py --input_dir /export/Lab-Xue/projects/mri/data/mri_test/${cases[$index]} --output_dir /export/Lab-Xue/projects/mri/data/mri_test/${cases[$index]}/${res_dir} --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname noisy_c --gmap_fname gmap_c --saved_model_path ${model} --model_type STCNNT_MRI --data_root /data/FM_data_repo/mri --train_files train_3D_3T_retro_cine_2020.h5 --ratio 10 5 10 --data_x_y_mode True --complex_i --low_acc
done

# ---------------------------------
# second stage training

# base on 1st net


python3 ./projects/mri/inference/run_mri.py --nnodes 4 --rdzv_endpoint 172.16.0.4 --node_rank 0 --nproc_per_node 4 --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch --model_type MRI_double_net --losses mse perpendicular perceptual charbonnier gaussian3D  --loss_weights 0.1 1 10 1.0 10.0 --min_noise_level 0.01 --max_noise_level 100.0 --lr_pre 1e-6 --lr_backbone 1e-6 --lr_post 1e-6 --global_lr 1e-6 --freeze_pre --freeze_backbone  --run_extra_note 2nd --data_root /export/Lab-Xue/projects/mri/data/ --num_epochs 30 --batch_size 4 --model_backbone STCNNT_HRNET --model_block_str T1L1G1 T1L1G1 --scheduler_factor 0.8 --not_load_post --disable_LSUV --post_model_of_1st_net /export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch_post.pth --add_salt_pepper --add_possion --scheduler_type ReduceLROnPlateau  


python3 ./projects/mri/inference/run_mri.py --nproc_per_node 8 --standalone --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch --model_type MRI_double_net --losses perpendicular perceptual charbonnier gaussian3D  --loss_weights 1 10 1.0 10.0 --min_noise_level 0.01 --max_noise_level 100.0 --lr_pre 1e-6 --lr_backbone 1e-6 --lr_post 1e-6 --global_lr 1e-6 --freeze_pre --freeze_backbone  --run_extra_note 2nd --data_root /data/FM_data_repo/mri --num_epochs 20 --batch_size 4 --model_backbone STCNNT_HRNET --model_block_str T1L1G1 T1L1G1 --scheduler_factor 0.5 --not_load_post --disable_LSUV --post_model_of_1st_net /export/Lab-Xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch_post.pth --add_salt_pepper --add_possion --weighted_loss_snr

# continue with the 2nd net
python3 ./projects/mri/inference/run_mri.py --nproc_per_node 4 --standalone --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/data/logs/mri-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231214_213835_461207_MRI_double_net_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1_T1L1G1/best_checkpoint_epoch_2 --model_type MRI_double_net --losses mse perpendicular perceptual charbonnier gaussian3D  --loss_weights 1 1 10 1.0 10.0 --min_noise_level 0.01 --max_noise_level 100.0 --lr_pre 1e-5 --lr_backbone 1e-5 --lr_post 1e-5 --global_lr 1e-5  --run_extra_note 2nd --data_root /export/Lab-Xue/projects/mri/data/ --num_epochs 20 --batch_size 4 --model_backbone STCNNT_HRNET --model_block_str T1L1G1 T1L1G1 --scheduler_factor 0.8 --disable_LSUV --continued_training --add_salt_pepper --add_possion --weighted_loss_snr --scheduler_type ReduceLROnPlateau --freeze_pre --freeze_backbone

python3 ./projects/mri/inference/run_mri.py --nnodes 8 --rdzv_endpoint 172.16.0.4 --node_rank 0 --nproc_per_node 4 --tra_ratio 90 --val_ratio 10 --load_path /export/Lab-Xue/projects/data/logs/mri-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231214_213835_461207_MRI_double_net_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1_T1L1G1/best_checkpoint_epoch_2 --model_type MRI_double_net --losses mse perpendicular perceptual charbonnier gaussian3D  --loss_weights 1 1 10 1.0 10.0 --min_noise_level 0.01 --max_noise_level 100.0 --lr_pre 1e-5 --lr_backbone 1e-5 --lr_post 1e-5 --global_lr 1e-5  --run_extra_note 2nd --data_root /export/Lab-Xue/projects/mri/data/ --num_epochs 20 --batch_size 4 --model_backbone STCNNT_HRNET --model_block_str T1L1G1 T1L1G1 --scheduler_factor 0.8 --disable_LSUV --continued_training --add_salt_pepper --add_possion --weighted_loss_snr --freeze_pre --freeze_backbone


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
