### Run the example cases

```
# patch_v2 branch

# 1st, unet, TLG, TLG
model=/export/Lab-Xue/projects/mri-main/logs/mri-main-1st_STCNNT_UNET_T1L1G1_T1L1G1_20231027_201703_437806_STCNNT_MRI_C-32-1_amp-True_complex_residual-T1L1G1_T1L1G1/mri-main-1st_STCNNT_UNET_T1L1G1_T1L1G1_20231027_201703_437806_STCNNT_MRI_C-32-1_amp-True_complex_residual-T1L1G1_T1L1G1_epoch-50.pth 

RES_DIR=res_1st_unet_TLG_TLG
model_type_str=STCNNT_MRI
scaling_factor=1.0

model=/export/Lab-Xue/projects/mri/test/mri-main-1st_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231029_195046_933339_STCNNT_MRI_C-32-1_amp-True_complex_residual-T1L1G1_T1L1G1T1L1G1/mri-main-1st_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231029_195046_933339_STCNNT_MRI_C-32-1_amp-True_complex_residual-T1L1G1_T1L1G1T1L1G1_epoch-50.pth

RES_DIR=res_1st_hrnet_TLG_TLGTLG
model_type_str=STCNNT_MRI
scaling_factor=1.0

# 2nd, unet
model=/export/Lab-Xue/projects/mri/test/mri-main-2nd_STCNNT_UNET_T1L1G1_T1L1G1_20231030_194521_497809_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8

# 2nd, hrnet, good ssim

model=/export/Lab-Xue/projects/mri-main/logs/mri-main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231104_204006_398966_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_12

model=/export/Lab-Xue/projects/mri-main/logs/mri-main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231104_204006_398966_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_12

model=/export/Lab-Xue/projects/mri-main/logs/mri-main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231106_221712_582764_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_14

model=/export/Lab-Xue/projects/mri/test/mri-main-1st_STCNNT_HRNET_T1L1G1_T1L1G1_20231107_131054_114694_STCNNT_MRI_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1/mri-main-1st_STCNNT_HRNET_T1L1G1_T1L1G1_20231107_131054_114694_STCNNT_MRI_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1_epoch-50.pth

model=/export/Lab-Xue/projects/mri-main/logs/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231113_125144_783329_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_16

model=/export/Lab-Xue/projects/mri-main/logs/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231113_125144_783329_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231113_125144_783329_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-20.pth

model=/export/Lab-Xue/projects/mri-main/logs/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231114_090037_843822_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_3

model=/export/Lab-Xue/projects/mri-main/logs/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231115_213057_794008_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231115_213057_794008_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-20.pth

model=/export/Lab-Xue/projects/mri-main/logs/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231116_221321_671150_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_0

model=/export/Lab-Xue/projects/mri-main/logs/mri_main-2nd_only_cine_no_percp_STCNNT_HRNET_T1L1G1_T1L1G1_20231116_083832_235452_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/mri_main-2nd_only_cine_no_percp_STCNNT_HRNET_T1L1G1_T1L1G1_20231116_083832_235452_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-10.pth

model=/export/Lab-Xue/projects/mri-main/logs/mri_main-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231117_212634_200520_MRI_double_net_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_15

model=/export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-10.pth

RES_DIR=res_2nd_hrnet_NN_40_TLG_TLG
model_type_str=MRI_double_net
scaling_factor=1.0

model=model=/export/Lab-Xue/projects/data/logs/mri-main-1st_NN_40_STCNNT_HRNET_T1T1T1_T1T1T1_20231119_124856_509059_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1T1T1_T1T1T1/mri-main-1st_NN_40_STCNNT_HRNET_T1T1T1_T1T1T1_20231119_124856_509059_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1T1T1_T1T1T1_epoch-30.pth

RES_DIR=res_1st_net_TTT_TTT
model_type_str=STCNNT_MRI
scaling_factor=1.0


model=/export/Lab-Xue/projects/data/logs/mri-main-1st_perp_charb_vgg_with_perf_STCNNT_HRNET_T1L1G1_T1L1G1_20231115_223334_623623_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/mri-main-1st_perp_charb_vgg_with_perf_STCNNT_HRNET_T1L1G1_T1L1G1_20231115_223334_623623_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-30.pth

model=/export/Lab-Xue/projects/data/logs/mri-main-1st_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231118_040652_735337_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_24

model=/export/Lab-Xue/projects/data/logs/mri-main-1st_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231118_040652_735337_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/mri-main-1st_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231118_040652_735337_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-30.pth

model=/export/Lab-Xue/projects/data/logs/mri_main-1st_BN_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231121_031351_308529_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/ mri_main-1st_BN_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231121_031351_308529_STCNNT_MRI_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-30.pth

RES_DIR=res_1st_hrnet_TLG_TLG
model_type_str=STCNNT_MRI
scaling_factor=1.0

export CUDA_VISIBLE_DEVICES=7
export DISABLE_FLOAT16_INFERENCE=True

# ======================================================================
# quick test case

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20231102_HV/meas_MID00542_FID20263_G25_4CH_CINE_256_R3ipat_85phase_res_BH/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20231102_HV/meas_MID00542_FID20263_G25_4CH_CINE_256_R3ipat_85phase_res_BH/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

# ======================================================================
# snr level test

case_dir=Retro_Lin_Cine_2DT_LAX_GLS_66016_078855422_078855431_409_20230613-154734_slc_1

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/projects/mri/data/mri_test/${case_dir}/ --output_dir /export/Lab-Xue/projects/mri/data/mri_test/${case_dir}/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname noisy --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

# ======================================================================
# val and test data

torchrun --standalone --nproc_per_node 4 ./projects/mri/run.py --ddp --data_dir /data1/mri/data --log_dir /export/Lab-Xue/projects/mri_main/logs --complex_i --train_model False --continued_training True --project mri_val_test --prefetch_factor 8 --batch_size 16 --time 12 --num_uploaded 128 --ratio 20 20 5 --max_load -1 --model_type MRI_double_net --train_files BARTS_RetroCine_3T_2023.h5 --test_files test_2D_sig_2_80_1000.h5 test_2DT_sig_2_80_2000.h5 --train_data_types 2dt 2dt 2dt 2dt 2dt 2dt 2dt 2dt 3d --test_data_types 2d 2dt 2d 2dt --backbone_model STCNNT_HRNET --wandb_dir /export/Lab-Xue/projects/mri/wandb --override --pre_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231120_113650_746129_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_9_pre.pth --backbone_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231120_113650_746129_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_9_backbone.pth --post_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231120_113650_746129_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_9_post.pth --post_model_of_1st_net /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231120_113650_746129_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_9_post.pth --freeze_pre True --freeze_backbone True --disable_LSUV --post_backbone STCNNT_HRNET --post_hrnet.block_str T1L1G1 T1L1G1 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --min_noise_level 2.0 --max_noise_level 80.0 --mri_height 32 64 --mri_width 32 64 --run_name Tra-2nd_NN_80_test-NN_80 --run_notes mri_test_2NN_80_on_80 --n_head 64 

torchrun --standalone --nproc_per_node 4 ./projects/mri/run.py --ddp --data_dir /data1/mri/data --log_dir /export/Lab-Xue/projects/mri_main/logs --complex_i --train_model False --continued_training True --project mri_val_test --prefetch_factor 8 --batch_size 16 --time 12 --num_uploaded 128 --ratio 20 20 5 --max_load -1 --model_type MRI_double_net --train_files BARTS_RetroCine_3T_2023.h5 --test_files test_2D_sig_2_80_1000.h5 test_2DT_sig_2_80_2000.h5 --train_data_types 2dt 2dt 2dt 2dt 2dt 2dt 2dt 2dt 3d --test_data_types 2d 2dt 2d 2dt --backbone_model STCNNT_HRNET --wandb_dir /export/Lab-Xue/projects/mri/wandb --override --pre_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8_pre.pth --backbone_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8_backbone.pth --post_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8_post.pth --post_model_of_1st_net /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8_post.pth --freeze_pre True --freeze_backbone True --disable_LSUV --post_backbone STCNNT_HRNET --post_hrnet.block_str T1L1G1 T1L1G1 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --min_noise_level 2.0 --max_noise_level 80.0 --mri_height 32 64 --mri_width 32 64 --run_name Tra-2nd_NN_40_test-NN_80 --run_notes mri_test_2NN_40_on_80 --n_head 64 


torchrun --standalone --nproc_per_node 4 ./projects/mri/run.py --ddp --data_dir /data1/mri/data --log_dir /export/Lab-Xue/projects/mri_main/logs --complex_i --train_model False --continued_training True --project mri_val_test --prefetch_factor 8 --batch_size 16 --time 12 --num_uploaded 128 --ratio 20 20 5 --max_load -1 --model_type MRI_double_net --train_files BARTS_RetroCine_3T_2023.h5 --test_files test_2D_sig_2_40_1000.h5 test_2DT_sig_2_40_2000.h5 --train_data_types 2dt 2dt 2dt 2dt 2dt 2dt 2dt 2dt 3d --test_data_types 2d 2dt 2d 2dt --backbone_model STCNNT_HRNET --wandb_dir /export/Lab-Xue/projects/mri/wandb --override --pre_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231120_113650_746129_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_9_pre.pth --backbone_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231120_113650_746129_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_9_backbone.pth --post_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231120_113650_746129_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_9_post.pth --post_model_of_1st_net /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_80_STCNNT_HRNET_T1L1G1_T1L1G1_20231120_113650_746129_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_9_post.pth --freeze_pre True --freeze_backbone True --disable_LSUV --post_backbone STCNNT_HRNET --post_hrnet.block_str T1L1G1 T1L1G1 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --min_noise_level 2.0 --max_noise_level 80.0 --mri_height 32 64 --mri_width 32 64 --run_name Tra-2nd_NN_80_test-NN_40 --run_notes mri_test_2NN_80_on_40 --n_head 64 

torchrun --standalone --nproc_per_node 4 ./projects/mri/run.py --ddp --data_dir /data1/mri/data --log_dir /export/Lab-Xue/projects/mri_main/logs --complex_i --train_model False --continued_training True --project mri_val_test --prefetch_factor 8 --batch_size 16 --time 12 --num_uploaded 128 --ratio 20 20 5 --max_load -1 --model_type MRI_double_net --train_files BARTS_RetroCine_3T_2023.h5 --test_files test_2D_sig_2_40_1000.h5 test_2DT_sig_2_40_2000.h5 --train_data_types 2dt 2dt 2dt 2dt 2dt 2dt 2dt 2dt 3d --test_data_types 2d 2dt 2d 2dt --backbone_model STCNNT_HRNET --wandb_dir /export/Lab-Xue/projects/mri/wandb --override --pre_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8_pre.pth --backbone_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8_backbone.pth --post_model_load_path /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8_post.pth --post_model_of_1st_net /export/Lab-Xue/projects/data/logs/mri_main-2nd_NN_40_STCNNT_HRNET_T1L1G1_T1L1G1_20231119_200635_979194_MRI_double_net_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/best_checkpoint_epoch_8_post.pth --freeze_pre True --freeze_backbone True --disable_LSUV --post_backbone STCNNT_HRNET --post_hrnet.block_str T1L1G1 T1L1G1 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --min_noise_level 2.0 --max_noise_level 80.0 --mri_height 32 64 --mri_width 32 64 --run_name Tra-2nd_NN_40_test-NN_40 --run_notes mri_test_2NN_40_on_40 --n_head 64 


# ======================================================================

python3 ./mri/eval_mri.py --test_files /export/Lab-Xue/projects/mri/data/retro_cine_3T_sigma_1_20_repeated_test.h5 --saved_model_path ${model} --num_uploaded 256 --save_samples --num_saved_samples 1024 --results_path /export/Lab-Xue/projects/mri/results/${RES_DIR} --model_type ${model_type_str} --scaling_factor 1.0

python3 ./mri/eval_mri.py --test_files /export/Lab-Xue/projects/mri/data/retro_cine_3T_sigma_1_20_repeated_test_3rd.h5 --saved_model_path ${model} --num_uploaded 256 --save_samples --num_saved_samples 1024 --results_path /export/Lab-Xue/projects/mri/results/${RES_DIR} --model_type ${model_type_str} --scaling_factor 1.0

# ======================================================================
## Run the batch

### Run the WB LGE

```
# on the raw images
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --saved_model_path $model  --model_type ${model_type_str}

# on the moco+ave images

python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_ave --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

```

### Run the DB LGE
```
# on the raw images
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising_AI_on_raw --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --saved_model_path $model --model_type ${model_type_str}

# on the moco+ave images

python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising_AI_on_ave --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

```

### Perfusion
```

# 3T
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising_AI --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

# 1.5T
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str} # --num_batches_to_process 2

```

# ======================================================================
# local PSF test

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/test_-1/1280/ --output_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/test_-1/1280//${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname test_1280_epoch_-1_1280_sigma_1.00_x --gmap_fname test_1280_epoch_-1_1280_sigma_1.00_gmap --saved_model_path $model --model_type ${model_type_str}


python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/test_-1/400/ --output_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/test_-1/400//${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname test_400_epoch_-1_400_sigma_1.00_x --gmap_fname test_400_epoch_-1_400_sigma_1.00_gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/ori/ --output_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/res/ori --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname x --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/perturb/ --output_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/res/perturb --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname x --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/perturb_2x/ --output_dir /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/res/perturb_2x --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname x --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

# ======================================================================

# knee

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/DataForHui/kneeH5/imagedata/20190104_200942_meas_MID00034_FID06440_t2_tse_tra/res/DebugOutput/ --output_dir /export/Lab-Kellman/Share/data/DataForHui/kneeH5/imagedata/20190104_200942_meas_MID00034_FID06440_t2_tse_tra/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input_ave0 --gmap_fname gmap_ave0 --saved_model_path $model --model_type ${model_type_str}

# spine

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/DataForHui/spineH5/imagedata/20181207_181828_meas_MID00150_FID01036_t2_tse_sag_p2/res/DebugOutput/ --output_dir /export/Lab-Kellman/Share/data/DataForHui/spineH5/imagedata/20181207_181828_meas_MID00150_FID01036_t2_tse_sag_p2/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input_ave0 --gmap_fname gmap_ave0 --saved_model_path $model --model_type ${model_type_str}

# neuro
python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/neuro/meas_MID00083_FID14721_t1_mprage_1mm_p4_pos50_ACPC_check/ --output_dir /export/Lab-Kellman/Share/data/neuro/meas_MID00083_FID14721_t1_mprage_1mm_p4_pos50_ACPC_check/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname im --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/neuro/meas_MID00094_FID14732_t2_spc_sag_1mm_p2X2/res/DebugOutput/ --output_dir /export/Lab-Kellman/Share/data/neuro/meas_MID00094_FID14732_t2_spc_sag_1mm_p2X2/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

#lung
python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/LungTSE_rawData/20181218_124140_meas_MID00542_FID03365_t2_tse_tra_p2_320_trig/res/DebugOutput/ --output_dir /export/Lab-Kellman/Share/data/LungTSE_rawData/20181218_124140_meas_MID00542_FID03365_t2_tse_tra_p2_320_trig/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model --model_type ${model_type_str}

#lung
python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/LungTSE_rawData/20181218_124140_meas_MID00542_FID03365_t2_tse_tra_p2_320_trig/res/DebugOutput/ --output_dir /export/Lab-Kellman/Share/data/LungTSE_rawData/20181218_124140_meas_MID00542_FID03365_t2_tse_tra_p2_320_trig/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model --model_type ${model_type_str}


T:\Share\data\neuro\meas_MID00083_FID14721_t1_mprage_1mm_p4_pos50_ACPC_check

T:\Share\data\LungTSE_rawData

T:\Share\data\DataForHui\kneeH5\imagedata\20190104_200259_meas_MID00033_FID06439_pd_tse_sag_384

T:\Share\data\DataForHui\spineH5\imagedata\20181207_181828_meas_MID00150_FID01036_t2_tse_sag_p2


# ======================================================================

# WB LGE


python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BWH/BWH_WB_LGE_2023_AI_denoising/20230208/WB_LGE_MOCO_AVE_OnTheFly_41144_01418721_01418731_1929_20230208-164114 --output_dir /export/Lab-Kellman/ReconResults/denoising/BWH/BWH_WB_LGE_2023_AI_denoising/20230208/WB_LGE_MOCO_AVE_OnTheFly_41144_01418721_01418731_1929_20230208-164114/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1525437056_1525437065_1150_20230405-115358 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1525437056_1525437065_1150_20230405-115358/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT_LGE/20230904/WB_LGE_MOCO_AVE_STCNNT_41837_2049151069_2049151078_178_20230904-105747/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT_LGE/20230904/WB_LGE_MOCO_AVE_STCNNT_41837_2049151069_2049151078_178_20230904-105747/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT_LGE/20230904/WB_LGE_MOCO_AVE_STCNNT_41837_2049151069_2049151078_179_20230904-105942/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT_LGE/20230904/WB_LGE_MOCO_AVE_STCNNT_41837_2049151069_2049151078_179_20230904-105942/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

# -------------------------------------------------------
# 3T perfusion

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210316/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_2173099_2173108_894_20210316-154715 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210316/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_2173099_2173108_894_20210316-154715/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210721/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_11661307_11661316_1331_20210721-151837 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210721/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_11661307_11661316_1331_20210721-151837/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}


# -------------------------------------------------------
# 1.5T perfusion

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230815/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_1986417889_1986417898_401_20230815-094846/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230815/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_1986417889_1986417898_401_20230815-094846/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230815/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_1986418053_1986418062_697_20230815-154903/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230815/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_1986418053_1986418062_697_20230815-154903/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230903/Perfusion_AIF_STCNNT_42110_56257534_56257543_3000004_20230903-172928/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230903/Perfusion_AIF_STCNNT_42110_56257534_56257543_3000004_20230903-172928/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising/20221005/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_0343475_0343484_1759_20210118-083835 --output_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_0343475_0343484_1759_20210118-083835/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_5346528_5346537_161_20210105-104117 --output_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_5346528_5346537_161_20210105-104117/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230424/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_21768048_21768055_241_20230424-122508 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230424/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_21768048_21768055_241_20230424-122508/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230119/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_51358165_51358175_441_20230119-155638 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230119/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_51358165_51358175_441_20230119-155638/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230117/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_46001532_46001542_263_20230117-110441 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230117/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_46001532_46001542_263_20230117-110441/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/MINNESOTA_UHVC_perf_stcnnt/20220517/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_26210628_26210638_148_20220517-100037/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/MINNESOTA_UHVC_perf_stcnnt/20220517/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_26210628_26210638_148_20220517-100037/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

# high res perfusion

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/Perfusion_AIF_2E_NL_Cloud_42170_49443333_49443342_657_20190330-124527/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/Perfusion_AIF_2E_NL_Cloud_42170_49443333_49443342_657_20190330-124527/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_072714971_072714980_550_20180718-175707/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_072714971_072714980_550_20180718-175707/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_303_20180808-183735/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_303_20180808-183735/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_322_20180808-185431/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_322_20180808-185431/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_408_20180906-123716/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_408_20180906-123716/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_410_20180906-125352/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_410_20180906-125352/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_49443137_49443146_219_20190329-133337/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_49443137_49443146_219_20190329-133337/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

cases=(
        Perfusion_AIF_2E_NL_Cloud_42170_072714971_072714980_550_20180718-175707
        Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_303_20180808-183735
        Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_322_20180808-185431
        Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_408_20180906-123716
        Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_410_20180906-125352
        Perfusion_AIF_2E_NL_Cloud_42170_49443137_49443146_219_20190329-133337
        Perfusion_AIF_2E_NL_Cloud_42170_49443461_49443470_1001_20190331-092527
        Perfusion_AIF_2E_NL_Cloud_42170_49443486_49443495_1068_20190331-102251
        Perfusion_AIF_2E_NL_Cloud_42170_55882022_55882031_89_20190401-084425
        Perfusion_AIF_2E_NL_Cloud_42170_90141277_90141286_902_20170622-150439
        Perfusion_AIF_2E_NL_Cloud_42170_90141277_90141286_916_20170622-151616
        Perfusion_AIF_2E_NL_Cloud_42170_99913385_99913394_383_20171129-160152
        Perfusion_AIF_2E_NL_Cloud_66097_19853195_19853203_3000002_20171207-151256
        Perfusion_AIF_2E_NL_Cloud_66097_29373222_29373230_45_20180926-092303
        Perfusion_AIF_2E_NL_Cloud_66097_46576496_46576504_228_20181010-161347
        Perfusion_AIF_2E_NL_Cloud_66097_52964208_52964216_3000002_20180110-161515
        Perfusion_AIF_2E_NL_Cloud_66097_5709937_5709942_106_20181016-173229
    )

for index in ${!cases[*]}; do 
    echo "${cases[$index]}"

    python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/${cases[$index]}/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/cloud_ai/${cases[$index]}/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

done

cases=(
    Perfusion_AIF_2E_NL_Cloud_66097_29373222_29373230_45_20180926-092303
        Perfusion_AIF_2E_NL_Cloud_66097_5709937_5709942_106_20181016-173229
    )

# -------------------------------------------------------

# R4
python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

# R5
python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}


# -------------------------------------------------------
# free max cine

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00163_FID07569_G25_4CH_CINE_256_R4/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00163_FID07569_G25_4CH_CINE_256_R4/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00685_FID08073_G25_4CH_CINE_256_R4ipat/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00685_FID08073_G25_4CH_CINE_256_R4ipat/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00687_FID08077_REPEAT_G25_4CH_CINE_256_R4ipat/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00687_FID08077_REPEAT_G25_4CH_CINE_256_R4ipat/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00417_FID09075_G25_4CH_CINE_192_R4ipat_BW401/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00417_FID09075_G25_4CH_CINE_192_R4ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00418_FID09076_G25_4CH_CINE_192_R3ipat_BW401/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00418_FID09076_G25_4CH_CINE_192_R3ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00419_FID09077_G25_4CH_CINE_192_R2ipat_BW401/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00419_FID09077_G25_4CH_CINE_192_R2ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00412_FID09070_G25_4CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00412_FID09070_G25_4CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00413_FID09071_G25_3CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00413_FID09071_G25_3CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00414_FID09072_G25_2CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00414_FID09072_G25_2CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00416_FID09074_G25_SAX_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00416_FID09074_G25_SAX_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00411_FID09069_G25_SAX_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00411_FID09069_G25_SAX_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00410_FID09068_REPEAT_FOV360_G25_3CH_CINE_256_R2ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00410_FID09068_REPEAT_FOV360_G25_3CH_CINE_256_R2ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00407_FID09065_G25_2CH_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00407_FID09065_G25_2CH_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00406_FID09064_G25_4CH_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00406_FID09064_G25_4CH_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}


python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20231005_HV1/meas_MID00163_FID16270_G25_CH4_CINE_256_R3ipat_85phase_res_BH/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20231005_HV1/meas_MID00163_FID16270_G25_CH4_CINE_256_R3ipat_85phase_res_BH/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------
# free max perf

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00175_FID07581_G25_Perfusion_trufi_sr_tpat_3_192res/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00175_FID07581_G25_Perfusion_trufi_sr_tpat_3_192res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_0 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_1 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_2 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input2 --gmap_fname gmap2 --saved_model_path $model  --model_type ${model_type_str}


python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_0 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_1 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_2 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input2 --gmap_fname gmap2 --saved_model_path $model  --model_type ${model_type_str}

# free max perf 256
python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00176_FID07582_G25_Perfusion_trufi_sr_tpat_4_256res/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00176_FID07582_G25_Perfusion_trufi_sr_tpat_4_256res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# new NV
python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00421_FID09079_G25_Perfusion_trufi_sr_tpat_3_192res_BW401/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00421_FID09079_G25_Perfusion_trufi_sr_tpat_3_192res_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00422_FID09080_G25_Perfusion_trufi_sr_tpat_4_256res_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00422_FID09080_G25_Perfusion_trufi_sr_tpat_4_256res_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------
# free max LGE

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00092_FID08362_G25_4CH_FB_de_tpat4_res256_Ave16/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00092_FID08362_G25_4CH_FB_de_tpat4_res256_Ave16/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00091_FID08361_G25_2CH_FB_de_tpat3_res256_Ave16/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00091_FID08361_G25_2CH_FB_de_tpat3_res256_Ave16/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00211_FID04109_G25_4CH_FB_de_snapshot_p3_BW500/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00211_FID04109_G25_4CH_FB_de_snapshot_p3_BW500/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00215_FID04113_G25_SAX_FB_de_snapshot_p3_BW500/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00215_FID04113_G25_SAX_FB_de_snapshot_p3_BW500/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00429_FID09087_G25_SAX3_FB_de_tpat3_res256_Ave16_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00429_FID09087_G25_SAX3_FB_de_tpat3_res256_Ave16_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00430_FID09088_G25_4CH_FB_de_tpat4_BW450_res256_Ave24_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00430_FID09088_G25_4CH_FB_de_tpat4_BW450_res256_Ave24_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00431_FID09089_G25_SAX3_FB_de_tpat4_BW450_res256_Ave24_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00431_FID09089_G25_SAX3_FB_de_tpat4_BW450_res256_Ave24_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------
# high-res cmr

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230824/Perfusion_AIF_STCNNT_41837_2020136443_2020136452_811_20230824-122633/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230824/Perfusion_AIF_STCNNT_41837_2020136443_2020136452_811_20230824-122633/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230824/RT_Cine_LIN_STCNNT_41837_2020136389_2020136398_731_20230824-111014/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230824/RT_Cine_LIN_STCNNT_41837_2020136389_2020136398_731_20230824-111014/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230824/RT_Cine_LIN_STCNNT_41837_2020136416_2020136425_774_20230824-115130/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230824/RT_Cine_LIN_STCNNT_41837_2020136416_2020136425_774_20230824-115130/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230824/LGE_MOCO_AVE_STCNNT_41837_2020136416_2020136425_772_20230824-115018/DebugOutput --output_dir /export/Lab-Kellman/ReconResults/Barts_STCNNT/20230824/LGE_MOCO_AVE_STCNNT_41837_2020136416_2020136425_772_20230824-115018/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

```

## Run the LPSF scripts

cd /export/Lab-Xue/projects/mri/results/test/mri-MRI_double_net_20230/scripts

for f in *.sh; do
  bash "$f" 
done


## run the snr test

python3 ./mri/run_inference_snr_pseudo_replica.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res_ai/ --im_scaling 1 --gmap_scaling 1 --saved_model_path /export/Lab-Xue/projects/mri/test/hy_search_contined_gaussian_2nd_stage/hy_search_contined_gaussian_2nd_stage_epoch-74.pth --input_fname input --gmap_fname gmap --scaling_factor 1.0 --added_noise_sd 0.1 --rep 32
