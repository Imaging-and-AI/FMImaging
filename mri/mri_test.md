### Run the example cases

```

# good model after gradient-deriv fine-tuning
model=/export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230702_013521_019623_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_epoch-59.pth

# large model
model=/export/Lab-Xue/projects/mri/models/mri-HRNET-20230708_034122_305779_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_03-41-31-20230708_best.pt

# medium model
model=/export/Lab-Xue/projects/mri/test/after_flash_attention/mri-HRNET-20230710_010701_408688_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_01-57-06-20230711_best.pt

# small model
model=/export/Lab-Xue/projects/mri/test/after_flash_attention/mri-HRNET-20230710_010701_409083_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_epoch-67.pth


# different model sizes
model=/export/Lab-Xue/projects/mri/test/after_flash_attention/mri-HRNET-20230710_010701_408688_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-98.pth

model=/export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230712_215823_066536_C-64-4_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-1024.pth

model=/export/Lab-Xue/projects/mri/test/after_flash_attention/mri-HRNET-20230710_010701_408769_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_19-56-09-20230711_best.pt

model=/export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230710_010701_408769_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-1024.pth

# double net

# 1st stage
model=/export/Lab-Xue/projects/mri/test/first_stage/mri-STCNNT_MRI_20230721_225151_726014_C-32-1_amp-True_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-67.pth

RES_DIR=res
model_type_str=STCNNT_MRI

# 2nd stage
model=/export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230716_190117_960318_C-32-1_amp-False_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-50.pth
RES_DIR=res_double_net
model_type_str=MRI_double_net

# new training
model=/export/Lab-Xue/projects/mri/test/mri_hrnet/mri-HRNET-20230720_002927_C-32-1_amp-False_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_19-01-29-20230716_best.pt

# less denoising, more sharpness
model=/export/Lab-Xue/projects/mri/test/second_stage/mri-MRI_double_net_20230722_190320_614230_C-32-1_amp-False_complex_residual_weighted_loss_snr_temporal_added_noise-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-69.pth

# more denoising, less sharpness
model=/export/Lab-Xue/projects/mri/test/second_stage/mri-MRI_double_net_20230722_235953_782390_C-32-1_amp-False_2nd_stage_perp_gaussian_ssim_complex_residual_weighted_loss_snr_temporal_added_noise-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-62.pth

# model after speed up and retraining
model=/export/Lab-Xue/projects/mri/test/second_stage/mri-MRI_double_net_20230731_225634_401217_C-32-1_amp-True_2nd_stage_noise_1to8_amp_complex_residual_weighted_loss_snr_temporal_added_noise-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-51.pth*

RES_DIR=res_double_net
model_type_str=MRI_double_net

scaling_factor=1.0

export CUDA_VISIBLE_DEVICES=7

# ======================================================================

python3 ./mri/eval_mri.py --test_files /export/Lab-Xue/projects/mri/data/retro_cine_3T_sigma_1_20_repeated_test.h5 --saved_model_path ${model} --num_uploaded 256 --save_samples --num_saved_samples 1024 --results_path /export/Lab-Xue/projects/mri/results/${RES_DIR} --model_type ${model_type_str} --scaling_factor 1.0

python3 ./mri/eval_mri.py --test_files /export/Lab-Xue/projects/mri/data/retro_cine_3T_sigma_1_20_repeated_test_2nd_random_mask.h5 --saved_model_path ${model} --num_uploaded 256 --save_samples --num_saved_samples 1024 --results_path /export/Lab-Xue/projects/mri/results/${RES_DIR}_random_mask --model_type ${model_type_str} --scaling_factor 1.0

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
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

```

# ======================================================================

# WB LGE


python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1194791055_1194791064_248_20230109-123219 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1194791055_1194791064_248_20230109-123219/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1199034792_1199034801_784_20230111-110935 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1199034792_1199034801_784_20230111-110935/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BWH/BWH_WB_LGE_2023_AI_denoising/20230208/WB_LGE_MOCO_AVE_OnTheFly_41144_01418721_01418731_1929_20230208-164114 --output_dir /export/Lab-Kellman/ReconResults/denoising/BWH/BWH_WB_LGE_2023_AI_denoising/20230208/WB_LGE_MOCO_AVE_OnTheFly_41144_01418721_01418731_1929_20230208-164114/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1525437056_1525437065_1150_20230405-115358 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1525437056_1525437065_1150_20230405-115358/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

# -------------------------------------------------------
# 3T perfusion

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210316/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_2173099_2173108_894_20210316-154715 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210316/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_2173099_2173108_894_20210316-154715/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210721/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_11661307_11661316_1331_20210721-151837 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210721/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_11661307_11661316_1331_20210721-151837/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}


# -------------------------------------------------------
# 1.5T perfusion

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising/20221005/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_659532_659541_396_20210203-084704 --output_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_659532_659541_396_20210203-084704/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_5346528_5346537_161_20210105-104117 --output_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_5346528_5346537_161_20210105-104117/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230424/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_21768048_21768055_241_20230424-122508 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230424/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_21768048_21768055_241_20230424-122508/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230119/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_51358165_51358175_441_20230119-155638 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230119/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_51358165_51358175_441_20230119-155638/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230117/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_46001532_46001542_263_20230117-110441 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230117/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_46001532_46001542_263_20230117-110441/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

# high res perfusion

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/Share/data/perfusion/cloud/Perfusion_AIF_2E_NL_Cloud_42170_49443333_49443342_657_20190330-124527/cloud_flow_res/DebugOutput --output_dir /export/Lab-Kellman/Share/data/perfusion/cloud/Perfusion_AIF_2E_NL_Cloud_42170_49443333_49443342_657_20190330-124527/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------

# R4
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

# R5
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}


# -------------------------------------------------------
# free max cine

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00163_FID07569_G25_4CH_CINE_256_R4/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00163_FID07569_G25_4CH_CINE_256_R4/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00685_FID08073_G25_4CH_CINE_256_R4ipat/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00685_FID08073_G25_4CH_CINE_256_R4ipat/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00687_FID08077_REPEAT_G25_4CH_CINE_256_R4ipat/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00687_FID08077_REPEAT_G25_4CH_CINE_256_R4ipat/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00417_FID09075_G25_4CH_CINE_192_R4ipat_BW401/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00417_FID09075_G25_4CH_CINE_192_R4ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00418_FID09076_G25_4CH_CINE_192_R3ipat_BW401/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00418_FID09076_G25_4CH_CINE_192_R3ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00419_FID09077_G25_4CH_CINE_192_R2ipat_BW401/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00419_FID09077_G25_4CH_CINE_192_R2ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00412_FID09070_G25_4CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00412_FID09070_G25_4CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00413_FID09071_G25_3CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00413_FID09071_G25_3CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00414_FID09072_G25_2CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00414_FID09072_G25_2CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00416_FID09074_G25_SAX_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00416_FID09074_G25_SAX_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00411_FID09069_G25_SAX_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00411_FID09069_G25_SAX_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00410_FID09068_REPEAT_FOV360_G25_3CH_CINE_256_R2ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00410_FID09068_REPEAT_FOV360_G25_3CH_CINE_256_R2ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00407_FID09065_G25_2CH_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00407_FID09065_G25_2CH_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00406_FID09064_G25_4CH_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00406_FID09064_G25_4CH_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------
# free max perf

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00175_FID07581_G25_Perfusion_trufi_sr_tpat_3_192res/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00175_FID07581_G25_Perfusion_trufi_sr_tpat_3_192res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_0 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_1 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_2 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input2 --gmap_fname gmap2 --saved_model_path $model  --model_type ${model_type_str}


python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_0 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_1 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_2 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input2 --gmap_fname gmap2 --saved_model_path $model  --model_type ${model_type_str}

# free max perf 256
python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00176_FID07582_G25_Perfusion_trufi_sr_tpat_4_256res/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00176_FID07582_G25_Perfusion_trufi_sr_tpat_4_256res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# new NV
python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00421_FID09079_G25_Perfusion_trufi_sr_tpat_3_192res_BW401/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00421_FID09079_G25_Perfusion_trufi_sr_tpat_3_192res_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00422_FID09080_G25_Perfusion_trufi_sr_tpat_4_256res_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00422_FID09080_G25_Perfusion_trufi_sr_tpat_4_256res_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------
# free max LGE

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00092_FID08362_G25_4CH_FB_de_tpat4_res256_Ave16/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00092_FID08362_G25_4CH_FB_de_tpat4_res256_Ave16/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00091_FID08361_G25_2CH_FB_de_tpat3_res256_Ave16/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00091_FID08361_G25_2CH_FB_de_tpat3_res256_Ave16/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00211_FID04109_G25_4CH_FB_de_snapshot_p3_BW500/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00211_FID04109_G25_4CH_FB_de_snapshot_p3_BW500/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00215_FID04113_G25_SAX_FB_de_snapshot_p3_BW500/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00215_FID04113_G25_SAX_FB_de_snapshot_p3_BW500/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00429_FID09087_G25_SAX3_FB_de_tpat3_res256_Ave16_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00429_FID09087_G25_SAX3_FB_de_tpat3_res256_Ave16_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00430_FID09088_G25_4CH_FB_de_tpat4_BW450_res256_Ave24_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00430_FID09088_G25_4CH_FB_de_tpat4_BW450_res256_Ave24_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00431_FID09089_G25_SAX3_FB_de_tpat4_BW450_res256_Ave24_BW399/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00431_FID09089_G25_SAX3_FB_de_tpat4_BW450_res256_Ave24_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}


```