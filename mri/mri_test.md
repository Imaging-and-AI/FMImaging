### Run the example cases

```

# good model after gradient-deriv fine-tuning
model=/export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230702_013521_019623_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_epoch-59.pth

# large model
model=/export/Lab-Xue/projects/mri/models/mri-HRNET-20230708_034122_305779_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_03-41-31-20230708_best.pt

# different model sizes
model=/export/Lab-Xue/projects/mri/test/after_flash_attention/mri-HRNET-20230710_010701_408688_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-98.pth

# -------------------------------------------------------
# WB LGE

scaling_factor=1.0

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1194791055_1194791064_248_20230109-123219 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1194791055_1194791064_248_20230109-123219 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1199034792_1199034801_784_20230111-110935 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1199034792_1199034801_784_20230111-110935/res --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BWH/BWH_WB_LGE_2023_AI_denoising/20230208/WB_LGE_MOCO_AVE_OnTheFly_41144_01418721_01418731_1929_20230208-164114 --output_dir /export/Lab-Kellman/ReconResults/denoising/BWH/BWH_WB_LGE_2023_AI_denoising/20230208/WB_LGE_MOCO_AVE_OnTheFly_41144_01418721_01418731_1929_20230208-164114/res --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --gmap_fname gfactor --saved_model_path $model

# -------------------------------------------------------
# 3T perfusion

scaling_factor=0.5

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210316/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_2173099_2173108_894_20210316-154715 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210316/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_2173099_2173108_894_20210316-154715 --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# -------------------------------------------------------
# 1.5T perfusion

scaling_factor=0.25

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising/20221005/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636 --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230424/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_21768048_21768055_241_20230424-122508 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230424/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_21768048_21768055_241_20230424-122508 --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230119/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_51358165_51358175_441_20230119-155638 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230119/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_51358165_51358175_441_20230119-155638 --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230117/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_46001532_46001542_263_20230117-110441 --output_dir /export/Lab-Kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230117/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_46001532_46001542_263_20230117-110441/res --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# -------------------------------------------------------

scaling_factor=0.5

# R4
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804/res --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# R5
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/res --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# -------------------------------------------------------
# free max cine

scaling_factor=0.9

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res_ai --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/res_ai --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/res_ai --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00163_FID07569_G25_4CH_CINE_256_R4/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00163_FID07569_G25_4CH_CINE_256_R4/res_ai --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00685_FID08073_G25_4CH_CINE_256_R4ipat/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00685_FID08073_G25_4CH_CINE_256_R4ipat/res_ai --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00687_FID08077_REPEAT_G25_4CH_CINE_256_R4ipat/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00687_FID08077_REPEAT_G25_4CH_CINE_256_R4ipat/res_ai --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

# -------------------------------------------------------
# free max perf

scaling_factor=0.8

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00175_FID07581_G25_Perfusion_trufi_sr_tpat_3_192res/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00175_FID07581_G25_Perfusion_trufi_sr_tpat_3_192res/res_ai --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

# free max perf 256
python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00176_FID07582_G25_Perfusion_trufi_sr_tpat_4_256res/res/DebugOutput/ --output_dir /export/Lab-Xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00176_FID07582_G25_Perfusion_trufi_sr_tpat_4_256res/res_ai --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model


```