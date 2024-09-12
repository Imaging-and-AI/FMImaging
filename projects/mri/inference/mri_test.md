### Run the example cases

export CUDA_VISIBLE_DEVICES=0
export DISABLE_FLOAT16_INFERENCE=True

# ------------------------

model=/data/mri/models/mri_denoising_s7__kings01_4xG16_V100/kings01_s7_hrnet_TLG_TLGTLG_160_OneCycleLR_instance2d_24.0_lr1e-4/checkpoints/mri-STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_kings01_s7_hrnet_TLG_TLGTLG_160_OneCycleLR_instance2d_24.0_lr1e-4_4x32G16-V100_NN_24.0_C-64_complex_residual/mri-STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_kings01_s7_hrnet_TLG_TLGTLG_160_OneCycleLR_instance2d_24.0_lr1e-4_4x32G16-V100_NN_24.0_C-64_complex_residual_epoch-160.pth

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data/mri/models/mri_denoising_s7__kings01_4xG16_V100/kings01_s7_unet_TLG_TLG_160_OneCycleLR_instance2d_24.0_lr1e-4/checkpoints/mri-STCNNT_UNET_T1L1G1_T1L1G1_kings01_s7_unet_TLG_TLG_160_OneCycleLR_instance2d_24.0_lr1e-4_4x32G16-V100_NN_24.0_C-64_complex_residual/mri-STCNNT_UNET_T1L1G1_T1L1G1_kings01_s7_unet_TLG_TLG_160_OneCycleLR_instance2d_24.0_lr1e-4_4x32G16-V100_NN_24.0_C-64_complex_residual_epoch-160.pth

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data/mri/models/mri_denoising_s9__kings01_4xG16_V100/kings01_s9_hrnet_TLG_TLG_60_OneCycleLR_layer_24.0_lr1e-4/checkpoints/mri-STCNNT_HRNET_T1L1G1_T1L1G1_kings01_s9_hrnet_TLG_TLG_60_OneCycleLR_layer_24.0_lr1e-4_4x32G16-V100_NN_24.0_C-64_complex_residual/mri-STCNNT_HRNET_T1L1G1_T1L1G1_kings01_s9_hrnet_TLG_TLG_60_OneCycleLR_layer_24.0_lr1e-4_4x32G16-V100_NN_24.0_C-64_complex_residual_epoch-60.pth

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data/mri/models/mri_denoising_s9__kings01_4xG16_V100/kings01_s9_unet_TTT_TTT_60_OneCycleLR_layer_24.0_lr1e-4/checkpoints/mri-STCNNT_UNET_T1T1T1_T1T1T1_kings01_s9_unet_TTT_TTT_60_OneCycleLR_layer_24.0_lr1e-4_4x32G16-V100_NN_24.0_C-64_complex_residual/mri-STCNNT_UNET_T1T1T1_T1T1T1_kings01_s9_unet_TTT_TTT_60_OneCycleLR_layer_24.0_lr1e-4_4x32G16-V100_NN_24.0_C-64_complex_residual_epoch-60.pth

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data/mri/models/mri_denoising_s10__kings01_8xG16_V100_epoch_160/kings01_s10_hrnet_TLG_TLGTLG_160_OneCycleLR_instance2d_120.0_lr1e-4/checkpoints/mri-STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_kings01_s10_hrnet_TLG_TLGTLG_160_OneCycleLR_instance2d_120.0_lr1e-4_8x32G16-V100_NN_120.0_C-64_complex_residual/mri-STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_kings01_s10_hrnet_TLG_TLGTLG_160_OneCycleLR_instance2d_120.0_lr1e-4_8x32G16-V100_NN_120.0_C-64_complex_residual_epoch-160.pth

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# long training, ------------------------

model=/data/mri/models/mri_denoising_s15__kings01_4xG16_V100_epoch_1000/kings01_s15_unet_TTT_TTT_1000_OneCycleLR_layer_120.0_lr2e-5/checkpoints/mri-STCNNT_UNET_T1T1T1_T1T1T1_kings01_s15_unet_TTT_TTT_1000_OneCycleLR_layer_120.0_lr2e-5_4x32G16-V100_NN_120.0_C-64_complex_residual/checkpoint_epoch_144

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

model=/data4/mri/models/mri-20240823_223234_444009_STCNNT_UNET_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_mri_denoising_s10__32.0_epoch_1000_NN_32.0_C-64_complex_residual/checkpoint_epoch_114

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data4/mri/models/mri_denoising_s14__msroctovc_4xG8_A100_epoch_1000/msroctovc_s14_hrnet_TLGTLGTLG_TLGTLGTLG_1000_OneCycleLR_layer_64.0_lr2e-5/checkpoints/mri-STCNNT_HRNET_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_msroctovc_s14_hrnet_TLGTLGTLG_TLGTLGTLG_1000_OneCycleLR_layer_64.0_lr2e-5_4x80G8-A100_NN_64.0_C-64_complex_residual/checkpoint_epoch_275

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data4/mri/models/mri_denoising_s15__kings01_4xG16_V100_epoch_1000/kings01_s15_unet_TTT_TTT_1000_OneCycleLR_layer_120.0_lr2e-5/checkpoints/mri-STCNNT_UNET_T1T1T1_T1T1T1_kings01_s15_unet_TTT_TTT_1000_OneCycleLR_layer_120.0_lr2e-5_4x32G16-V100_NN_120.0_C-64_complex_residual/checkpoint_epoch_413

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data/mri/models/mri_denoising_s14__msroctovc_4xG8_A100_epoch_1000/msroctovc_s14_hrnet_TLG_TLGTLG_1000_OneCycleLR_instance2d_120.0_lr1e-4/checkpoints/mri-STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_msroctovc_s14_hrnet_TLG_TLGTLG_1000_OneCycleLR_instance2d_120.0_lr1e-4_4x80G8-A100_NN_120.0_C-64_complex_residual/mri-STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_msroctovc_s14_hrnet_TLG_TLGTLG_1000_OneCycleLR_instance2d_120.0_lr1e-4_4x80G8-A100_NN_120.0_C-64_complex_residual_epoch-1000.pth

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data/mri/models/mri_denoising_s22__msroctovc_4xG8_A100_epoch_1000/msroctovc_s22_hrnet_TTT_TTT_1000_OneCycleLR_layer_32.0_lr1e-5/checkpoints/mri-STCNNT_HRNET_T1T1T1_T1T1T1_msroctovc_s22_hrnet_TTT_TTT_1000_OneCycleLR_layer_32.0_lr1e-5_4x80G8-A100_NN_32.0_C-64_complex_residual/checkpoint_epoch_522

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res


# ------------------------

model=/home/xueh/mri/logs/mri-20240823_223234_444009_STCNNT_UNET_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_mri_denoising_s10__32.0_epoch_1000_NN_32.0_C-64_complex_residual/checkpoint_epoch_308

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

# ------------------------

model=/data/mri/models/mri_denoising_s22__msroctovc_4xG8_A100_epoch_1000/msroctovc_s22_hrnet_TTT_TTT_1000_OneCycleLR_layer_32.0_lr1e-5/checkpoints/mri-STCNNT_HRNET_T1T1T1_T1T1T1_msroctovc_s22_hrnet_TTT_TTT_1000_OneCycleLR_layer_32.0_lr1e-5_4x80G8-A100_NN_32.0_C-64_complex_residual/checkpoint_epoch_996

model_type_str=STCNNT_MRI
scaling_factor=1.0
RES_DIR=res

model=/data/models/mri_denoising_s23_window_patch_sizes__kings01_4xG16_V100_epoch_256/kings01_s23_window_patch_sizes_hrnet_TLG_TLG_256_batchsize_1_OneCycleLR_layer_32.0_lr1e-5_mri_width_64_win_size_16_16_patch_size_2_2/checkpoints/mri-STCNNT_HRNET_T1L1G1_T1L1G1_kings01_s23_window_patch_sizes_hrnet_TLG_TLG_256_batchsize_1_OneCycleLR_layer_32.0_lr1e-5_mri_width_64_win_size_16_16_patch_size_2_2_4x32G16-V100_NN_32.0_32.0_C-64_complex_residual/checkpoint_epoch_254

model=/data/models/mri_denoising_s21__kings01_4xG16_V100_epoch_1000/kings01_s21_unet_TTT_TTT_1000_OneCycleLR_layer_32.0_lr1e-5/checkpoints/mri-STCNNT_UNET_T1T1T1_T1T1T1_kings01_s21_unet_TTT_TTT_1000_OneCycleLR_layer_32.0_lr1e-5_4x32G16-V100_NN_32.0_C-64_complex_residual/checkpoint_epoch_846

# ------------------------

data_dir=/data2/raw_data/data_local/perf/Perfusion_AIFR3_2E_Interleaved_000000_18756389_18756397_62_00000000-000000/res/DebugOutput

python3 ./projects/mri/inference/run_inference.py --input_dir ${data_dir} --output_dir ${data_dir}/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# ------------------------

python3 ./projects/mri/save_pt_as_onnx.py --input $model --only_save --output ~/models/model.pth

cp $model  ~/buildkite/cmr_ml-source/deployment/networks/
cp $model  $GADGETRON_MODEL_HOME 