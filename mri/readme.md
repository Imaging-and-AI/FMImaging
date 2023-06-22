# MRI image enhancement training

This project provides model and training code for MRI image enhancement training. 

### wandb sweep

To run the parameter sweep, a sweep configuration file is needed (e.g. [sweep_conf](./sweep_conf.py)). 

```
# first, generate the sweep and record the sweep in
python3 ./sweep_conf.py

# second, run the sweep

# run on 2 gpus, with 2 processes
sh ./run_sweep.sh -g 0,1 -n 2 -s $sweep_id  -r 100 -p 9001

# run on 4 gpus, with 4 processes
sh ./run_sweep.sh -g 0,1,2,3 -n 4 -s $sweep_id  -r 100 -p 9001

# run on one gpu
sh ./run_sweep.sh -g 0 -n 1 -s $sweep_id -r 100 -p 9001

```

**Warning** Current wandb sweep does not work well with toruchrun for multi-gpu training. So current solution is good for one gpu and one process usecage (e.g. -g 0 -n 1).

### multi-node training

on the local set up

- single node, single gpu training
```
torchrun --nproc_per_node 1 --standalone $HOME/mrprogs/STCNNT.git/mri/main_mri.py --ddp

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 1 --standalone

```

- single node, multiple gpu training
```
torchrun --nproc_per_node 2 --standalone $HOME/mrprogs/STCNNT.git/mri/main_mri.py --ddp

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 2 --standalone

```

- two nodes, multiple gpu training
```
# every node has two processes, two nodes are trained together
# gt7
torchrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 mri/main_mri.py --ddp
# gt3
torchrun --nproc_per_node 2 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001 mri/main_mri.py --ddp

# gt7
python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 2 --nnodes 2 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001
# gt3
python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 2 --nnodes 2 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint gt7.nhlbi.nih.gov:9001


```

- four nodes, on cloud
```

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 4 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001 --tra_ratio 99 --val_ratio 1

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 4 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001 --tra_ratio 99 --val_ratio 1

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 4 --node_rank 2 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001 --tra_ratio 99 --val_ratio 1

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 4 --node_rank 3 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001 --tra_ratio 99 --val_ratio 1

- eight nodes, on cloud

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 0 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 2 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 3 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 4 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 5 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 6 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node 4 --nnodes 8 --node_rank 7 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint 172.16.0.4:9001

```

## Use the run shell on the VM cluster
```
sh /home/gtuser/mrprogs/STCNNT.git/mri/run_cloud.sh -d 16 -e 172.16.0.4 -n 4 -p 9001 -r 100
```

## Run the evaluation code
```
python3 ./mri/eval_mri.py --data_root /data/mri/denoising/data --test_files train_3D_3T_retro_cine_2020_small_test.h5 --results_path /export/Lab-Xue/projects/mri/results --pad_time --saved_model_path /export/Lab-Xue/projects/mri/test/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-False-1-BLOCK_DENSE-0-QKNORM-True-CONSINE_ATT-0-shuffle_in_window-0-att_with_relative_postion_bias-1-BLOCK_STR-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1.pts
```

Run the inference call:
```
python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/RT_Cine_R6/RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149 --output_dir /export/Lab-Xue/projects/mri/test/results/RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149 --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --saved_model_path /export/Lab-Xue/projects/mri/test/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-False-1-BLOCK_STR-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_05-27-2023.pt

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --output_dir /export/Lab-Xue/projects/mri/test/results/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path /export/Lab-Xue/projects/mri/test/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-False-1-BLOCK_STR-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_05-27-2023.pt

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --output_dir /export/Lab-Xue/projects/mri/test/results/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path /export/Lab-Xue/projects/mri/test/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-False-1-BLOCK_STR-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_05-27-2023.pt


python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/Share/data/FreeMax/SCMR/2022-08-23-HV-cardiac-SNR-DL/meas_MID00055_FID06126_MID_SAX_CINE_IPAT4_256Res_36ref/numpy --output_dir /export/Lab-Xue/projects/mri/test/results/2022-08-23-HV-cardiac-SNR-DL --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path /export/Lab-Xue/projects/mri/test/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-BLOCK_STR-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_complex_residual_20-37-44-20230529_best.pt


python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/RT_Cine_R6/RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149 --output_dir /export/Lab-Xue/projects/mri/test/results/RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149 --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --saved_model_path /export/Lab-Xue/projects/mri/test/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230529_230547_complex_residual_23-06-13-20230529_best.pt

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --output_dir /export/Lab-Xue/projects/mri/test/results/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path /export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230531_001857_complex_residual_00-19-29-20230531_best.pt


python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/RT_Cine_R6/RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149 --output_dir /export/Lab-Xue/projects/mri/test/results/RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149 --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --saved_model_path /export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230530_021440_residual_08-52-03-20230530_best.pt

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --output_dir /export/Lab-Xue/projects/mri/test/results/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path /export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230530_021440_residual_08-52-03-20230530_best.pt

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/Share/data/FreeMax/SCMR/2022-08-23-HV-cardiac-SNR-DL/meas_MID00055_FID06126_MID_SAX_CINE_IPAT4_256Res_36ref/numpy --output_dir /export/Lab-Xue/projects/mri/test/results/2022-08-23-HV-cardiac-SNR-DL --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path /export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230530_021440_residual_08-52-03-20230530_best.pt
```

```
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-False-1-BLOCK_DENSE-0-QKNORM-True-CONSINE_ATT-0-shuffle_in_window-0-att_with_relative_postion_bias-1-BLOCK_STR-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1.pt

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230601_010244_complex_residual_01-02-49-20230601_best.pt

model=/export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230531_030804_residual_08-20-49-20230531_best.pt

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-16-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230601_141534_residual_15-44-12-20230601_best.pt

# small model picked
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230531_175035_complex_residual_13-50-40-20230531_best.pt
# large model picked
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230531_030804_complex_residual_03-08-11-20230531_best.pt
# fast model
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-16-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230601_141534_complex_residual_14-15-40-20230601_best.pt

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-UNET-conv-parallel-instance2d-sophia-C-16-MIXER-conv-1-T1T1T1_T1T1T1_T1T1T1_T1T1T1-20230601_141534_complex_residual_09-49-23-20230602_best.pt

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230531_175035_complex_residual_13-50-40-20230531_best.onnx

# hybrid loss
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230603_012008_complex_residual_21-20-11-20230602_best.pt

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-UNET-conv-parallel-batch2d-sophia-C-16-MIXER-conv-1-T1T1T1_T1T1T1_T1T1T1_T1T1T1-20230601_141534_complex_residual_weighted_loss_07-48-30-20230602_best

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-UNET-conv-parallel-batch2d-sophia-C-16-MIXER-conv-1-T1T1T1_T1T1T1_T1T1T1_T1T1T1-20230601_141534_complex_residual_weighted_loss_07-48-30-20230602_best

model=/export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230606_000026_residual_00-00-32-20230606_best.pt

model=/export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-64-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230606_000026_residual_05-20-51-20230606_best.pt

model=/export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-64-H-32-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230606_000026_residual_19-13-33-20230606_best.pt

# new training, with more data
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_034255_complex_residual_03-43-00-20230611_best.pt

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/Share/data/FreeMax/SCMR/2022-08-23-HV-cardiac-SNR-DL/meas_MID00055_FID06126_MID_SAX_CINE_IPAT4_256Res_36ref/numpy --output_dir /export/Lab-Xue/projects/mri/test/results/2022-08-23-HV-cardiac-SNR-DL --scaling_factor 2.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/RT_Cine_R6/RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149 --output_dir /export/Lab-Xue/projects/mri/test/results/RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149 --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --output_dir /export/Lab-Xue/projects/mri/test/results/Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_7065950_7065959_2582_20210120-123912 --output_dir /export/Lab-Xue/projects/mri/test/results/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_7065950_7065959_2582_20210120-123912 --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Xue/mrprogs/gadgetron_CMR_ML-source/ut/data/denoising/snr_gmap_denoising/WB_LGE_MOCO_AVE_OnTheFly_42110_7066558_7066567_3672_20210125-140041 --output_dir /export/Lab-Xue/projects/mri/test/results/WB_LGE_MOCO_AVE_OnTheFly_42110_7066558_7066567_3672_20210125-140041 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --saved_model_path $model

```

python3 ./mri/save_pt_as_onnx.py --input /export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T
1L1G1_T1L1G1_T1L1G1_T1L1G1-20230531_175035_complex_residual_13-50-40-20230531_best.pt

### Run the WB LGE

```
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-instance2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230605_020426_complex_residual_19-06-34-20230605_best.pt

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-instance2d-sophia-C-32-H-32-MIXER-conv-4-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230607_134035_complex_residual_15-27-40-20230607_best.pt

# increase L1 weight to 2
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-instance2d-sophia-C-32-H-32-MIXER-conv-4-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230607_134035_complex_residual_18-48-18-20230607_best.pt

# best ssim
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-64-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230605_020426_complex_residual_12-08-08-20230605_best.pt

# new training, with more data
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_034255_complex_residual_03-43-00-20230611_best.pt

# a case
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1194791055_1194791064_248_20230109-123219 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1194791055_1194791064_248_20230109-123219 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1199034792_1199034801_784_20230111-110935/ --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1199034792_1199034801_784_20230111-110935/res --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model


python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_ave/WB_LGE_MOCO_AVE_OnTheFly_41837_1199034792_1199034801_784_20230111-110935/ --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_ave/WB_LGE_MOCO_AVE_OnTheFly_41837_1199034792_1199034801_784_20230111-110935/res --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model



python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_WB_LGE_2023_AI_denoising/20230619/WB_LGE_MOCO_AVE_OnTheFly_41837_1780995164_1780995173_3000006_20230619-191346/ --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_WB_LGE_2023_AI_denoising_AI/WB_LGE_MOCO_AVE_OnTheFly_41837_1780995164_1780995173_3000006_20230619-191346 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --gmap_fname raw_gfactor --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_WB_LGE_2023_AI_denoising/20230619/WB_LGE_MOCO_AVE_OnTheFly_41837_1780995164_1780995173_3000002_20230619-191248/ --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_WB_LGE_2023_AI_denoising_AI/WB_LGE_MOCO_AVE_OnTheFly_41837_1780995164_1780995173_3000002_20230619-191248 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --gmap_fname raw_gfactor --saved_model_path $model

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_WB_LGE_2023_AI_denoising/20230619/WB_LGE_MOCO_AVE_OnTheFly_41837_1780995164_1780995173_3000004_20230619-191318/ --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_WB_LGE_2023_AI_denoising_AI/WB_LGE_MOCO_AVE_OnTheFly_41837_1780995164_1780995173_3000004_20230619-191318 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --gmap_fname raw_gfactor --saved_model_path $model

# on the raw images
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --saved_model_path $model

# on the moco+ave images

python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_ave --scaling_factor 1.25 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname im --saved_model_path $model

```

### Run the DB LGE
```
# increase L1 weight to 2
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-instance2d-sophia-C-32-H-32-MIXER-conv-4-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230607_134035_complex_residual_18-48-18-20230607_best.pt

# best ssim
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-64-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230605_020426_complex_residual_12-08-08-20230605_best.pt

# new training, with more data
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_034255_complex_residual_03-43-00-20230611_best.pt

# on the raw images
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising_AI_on_raw --scaling_factor 1.25 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --saved_model_path $model

# on the moco+ave images

python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising_AI_on_ave --scaling_factor 1.25 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname im --saved_model_path $model

```

### Perfusion
```
# new training, with more data
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_034255_complex_residual_03-43-00-20230611_best.pt

# mag model
model=/export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_194153_residual_02-48-28-20230612_last.pt

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210507/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_21149167_21149176_1466_20210507-111038 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_21149167_21149176_1466_20210507-111038 --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# flash
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising/20221005/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636 --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636 --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# 3T
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising_AI --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# 1.5T
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

```

### RT Cine
```
# new training, with more data
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-4-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_194153_complex_residual_09-48-59-20230612_last.pt

# mag model
model=/export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_194153_residual_02-48-28-20230612_last.pt

# previous model
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_034255_complex_residual_03-43-00-20230611_best.pt
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230603_012008_complex_residual_21-20-11-20230602_best.pt

# new training
model=/export/Lab-Xue/projects/mri/test/complex_model/cifar_21-28-58-20230617_last.pt
model=/export/Lab-Xue/projects/mri/test/complex_model/cifar_21-28-58-20230617_best.pt
model=/export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230618_134522_residual_weighted_loss_13-45-45-20230618_epoch-30.pth
model=/export/Lab-Xue/projects/mri/checkpoints/mri-HRNET-20230619_085716_714328_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_08-57-24-20230619_epoch-2.pth
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-20230619_131814_275276_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_13-18-39-20230619_last.pt

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-20230619_202403_307218_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-56.pth
mode=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-20230619_195348_541312_complex_residual_weighted_loss-T1L1G1_T1L1G1_T1L1G1_T1L1G1_epoch-71.pth

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-20230619_202403_307218_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_20-24-28-20230619_last.pt

model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-20230621_132139_784364_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_13-22-06-20230621_last.pt

# R4
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804/res --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# R5
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/res --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# R6
python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/res --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# test two models
model1=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-20230619_202403_307218_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-56.pth
model2=/export/Lab-Xue/projects/mri/test/mag_model/mri-HRNET-conv-parallel-batch2d-sophia-C-32-H-32-MIXER-conv-1-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1-20230611_194153_residual_02-48-28-20230612_last.pt

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/res --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model1

python3 ./mri/run_inference.py --input_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/res --output_dir /export/Lab-Kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/res2 --scaling_factor 1.0 --im_scaling 1.0 --gmap_scaling 1.0 --input_fname output --saved_model_path $model2


# on the raw images
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/Barts_RTCine_Denoising_2021_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/Barts_RTCine_Denoising_2021_AI_denoising_AI --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

# 1.5T
python3 ./mri/run_inference_batch.py --input_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_RTCine_Denoising_2021_AI_denoising --output_dir /export/Lab-Kellman/ReconResults/denoising/1p5T_for_testing/Barts_RTCine_Denoising_2021_AI_denoising_AI --scaling_factor 1.0 --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model

```

# other models

# smaller model
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-conv-parallel-instance2d-sophia-C-32-H-32-MIXER-conv-4-T1L1G1_T1L1G1_T1L1G1_T1L1G1-20230607_134035_complex_residual_18-48-18-20230607_best.pt