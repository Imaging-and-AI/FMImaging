
#!/usr/bin/bash

sweep_id=0
gpu_id=0
nproc_per_node=1
data_name=mri
port=9001
tra_ratio=100

help_info() {
    echo "-s sweep_id -g gpu_ids -n nproc_per_node -d data_name -p port -r tra_ratio"
    exit 0
}

while getopts s:g:n:p:r:h OPTION; do
    case "$OPTION" in
        s) sweep_id=${OPTARG};;
        g) gpu_id=${OPTARG};;
        n) nproc_per_node=${OPTARG};;
        p) port=${OPTARG};;
        r) tra_ratio=${OPTARG};;
        h) 
          echo "-s sweep_id -g gpu_ids -n nproc_per_node -p port -r tra_ratio"
          exit 0
        ;;
    esac
done

echo "sweep id: $sweep_id";
echo "gpu id: $gpu_id";
echo "proc per node: $nproc_per_node";
echo "port: $port";
echo "training data ratio: $tra_ratio";

CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node $nproc_per_node --master-port $port --max_restarts 6 --standalone mri/main_mri.py --data_root /data/mri/denoising/data --check_path /export/Lab-Xue/projects/mri/checkpoints --model_path /export/Lab-Xue/projects/mri/models --log_path /export/Lab-Xue/projects/mri/logs --results_path /export/Lab-Xue/projects/mri/results --summary_depth 6 --save_cycle 200 --device cuda --ddp --project mri --backbone_hrnet.use_interpolation 1 --backbone_unet.use_unet_attention 1 --backbone_unet.use_interpolation 1 --backbone_unet.with_conv 1 --backbone_LLM.add_skip_connections 1 --num_epochs 100 --batch_size 32 --window_size 8 8 --patch_size 4 4 --n_head 32 --global_lr 1e-4 --clip_grad_norm 1.0 --weight_decay 0.1 --iters_to_accumulate 1 --num_workers 32 --prefetch_factor 4 --scheduler_type OneCycleLR --backbone_hrnet.num_resolution_levels 2 --backbone_unet.num_resolution_levels 2 --backbone_LLM.num_stages 3 --backbone_small_unet.channels 16 32 64 --backbone_small_unet.block_str T1L1G1 T1L1G1 T1L1G1 --min_noise_level 2.0 --max_noise_level 8.0 --complex_i --residual --losses mse l1 --loss_weights 1.0 1.0 --height 32 64 --width 32 64 --time 12 --train_files train_3D_3T_retro_cine_2018.h5 train_3D_3T_perf_2021.h5 --train_data_types 2dt 2dt --ratio 100 5 5 --max_load 1000 --run_name mri-UNET-conv-parallel-batch2d-C-32-MIXER-conv-False-1-BLOCK_DENSE-0-QKNORM-True-CONSINE_ATT-1-shuffle_in_window-0-att_with_relative_postion_bias-1-BLOCK_STR-T1L1G1_T1L1G1_T1L1G1 --run_notes mri-UNET-conv-parallel-batch2d-C-32-MIXER-conv-False-1-BLOCK_DENSE-0-QKNORM-True-CONSINE_ATT-1-shuffle_in_window-0-att_with_relative_postion_bias-1-BLOCK_STR-T1L1G1_T1L1G1_T1L1G1 --backbone unet --a_type conv --cell_type parallel --cosine_att 1 --att_with_relative_postion_bias 1 --backbone_hrnet.C 32 --backbone_unet.C 32 --backbone_LLM.C 32 --block_dense_connection 0 --norm_mode batch2d --mixer_type conv --shuffle_in_window 0 --scale_ratio_in_mixer 1.0 --mixer_kernel_size 3 --mixer_padding 1 --mixer_stride 1 --normalize_Q_K --backbone_unet.block_str T1L1G1 T1L1G1 T1L1G1 --ratio $tra_ratio 10 10 --sweep_id $sweep_id --ddp