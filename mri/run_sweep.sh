
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

CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node $nproc_per_node --master-port $port --max_restarts 6 --standalone mri/main_mri.py --data_set mri --data_root /export/Lab-Xue/projects/mri/data --check_path /export/Lab-Xue/projects/mri/checkpoints --model_path /export/Lab-Xue/projects/mri/models --log_path /export/Lab-Xue/projects/mri/logs --results_path /export/Lab-Xue/projects/mri/results --train_files train_3D_3T_retro_cine_2020.h5 train_3D_3T_retro_cine_2019.h5 train_3D_3T_retro_cine_2018.h5 --train_data_types 2d 2dt 3d --data_set mri --summary_depth 6 --save_cycle 200 --num_epochs 150 --batch_size 16 --device cuda --window_size 8 8 --patch_size 4 4 --n_head 32 --global_lr 1e-4 --clip_grad_norm 1.0 --weight_decay 1.0 --use_amp --iters_to_accumulate 1 --project mri --num_workers 16 --scheduler_type OneCycleLR --scheduler.ReduceLROnPlateau.patience 2 --scheduler.ReduceLROnPlateau.cooldown 2 --scheduler.ReduceLROnPlateau.min_lr 1e-7 --scheduler.ReduceLROnPlateau.factor 0.9 --scheduler.StepLR.step_size 5 --scheduler.StepLR.gamma 0.8 --backbone_hrnet.num_resolution_levels 3 --backbone_hrnet.use_interpolation 1 --backbone_unet.num_resolution_levels 3 --backbone_unet.use_unet_attention 1 --backbone_unet.use_interpolation 1 --backbone_unet.with_conv 1 --backbone_LLM.num_stages 3 --backbone_LLM.add_skip_connections 1 --backbone_small_unet.channels 16 32 64 --backbone_small_unet.block_str T1L1G1 T1L1G1 T1L1G1 --complex_i --losses mse l1 --height 32 64 --width 32 64 --time 12 --run_name cifar-hrnet-conv-parallel-layer-C-64-mixer-conv-False-1.0-1-block_dense-1-qknorm-True-cosine_att-1-shuffle_in_window-0-att_with_relative_postion_bias-1-block_str-T1T1T1_T1T1T1_T1T1T1 --run_notes cifar-hrnet-conv-parallel-layer-C-64-mixer-conv-False-1.0-1-block_dense-1-qknorm-True-cosine_att-1-shuffle_in_window-0-att_with_relative_postion_bias-1-block_str-T1T1T1_T1T1T1_T1T1T1 --backbone hrnet --a_type conv --cell_type parallel --cosine_att 1 --att_with_relative_postion_bias 1 --backbone_hrnet.C 64 --backbone_unet.C 64 --backbone_LLM.C 64 --block_dense_connection 1 --norm_mode layer --mixer_type conv --shuffle_in_window 0 --scale_ratio_in_mixer 4.0 --mixer_kernel_size 3 --mixer_padding 1 --mixer_stride 1 --normalize_Q_K --backbone_hrnet.block_str T1L1G1 T1L1GT1 T1L1G1 --ddp --ratio $tra_ratio 10 10 --sweep_id $sweep_id