
#!/usr/bin/bash

while getopts s:g: flag
do
    case "${flag}" in
        s) sweep_id=${OPTARG};;
        g) gpu_id=${OPTARG};;
    esac
done
echo "sweep id: $sweep_id";
echo "gpu id: $gpu_id";

CUDA_VISIBLE_DEVICES=$gpu_id python3 ./cifar10/main_cifar.py --data_set cifar10 --data_root /export/Lab-Xue/projects/cifar10/data --check_path /export/Lab-Xue/projects/cifar10/checkpoints --model_path /export/Lab-Xue/projects/cifar10/models --log_path /export/Lab-Xue/projects/cifar10/logs --results_path /export/Lab-Xue/projects/cifar10/results --summary_depth 6 --save_cycle 200 --num_epochs 200 --batch_size 128 --device cuda --n_head 8 --window_size 8 8 --patch_size 4 4 --global_lr 1e-4 --clip_grad_norm 1.0 --weight_decay 0.0 --use_amp --iters_to_accumulate 1 --project cifar --num_workers 8 --scheduler_type OneCycleLR --scheduler.ReduceLROnPlateau.patience 2 --scheduler.ReduceLROnPlateau.cooldown 2 --scheduler.ReduceLROnPlateau.min_lr 1e-7 --scheduler.ReduceLROnPlateau.factor 0.9 --scheduler.StepLR.step_size 5 --scheduler.StepLR.gamma 0.8 --backbone_hrnet.num_resolution_levels 3 --backbone_hrnet.use_interpolation 1 --backbone_unet.num_resolution_levels 3 --backbone_unet.use_unet_attention 1 --backbone_unet.use_interpolation 1 --backbone_unet.with_conv 1 --backbone_LLM.num_stages 3 --backbone_LLM.add_skip_connections 1 --backbone_small_unet.channels 16 32 64 --backbone_small_unet.block_str T1L1G1 T1L1G1 T1L1G1 --run_name cifar-hrnet-conv-sequential-batch2d-C-128-mixer-conv-False-1.0-1-block_dense-1-qknorm-True-cosine_att-1-shuffle_in_window-0-att_with_relative_postion_bias-1-block_str-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1 --run_notes cifar-hrnet-conv-sequential-batch2d-C-128-mixer-conv-False-1.0-1-block_dense-1-qknorm-True-cosine_att-1-shuffle_in_window-0-att_with_relative_postion_bias-1-block_str-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1 --backbone hrnet --a_type conv --cell_type sequential --cosine_att 1 --att_with_relative_postion_bias 1 --backbone_hrnet.C 128 --backbone_unet.C 128 --backbone_LLM.C 128 --block_dense_connection 1 --norm_mode batch2d --mixer_type conv --shuffle_in_window 0 --scale_ratio_in_mixer 1.0 --mixer_kernel_size 3 --mixer_padding 1 --mixer_stride 1 --normalize_Q_K --backbone_hrnet.block_str T1L1G1 T1L1G1T1L1G1 T1L1G1T1L1G1 --sweep_id $sweep_id