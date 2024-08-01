export CUDA_VISIBLE_DEVICES=0,1,2,3


torchrun --nnodes=1 \
        --nproc_per_node=4 \
        --max_restarts=0 \
        --master_port=9050 \
        --rdzv_id=100 \
        --rdzv_backend="c10d" \
        ../../run.py \
                --run_name="micro_ss_image_restoration_mask32_.25_sgm.1_lr1e-4" \
                --group="debug" \
                --project='self_sup' \
                --log_dir="/home/hoopersm/long_context_paper/logs/tuning" \
                --wandb_dir="/home/hoopersm/long_context_paper/wandb/tuning" \
                --data_dir="/home/hoopersm/preprocessed_data/microscopy" \
                --split_csv_path="/home/hoopersm/long_context_paper/csv_samplers/microscopy_split.csv" \
                --tasks "microscopy_ss" \
                --task_type=ss_image_restoration \
                --ss_image_restoration.mask_percent=0.25 \
                --ss_image_restoration.mask_patch_size 1 32 32 \
                --ss_image_restoration.noise_std=0.1 \
                --ss_image_restoration.resolution_factor=1 \
                --exact_metrics=False \
                --height=1024 \
                --width=1024 \
                --time=1 \
                --no_in_channel=1 \
                --no_out_channel=1 \
                --affine_aug=True \
                --brightness_aug=True \
                --gaussian_blur_aug=False \
                --batch_size=4 \
                --num_epochs=100 \
                --train_model=True \
                --pre_component=Identity \
                --backbone_component=ViT \
                --ViT.size='small' \
                --ViT.patch_size 1 16 16 \
                --ViT.use_hyena False \
                --post_component=ViTMAEHead \
                --loss_func=SSImageRestoration \
                --optim_type=adam \
                --optim.lr=0.0001 \
                --optim.global_lr=0.0001 \
                --optim.beta1=0.9 \
                --optim.beta2=0.99 \
                --scheduler_type=OneCycleLR \
                --device=cuda \
                --num_workers=16 \
                --seed 1 \
                --save_model_components=False \
                --checkpoint_frequency 500 \
                --save_test_samples True \
                --override \

