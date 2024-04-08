export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for lr in 1e-3 #1e-2 1e-3 1e-4 1e-5 1e-6
do
        torchrun --nnodes=1 \
                --nproc_per_node=8 \
                --max_restarts=0 \
                --master_port=9050 \
                --rdzv_id=100 \
                --rdzv_backend="c10d" \
                ../../run.py \
                        --run_name="vessel_vit_s_attn_patch32_lr${lr}_final" \
                        --group="final" \
                        --project='long_context' \
                        --log_dir="/people/hoopersm/long_context_paper/logs" \
                        --wandb_dir="/people/hoopersm/long_context_paper/wandb" \
                        --data_dir="/home/hoopersm/preprocessed_data/vessels" \
                        --split_csv_path="/home/hoopersm/long_context_paper/csv_samplers/vessels_split.csv" \
                        --tasks "vessel_segment" \
                        --task_type=seg \
                        --exact_metrics=False \
                        --height=2048 \
                        --width=2048 \
                        --time=1 \
                        --no_in_channel=3 \
                        --no_out_channel=2 \
                        --affine_aug=True \
                        --brightness_aug=True \
                        --gaussian_blur_aug=False \
                        --batch_size=4 \
                        --num_epochs=250 \
                        --train_model=True \
                        --pre_component=Identity \
                        --backbone_component=ViT \
                        --ViT.size='small' \
                        --ViT.patch_size 1 32 32 \
                        --ViT.use_hyena False \
                        --post_component=ViTUNETR \
                        --loss_func=CrossEntropy \
                        --optim_type=adam \
                        --optim.lr=$lr \
                        --optim.global_lr=$lr \
                        --optim.beta1=0.9 \
                        --optim.beta2=0.99 \
                        --scheduler_type=OneCycleLR \
                        --device=cuda \
                        --num_workers=16 \
                        --seed 1 \
                        --save_model_components=False \
                        --checkpoint_frequency 100 \
                        --save_test_samples True \
                        --override \

done