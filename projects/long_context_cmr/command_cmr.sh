export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for lr in 1e-2 #1e-3 1e-4 1e-5 1e-6
do
        torchrun --nnodes=1 \
                --nproc_per_node=8 \
                --max_restarts=0 \
                --master_port=9050 \
                --rdzv_id=100 \
                --rdzv_backend="c10d" \
                ../../run.py \
                        --run_name="cmr_swin_unetr_attn_patch2_window8_lr${lr}_final" \
                        --group="final" \
                        --project='long_context' \
                        --log_dir="/people/hoopersm/long_context_paper/logs" \
                        --wandb_dir="/people/hoopersm/long_context_paper/wandb" \
                        --data_dir="/home/hoopersm/preprocessed_data/cmr" \
                        --split_csv_path="/home/hoopersm/long_context_paper/csv_samplers/cmr_split.csv" \
                        --tasks "cmr_enh" \
                        --task_type=enhance \
                        --exact_metrics=False \
                        --height=128 \
                        --width=128 \
                        --time=32 \
                        --no_in_channel=1 \
                        --no_out_channel=1 \
                        --affine_aug=True \
                        --brightness_aug=True \
                        --gaussian_blur_aug=False \
                        --batch_size=16 \
                        --num_epochs=250 \
                        --train_model=True \
                        --pre_component=Identity \
                        --backbone_component=SWIN \
                        --SWIN.size='unetr' \
                        --SWIN.patch_size 2 2 2 \
                        --SWIN.window_size 8 8 8 \
                        --SWIN.use_hyena False \
                        --post_component=SwinUNETR \
                        --loss_func=CombinationEnhance \
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
                        --checkpoint_frequency 50 \
                        --save_test_samples True \
                        --override \

done