export CUDA_VISIBLE_DEVICES=4

python ../../run.py --run_name="mrnet_refactor" \
                    --log_dir="/home/hoopersm/refactor_debug/logs" \
                    --data_dir="/home/hoopersm/preprocessed_data/mrnet" \
                    --split_csv_path="/home/hoopersm/archive/baseline_backbones/samplers/simple_mrnet_splits_seed_1.csv" \
                    --height=256 \
                    --width=256 \
                    --time=48 \
                    --no_in_channel=1 \
                    --no_out_channel=2 \
                    --affine_aug=True \
                    --brightness_aug=True \
                    --gaussian_blur_aug=True \
                    --pre_model=Identity \
                    --backbone_model=omnivore \
                    --omnivore.size='tiny' \
                    --post_model=NormPoolLinear \
                    --task_type=class \
                    --optim_type=adam \
                    --scheduler_type=None \
                    --loss_type=CrossEntropy \
                    --device=cuda \
                    --num_workers=4 \
                    --num_epochs=50 \
                    --batch_size=8 \
                    --optim.lr=0.00001 \
                    --optim.beta1=0.9 \
                    --optim.beta2=0.99 \
                    --exact_metrics=True \
