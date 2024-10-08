export CUDA_VISIBLE_DEVICES=1

python ../../run.py --run_name="abct_refactor" \
                    --log_dir="/home/hoopersm/refactor_debug/logs" \
                    --data_dir="/home/hoopersm/preprocessed_data/abct" \
                    --split_csv_path="/home/hoopersm/archive/baseline_backbones/samplers/simple_abct_splits_seed_1.csv" \
                    --height=112 \
                    --width=112 \
                    --time=32 \
                    --no_in_channel=1 \
                    --no_out_channel=14 \
                    --affine_aug=True \
                    --brightness_aug=True \
                    --gaussian_blur_aug=True \
                    --pre_model=Identity \
                    --backbone_model=omnivore \
                    --omnivore.size='tiny' \
                    --post_model=UperNet3D \
                    --task_type=seg \
                    --optim_type=adam \
                    --scheduler_type=None \
                    --loss_type=CrossEntropy \
                    --device=cuda \
                    --num_workers=4 \
                    --num_epochs=50 \
                    --batch_size=2 \
                    --optim.lr=0.0001 \
                    --optim.beta1=0.9 \
                    --optim.beta2=0.99 \
                    --seed 1 