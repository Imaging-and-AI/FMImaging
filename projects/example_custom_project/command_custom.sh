export CUDA_VISIBLE_DEVICES=4

python custom_run.py  --run_name='custom_project_refactor' \
                      --log_dir="/home/hoopersm/refactor_debug/logs" \
                      --data_dir='/home/hoopersm/preprocessed_data/ptx' \
                      --height=512 \
                      --width=512 \
                      --time=1 \
                      --batch_size=2 \
                      --task_type="class" \
                      --no_in_channel=1 \
                      --no_out_channel=2 \
                      --custom_arg_2="Example custom arg modification" \
                      --pre_model=Identity \
                      --backbone_model='omnivore' \
                      --omnivore.size='tiny'
                      --post_model=NormPoolLinear \
                      --optim_type=adam \
                      --scheduler_type=None \
                      --device=cuda \
                      --num_workers=4 \
                      --num_epochs=50 \
                      --optim.lr=0.0001 \
                      --optim.beta1=0.9 \
                      --optim.beta2=0.99 \
                      --override
