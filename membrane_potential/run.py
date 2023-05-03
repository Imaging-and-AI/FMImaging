# run the training

export FMIMAGING_PROJECT_BASE=/export/Lab-Xue/projects

python3 ./membrane_potential/main.py --run_name membrane_potential_prediction --batch_size 32 --num_starts 40 --num_epoch 200 --save_cycle 2 --weight_decay 0.1 --device cuda --data_root /export/Lab-Xue/MembranePotential/experiments --global_lr 0.001 --n_embd 512 --project membrane_potential