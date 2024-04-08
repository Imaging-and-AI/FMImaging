# This file will load a config and checkpoint for a specified directory and run inference, saving samples and metrics in the specified dir.
# Note: if the original training was done with ddp, this script should also use torchrun; if the original script did not use ddp, this script should not use torchrun.

torchrun --nnodes=1 \
        --nproc_per_node=8 \
        --max_restarts=0 \
        --master_port=9050 \
        --rdzv_id=100 \
        --rdzv_backend="c10d" \
        ../run.py \
                --inference_only True \
                --inference_dir "/home/hoopersm/long_context_paper/logs/cmr_vit_s_attn_patch32_lr1e-2_final" \
                --inference_log_dir "/home/hoopersm/long_context_paper/logs/cmr_vit_s_attn_patch32_lr1e-2_final" \
                --inference_run_name "inference_only"

