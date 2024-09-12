
#!/usr/bin/bash

nproc_per_node=4
data_name=mri
port=9001
tra_ratio=95
val_ratio=5
max_load=-1
rdzv_endpoint=172.16.0.4
nnodes=4
run_list=-1
model_type=STCNNT_MRI

while getopts d:e:n:p:t:m:v:r:l:h OPTION; do
    case "$OPTION" in
        d) nnodes=${OPTARG};;
        e) rdzv_endpoint=${OPTARG};;
        n) nproc_per_node=${OPTARG};;
        p) port=${OPTARG};;
        t) tra_ratio=${OPTARG};;
        v) val_ratio=${OPTARG};;
        m) max_load=${OPTARG};;
        r) run_list=${OPTARG};;
        l) model_type=${OPTARG};;
        h) 
          echo "-d nnodes -e rdzv_endpoint -n nproc_per_node -p port -t tra_ratio -v val_ratio -m max_load -r run_list -l model_type"
          exit 0
        ;;
    esac
done

echo "nnodes: $nnodes"
echo "rdzv_endpoint: $rdzv_endpoint"
echo "proc per node: $nproc_per_node"
echo "port: $port"
echo "training data ratio: $tra_ratio"
echo "val data ratio: $val_ratio"
echo "max_load: $max_load"
echo "run_list: $run_list"
echo "model_type: $model_type"

node_rank=$(($(hostname | sed 's/[^0-9]*//g')-1)) 
echo "node_rank: $node_rank"

bash ./doc/notes/clean_VM.sh

ulimit -n 65536

echo "python3 ./mri/run_mri.py --nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint $rdzv_endpoint:$port --run_list $run_list --model_type $model_type"

WANDB_API_KEY=3420ade4920debef5eb73377a2bb8f600e26a2c8
WANDB_MODE=offline
python3 ./mri/run_mri.py --nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint $rdzv_endpoint:$port --tra_ratio $tra_ratio --max_load $max_load --val_ratio $val_ratio --test_ratio 100 --run_list $run_list --model_type $model_type
WANDB_MODE=online
wandb sync --sync-all ./wandb/