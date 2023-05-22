
#!/usr/bin/bash

nproc_per_node=4
data_name=mri
port=9001
tra_ratio=100
rdzv_endpoint=172.16.0.4
nnodes=16

while getopts d:e:n:p:r:h OPTION; do
    case "$OPTION" in
        d) nnodes=${OPTARG};;
        e) rdzv_endpoint=${OPTARG};;
        n) nproc_per_node=${OPTARG};;
        p) port=${OPTARG};;
        r) tra_ratio=${OPTARG};;
        h) 
          echo "-d nnodes -e rdzv_endpoint -n nproc_per_node -p port -r tra_ratio"
          exit 0
        ;;
    esac
done

echo "nnodes: $nnodes"
echo "rdzv_endpoint: $rdzv_endpoint"
echo "proc per node: $nproc_per_node"
echo "port: $port"
echo "training data ratio: $tra_ratio"

node_rank=$(($(hostname | sed 's/[^0-9]*//g')-1)) 
echo "node_rank: $node_rank"

echo "python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint $rdzv_endpoint:$port"

python3 $HOME/mrprogs/STCNNT.git/mri/run_mri.py --nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint $rdzv_endpoint:$port