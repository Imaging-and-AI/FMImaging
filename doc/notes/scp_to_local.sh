#!/usr/bin/bash

if [ $# -ne 2 ]; then
    echo $0: usage: scp_to_local.sh fsi1 path_at_fsi1
    exit 1
fi

scp -r -i ~/.ssh/xueh2-a100.pem gtuser@$1.eastus2.cloudapp.azure.com:$2/ /export/Lab-Xue/projects/data/logs/