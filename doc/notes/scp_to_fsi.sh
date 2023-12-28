#!/usr/bin/bash

if [ $# -ne 2 ]; then
    echo $0: usage: scp_to_fsi.sh fsi1 path_at_fsi1
    exit 1
fi

scp -r -i ~/.ssh/xueh2-a100.pem $2 gtuser@$1.eastus2.cloudapp.azure.com:/export/Lab-Xue/projects/data/logs/
