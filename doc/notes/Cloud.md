# Cloud usage

A multi-node cluster is set up in Azure. Currently, it has N=16 nodes with each having 4x A100 GPUs.

## Installation
Install az cli:
```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login --use-device-code

```

## Start/stop VMs
```
rg=xueh2-a100-eastus2
node_list=(node1 node2 node3 node4 node5 node6 node7 node8 node9 node10 node11 node12 node13 node14 node15 node16)

# start the VMs
for n in ${node_list[*]}
do
    echo "start node $n ..."
    az vm start --name $n -g $rg
done

# stop the VMs
for n in ${node_list[*]}
do
    echo "stop node $n ..."
    az vm stop --name $n -g $rg
    az vm deallocate --name $n -g $rg
done

# check GPU status
for n in fsi{1..16}
do
    echo "check node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "nvidia-smi"
done

# mount drive
# no need to mount node5
for n in fsi{1,6,10,12,13}
do
    echo "mount node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "sudo mount /dev/sda1 /export/Lab-Xue"
done

for n in fsi{2,3,4,7,8,11,14,15,16,9}
do
    echo "mount node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "sudo mount /dev/sdc1 /export/Lab-Xue"
done

# update the code
for n in fsi{1..16}
do
    echo "update node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "cd /home/gtuser/mrprogs/STCNNT.git && git pull"
done
```

# Download training data to VMs
```
rg=xueh2-a100-eastus2

node_list=(fsi1 fsi2 fsi3 fsi4 fsi5 fsi6 fsi7 fsi8 fsi9 fsi10 fsi11 fsi12 fsi13 fsi14 fsi15 fsi16)

# copy key
for n in ${node_list[*]}
do
    echo "copy data to $n ..."
    VM_name=$n.eastus2.cloudapp.azure.com    
scp -i ~/.ssh/xueh2-a100.pem ~/.ssh/xueh2-a100.pem gtuser@$VM_name:/home/gtuser/.ssh/
done

# copy data

for n in ${node_list[*]}
do
    echo "copy data to $n ..."
    VM_name=$n.eastus2.cloudapp.azure.com

    # imagenet data
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "sh -c 'cd /export/Lab-Xue/projects/imagenet/data; nohup azcopy copy https://gadgetronrawdata.blob.core.windows.net/stcnnt/ILSVRC2012_devkit_t12.tar.gz . > /dev/null 2>&1 &'"
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "sh -c 'cd /export/Lab-Xue/projects/imagenet/data; nohup azcopy copy https://gadgetronrawdata.blob.core.windows.net/stcnnt/ILSVRC2012_img_val.tar . > /dev/null 2>&1 &'"
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "sh -c 'cd /export/Lab-Xue/projects/imagenet/data; nohup azcopy copy https://gadgetronrawdata.blob.core.windows.net/stcnnt/ILSVRC2012_devkit_t12.tar.gz . > /dev/null 2>&1 &'"
done

for n in ${node_list[*]}
do
    echo "copy data to $n ..."
    VM_name=$n.eastus2.cloudapp.azure.com

    # MRI data
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "sh -c 'cd /export/Lab-Xue/projects/mri/data; nohup azcopy copy https://gadgetronrawdata.blob.core.windows.net/mr-denoising-training-data/train_3D_3T_retro_cine_2018.h5 . > /dev/null 2>&1 &'"

    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "sh -c 'cd /export/Lab-Xue/projects/mri/data; nohup azcopy copy https://gadgetronrawdata.blob.core.windows.net/mr-denoising-training-data/train_3D_3T_perf_2021.h5 . > /dev/null 2>&1 &'"
done

```