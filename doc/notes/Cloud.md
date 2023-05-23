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

# copy key
for n in ${node_list[*]}
do
    echo "copy data to $n ..."
    VM_name=$n.eastus2.cloudapp.azure.com    
scp -i ~/.ssh/xueh2-a100.pem ~/.ssh/xueh2-a100.pem gtuser@$VM_name:/home/gtuser/.ssh/
scp -i ~/.ssh/xueh2-a100.pem $HOME/mrprogs/STCNNT.git/doc/notes/set_up_VM.sh gtuser@$VM_name:/home/gtuser/
done

# mount drive
bash ~/set_up_VM.sh

# update the code
for n in fsi{1..16}
do
    echo "update node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "cd /home/gtuser/mrprogs/STCNNT.git && git pull"
done
```
