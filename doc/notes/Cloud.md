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

for n in ${node_list[*]}
do
    echo "start node $n ..."
    az vm start --name $n -g $rg
done

for n in ${node_list[*]}
do
    echo "stop node $n ..."
    az vm stop --name $n -g $rg
    az vm deallocate --name $n -g $rg
done

for n in fsi{1..16}
do
    echo "update node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "git clone git@github.com:AzR919/STCNNT.git /home/gtuser/mrprogs/STCNNT.git"
done

for n in fsi{1..16}
do
    echo "update node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "cd /home/gtuser/mrprogs/STCNNT.git && git pull"
done

for n in fsi{1..16}
do
    echo "check node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "nvidia-smi"
done
```

## Reinstall nvidia driver
```
# remote old installation if any
sudo apt-get --purge remove cuda*
sudo apt-get remove --purge nvidia-*

# add nvidia driver ppa
sudo add-apt-repository ppa:graphics-drivers/ppa -y

# update software cache
sudo apt update
sudo apt upgrade -y

sudo apt-get install ubuntu-drivers-common -y
sudo ubuntu-drivers install 525 -y
```
