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
for n in fsi{1..16}
do
    echo "copy data to $n ..."
    VM_name=$n.eastus2.cloudapp.azure.com    
    scp -i ~/.ssh/xueh2-a100.pem ~/.ssh/xueh2-a100.pem gtuser@$VM_name:/home/gtuser/.ssh/
    scp -i ~/.ssh/xueh2-a100.pem $HOME/mrprogs/STCNNT.git/doc/notes/set_up_VM.sh gtuser@$VM_name:/home/gtuser/
    scp -i ~/.ssh/xueh2-a100.pem $HOME/mrprogs/STCNNT.git/doc/notes/clean_VM.sh gtuser@$VM_name:/home/gtuser/
done

# scp model
model=/export/Lab-Xue/projects/mri/test/complex_model/mri-HRNET-20230621_132139_784364_complex_residual_weighted_loss-T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_T1L1G1T1L1G1T1L1G1_13-22-06-20230621_last.pt
model=/export/Lab-Xue/projects/mri/test/first_stage/mri-STCNNT_MRI_20230721_225151_726014_C-32-1_amp-True_complex_residual_weighted_loss-T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-67.pth

for n in fsi{1..16}
do
    echo "copy to $n ..."
    VM_name=$n.eastus2.cloudapp.azure.com
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name 'mkdir -p /export/Lab-Xue/projects/mri/models/'
    scp -i ~/.ssh/xueh2-a100.pem $model gtuser@$VM_name:/export/Lab-Xue/projects/mri/models/
done


# mount drive
bash ~/set_up_VM.sh

# update the code
for n in fsi{1..16}
do
    echo "update node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "cd /home/gtuser/mrprogs/STCNNT.git && git pull"
done

# clean nodes
for n in fsi{1..16}
do
    echo "clean node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "bash ~/clean_VM.sh"
done

# update
for n in fsi{1..16}
do
    echo "update node $n ..."
    ssh -i ~/.ssh/xueh2-a100.pem gtuser@$n.eastus2.cloudapp.azure.com "cd /home/gtuser/mrprogs/STCNNT.git && git pull && git checkout test_snr_perturb"
done
```
## Cluster ssh configuration

Create the cluster configuration file as ~/.clusterssh/clusters
```
# cloud cluster
fsi  gtuser@fsi{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
fsi-quarter-1st gtuser@fsi{1,2,3,4}
fsi-quarter-2nd gtuser@fsi{5,6,7,8}
fsi-quarter-3rd gtuser@fsi{9,10,11,12}
fsi-quarter-4th gtuser@fsi{13,14,15,16}
fsi-half-1st gtuser@fsi{1,2,3,4,5,6,7,8}
fsi-half-2nd gtuser@fsi{9,10,11,12,13,14,15,16}
fsi-1 gtuser@fsi1
fsi-2 gtuser@fsi2
fsi-3 gtuser@fsi3
fsi-4 gtuser@fsi4
```

Then the cloud cluster can be opened as :
```
cssh --fillscreen fsi-quarter-1st
```