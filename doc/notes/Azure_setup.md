# Set up a new Azure VM

```
# set up home folder
sudo apt update
mkdir ~/software
mkdir ~/Debug
mkdir ~/Debug/DebugOutput
mkdir ~/gadgetron_ismrmrd_data
ln -s ~/gadgetron_ismrmrd_data /tmp/gadgetron_data
chmod a+rwx /tmp/gadgetron_data
mkdir ~/mrprogs
mkdir ~/key

sudo apt install software-properties-common -y
sudo apt install build-essential -y
sudo apt install python3-pip emacs tmux hdparm -y

cd ~ && touch .tmux.conf
echo "set -g mouse on" >> .tmux.conf

# install vscode
#https://code.visualstudio.com/docs/setup/linux

wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt install code -y

# install cuda and gpu driver
cd ~/software

wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run

sudo sh cuda_12.1.1_530.30.02_linux.run

sudo nvidia-smi -pm 1

sudo reboot now

# clone repo
cd ~/.ssh
ssh-keygen -t rsa -b 4096 -C "hui.xue@nih.gov"
cat ./id_rsa.pub

# add key to github
cd ~/mrprogs
git clone git@github.com:AzR919/STCNNT.git

# mount and format disks
# https://gist.github.com/keithmorris/b2aeec1ea947d4176a14c1c6a58bfc36

sudo fdisk -l

# test hard drive speed
sudo hdparm -Tt /dev/nvme3n1

# format and mount drives

drive=(/dev/nvme0n1
    /dev/nvme1n1
    /dev/nvme2n1
    /dev/nvme3n1
    )

mpoint=(
    /export/Lab-Xue/projects/mri
    /export/Lab-Xue/projects/imagenet
    /export/Lab-Xue/projects/fm
    /export/Lab-Xue/projects/data
    )

for index in ${!drive[*]}; do 
    echo "${drive[$index]} is in ${mpoint[$index]}"
    sudo fdisk ${drive[$index]}
    sudo mkfs -t ext4 ${drive[$index]}
    sudo mkdir -p ${mpoint[$index]}
    sudo mount -t ext4 ${drive[$index]} ${mpoint[$index]}
    sudo chmod a+rw ${mpoint[$index]}
done
mkdir -p /export/Lab-Xue/projects/mri/data
mkdir -p /export/Lab-Xue/projects/imagenet/data
mkdir -p /export/Lab-Xue/projects/fm/data

# install azcopy
cd ~/software
wget https://aka.ms/downloadazcopy-v10-linux
tar -xvf downloadazcopy-v10-linux
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
sudo chmod a+x /usr/bin/azcopy

# install packages

pip3 uninstall -y torch torchvision torchaudio
pip3 install torch torchvision torchaudio
pip3 install wandb tqdm h5py torchinfo tifffile opencv-python nibabel piq colorama scikit-image einops onnx numpy scipy moviepy imageio matplotlib torchmetrics gputil onnxruntime gif prettytable timm monai PyWavelets pytorch_wavelets
pip3 install torch-interpol laplace-torch pytorch_wavelets

cd ~/mrprogs
git clone git@github.com:NHLBI-MR/FMImaging.git

cd ~/mrprogs/FMImaging
bash ./doc/notes/set_up_VM.sh

bash ./doc/notes/download_data.sh
ll -ltr /export/Lab-Xue/projects/mri/data

wandb login
ulimit -n 65536

wandb login
3420ade4920debef5eb73377a2bb8f600e26a2c8

cd ~/mrprogs
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip3 install -e . --break-system-packages

cd ~/mrprogs
git clone git@github.com:aleximmer/Laplace.git
cd Laplace
pip3 install -e . --break-system-packages

# login prompt
# add ~/.local/bin into the path

wandb login

ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi1.eastus2.cloudapp.azure.com
ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi2.eastus2.cloudapp.azure.com

```

## Create a new VM

- use the azure image SFImaging/a100-general-training in the xueh2-a100-eastus2-2 group
- Create VM, with the node type Standard NC96ads A100 v4 (96 vcpus, 880 GiB memory)

## Add new VM to code remote debug
```
ssh -i C:/Users/xueh2/.ssh/xueh2-a100.pem gtuser@20.114.147.179 -K
```

# After creating a VM from an image
```
VM_name=fsi6.eastus2.cloudapp.azure.com

ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "pip3 install torchmetrics colorama moviepy imageio"
ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "sudo nvidia-smi -pm 1"
ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "nvidia-smi"


ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name
git clone git@github.com:AzR919/STCNNT.git /home/gtuser/mrprogs/STCNNT.git"

ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "mkdir -p /export/Lab-Xue/projects/imagenet/data"
scp -i ~/.ssh/xueh2-a100.pem /export/Lab-Xue/data/common/ILSVRC2012_img_val.tar gtuser@$VM_name:/export/Lab-Xue/projects/imagenet/data/
scp -i ~/.ssh/xueh2-a100.pem /export/Lab-Xue/data/common/ILSVRC2012_devkit_t12.tar.gz gtuser@$VM_name:/export/Lab-Xue/projects/imagenet/data/
scp -i ~/.ssh/xueh2-a100.pem /export/Lab-Xue/data/common/ILSVRC2012_img_train.tar gtuser@$VM_name:/export/Lab-Xue/projects/imagenet/data/
ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "python3 -c \"import torchvision as tv; a = tv.datasets.ImageNet(root='/export/Lab-Xue/projects/imagenet/data', split='train'); a = tv.datasets.ImageNet(root='/export/Lab-Xue/projects/imagenet/data', split='val') \" "

ssh -i ~/.ssh/xueh2-a100.pem gtuser@$VM_name "mkdir -p /export/Lab-Xue/projects/mri/data"

```

# Reinstall nvidia driver
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
sudo ubuntu-drivers install 535
```

# azcopy files
```
azcopy copy ./MINNESOTA_UHVC_RetroCine_1p5T_2022.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/MINNESOTA_UHVC_RetroCine_1p5T_2022.h5?${SAS}" --recursive
azcopy copy ./MINNESOTA_UHVC_RetroCine_1p5T_2023.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/MINNESOTA_UHVC_RetroCine_1p5T_2023.h5?${SAS}" --recursive

azcopy copy ./BWH_RetroCine_3T_2021.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RetroCine_3T_2021.h5?${SAS}" --recursive
azcopy copy ./BWH_RetroCine_3T_2022.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RetroCine_3T_2022.h5?${SAS}" --recursive
azcopy copy ./BWH_RetroCine_3T_2023.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RetroCine_3T_2023.h5?${SAS}" --recursive

azcopy copy ./BWH_Perfusion_3T_2021.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_Perfusion_3T_2021.h5?${SAS}" --recursive
azcopy copy ./BWH_Perfusion_3T_2022.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_Perfusion_3T_2022.h5?${SAS}" --recursive
azcopy copy ./BWH_Perfusion_3T_2023.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_Perfusion_3T_2023.h5?${SAS}" --recursive

azcopy copy ./BWH_RTCine_3T_2021.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RTCine_3T_2021.h5?${SAS}" --recursive
azcopy copy ./BWH_RTCine_3T_2022.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RTCine_3T_2022.h5?${SAS}" --recursive
azcopy copy ./BWH_RTCine_3T_2023.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RTCine_3T_2023.h5?${SAS}" --recursive

azcopy copy ./VIDA_train_clean_0430.h5 "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/VIDA_train_clean_0430.h5?${SAS}" --recursive

azcopy cp "./*.h5" "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/?${SAS}"
```

# install pytorch wavelet

```
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip3 install .
```
# clone others

```
cd ~/mrprogs
git clone git@github.com:xueh2/matlab_mrrecon.git
git clone git@github.com:xueh2/imagescn.git
```

```
doskey fsi1=ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi1.eastus2.cloudapp.azure.com
doskey fsi2=ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi2.eastus2.cloudapp.azure.com
doskey fsi3=ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi3.eastus2.cloudapp.azure.com
doskey fsi4=ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi4.eastus2.cloudapp.azure.com
doskey fsi5=ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi5.eastus2.cloudapp.azure.com
doskey fsi6=ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi6.eastus2.cloudapp.azure.com
doskey fsi7=ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi7.eastus2.cloudapp.azure.com
doskey fsi8=ssh -i ~/.ssh/xueh2-a100.pem gtuser@fsi8.eastus2.cloudapp.azure.com
```
```
for VARIABLE in 1 2 3 4 6 7 8 9
do
    bash ./doc/notes/scp_to_fsi.sh fsi${VARIABLE} /export/Lab-Xue/projects/data/logs/mri-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231214_213835_461207_MRI_double_net_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1_T1L1G1
done
```

for file in ./*sh; do bash $file &; done