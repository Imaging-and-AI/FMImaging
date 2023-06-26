#!/usr/bin/bash

drive=(
    /dev/nvme0n1
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

SAS="sp=racwdli&st=2023-06-10T20:45:41Z&se=2024-06-11T04:45:41Z&sv=2022-11-02&sr=c&sig=60Z2H8v9237zvtckS0lCa5g%2FWkkUc%2FivqhEn8KDcSmM%3D"
SAS="sp=racwdli&st=2023-06-24T03:52:16Z&se=2024-06-24T11:52:16Z&spr=https&sv=2022-11-02&sr=c&sig=VMXIrGEFZEFSU6IrmxdjQSoj3wj8QTBWEE6CFzV9dic%3D"

# cine
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_retro_cine_2018.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_retro_cine_2019.h5?${SAS}" /export/Lab-Xue/projects/data
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_retro_cine_2020.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_retro_cine_2021.h5?${SAS}" /export/Lab-Xue/projects/data

ln -s /export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2019.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_retro_cine_2019.h5
ln -s /export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2021.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_retro_cine_2021.h5

azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/MINNESOTA_UHVC_RetroCine_1p5T_2023.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/MINNESOTA_UHVC_RetroCine_1p5T_2022.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RetroCine_3T_2023.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RetroCine_3T_2022.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_RetroCine_3T_2021.h5?${SAS}" /export/Lab-Xue/projects/imagenet

ln -s /export/Lab-Xue/projects/imagenet/MINNESOTA_UHVC_RetroCine_1p5T_2023.h5 /export/Lab-Xue/projects/mri/data/MINNESOTA_UHVC_RetroCine_1p5T_2023.h5
ln -s /export/Lab-Xue/projects/imagenet/MINNESOTA_UHVC_RetroCine_1p5T_2022.h5 /export/Lab-Xue/projects/mri/data/MINNESOTA_UHVC_RetroCine_1p5T_2022.h5
ln -s /export/Lab-Xue/projects/imagenet/BWH_RetroCine_3T_2023.h5 /export/Lab-Xue/projects/data/BWH_RetroCine_3T_2023.h5
ln -s /export/Lab-Xue/projects/imagenet/BWH_RetroCine_3T_2022.h5 /export/Lab-Xue/projects/data/BWH_RetroCine_3T_2022.h5 
ln -s /export/Lab-Xue/projects/imagenet/BWH_RetroCine_3T_2021.h5 /export/Lab-Xue/projects/data/BWH_RetroCine_3T_2021.h5 

# perfusion
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_perf_2018.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_perf_2019.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_perf_2020.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_perf_2021.h5?${SAS}" /export/Lab-Xue/projects/fm

ln -s /export/Lab-Xue/projects/fm/train_3D_3T_perf_2018.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_perf_2018.h5
ln -s /export/Lab-Xue/projects/fm/train_3D_3T_perf_2019.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_perf_2019.h5
ln -s /export/Lab-Xue/projects/fm/train_3D_3T_perf_2020.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_perf_2020.h5
ln -s /export/Lab-Xue/projects/fm/train_3D_3T_perf_2021.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_perf_2021.h5

azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_Perfusion_3T_2023.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_Perfusion_3T_2022.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/BWH_Perfusion_3T_2021.h5?${SAS}" /export/Lab-Xue/projects/fm

ln -s /export/Lab-Xue/projects/fm/BWH_Perfusion_3T_2023.h5 /export/Lab-Xue/projects/mri/data/BWH_Perfusion_3T_2023.h5
ln -s /export/Lab-Xue/projects/fm/BWH_Perfusion_3T_2022.h5 /export/Lab-Xue/projects/mri/data/BWH_Perfusion_3T_2022.h5
ln -s /export/Lab-Xue/projects/fm/BWH_Perfusion_3T_2021.h5 /export/Lab-Xue/projects/mri/data/BWH_Perfusion_3T_2021.h5

# test data
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_retro_cine_2020_small_2DT_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_retro_cine_2020_small_2D_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_retro_cine_2020_small_3D_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared/train_3D_3T_retro_cine_2020_500_samples.h5?${SAS}" /export/Lab-Xue/projects/mri/data

# imagenet
#azcopy copy "https://stcnnt.blob.core.windows.net/imagenet/downloaded/ILSVRC2012_img_train.tar?sp=racwdli&st=2023-05-23T12:12:36Z&se=2026-05-23T20:12:36Z&sv=2022-11-02&sr=c&sig=BD8VIaux4YSYsmkg6JdeIf1ckVAVmcGCnqlHGp93h8Y%3D" /export/Lab-Xue/projects/imagenet/data
#azcopy copy "https://stcnnt.blob.core.windows.net/imagenet/downloaded/ILSVRC2012_devkit_t12.tar.gz?sp=racwdli&st=2023-05-23T12:12:36Z&se=2026-05-23T20:12:36Z&sv=2022-11-02&sr=c&sig=BD8VIaux4YSYsmkg6JdeIf1ckVAVmcGCnqlHGp93h8Y%3D" /export/Lab-Xue/projects/imagenet/data
#azcopy copy "https://stcnnt.blob.core.windows.net/imagenet/downloaded/ILSVRC2012_img_val.tar?sp=racwdli&st=2023-05-23T12:12:36Z&se=2026-05-23T20:12:36Z&sv=2022-11-02&sr=c&sig=BD8VIaux4YSYsmkg6JdeIf1ckVAVmcGCnqlHGp93h8Y%3D" /export/Lab-Xue/projects/imagenet/data

#python3 -c "import torch; import torchvision as tv; print(torch.__version__); print(torch.cuda.is_available()); a = tv.datasets.ImageNet(root='/export/Lab-Xue/projects/imagenet/data', split='train');a = tv.datasets.ImageNet(root='/export/Lab-Xue/projects/imagenet/data', split='val')"

python3 -c "import torch; import torchvision as tv; print(torch.__version__); print(torch.cuda.is_available()); a = torch.zeros(12, device='cuda:0')"
