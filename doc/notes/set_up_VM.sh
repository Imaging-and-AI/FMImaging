#!bash

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
