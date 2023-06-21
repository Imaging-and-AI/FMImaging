# MRI image enhancement training

## Run the data

```

sudo mkdir -p /data/Debug/DebugOutput_RetroCine
sudo mkdir -p /data/Debug/DebugOutput_Perfusion
sudo mkdir -p /data/Debug/DebugOutput_LGE
sudo mkdir -p /data/Debug/DebugOutput_RTCine

sudo chmod a+rw /data
sudo chmod a+rw /data/Debug

sudo mkdir /data/gt
sudo chmod a+rw /data/gt

sudo chmod a+rw /data/Debug/DebugOutput_RetroCine
sudo chmod a+rw /data/Debug/DebugOutput_Perfusion
sudo chmod a+rw /data/Debug/DebugOutput_LGE
sudo chmod a+rw /data/Debug/DebugOutput_RTCine

docker run  --publish=9016:9002 --name=gt_retro_cine --volume=/data/Debug/DebugOutput_RetroCine:/tmp/DebugOutput --volume=/home/xueh2/gadgetron_ismrmrd_data:/tmp/gadgetron_data --restart=unless-stopped -e OMP_THREAD_LIMIT=$(nproc)  -e GADGETRON_DEBUG_FOLDER=/tmp --detach -t gadgetronnhlbi/gtprep4px_ubuntu_2204_cuda12_pytorch12:20230619

docker run  --publish=9017:9002 --name=gt_perfusion --volume=/data/Debug/DebugOutput_Perfusion:/tmp/DebugOutput --volume=/home/xueh2/gadgetron_ismrmrd_data:/tmp/gadgetron_data --restart=unless-stopped -e OMP_THREAD_LIMIT=$(nproc)  -e GADGETRON_DEBUG_FOLDER=/tmp --detach -t gadgetronnhlbi/gtprep4px_ubuntu_2204_cuda12_pytorch12:20230619

docker run  --publish=9018:9002 --name=gt_LGE --volume=/data/Debug/DebugOutput_LGE:/tmp/DebugOutput --volume=/home/xueh2/gadgetron_ismrmrd_data:/tmp/gadgetron_data --restart=unless-stopped -e OMP_THREAD_LIMIT=$(nproc)  -e GADGETRON_DEBUG_FOLDER=/tmp --detach -t gadgetronnhlbi/gtprep4px_ubuntu_2204_cuda12_pytorch12:20230619

docker run  --publish=9019:9002 --name=gt_RTCine --volume=/data/Debug/DebugOutput_RTCine:/tmp/DebugOutput --volume=/home/xueh2/gadgetron_ismrmrd_data:/tmp/gadgetron_data --restart=unless-stopped -e OMP_THREAD_LIMIT=$(nproc)  -e GADGETRON_DEBUG_FOLDER=/tmp --detach -t gadgetronnhlbi/gtprep4px_ubuntu_2204_cuda12_pytorch12:20230619
```

## data convertion


### BWH
```
site=BWH

# retro cine
python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2023_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2022_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2021_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_RetroCine_3T_2023_2022_2021.h5 --only_3T --im_scaling 10.0

python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2023_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2022_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2021_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_RetroCine_1p5T_2023_2022_2021.h5 --no_3T --im_scaling 10.0

# rt cine
python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RTCine_2023_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RTCine_2022_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RTCine_2021_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_RTCine_3T_2023_2022_2021.h5 --only_3T --im_scaling 10.0

# perfusion
python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_Perfusion_2023_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_Perfusion_2022_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_Perfusion_2021_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_Perfusion_3T_2023_2022_2021.h5 --only_3T --im_scaling 1.0

# LGE - moco-ave
python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_2023_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_2022_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_2021_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_LGE_ave_3T_2023_2022_2021.h5 --only_3T --im_scaling 1.0 --input_fname im

# LGE - raw
python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_2023_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_2022_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_2021_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_LGE_3T_raw_2023_2022_2021.h5 --only_3T --im_scaling 10.0 --input_fname raw_im
```

### MINNESOTA_UHVC
```
site=MINNESOTA_UHVC

for year in 2023 2022 2021 2020 2019 2018
do
    #year=2023

    # retro cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_${year}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_RetroCine_3T_${year}.h5 --only_3T --im_scaling 10.0

    # rt cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RTCine_${site}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_RTCine_3T_${site}.h5 --only_3T --im_scaling 10.0

    # perfusion
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_Perfusion_${site}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_Perfusion_3T_${site}.h5 --only_3T --im_scaling 1.0

    # LGE - moco-ave
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${site}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_LGE_ave_3T_${site}.h5 --only_3T --im_scaling 1.0 --input_fname im

    # LGE - raw
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${site}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_LGE_3T_raw_${site}.h5 --only_3T --im_scaling 10.0 --input_fname raw_im
done

```

### Barts
```
site=BARTS

for year in 2023 2022 2021 2020 2019 2018
do
    # retro cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_${year}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_RetroCine_3T_${year}.h5 --only_3T --im_scaling 10.0

    # rt cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RTCine_${site}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_RTCine_3T_${site}.h5 --only_3T --im_scaling 10.0

    # perfusion
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_Perfusion_${site}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_Perfusion_3T_${site}.h5 --only_3T --im_scaling 1.0

    # LGE - moco-ave
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${site}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_LGE_ave_3T_${site}.h5 --only_3T --im_scaling 1.0 --input_fname im

    # LGE - raw
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${site}_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_LGE_3T_raw_${site}.h5 --only_3T --im_scaling 10.0 --input_fname raw_im
done
```