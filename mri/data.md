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

docker run  --publish=9016:9002 --name=gt_retro_cine --volume=/data/Debug/DebugOutput_RetroCine:/tmp/DebugOutput --volume=/home/xueh2/gadgetron_ismrmrd_data:/tmp/gadgetron_data --restart=unless-stopped -e OMP_THREAD_LIMIT=$(nproc)  -e GADGETRON_DEBUG_FOLDER=/tmp --detach -t gadgetronnhlbi/gtprep4px_ubuntu_2204_cuda12_pytorch12:20230620

docker run  --publish=9017:9002 --name=gt_perfusion --volume=/data/Debug/DebugOutput_Perfusion:/tmp/DebugOutput --volume=/home/xueh2/gadgetron_ismrmrd_data:/tmp/gadgetron_data --restart=unless-stopped -e OMP_THREAD_LIMIT=$(nproc)  -e GADGETRON_DEBUG_FOLDER=/tmp --detach -t gadgetronnhlbi/gtprep4px_ubuntu_2204_cuda12_pytorch12:20230620

docker run  --publish=9018:9002 --name=gt_LGE --volume=/data/Debug/DebugOutput_LGE:/tmp/DebugOutput --volume=/home/xueh2/gadgetron_ismrmrd_data:/tmp/gadgetron_data --restart=unless-stopped -e OMP_THREAD_LIMIT=$(nproc)  -e GADGETRON_DEBUG_FOLDER=/tmp --detach -t gadgetronnhlbi/gtprep4px_ubuntu_2204_cuda12_pytorch12:20230620

docker run  --publish=9019:9002 --name=gt_RTCine --volume=/data/Debug/DebugOutput_RTCine:/tmp/DebugOutput --volume=/home/xueh2/gadgetron_ismrmrd_data:/tmp/gadgetron_data --restart=unless-stopped -e OMP_THREAD_LIMIT=$(nproc)  -e GADGETRON_DEBUG_FOLDER=/tmp --detach -t gadgetronnhlbi/gtprep4px_ubuntu_2204_cuda12_pytorch12:20230620
```

## data convertion


### BWH
```
site=BWH

for year in 2023 2022 2021
do
    #year=2023

    # retro cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_RetroCine_3T_${year}.h5 --only_3T --im_scaling 10.0

    # rt cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RTCine_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_RTCine_3T_${year}.h5 --only_3T --im_scaling 10.0

    # perfusion
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_Perfusion_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_Perfusion_3T_${year}.h5 --only_3T --im_scaling 1.0

    # LGE - moco-ave
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_LGE_ave_3T_${year}.h5 --only_3T --im_scaling 1.0 --input_fname im

    # LGE - raw
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_LGE_3T_raw_${year}.h5 --only_3T --im_scaling 10.0 --input_fname raw_im
done
```

### MINNESOTA_UHVC
```
site=MINNESOTA_UHVC

for year in 2023 2022 2021 2020 2019 2018
do
    #year=2023

    # retro cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_RetroCine_1p5T_${year}.h5 --no_3T --im_scaling 10.0

    # rt cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RTCine_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_RTCine_1p5T_${site}.h5 --no_3T --im_scaling 10.0

    # perfusion
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_Perfusion_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_Perfusion_1p5T_${site}.h5 --no_3T --im_scaling 1.0

    # LGE - moco-ave
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_LGE_ave_1p5T_${site}.h5 --no_3T --im_scaling 1.0 --input_fname im

    # LGE - raw
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_LGE_1p5T_raw_${site}.h5 --no_3T --im_scaling 10.0 --input_fname raw_im
done

```

### Barts
```
site=BARTS

for year in 2023 2022 2021 2020 2019 2018
do
    # retro cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_RetroCine_3T_${year}.h5 --only_3T --im_scaling 10.0

    # rt cine
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RTCine_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_RTCine_3T_${site}.h5 --only_3T --im_scaling 10.0

    # perfusion
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_Perfusion_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_Perfusion_3T_${site}.h5 --only_3T --im_scaling 1.0

    # LGE - moco-ave
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_LGE_ave_3T_${site}.h5 --only_3T --im_scaling 1.0 --input_fname im

    # LGE - raw
    python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_LGE_${year}_AI_denoising --output /export/Lab-Xue/projects/mri/data/${site}_LGE_3T_raw_${site}.h5 --only_3T --im_scaling 10.0 --input_fname raw_im
done
```