# MRI image enhancement training

Create datasets

## data convertion


### BWH
```
site=BWH

# retro cine
python3 ./mri/create_hdf5_3D_dataset.py /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2023_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2022_AI_denoising /export/Lab-Kellman/ReconResults/denoising/${site}/${site}_RetroCine_2021_AI_denoising --output /export/Lab-Kellman/ReconResults/denoising/data_prepared/${site}_RetroCine_3T_2023_2022_2021.h5 --only_3T --im_scaling 10.0

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