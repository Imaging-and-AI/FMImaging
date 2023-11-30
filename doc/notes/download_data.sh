#!/usr/bin/bash

export SAS="sp=racwdli&st=2023-06-10T20:45:41Z&se=2024-06-11T04:45:41Z&sv=2022-11-02&sr=c&sig=60Z2H8v9237zvtckS0lCa5g%2FWkkUc%2FivqhEn8KDcSmM%3D"
export SAS="sp=racwdli&st=2023-06-24T03:52:16Z&se=2024-06-24T11:52:16Z&spr=https&sv=2022-11-02&sr=c&sig=VMXIrGEFZEFSU6IrmxdjQSoj3wj8QTBWEE6CFzV9dic%3D"

export data_src=https://stcnnt.blob.core.windows.net/mri/data/denoising/data_prepared

export SAS_qperf="sp=racwdli&st=2023-10-22T21:32:48Z&se=2024-10-23T05:32:48Z&spr=https&sv=2022-11-02&sr=c&sig=cFBFXqcvpRZyfw9xyHKhPM%2B67l6F3LyBZLRy0fsrIYY%3D"
export data_src_qperf=https://stcnnt.blob.core.windows.net/qperf

export SAS2="sp=racwdli&st=2023-10-29T19:02:56Z&se=2025-10-30T03:02:56Z&spr=https&sv=2022-11-02&sr=c&sig=WbK6S7bLpXJ%2F4838iQTMrYPCcY%2FdH5w9k8bw6gz9uTk%3D"

# azcopy copy "./val/" "{$data_src}?${SAS}" --recursive
# azcopy copy "./val/" "https://stcnnt.blob.core.windows.net/qperf/?${SAS_qperf}" --recursive
# azcopy copy "https://stcnnt.blob.core.windows.net/qperf/h5_data?${SAS_qperf}" . --recursive

# mkdir -p /export/Lab-Xue/projects/imagenet/qperf/v2
# mkdir -p /export/Lab-Xue/projects/imagenet/qperf/v2/tra
# mkdir -p /export/Lab-Xue/projects/imagenet/qperf/v2/val
# mkdir -p /export/Lab-Xue/projects/imagenet/qperf/v2/test
# azcopy copy "${data_src_qperf}/v2/tra/aif.npy?${SAS2}" /export/Lab-Xue/projects/imagenet/qperf/v2/tra --recursive
# azcopy copy "${data_src_qperf}/v2/tra/myo.npy?${SAS2}" /export/Lab-Xue/projects/imagenet/qperf/v2/tra --recursive
# azcopy copy "${data_src_qperf}/v2/tra/params.npy?${SAS2}" /export/Lab-Xue/projects/imagenet/qperf/v2/tra --recursive
# azcopy copy "${data_src_qperf}/v2/val?${SAS2}" /export/Lab-Xue/projects/imagenet/qperf/v2 --recursive
# azcopy copy "${data_src_qperf}/v2/test?${SAS2}" /export/Lab-Xue/projects/imagenet/qperf/v2 --recursive

#mkdir -p /data/qperf
#azcopy copy "${data_src_qperf}/h5_data/tra?${SAS_qperf}" /data/qperf/ --recursive
#azcopy copy "${data_src_qperf}/h5_data/val?${SAS_qperf}" /data/qperf/ --recursive
#azcopy copy "${data_src_qperf}/h5_data/test?${SAS_qperf}" /data/qperf/ --recursive

# test data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_2DT_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_2D_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_3D_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_500_samples.h5?${SAS}" /export/Lab-Xue/projects/mri/data

#azcopy copy "${data_src}/test_2D_sig_1_16_1000.h5?${SAS}" /export/Lab-Xue/projects/data
#azcopy copy "${data_src}/test_2DT_sig_1_16_2000.h5?${SAS}" /export/Lab-Xue/projects/data
#ln -s /export/Lab-Xue/projects/data/test_2D_sig_1_16_1000.h5 /export/Lab-Xue/projects/mri/data/test_2D_sig_1_16_1000.h5
#ln -s /export/Lab-Xue/projects/data/test_2DT_sig_1_16_2000.h5 /export/Lab-Xue/projects/mri/data/test_2DT_sig_1_16_2000.h5

azcopy copy "${data_src}/test_2D_sig_2_40_1000.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "${data_src}/test_2DT_sig_2_40_2000.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "${data_src}/VIDA_train_clean_0430.h5?${SAS}" /export/Lab-Xue/projects/imagenet

ln -s /export/Lab-Xue/projects/imagenet/test_2D_sig_2_40_1000.h5 /export/Lab-Xue/projects/mri/data/test_2D_sig_2_40_1000.h5
ln -s /export/Lab-Xue/projects/imagenet/test_2DT_sig_2_40_2000.h5 /export/Lab-Xue/projects/mri/data/test_2DT_sig_2_40_2000.h5
ln -s /export/Lab-Xue/projects/imagenet/VIDA_train_clean_0430.h5 /export/Lab-Xue/projects/mri/data/VIDA_train_clean_0430.h5

# test data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_2DT_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_2D_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_3D_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_500_samples.h5?${SAS}" /export/Lab-Xue/projects/mri/data

# cine
azcopy copy "${data_src}/BARTS_3D_3T_retro_cine_2023.h5?${SAS}" /export/Lab-Xue/projects/data
ln -s /export/Lab-Xue/projects/data/BARTS_3D_3T_retro_cine_2023.h5 /export/Lab-Xue/projects/mri/data/BARTS_3D_3T_retro_cine_2023.h5

azcopy copy "${data_src}/train_3D_3T_retro_cine_2018.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2019.h5?${SAS}" /export/Lab-Xue/projects/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2021.h5?${SAS}" /export/Lab-Xue/projects/data

ln -s /export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2019.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_retro_cine_2019.h5
ln -s /export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2021.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_retro_cine_2021.h5

azcopy copy "${data_src}/MINNESOTA_UHVC_RetroCine_1p5T_2023.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "${data_src}/MINNESOTA_UHVC_RetroCine_1p5T_2022.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "${data_src}/BWH_RetroCine_3T_2023.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "${data_src}/BWH_RetroCine_3T_2022.h5?${SAS}" /export/Lab-Xue/projects/imagenet
azcopy copy "${data_src}/BWH_RetroCine_3T_2021.h5?${SAS}" /export/Lab-Xue/projects/imagenet

ln -s /export/Lab-Xue/projects/imagenet/MINNESOTA_UHVC_RetroCine_1p5T_2023.h5 /export/Lab-Xue/projects/mri/data/MINNESOTA_UHVC_RetroCine_1p5T_2023.h5
ln -s /export/Lab-Xue/projects/imagenet/MINNESOTA_UHVC_RetroCine_1p5T_2022.h5 /export/Lab-Xue/projects/mri/data/MINNESOTA_UHVC_RetroCine_1p5T_2022.h5
ln -s /export/Lab-Xue/projects/imagenet/BWH_RetroCine_3T_2023.h5 /export/Lab-Xue/projects/mri/data/BWH_RetroCine_3T_2023.h5
ln -s /export/Lab-Xue/projects/imagenet/BWH_RetroCine_3T_2022.h5 /export/Lab-Xue/projects/mri/data/BWH_RetroCine_3T_2022.h5 
ln -s /export/Lab-Xue/projects/imagenet/BWH_RetroCine_3T_2021.h5 /export/Lab-Xue/projects/mri/data/BWH_RetroCine_3T_2021.h5 

azcopy copy "${data_src}/BARTS_RetroCine_3T_2023.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "${data_src}/BARTS_RetroCine_1p5T_2023.h5?${SAS}" /export/Lab-Xue/projects/fm

ln -s /export/Lab-Xue/projects/fm/BARTS_RetroCine_3T_2023.h5 /export/Lab-Xue/projects/mri/data/BARTS_RetroCine_3T_2023.h5
ln -s /export/Lab-Xue/projects/fm/BARTS_RetroCine_1p5T_2023.h5 /export/Lab-Xue/projects/mri/data/BARTS_RetroCine_1p5T_2023.h5

exit 0

# perfusion
azcopy copy "${data_src}/train_3D_3T_perf_2018.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "${data_src}/train_3D_3T_perf_2019.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "${data_src}/train_3D_3T_perf_2020.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "${data_src}/train_3D_3T_perf_2021.h5?${SAS}" /export/Lab-Xue/projects/fm

ln -s /export/Lab-Xue/projects/fm/train_3D_3T_perf_2018.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_perf_2018.h5
ln -s /export/Lab-Xue/projects/fm/train_3D_3T_perf_2019.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_perf_2019.h5
ln -s /export/Lab-Xue/projects/fm/train_3D_3T_perf_2020.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_perf_2020.h5
ln -s /export/Lab-Xue/projects/fm/train_3D_3T_perf_2021.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_perf_2021.h5

azcopy copy "${data_src}/BWH_Perfusion_3T_2023.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "${data_src}/BWH_Perfusion_3T_2022.h5?${SAS}" /export/Lab-Xue/projects/fm
azcopy copy "${data_src}/BWH_Perfusion_3T_2021.h5?${SAS}" /export/Lab-Xue/projects/fm

ln -s /export/Lab-Xue/projects/fm/BWH_Perfusion_3T_2023.h5 /export/Lab-Xue/projects/mri/data/BWH_Perfusion_3T_2023.h5
ln -s /export/Lab-Xue/projects/fm/BWH_Perfusion_3T_2022.h5 /export/Lab-Xue/projects/mri/data/BWH_Perfusion_3T_2022.h5
ln -s /export/Lab-Xue/projects/fm/BWH_Perfusion_3T_2021.h5 /export/Lab-Xue/projects/mri/data/BWH_Perfusion_3T_2021.h5

# imagenet
#azcopy copy "https://stcnnt.blob.core.windows.net/imagenet/downloaded/ILSVRC2012_img_train.tar?sp=racwdli&st=2023-05-23T12:12:36Z&se=2026-05-23T20:12:36Z&sv=2022-11-02&sr=c&sig=BD8VIaux4YSYsmkg6JdeIf1ckVAVmcGCnqlHGp93h8Y%3D" /export/Lab-Xue/projects/imagenet/data
#azcopy copy "https://stcnnt.blob.core.windows.net/imagenet/downloaded/ILSVRC2012_devkit_t12.tar.gz?sp=racwdli&st=2023-05-23T12:12:36Z&se=2026-05-23T20:12:36Z&sv=2022-11-02&sr=c&sig=BD8VIaux4YSYsmkg6JdeIf1ckVAVmcGCnqlHGp93h8Y%3D" /export/Lab-Xue/projects/imagenet/data
#azcopy copy "https://stcnnt.blob.core.windows.net/imagenet/downloaded/ILSVRC2012_img_val.tar?sp=racwdli&st=2023-05-23T12:12:36Z&se=2026-05-23T20:12:36Z&sv=2022-11-02&sr=c&sig=BD8VIaux4YSYsmkg6JdeIf1ckVAVmcGCnqlHGp93h8Y%3D" /export/Lab-Xue/projects/imagenet/data

azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_3D_test_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_2DT_test_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_2D_test_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_500_samples_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data

azcopy copy "${data_src}/train_3D_3T_retro_cine_2018_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2019_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/BARTS_RetroCine_3T_2023_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/BARTS_RetroCine_1p5T_2023_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/MINNESOTA_UHVC_RetroCine_1p5T_2023_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/MINNESOTA_UHVC_RetroCine_1p5T_2022_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/VIDA_train_clean_0430_with_2x_resized.h5?${SAS}" /export/Lab-Xue/projects/mri/data


# cine
azcopy copy "${data_src}/train_3D_3T_retro_cine_2018.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2019.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2021.h5?${SAS}" /export/Lab-Xue/projects/mri/data

azcopy copy "${data_src}/MINNESOTA_UHVC_RetroCine_1p5T_2023.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/MINNESOTA_UHVC_RetroCine_1p5T_2022.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/BWH_RetroCine_3T_2023.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/BWH_RetroCine_3T_2022.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/BWH_RetroCine_3T_2021.h5?${SAS}" /export/Lab-Xue/projects/mri/data

azcopy copy "${data_src}/BARTS_RetroCine_3T_2023.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/BARTS_RetroCine_1p5T_2023.h5?${SAS}" /export/Lab-Xue/projects/mri/data

azcopy copy "${data_src}/train_3D_3T_perf_2018.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_perf_2019.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_perf_2020.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_perf_2021.h5?${SAS}" /export/Lab-Xue/projects/mri/data

azcopy copy "${data_src}/BWH_Perfusion_3T_2023.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/BWH_Perfusion_3T_2022.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/BWH_Perfusion_3T_2021.h5?${SAS}" /export/Lab-Xue/projects/mri/data

azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_2DT_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_2D_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_small_3D_test.h5?${SAS}" /export/Lab-Xue/projects/mri/data
azcopy copy "${data_src}/train_3D_3T_retro_cine_2020_500_samples.h5?${SAS}" /export/Lab-Xue/projects/mri/data

ln -s /export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2020_small_3D_test_with_2x_resized.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_retro_cine_2020_small_3D_test_with_2x_resized.h5
ln -s /export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2020_small_2DT_test_with_2x_resized.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_retro_cine_2020_small_2DT_test_with_2x_resized.h5
ln -s /export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2020_small_2D_test_with_2x_resized.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_retro_cine_2020_small_2D_test_with_2x_resized.h5
ln -s /export/Lab-Xue/projects/data/train_3D_3T_retro_cine_2020_500_samples_with_2x_resized.h5 /export/Lab-Xue/projects/mri/data/train_3D_3T_retro_cine_2020_500_samples_with_2x_resized.h5

