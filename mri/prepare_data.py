#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import time

import os
import imp
import sys
import math
import time
import random
import shutil
import copy
import gc
import logging
import json
import pynvml

import torch
from skimage.util.shape import view_as_windows
from pathlib import Path 

GT_HOME = os.environ['GADGETRON_HOME']
GT_CMR_ML_UT_HOME = os.environ['GT_CMR_ML_UNITTEST_DIRECTORY']

print("GT_HOME is", GT_HOME)
print("GT_CMR_ML_UT_HOME is", GT_CMR_ML_UT_HOME)

case_list = []

# --------------------------- ----------------------------------
# scmr, free max retro cine
case_dir = '/export/Lab-Kellman/Share/data/FreeMax/SCMR/2022-08-23-HV-cardiac-SNR-DL/meas_MID00055_FID06126_MID_SAX_CINE_IPAT4_256Res_36ref/numpy'
im_scaling=1.0
gmap_scaling=1.0

case_list.append((case_dir, im_scaling, gmap_scaling))

# -------------------------------------------------------------
# rt cine, R4
case_name = 'RT_Cine_LIN_42110_3663094_3663103_959_20210128-095255'
case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'snr_gmap_denoising', case_name)

im_scaling=10.0
gmap_scaling=100.0
case_list.append((case_dir, im_scaling, gmap_scaling))
# --------------------------- ----------------------------------
# rt cine, R5
case_name = 'RT_Cine_LIN_42110_237143644_237143653_177_20220216-113137'
case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'snr_gmap_denoising', 'RT_Cine_R5', case_name)
im_scaling=10.0
gmap_scaling=100.0
case_list.append((case_dir, im_scaling, gmap_scaling))
# -------------------------------------------------------------
# rt cine, R6
case_name = 'RT_Cine_LIN_42110_237143644_237143653_178_20220216-113149'
case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'snr_gmap_denoising', 'RT_Cine_R6', case_name)
im_scaling=10.0
gmap_scaling=100.0
case_list.append((case_dir, im_scaling, gmap_scaling))        
# -------------------------------------------------------------
# perfusion
case_name = 'Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_7065950_7065959_2582_20210120-123912'

case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'snr_gmap_denoising', case_name)
im_scaling=10.0
gmap_scaling=100.0
case_list.append((case_dir, im_scaling, gmap_scaling))
# -------------------------------------------------------------
# perfusion, high res
case_name = 'Perfusion_AIF_2E_NL_Cloud_66097_9478344_9478349_482_20181023-122206'
case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'snr_gmap_denoising', case_name)
im_scaling=1.0
gmap_scaling=1.0
case_list.append((case_dir, im_scaling, gmap_scaling))
# -------------------------------------------------------------
# T1T2
case_name = 'T1SR_Mapping_SASHA_HC_T1T2_42363_681825496_681825501_573_20200623-164324'
case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'snr_gmap_denoising', 't1t2', case_name)
im_scaling=1.0
gmap_scaling=1.0
case_list.append((case_dir, im_scaling, gmap_scaling))
# -------------------------------------------------------------
# T1T2
case_name = 'T1SR_Mapping_SASHA_HC_T1T2_42363_681825496_681825501_573_20200623-164324_S2'
case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'snr_gmap_denoising', 't1t2', case_name)
im_scaling=1.0
gmap_scaling=1.0
case_list.append((case_dir, im_scaling, gmap_scaling))
# -------------------------------------------------------------
# WB LGE
case_name = 'WB_LGE_MOCO_AVE_OnTheFly_42110_7066558_7066567_3672_20210125-140041'
case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'snr_gmap_denoising', case_name)
im_scaling=1.0
gmap_scaling=1.0
case_list.append((case_dir, im_scaling, gmap_scaling))
# # -------------------------------------------------------------
# # low field, spine
# case_name = '20190627_160917_meas_MID00024_FID23439_t1_tse_sag_c-spine'
# case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'LowField', 'spine', case_name)
# im_scaling=1.0
# gmap_scaling=1.0
# case_list.append((case_dir, im_scaling, gmap_scaling))
# # -------------------------------------------------------------
# # low field, spine
# case_name = '20200305_213517_meas_MID00037_FID67006_t2_tse_sag_t-spine'
# case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'LowField', 'spine', case_name)
# im_scaling=1.0
# gmap_scaling=1.0
# case_list.append((case_dir, im_scaling, gmap_scaling))
# # -------------------------------------------------------------
# # low field, spine
# case_name = '20200305_214034_meas_MID00040_FID67009_t2_tse_sag_l-spine'
# case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'LowField', 'spine', case_name)
# im_scaling=1.0
# gmap_scaling=1.0
# case_list.append((case_dir, im_scaling, gmap_scaling))
# # -------------------------------------------------------------
# # low field, spine
# case_name = '20200305_212932_meas_MID00031_FID67000_t2_tse_sag_c-spine'
# case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'LowField', 'spine', case_name)
# im_scaling=1.0
# gmap_scaling=1.0
# case_list.append((case_dir, im_scaling, gmap_scaling))
# # -------------------------------------------------------------
# # low field, neuro, t1
# case_name = '20190806_171316_meas_MID00018_FID31911_t1_tse_dark-fluid_sag'
# case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'LowField', 'neuro', case_name)
# im_scaling=1.0
# gmap_scaling=1.0
# case_list.append((case_dir, im_scaling, gmap_scaling))
# # -------------------------------------------------------------
# # free max, neuro, t1
# case_name = 'meas_MID00636_FID06718_Ax_T1_PAT3_2conc'
# case_dir = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'denoising', 'FreeMax', 'neuro_T1', case_name)
# im_scaling=1.0
# gmap_scaling=1.0
# case_list.append((case_dir, im_scaling, gmap_scaling))

print(case_list)

import h5py

filename ='/export/Lab-Xue/projects/mri/data/mri_test.h5'
with h5py.File(filename , mode="w",libver='earliest') as h5file:
    for case in case_list:
        
        case_dir = case[0]
        im_scaling = case[1]
        gmap_scaling = case[2]
        
        image = np.load(os.path.join(case_dir, f"im_real.npy")) + np.load(os.path.join(case_dir, f"im_imag.npy")) * 1j
        image /= im_scaling
        image = np.squeeze(image)

        gmap = np.load(f"{case_dir}/gfactor.npy")
        gmap /= gmap_scaling

        print(f"{case_dir}, images - {image.shape}, gmap - {gmap.shape}")
        
        data_folder = h5file.create_group(case_dir)
        data_folder["image"] = image
        data_folder["gmap"] = gmap
