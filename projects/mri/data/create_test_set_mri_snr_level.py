"""
Create test set for STCNNT MRI
"""
import sys
import os
import tqdm
import h5py
import random
import numpy as np
import nibabel as nib

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[3].resolve()
sys.path.append(str(REPO_DIR))

from noise_augmentation import *


base_file_path = "/data/FM_data_repo/mri"
base_file_path = "/data1/mri/data"
base_file_name = "BARTS_RetroCine_3T_2023.h5"

snr_levels = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

matrix_size_adjust_ratio=[0.5, 0.75, 1.0, 1.25, 1.5]
kspace_filter_sigma=[0.8, 1.0, 1.5, 2.0, 2.25]
pf_filter_ratio=[1.0, 0.875, 0.75, 0.625]
phase_resolution_ratio=[1.0, 0.75, 0.65, 0.55]
readout_resolution_ratio=[1.0, 0.75, 0.65, 0.55]

res_file_path = "/export/Lab-Xue/projects/mri/data/mri_test"
os.makedirs(res_file_path, exist_ok=True)

# min_noise_level=1.0
# max_noise_level=20.0
# matrix_size_adjust_ratio=[1.0]
# kspace_filter_sigma=[1.2]
# pf_filter_ratio=[1.0]
# phase_resolution_ratio=[1.0]
# readout_resolution_ratio=[1.0]

h5_file = h5py.File(os.path.join(base_file_path, base_file_name))
keys = list(h5_file.keys())
n = len(keys)

def load_gmap(gmap, random_factor=0):
    """
    Loads a random gmap for current index
    """
    if(gmap.ndim==2):
        gmap = np.expand_dims(gmap, axis=0)

    factors = gmap.shape[0]
    if(random_factor<0):
        random_factor = np.random.randint(0, factors)

    return gmap[random_factor, :,:]

def create_3d_repeated(N=20):

    indices = np.arange(n)
    random.shuffle(indices)

    for k in range(N):

        i = indices[k]
        print(f"{k} out of {N}, {keys[i]}")

        ori_data = np.array(h5_file[keys[i]+"/image"])
        gmap = load_gmap(h5_file[keys[i]+"/gmap"][:], random_factor=0)

        T, RO, E1 = ori_data.shape

        signal_level = np.median(np.abs(ori_data))

        num_levels = len(snr_levels)

        clean_im = np.zeros((RO, E1, T, num_levels), dtype=np.complex64)
        noisy_im = np.zeros((RO, E1, T, num_levels), dtype=np.complex64)
        gmaps = np.zeros((RO, E1, num_levels))

        for ind, snr in enumerate(snr_levels):

            data = np.copy(ori_data)

            signal_level = np.median(np.abs(data))

            if snr > 0:
                noise_level = signal_level / snr
            else:
                noise_level = 1.0

            print(f"---> snr {snr}, signal_level {signal_level}, noise_level {noise_level}")


            nn, noise_sigma = generate_3D_MR_correlated_noise(T=T, RO=RO, E1=E1, REP=1,
                                                                min_noise_level=noise_level,
                                                                max_noise_level=noise_level,
                                                                kspace_filter_sigma=kspace_filter_sigma,
                                                                pf_filter_ratio=pf_filter_ratio,
                                                                phase_resolution_ratio=phase_resolution_ratio,
                                                                readout_resolution_ratio=readout_resolution_ratio,
                                                                only_white_noise=False,
                                                                verbose=False)

            nn *= gmap

            if snr > 0:
                noisy_data = data + np.copy(nn)
            else:
                noisy_data = np.copy(nn)

            data /= noise_sigma
            noisy_data /= noise_sigma

            clean_im[:,:,:,ind] = np.transpose(data, (1, 2, 0))
            noisy_im[:,:,:,ind] = np.transpose(noisy_data, (1, 2, 0))
            gmaps[:,:,ind] = gmap

        case_res_file_path = os.path.join(res_file_path, keys[i])
        os.makedirs(case_res_file_path, exist_ok=True)

        np.save(os.path.join(case_res_file_path, 'clean_real.npy'), clean_im.real)
        np.save(os.path.join(case_res_file_path, 'clean_imag.npy'), clean_im.imag)
        nib.save(nib.Nifti1Image(np.abs(clean_im), affine=np.eye(4)), os.path.join(case_res_file_path, 'clean.nii'))

        np.save(os.path.join(case_res_file_path, 'noisy_real.npy'), noisy_im.real)
        np.save(os.path.join(case_res_file_path, 'noisy_imag.npy'), noisy_im.imag)
        nib.save(nib.Nifti1Image(np.abs(noisy_im), affine=np.eye(4)), os.path.join(case_res_file_path, 'noisy.nii'))

        np.save(os.path.join(case_res_file_path, 'gmap.npy'), gmaps)
        nib.save(nib.Nifti1Image(gmaps, affine=np.eye(4)), os.path.join(case_res_file_path, 'gmap.nii'))

def main():

    create_3d_repeated(N=32)

if __name__=="__main__":
    main()
