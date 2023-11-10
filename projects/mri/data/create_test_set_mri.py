"""
Create test set for STCNNT MRI
"""
import os
import tqdm
import h5py
import random
import numpy as np

from noise_augmentation import *


base_file_path = "/data1/mri"
base_file_name = "BARTS_RetroCine_3T_2023.h5"

min_noise_level=1.0
max_noise_level=12.0
matrix_size_adjust_ratio=[0.5, 0.75, 1.0, 1.25, 1.5]
kspace_filter_sigma=[0.8, 1.0, 1.5, 2.0, 2.25]
pf_filter_ratio=[1.0, 0.875, 0.75, 0.625]
phase_resolution_ratio=[1.0, 0.75, 0.65, 0.55]
readout_resolution_ratio=[1.0, 0.75, 0.65, 0.55]

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

def create_2d(write_path, N=1000):

    h5_file_2d = h5py.File(write_path, mode="w", libver="earliest")

    for i in range(N):

        data = np.array(h5_file[keys[i]+"/image"])
        gmap = load_gmap(h5_file[keys[i]+"/gmap"][:], random_factor=-1)

        T, RO, E1 = data.shape

        nn, noise_sigma = generate_3D_MR_correlated_noise(T=T, RO=RO, E1=E1, REP=1,
                                                            min_noise_level=min_noise_level,
                                                            max_noise_level=max_noise_level,
                                                            kspace_filter_sigma=kspace_filter_sigma,
                                                            pf_filter_ratio=pf_filter_ratio,
                                                            phase_resolution_ratio=phase_resolution_ratio,
                                                            readout_resolution_ratio=readout_resolution_ratio,
                                                            only_white_noise=False,
                                                            verbose=True)

        nn *= gmap

        noisy_data = data + nn

        data /= noise_sigma
        noisy_data /= noise_sigma

        rand_frame = np.random.randint(0,T)

        noisy = noisy_data[rand_frame]
        clean = data[rand_frame]
        gmap = gmap
        noise_sigma = noise_sigma

        data_folder = h5_file_2d.create_group(keys[i])

        data_folder["noisy"] = noisy
        data_folder["image"] = data
        data_folder["image_resized"] = data
        data_folder["gmap"] = gmap
        data_folder["noise_sigma"] = noise_sigma

def create_3d(write_path, N=1000):

    h5_file_3d = h5py.File(write_path, mode="w", libver="earliest")

    indices = np.arange(n)
    random.shuffle(indices)

    for k in tqdm.tqdm(range(N)):

        i = indices[k]

        data = np.array(h5_file[keys[i]+"/image"])
        gmap = load_gmap(h5_file[keys[i]+"/gmap"][:], random_factor=-1)

        clean = np.copy(data)

        T, RO, E1 = data.shape

        nn, noise_sigma = generate_3D_MR_correlated_noise(T=T, RO=RO, E1=E1, REP=1,
                                                            min_noise_level=min_noise_level,
                                                            max_noise_level=max_noise_level,
                                                            kspace_filter_sigma=kspace_filter_sigma,
                                                            pf_filter_ratio=pf_filter_ratio,
                                                            phase_resolution_ratio=phase_resolution_ratio,
                                                            readout_resolution_ratio=readout_resolution_ratio,
                                                            only_white_noise=False,
                                                            verbose=True)

        nn *= gmap

        noisy_data = data + nn

        data /= noise_sigma
        noisy_data /= noise_sigma

        grp_name = f"Image_{k:03d}_{keys[i]}"
        data_folder = h5_file_3d.create_group(grp_name)

        data_folder["noisy"] = noisy_data
        data_folder["image"] = data
        data_folder["image_resized"] = data
        data_folder["gmap"] = gmap
        data_folder["noise_sigma"] = noise_sigma

        noise = (noisy_data - clean/noise_sigma) / gmap
        print(f"{grp_name}, {noise_sigma}, data noise std - {np.mean(np.std(np.real(noise), axis=0))} - {np.mean(np.std(np.imag(noise), axis=0))}")

def create_3d_repeated(write_path, N=20, sigmas=[1,11,1], random_mask=False):

    h5_file_3d = h5py.File(write_path, mode="w", libver="earliest")

    indices = np.arange(n)
    random.shuffle(indices)

    #for k in tqdm.tqdm(range(N)):
    for k in range(N):

        i = indices[k]
        print(f"{k} out of {N}, {keys[i]}")
        
        ori_data = np.array(h5_file[keys[i]+"/image"])
        gmap = load_gmap(h5_file[keys[i]+"/gmap"][:], random_factor=0)

        for noise_sig in np.arange(sigmas[0],sigmas[1],sigmas[2]):

            data = np.copy(ori_data)
            if random_mask:
                # mask out the signal
                T, RO, E1 = data.shape
                start_ro = int(RO//2 - 16)
                end_ro = int(RO//2 + 16)
                start_e1 = int(E1//2 - 16)
                end_e1 = int(E1//2 + 16)

                data[:, start_ro:end_ro, start_e1:end_e1] = 0

            T, RO, E1 = data.shape

            nn, noise_sigma = generate_3D_MR_correlated_noise(T=T, RO=RO, E1=E1, REP=1,
                                                                min_noise_level=noise_sig,
                                                                max_noise_level=noise_sig,
                                                                kspace_filter_sigma=kspace_filter_sigma,
                                                                pf_filter_ratio=pf_filter_ratio,
                                                                phase_resolution_ratio=phase_resolution_ratio,
                                                                readout_resolution_ratio=readout_resolution_ratio,
                                                                only_white_noise=False,
                                                                verbose=False)

            nn *= gmap

            noisy_data = data + nn

            data /= noise_sigma
            noisy_data /= noise_sigma

            grp_name = f"Image_{k:03d}_{keys[i]}_sig_{noise_sig:02d}"
            data_folder = h5_file_3d.create_group(grp_name)

            data_folder["noisy"] = noisy_data
            data_folder["image"] = data
            data_folder["image_resized"] = data
            data_folder["gmap"] = gmap
            data_folder["noise_sigma"] = noise_sigma

            noise = (noisy_data - ori_data/noise_sigma) / gmap
            print(f"{grp_name}, {noise_sigma}, data noise std - {np.mean(np.std(np.real(noise), axis=0))} - {np.mean(np.std(np.imag(noise), axis=0))}")


def main():

    write_path_2d = f"{base_file_path}/test_2D_sig_1_16_1000.h5"
    create_3d(write_path=write_path_2d, N=1000)
    print(f"{write_path_2d} - done")

    write_path_3d = f"{base_file_path}/test_2DT_sig_1_16_2000.h5"
    create_3d(write_path=write_path_3d, N=2000)
    print(f"{write_path_3d} - done")

    # write_path_20_3d_repeated = f"{base_file_path}/train_3D_3T_retro_cine_2020_20_sample_sig_2_30_repeated_test.h5"    
    # create_3d_repeated(write_path=write_path_20_3d_repeated)

    # write_path = f"{base_file_path}/retro_cine_3T_sigma_1_20_repeated_test_3rd.h5"
    # create_3d_repeated(write_path=write_path, N=200, sigmas=[1,21,1], random_mask=False)

    # write_path = f"{base_file_path}/retro_cine_3T_sigma_1_20_repeated_test_2nd_random_mask.h5"
    # create_3d_repeated(write_path=write_path, N=20, sigmas=[1,21,1], random_mask=True)

if __name__=="__main__":
    main()
