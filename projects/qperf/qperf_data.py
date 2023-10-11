"""
Data utilities for QPerf.
"""

import os
import sys
import scipy
import shutil
import cv2
import h5py
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from skimage.util import view_as_blocks
from colorama import Fore, Style

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

# -------------------------------------------------------------------------------------------------

class QPerfDataSet(torch.utils.data.Dataset):
    """
    Every sample includes aif ([B, T, D]), myo ([B, T, D]) and parameters (Fp, Vp, PS, Visf, delay, foot, peak, valley)
    """
    def __init__(self, data_folder, 
                        max_load=-1,
                        T=80, 
                        foot_to_end=True, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[1.0, 0.25],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True]):
        """
        @args:
            - data_folder : folder to store the mat files
            - T : number of time points
            - min_noise_level : minimal noise sigma to add for aif and myo
            - max_noise_level : maximal noise sigma to add for aif and myo
            - matrix_size_adjust_ratio (float list): down/upsample the image, keeping the fov
            - only_white_noise : if True, only add white noise; otherwise, add correlated noise
            - add_noise : whether to add noise for aif or myo
        """
        self.data_folder = data_folder
        self.max_load = max_load
        self.T = T
        self.foot_to_end = foot_to_end
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.filter_sigma = filter_sigma
        self.only_white_noise = only_white_noise
        self.add_noise = add_noise

        all_data_files = []

        site_dirs = os.listdir(self.data_folder)
        for a_case in site_dirs:
            if a_case.endswith('mat'):
                all_data_files.append(os.path.join(self.data_folder, a_case))

        N = len(all_data_files)
        print(f"--> {N} cases found for data folder : {self.data_folder}")

        self.aif = []
        self.myo = []
        self.params = []

        cases_loaded = 0
        for ind in range(N):
            if self.max_load> 0 and cases_loaded > self.max_load:
                break

            a_case = all_data_files[ind]
            if os.path.isfile(a_case):
                mat = scipy.io.loadmat(a_case)

                N = mat['out'].shape[1]
                print(f"--> case {ind} loaded from {N} cases - {a_case} ... ")

                #pbar = tqdm(total=N)
                for ind in range(mat['out'].shape[1]):
                        params = mat['out'][0,ind]['params'][0,0].flatten()
                        aif = mat['out'][0,ind]['aif'][0,0].flatten().astype(np.float16)
                        myo = mat['out'][0,ind]['myo'][0,0].flatten().astype(np.float16)

                        self.aif.append(aif)
                        self.myo.append(myo)
                        self.params.append(params)

                        # if ind % 10000 == 0:
                        #     pbar.update(10000)
                        #     pbar.set_description_str(f"{a_case}, {ind} out of {N}, {aif.shape[0]}, {params[0]:.4f}")

                cases_loaded += 1
                print(f"--> {len(self.aif)} samples loaded from {cases_loaded} cases - {a_case} ... ")

        print(f"--> {len(self.aif)} samples loaded from {cases_loaded} cases ... ")

    def load_one_sample(self, i):
        """
        Loads one sample from the loaded data images
        @args:
            - i (int): index of the file to load
        @rets:
            - x : [T, 2] for aif and myo
            - y : [T, 1] for clean myo
            - p : [1, 5] for parameters (Fp, Vp, PS, Visf, delay)
        """

        aif = self.aif[i]
        y = self.myo[i]
        params = self.params[i]
        p = params[:5]

        if self.add_noise[0]:
            # add noise to aif
            N = aif.shape[0]
            sigma = np.random.uniform(self.min_noise_level[0], self.max_noise_level[0])
            nn = np.random.standard_normal(size=N) * sigma

            if not self.only_white_noise:
                fs = self.filter_sigma[np.random.randint(0, len(self.filter_sigma))]
                nn = scipy.ndimage.gaussian_filter1d(nn, fs, axis=0)

            aif += nn

        if self.add_noise[1]:
            # add noise to myo
            N = y.shape[0]
            sigma = np.random.uniform(self.min_noise_level[1], self.max_noise_level[1])
            nn = np.random.standard_normal(size=N) * sigma

            if not self.only_white_noise:
                fs = self.filter_sigma[np.random.randint(0, len(self.filter_sigma))]
                nn = scipy.ndimage.gaussian_filter1d(nn, fs, axis=0)

            myo = np.copy(y)
            myo += nn

        x = np.concatenate((aif[:, np.newaxis], myo[:, np.newaxis]), axis=1)

        return x, y[:, np.newaxis], p[np.newaxis, :]

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.aif)

    def __getitem__(self, index):
        """
        Gets the item given index
        """
        return self.load_one_sample(index)

# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    saved_path = "/export/Lab-Xue/projects/qperf/results/loader_test"
    if os.path.exists(saved_path):
        shutil.rmtree(saved_path)
    os.makedirs(saved_path, exist_ok=True)

    # -----------------------------------------------------------------

    qperf_dataset = QPerfDataSet(data_folder='/data/qperf/mat', 
                        max_load=1,
                        T=80, 
                        foot_to_end=True, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[0.4, 0.15],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True])


    ind = np.arange(len(qperf_dataset))
    case_lists = np.random.permutation(ind)

    for t in range(20):
        x, y, params = qperf_dataset[case_lists[t]]
        print(f"x - {x.shape}, y - {y.shape}, params - {params.shape}")
        np.save(os.path.join(saved_path, f"case_{case_lists[t]}_x.npy"), x.astype(np.float32))
        np.save(os.path.join(saved_path, f"case_{case_lists[t]}_y.npy"), y.astype(np.float32))
        np.save(os.path.join(saved_path, f"case_{case_lists[t]}_p.npy"), params.astype(np.float32))