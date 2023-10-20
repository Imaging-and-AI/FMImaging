"""
Data utilities for QPerf.
"""

import os
import sys
import scipy
import shutil
import pickle
import torch
from tqdm import tqdm
import time
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
def normalize_data(x, y, p):
    # normalize data
    x[:, 0] -= 2.0 # aif
    x[:, 1] -= 0.5 # myo
    y -= 0.5

    # Fp, Vp, Visf, PS, delay
    # p[0] -= 1.25
    # p[1] -= 0.05
    # p[2] -= 0.15
    # p[3] -= 1.0
    # p[4] -= 2.0

    return x, y, p

def denormalize_data(x, y, p):

    x[:, 0] += 2.0 # aif
    x[:, 1] += 0.5 # myo
    y += 0.5

    # Fp, Vp, Visf, PS, delay
    # p[0] += 1.25
    # p[1] += 0.05
    # p[2] += 0.15
    # p[3] += 1.0
    # p[4] += 2.0

    return x, y, p

class QPerfDataSet(torch.utils.data.Dataset):
    """
    Every sample includes aif ([B, T, D]), myo ([B, T, D]) and parameters (Fp, Vp, PS, Visf, delay, foot, peak, valley)
    """
    def __init__(self, data_folder, 
                        cache_folder=None,
                        max_load=-1,
                        max_samples=-1,
                        max_samples_per_file=-1,
                        T=80, 
                        foot_to_end=True, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[1.0, 0.25],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True],
                        load_cache=True):
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
        self.max_samples = max_samples
        self.max_samples_per_file = max_samples_per_file
        self.T = T
        self.foot_to_end = foot_to_end
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.filter_sigma = filter_sigma
        self.only_white_noise = only_white_noise
        self.add_noise = add_noise

        if cache_folder is None:
            self.cache_folder = data_folder
        else:
            self.cache_folder = cache_folder

        all_data_files = []

        site_dirs = os.listdir(self.data_folder)
        for a_case in site_dirs:
            if a_case.endswith('mat'):
                all_data_files.append(os.path.join(self.data_folder, a_case))

        N = len(all_data_files)
        print(f"--> {N} cases found for data folder : {self.data_folder}")

        if load_cache and self.load_cached_dataset(self.cache_folder):
            print(self.__str__())
        else:
            self.x = []
            self.y = []
            self.p = []

            cases_loaded = 0
            for ind in range(N):
                if self.max_load> 0 and cases_loaded > self.max_load:
                    break

                a_case = all_data_files[ind]
                if os.path.isfile(a_case):

                    a_case_fname, mat_ext = os.path.splitext(a_case)

                    aif_fname = f"{a_case_fname}_aif.npy"
                    myo_fname = f"{a_case_fname}_myo.npy"
                    params_fname = f"{a_case_fname}_params.npy"

                    if not os.path.exists(aif_fname):
                        continue

                    print(f"{Fore.RED}--> case {ind} loaded from {N} cases - {a_case} ... {Style.RESET_ALL}")

                    t0 = time.time()
                    aif = np.load(aif_fname)
                    myo = np.load(myo_fname)
                    params = np.load(params_fname)

                    aif = aif.astype(np.float16)
                    myo = myo.astype(np.float16)
                    params = params.astype(np.float16)
                    t1 = time.time()
                    print(f"{Fore.YELLOW}Load {a_case_fname} takes {t1-t0}s ...{Style.RESET_ALL}")

                    # t0 = time.time()
                    # mat = scipy.io.loadmat(a_case)
                    # t1 = time.time()
                    # print(f"Load mat file {a_case} takes {(t1-t0):.2f}s ...")

                    # N = mat['out'].shape[1]
                    M = aif.shape[0]
                    if self.max_samples_per_file > 0 and M > self.max_samples_per_file:
                        samples_ind = np.random.default_rng().choice(M, self.max_samples_per_file, replace=False)
                        aif = aif[samples_ind, :]
                        myo = myo[samples_ind, :]
                        params = params[samples_ind, :]
                        M = aif.shape[0]
                        assert M == self.max_samples_per_file

                    t0 = time.time()
                    self.x.append(aif)
                    self.y.append(myo)
                    self.p.append(params)

                    # if self.x is None:
                    #     self.x = aif
                    #     self.y = myo
                    #     self.p = params
                    # else:
                    #     self.x = np.vstack((self.x, aif))
                    #     self.y = np.vstack((self.y, myo))
                    #     self.p = np.vstack((self.p, params))

                    t1 = time.time()
                    print(f"{Fore.YELLOW}Concatenate {a_case_fname} takes {t1-t0}s ...{Style.RESET_ALL}")

                    cases_loaded += 1
                    print(f"--> {aif.shape[0]} samples loaded from {cases_loaded} cases - {a_case} ... ")

            self.x = np.vstack(self.x)
            self.y = np.vstack(self.y)
            self.p = np.vstack(self.p)

            print(f"--> {self.x.shape[0]} samples loaded from {cases_loaded} cases ... ")

            self.cache_dataset(self.cache_folder)

        # for k in range(self.x.shape[0]):
        #     a_x = self.x[k, :]
        #     a_y = self.y[k, :]
        #     a_p = self.p[k,:]
        #     assert np.linalg.norm(a_x) > 1e-3
        #     assert np.linalg.norm(a_y) > 1e-3
        #     assert np.linalg.norm(a_p) > 1e-3

        self.picked_samples = None
        self.generate_picked_samples()

    # def pre_load_one_sample(self, i):
    #     """
    #     Loads one sample from the loaded data images
    #     @args:
    #         - i (int): index of the file to load
    #     @rets:
    #         - x : [T, 2] for aif and myo
    #         - y : [T] for clean myo
    #         - p : [9] for parameters (Fp, Vp, PS, Visf, delay, foot, peak, valley, N)
    #     """

    #     T = self.T

    #     x = self.aif[i]
    #     y = self.myo[i]
    #     params = self.params[i]

    #     N = x.shape[0]
    #     foot = int(params[5])

    #     if self.foot_to_end and foot < N/2 and foot > 3:
    #         x = x[foot:]
    #         y = y[foot:]
    #         N = x.shape[0]

    #     if N > self.T:
    #         x = x[:self.T]
    #         y = y[:self.T]
    #         N = x.shape[0]
    #     elif N < self.T:
    #         x = np.append(x, x[N-1] * np.ones((self.T-N)), axis=0)
    #         y = np.append(y, y[N-1] * np.ones(self.T-N), axis=0)

    #     return x, y, np.append(params, [N], axis=0)

    def load_one_sample(self, i):
        """
        Loads one sample from the loaded data images
        @args:
            - i (int): index of the file to load
        @rets:
            - x : [T, 2] for aif and myo
            - y : [T] for clean myo
            - p : [9] for parameters (Fp, Vp, PS, Visf, delay, foot, peak, valley, N)
        """

        T = self.T

        x = self.x[i, :]
        y = self.y[i, :]
        p = self.p[i, :]

        N = x.shape[0]

        if self.add_noise[0]:
            sigma = np.random.uniform(self.min_noise_level[0], self.max_noise_level[0])
            nn = np.random.standard_normal(size=N) * sigma

            if not self.only_white_noise:
                fs = self.filter_sigma[np.random.randint(0, len(self.filter_sigma))]
                nn = scipy.ndimage.gaussian_filter1d(nn, fs, axis=0)

            x += nn

        if self.add_noise[1]:
            # add noise to myo
            sigma = np.random.uniform(self.min_noise_level[1], self.max_noise_level[1])
            nn = np.random.standard_normal(size=N) * sigma

            if not self.only_white_noise:
                fs = self.filter_sigma[np.random.randint(0, len(self.filter_sigma))]
                nn = scipy.ndimage.gaussian_filter1d(nn, fs, axis=0)

            myo = y + nn
        else:
            myo = np.copy(y)

        x = np.concatenate((x[:, np.newaxis], myo[:, np.newaxis]), axis=1)

        foot = int(p[5]) - 5

        if self.foot_to_end and foot < N/2 and foot > 3:
            x = x[foot:, :]
            y = y[foot:]
            N = x.shape[0]
            p[5] = 0
            p[6] -= foot
            p[7] -= foot

        if N > self.T:
            x = x[:self.T, :]
            y = y[:self.T]
            N = x.shape[0]
        elif N < self.T:
            x = np.append(x, x[N-1,:] * np.ones((self.T-N, 2)), axis=0)
            y = np.append(y, y[N-1] * np.ones(self.T-N), axis=0)

        p[8] = N

        # normalize data
        normalize_data(x, y, p)

        return x, y, p

    def generate_picked_samples(self):
        if self.max_samples>0 and self.x.shape[0]>self.max_samples:
            self.picked_samples = np.random.default_rng().choice(self.x.shape[0], size=self.max_samples, replace=False)
        else:
            self.picked_samples = np.arange(self.x.shape[0])

    def __len__(self):
        """
        Length of dataset
        """
        if self.max_samples>0 and self.x.shape[0]>self.max_samples:
            return self.max_samples
        else:
            return self.x.shape[0]

    def __getitem__(self, index):
        """
        Gets the item given index
        """
        return self.load_one_sample(self.picked_samples[index])
    
    def __str__(self):
        str = "QPerf, Dataset\n"
        str += "  Number of aif: %d" % self.x.shape[0] + "\n"
        str += "  Number of myo: %d" % self.y.shape[0] + "\n"
        str += "  Number of params: %d" % self.p.shape[0] + "\n"
        str += "  max_samples: %d" % self.max_samples + "\n"
        str += f"  foot_to_end: {self.foot_to_end}" + "\n"
        str += f"  add_noise: {self.add_noise}" + "\n"
        str += f"  T: {self.T}" + "\n"

        return str
        
    # --------------------------------------------------------

    def cache_dataset(self, cache_data_dir):
        """
        Cache the loaded dataset to speed up process.
        """
        print(cache_data_dir)

        t0 = time.time()
        os.makedirs(cache_data_dir, exist_ok=True)
        np.save(os.path.join(cache_data_dir, 'aif.npy'), self.x)
        np.save(os.path.join(cache_data_dir, 'myo.npy'), self.y)
        np.save(os.path.join(cache_data_dir, 'params.npy'), self.p)
        t1 = time.time()
        print("Saving dataset images takes %.2f ..." % (t1-t0))

    # --------------------------------------------------------
    def load_cached_dataset(self, cache_data_dir):
        """
        Load cached dataset to speed up.
        """

        fname = os.path.join(cache_data_dir, 'aif.npy')
        if os.path.exists(fname):
            print(f"--> loading from the cache {fname} ...")
            t0 = time.time()
            self.x = np.load(os.path.join(cache_data_dir, 'aif.npy'))
            self.y = np.load(os.path.join(cache_data_dir, 'myo.npy'))
            self.p = np.load(os.path.join(cache_data_dir, 'params.npy'))
            t1 = time.time()
            print("Load dataset images takes %.2f ..." % (t1-t0))
            
            return True
        else:
            return False

# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    saved_path = "/export/Lab-Xue/projects/qperf/results/loader_test"
    if os.path.exists(saved_path):
        shutil.rmtree(saved_path)
    os.makedirs(saved_path, exist_ok=True)

    # -----------------------------------------------------------------

    load_cache = False

    foot_to_end = True

    data_folder='/data/qperf/mat'
    data_folder='/data/qperf/new_data'

    qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'tra_small'), 
                        max_load=-1,
                        T=80, 
                        foot_to_end=foot_to_end, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[0.4, 0.15],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True],
                        cache_folder=os.path.join(data_folder, 'cache/tra_small'),
                        load_cache=load_cache)

    qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'val_small'), 
                        max_load=-1,
                        T=80, 
                        foot_to_end=foot_to_end, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[0.4, 0.15],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True],
                        cache_folder=os.path.join(data_folder, 'cache/val_small'),
                        load_cache=load_cache)

    qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'test_small'), 
                        max_load=-1,
                        T=80, 
                        foot_to_end=foot_to_end, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[0.4, 0.15],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True],
                        cache_folder=os.path.join(data_folder, 'cache/test_small'),
                        load_cache=load_cache)

    max_samples_per_file = int(1000000/10)

    qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'val'), 
                        max_load=-1,
                        max_samples_per_file=-1,
                        T=80, 
                        foot_to_end=foot_to_end, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[0.4, 0.15],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True],
                        cache_folder=os.path.join(data_folder, 'cache/val'),
                        load_cache=load_cache)

    qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'test'), 
                        max_load=-1,
                        max_samples_per_file=-1,
                        T=80, 
                        foot_to_end=foot_to_end, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[0.4, 0.15],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True],
                        cache_folder=os.path.join(data_folder, 'cache/test'),
                        load_cache=load_cache)

    qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'tra'), 
                            max_load=-1,
                            max_samples_per_file=max_samples_per_file,
                            T=80, 
                            foot_to_end=foot_to_end, 
                            min_noise_level=[0.01, 0.01], 
                            max_noise_level=[0.4, 0.15],
                            filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                            only_white_noise=False,
                            add_noise=[True, True],
                            cache_folder=os.path.join(data_folder, 'cache/tra'),
                            load_cache=load_cache)

    ind = np.arange(len(qperf_dataset))
    case_lists = np.random.permutation(ind)

    for ind, t in enumerate(case_lists):
        x, y, params = qperf_dataset[t]
        if ind % 10000 == 0:
            print(f"{ind} out of {case_lists.shape[0]}, x - {x.shape}, y - {y.shape}, params - {params.shape}")
        # np.save(os.path.join(saved_path, f"case_{case_lists[t]}_x.npy"), x.astype(np.float32))
        # np.save(os.path.join(saved_path, f"case_{case_lists[t]}_y.npy"), y.astype(np.float32))
        # np.save(os.path.join(saved_path, f"case_{case_lists[t]}_p.npy"), params.astype(np.float32))