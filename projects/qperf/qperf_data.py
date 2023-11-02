"""
Data utilities for QPerf.
"""

import os
import sys
import scipy
import shutil
import random
import pickle
import torch
from tqdm import tqdm
import time
import numpy as np
from pathlib import Path
from skimage.util import view_as_blocks
from colorama import Fore, Style
import h5py
import glob

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
    def __init__(self, h5_files, keys, 
                cache_folder = None, 
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
            - T : number of time points
            - min_noise_level : minimal noise sigma to add for aif and myo
            - max_noise_level : maximal noise sigma to add for aif and myo
            - matrix_size_adjust_ratio (float list): down/upsample the image, keeping the fov
            - only_white_noise : if True, only add white noise; otherwise, add correlated noise
            - add_noise : whether to add noise for aif or myo
        """
        self.h5_files = h5_files
        if not isinstance(self.h5_files, list):
            self.h5_files = [h5_files]

        self.keys = keys
        assert len(self.h5_files) == len(self.keys)

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

        N = sum([len(x) for x in keys])
        print(f"--> {N} cases found for {len(self.h5_files)} files ...")

        # count items and create search entries
        total_samples = 0
        num_samples = []
        starts = []
        ends = []
        for h in range(len(self.h5_files)):
            aif = np.array(self.h5_files[h][self.keys[h][0]+'/aif'])
            grp_num = aif.shape[0]
            print(f"--> grp_num is {grp_num} for the {h} h5 file ...")

            total_samples += grp_num * len(self.keys[h])
            if len(num_samples) == 0:
                starts.append(0)
            else:
                starts.append(ends[-1])

            num_samples.append(grp_num * len(self.keys[h]))
            ends.append(starts[-1] + num_samples[-1])

        # store idx for h5_files, keys and entry in a key
        self.search_entry = np.zeros((total_samples, 3), dtype=np.int32)
        for h in tqdm(range(len(self.h5_files))):
            aif = np.array(self.h5_files[h][self.keys[h][0]+'/aif'])
            grp_num = aif.shape[0]
            n_ind = np.array(range(grp_num))

            self.search_entry[starts[h]:ends[h], 0] = h
            for k in tqdm(range(len(self.keys[h]))):
                start_ind = starts[h]+grp_num*k
                self.search_entry[start_ind:start_ind+grp_num, 1] = k
                self.search_entry[start_ind:start_ind+grp_num, 2] = n_ind

        # if load_cache and self.load_cached_dataset(self.cache_folder):
        #     print(self.__str__())
        # else:
        #     self.x = []
        #     self.y = []
        #     self.p = []

        #     for h5_file, key in zip(self.h5_files, self.keys):
        #         with tqdm(total=len(key)) as pbar:
        #             for k in key:
        #                 aif = np.array(h5_file[k+"/aif"])
        #                 myo = np.array(h5_file[k+"/myo"])
        #                 params = np.array(h5_file[k+"/params"])

        #                 aif = aif.astype(np.float16)
        #                 myo = myo.astype(np.float16)
        #                 params = params.astype(np.float16)
        #                 t1 = time.time()

        #                 M = aif.shape[0]
        #                 if self.max_samples_per_file > 0 and M > self.max_samples_per_file:
        #                     samples_ind = np.random.default_rng().choice(M, self.max_samples_per_file, replace=False)
        #                     aif = aif[samples_ind, :]
        #                     myo = myo[samples_ind, :]
        #                     params = params[samples_ind, :]
        #                     M = aif.shape[0]
        #                     assert M == self.max_samples_per_file

        #                 self.x.append(aif)
        #                 self.y.append(myo)
        #                 self.p.append(params)

        #                 pbar.update(1)

        #     t0 = time.time()
        #     self.x = np.vstack(self.x)
        #     self.y = np.vstack(self.y)
        #     self.p = np.vstack(self.p)
        #     t1 = time.time()
        #     print(f"{Fore.YELLOW}vstack takes {t1-t0}s ...{Style.RESET_ALL}")

        #     print(f"--> {self.x.shape[0]} samples loaded from {len(self.h5_files)} cases ... ")

            # self.cache_dataset(self.cache_folder)

        self.picked_samples = None
        self.generate_picked_samples()

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

        # x = self.x[i, :]
        # y = self.y[i, :]
        # p = self.p[i, :]

        h = self.search_entry[i, 0]
        k = self.search_entry[i, 1]
        n = self.search_entry[i, 2]

        h5_file = self.h5_files[h]
        key = self.keys[h][k]

        aif = np.array(h5_file[key+"/aif"])
        myo = np.array(h5_file[key+"/myo"])
        params = np.array(h5_file[key+"/params"])

        x = aif[n, :]
        y = myo[n, :]
        p = params[n, :]

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
        if self.max_samples>0 and self.search_entry.shape[0]>self.max_samples:
            self.picked_samples = np.random.default_rng().choice(self.search_entry.shape[0], size=self.max_samples, replace=False)
        else:
            self.picked_samples = np.arange(self.search_entry.shape[0])

    def __len__(self):
        """
        Length of dataset
        """
        if self.max_samples>0 and self.search_entry.shape[0]>self.max_samples:
            return self.max_samples
        else:
            return self.search_entry.shape[0]

    def __getitem__(self, index):
        """
        Gets the item given index
        """
        return self.load_one_sample(self.picked_samples[index])
    
    def __str__(self):
        str = "QPerf, Dataset\n"
        str += "  Number of aif: %d" % self.search_entry.shape[0] + "\n"
        str += "  Number of myo: %d" % self.search_entry.shape[0] + "\n"
        str += "  Number of params: %d" % self.search_entry.shape[0] + "\n"
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
            self.x.astype(np.float16)
            self.y = np.load(os.path.join(cache_data_dir, 'myo.npy'))
            self.y.astype(np.float16)
            self.p = np.load(os.path.join(cache_data_dir, 'params.npy'))
            t1 = time.time()
            print("Load dataset images takes %.2f ..." % (t1-t0))
            
            return True
        else:
            return False

# -------------------------------------------------------------------------------------------------

def load_one_set(train_paths, ratio):

    h5files = []
    train_keys = []

    def process_one_item(file):
        if not os.path.exists(file):
            raise RuntimeError(f"File not found: {file}")

        print(f"reading from file: {file}")
        h5file = h5py.File(file, libver='latest', mode='r')
        keys = list(h5file.keys())

        random.shuffle(keys)

        n = len(keys)

        keys_all = []
        for k in keys:
            case_keys = list(h5file[k].keys())
            for ck in case_keys:
                keys_all.append(k+"/"+ck)

        n = len(keys_all)
        tra = int(ratio*n)
        tra = 1 if tra == 0 else tra

        h5files.append(h5file)
        train_keys.append(keys_all[:tra])

    if isinstance(train_paths, list):
        for file in train_paths:
            process_one_item(file)
    else:
        process_one_item([train_paths])

    return h5files, train_keys

def load_qperf_data(config):
    """
        - data_dir (str): main folder of the data
        - train_files (str list): names of h5files in dataroot for training
        - val_files (str list): names of h5files in dataroot for testing
        - test_files (str list): names of h5files in dataroot for testing
    """

    tra_dir = 'tra_small'
    val_dir = 'val'
    test_dir = 'test'

    ratio = [x/100 for x in config.ratio if x > 1]
    print(f"--> loading data with ratio {ratio} ...")

    only_white_noise = True

    start = time.time()
    data_folder=os.path.join(config.data_dir, tra_dir)
    h5_names = glob.glob(data_folder + "/tra_*.h5")
    h5_files, keys = load_one_set(h5_names, ratio[0])
    train_set = QPerfDataSet(h5_files=h5_files, keys=keys,
                        max_samples=config.max_samples[0],
                        T=config.qperf_T, 
                        foot_to_end=config.foot_to_end, 
                        min_noise_level=config.min_noise_level, 
                        max_noise_level=config.max_noise_level,
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=only_white_noise,
                        add_noise=config.add_noise,
                        cache_folder=os.path.join(config.log_dir, tra_dir))

    print(f"{Fore.RED}----> Info for the training set, {data_folder} ...{Style.RESET_ALL}")
    print(train_set)

    data_folder=os.path.join(config.data_dir, val_dir)
    h5_names = glob.glob(data_folder + "/val_*.h5")
    h5_files, keys = load_one_set(h5_names, ratio[1])
    val_set = QPerfDataSet(h5_files=h5_files, keys=keys,
                        max_samples=config.max_samples[1],
                        T=config.qperf_T, 
                        foot_to_end=config.foot_to_end, 
                        min_noise_level=config.min_noise_level, 
                        max_noise_level=config.max_noise_level,
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=only_white_noise,
                        add_noise=config.add_noise,
                        cache_folder=os.path.join(config.log_dir, val_dir))

    print(f"{Fore.RED}----> Info for the val set, {data_folder} ...{Style.RESET_ALL}")
    print(val_set)

    data_folder=os.path.join(config.data_dir, test_dir)
    h5_names = glob.glob(data_folder + "/test_*.h5")
    h5_files, keys = load_one_set(h5_names, ratio[1])
    test_set = QPerfDataSet(h5_files=h5_files, keys=keys,
                        max_samples=config.max_samples[2],
                        T=config.qperf_T, 
                        foot_to_end=config.foot_to_end, 
                        min_noise_level=config.min_noise_level, 
                        max_noise_level=config.max_noise_level,
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=only_white_noise,
                        add_noise=config.add_noise,
                        cache_folder=os.path.join(config.log_dir, test_dir))

    print(f"{Fore.RED}----> Info for the test set, {data_folder} ...{Style.RESET_ALL}")
    print(test_set)

6    print(f"load_qperf_data took {time.time() - start} seconds ...")

    print(f"--->{Fore.YELLOW}Number of samples for tra/val/test are {len(train_set)}/{len(val_set)}/{len(test_set)}{Style.RESET_ALL}")

    return train_set, val_set, test_set

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
    data_folder='/data/qperf/new_data_3'

    tra_dir = 'tra'
    val_dir = 'val'
    test_dir = 'test'

    only_white_noise = True
    filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0]

    # ----------------------------------------------------------

    tra_names = glob.glob(data_folder + "/tra_*.h5")
    tra_h5_files, tra_keys = load_one_set(tra_names[0:3], 1)

    #tra_h5_files, tra_keys = load_one_set(['/data/qperf/new_data_3/tra_0.h5'], 1)

    qperf_dataset = QPerfDataSet(tra_h5_files, tra_keys, 
                        max_samples_per_file=-1,
                        T=80, 
                        foot_to_end=foot_to_end, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[0.4, 0.15],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True],
                        cache_folder=os.path.join(data_folder, "cache/"+tra_dir),
                        load_cache=load_cache)

    ind = np.arange(len(qperf_dataset))
    case_lists = np.random.permutation(ind)

    for ind, t in enumerate(case_lists):
        x, y, params = qperf_dataset[t]
        if ind % 10000 == 0:
            print(f"{ind} out of {case_lists.shape[0]}, x - {x.shape}, y - {y.shape}, params - {params.shape}")

    # ----------------------------------------------------------

    val_h5_files, val_keys = load_one_set([data_folder+"/val_0.h5"], 1)
    qperf_dataset = QPerfDataSet(val_h5_files, val_keys, 
                        max_samples_per_file=-1,
                        T=80, 
                        foot_to_end=foot_to_end, 
                        min_noise_level=[0.01, 0.01], 
                        max_noise_level=[0.4, 0.15],
                        filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
                        only_white_noise=False,
                        add_noise=[True, True],
                        cache_folder=os.path.join(data_folder, "cache/"+val_dir),
                        load_cache=load_cache)

    ind = np.arange(len(qperf_dataset))
    case_lists = np.random.permutation(ind)

    t0 = time.time()
    for ind, t in enumerate(case_lists):
        x, y, params = qperf_dataset[t]
        if ind > 2000:
            break
        if ind % 10000 == 0:
            print(f"{ind} out of {case_lists.shape[0]}, x - {x.shape}, y - {y.shape}, params - {params.shape}")

    t1 = time.time()
    print(f"load data - {t1-t0}s ...")

    test_h5_files, test_keys = load_one_set([data_folder+"/test_0.h5"], 1)

    # qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'tra_small'), 
    #                     max_load=-1,
    #                     T=80, 
    #                     foot_to_end=foot_to_end, 
    #                     min_noise_level=[0.01, 0.01], 
    #                     max_noise_level=[0.4, 0.15],
    #                     filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
    #                     only_white_noise=False,
    #                     add_noise=[True, True],
    #                     cache_folder=os.path.join(data_folder, 'cache/tra_small'),
    #                     load_cache=load_cache)

    # qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'val_small'), 
    #                     max_load=-1,
    #                     T=80, 
    #                     foot_to_end=foot_to_end, 
    #                     min_noise_level=[0.01, 0.01], 
    #                     max_noise_level=[0.4, 0.15],
    #                     filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
    #                     only_white_noise=False,
    #                     add_noise=[True, True],
    #                     cache_folder=os.path.join(data_folder, 'cache/val_small'),
    #                     load_cache=load_cache)

    # qperf_dataset = QPerfDataSet(data_folder=os.path.join(data_folder, 'test_small'), 
    #                     max_load=-1,
    #                     T=80, 
    #                     foot_to_end=foot_to_end, 
    #                     min_noise_level=[0.01, 0.01], 
    #                     max_noise_level=[0.4, 0.15],
    #                     filter_sigma=[0.1, 0.25, 0.5, 0.8, 1.0],
    #                     only_white_noise=False,
    #                     add_noise=[True, True],
    #                     cache_folder=os.path.join(data_folder, 'cache/test_small'),
    #                     load_cache=load_cache)

    # max_samples_per_file = int(1000000/10)

    max_samples_per_file = -1

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