"""Membrane potential dataset

    Every case have its own folder, with following data variables:
    
               A: [725×2801 double] # waveform, feature by time
              MP: [2801×1 double]   # membrane potential
    i_valid_spec: [2801×1 logical]  # valid time points if 1; invalid time points if 0, due to chemical injection, indicating the valid waveforms
      idx_select: [2801×1 logical]  # selected time points if 1, indicating the valid MP measurements
             wav: [725×1 double]    # wave length in A
             
    The feature dimension is 725. The time dimension is 2801. Different data may have different time points.
        
"""

import sys
import numpy as np
from tqdm import tqdm 
import time

import scipy.io as sio
from scipy import interpolate

import matplotlib.pyplot as plt

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *

class MembranePotentialDataset(Dataset):
    """Dataset for membrane potential prediction."""

    def __init__(self, data_dir, chunk_length=1024, num_starts=5, transform=None, debug_mode=False):
        """Initialize the dataset

        Args:
            data_dir : directory to store the MP data; each subfolder is an experiment
            chunk_length : the number of time points in each sample
            num_starts : for every signal, the starting point for sampling is randomly picked for num_starts times

        """

        self.chunk_length = chunk_length
        self.num_starts = num_starts
        self.A = []
        self.MP = []
        self.i_valid_spec = []
        self.idx_select = []
        self.wav = []
        self.names = []

        self.debug_mode = debug_mode

        t0 = time.time()
        
        sub_dirs = os.listdir(data_dir)
        
        num_samples = len(sub_dirs)*num_starts

        with tqdm(total=num_samples) as tq:
            for case_dir in sub_dirs:
                data_file_name = os.path.join(data_dir, case_dir, 'selected_data.mat')        
                data = sio.loadmat(data_file_name)
                data['case'] = case_dir
                self.load_one_experiment(data)
                tq.update(num_starts)

        t1 = time.time()
        tq.set_postfix_str(f"Data loading - {t1-t0} seconds")
        
        self.transform = transform

    def load_one_experiment(self, data):
        """Sample one signal

        Args:   
            data (matlab mat) : loaded contents from the mat file.
        """
        
        A = np.copy(data['A'])
        MP = data['MP'].squeeze()
        i_valid_spec = data['i_valid_spec'].squeeze()
        idx_select = data['idx_select'].squeeze()
        wav = data['wav'].squeeze()
        
        D, T = A.shape
        
        if(T<self.chunk_length):
            logger.info(f"T<self.chunk_length, {T}, {self.chunk_length}")
            self.chunk_length = T

        if(T>MP.shape[0]):
            logger.info(f"T>MP.shape[0], {T}, {MP.shape[0]}")
            return
        
        A = np.transpose(A, (1, 0))        
                        
        # use interpolation to fix i_valid_spec==0
        ind = np.arange(T)
        x = ind[i_valid_spec==1]
        y = A[i_valid_spec==1, :]
        f = interpolate.interp1d(x, y, axis=0)
        y2 = f(ind[i_valid_spec==0])
        A[i_valid_spec==0, :] = y2
        A -= np.mean(A, axis=0, keepdims=True)
        
        if self.debug_mode:
            import matplotlib.pyplot as plt
            f = plt.figure()
            plt.imshow(A.T)
            plt.show()
        
        starting_locs = np.random.randint(0, T, size=self.num_starts)
        
        for n in range(self.num_starts):
            s = starting_locs[n]
            inds = np.arange(s, T, self.chunk_length)
            for ind in inds:
                if(ind+self.chunk_length>=T):
                    ind = T - self.chunk_length - 1
                    
                if(ind<0):
                    ind = 0
                    
                a_A = A[ind:ind+self.chunk_length, :]
                a_MP = MP[ind:ind+self.chunk_length]
                a_i_valid_spec = i_valid_spec[ind:ind+self.chunk_length]
                a_idx_select = idx_select[ind:ind+self.chunk_length]
                a_wav = wav
                name = f"{data['case'].rjust(8, '0')}_{ind:4d}_{n:4d}"
                
                if a_A.shape[0] != self.chunk_length:
                    logger.info(f"sample {name} has {a_A.shape} size ... ignore ...")
                    continue
        
                self.A.append(a_A)
                self.MP.append(a_MP)
                self.i_valid_spec.append(a_i_valid_spec)
                self.idx_select.append(a_idx_select)
                self.wav.append(a_wav)
                self.names.append(name)

    def __len__(self):
        """Get the number of samples in this dataset.

        Returns:
            number of samples
        """
        return len(self.A)

    def __getitem__(self, idx):
        """Get the idx sample

        Args:
            idx (int): the index of sample to get; first sample has idx being 0

        Returns:
            sample : a tuple (ecg_signal, ecg_trigger)
            ecg_signal : [chunk_length, C]
            ecg_trigger : [chunk_length]
        """
        # *** START CODE HERE ***
        if idx >= len(self.A):
            raise "invalid index"

        A = self.A[idx]
        MP = self.MP[idx]
        i_valid_spec = self.i_valid_spec[idx]
        idx_select = self.idx_select[idx]
        wav = self.wav[idx]
        name = self.names[idx]
        
        if self.transform:
            return self.transform((A, MP, i_valid_spec, idx_select, wav, name))

        return (A, MP, i_valid_spec, idx_select, name)
        
    def __str__(self):
        str = "Membrane potential Dataset\n"
        str += "  Number of A: %d" % len(self.A) + "\n"
        str += "  Number of MP: %d" % len(self.MP) + "\n"
        str += "  transform : %s" % (self.transform) + "\n"
        str += "  A shape: %d %d" % self.A[0].shape

        return str

# --------------------------------------------------------

def set_up_dataset(train_dir, test_dir, batch_size=64, num_starts=20, chunk_length=512, val_frac=0.1):
    """Set up the ecg dataset and loader

    Args:
        train_dir (str): data directory for training
        test_dir (str): data directory for testing
        batch_size (int): batch size
        num_starts (int, optional): number of staring locations to sample waveform
        chunk_length (int, optional): chunk size for every sample
        val_frac (float, optional): fraction of validation signal
    Returns:
        train_set, test_set (MP dataset): dataset objects for train and test
        loader_for_train, loader_for_val, loader_for_test : data loader for train, validation and test
    """
    # add some data transformation
    transform = None

    # set up the training set, use the transform
    train_set = MembranePotentialDataset(data_dir=train_dir, chunk_length=chunk_length, num_starts=num_starts, transform=transform, debug_mode=False)
    print(train_set)
    # no need to augment the test data
    if test_dir is not None:
        test_set = MembranePotentialDataset(data_dir=test_dir, chunk_length=chunk_length, num_starts=3, transform=None, debug_mode=False)
        print(test_set)
    else:
        test_set = None
    
    # create loader for train, val, and test
    dataset_size = len(train_set)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    val_split_index = int(np.floor(val_frac * dataset_size))
    
    train_idx = dataset_indices[val_split_index:] 
    if val_split_index > 0:
        val_idx = dataset_indices[:val_split_index]
    else:
        val_idx = [0]
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    loader_for_train = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=train_sampler, pin_memory=True)
    loader_for_val = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=val_sampler, pin_memory=True)
    if test_set is not None:
        loader_for_test = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        loader_for_test = None

    return train_set, test_set, loader_for_train, loader_for_val, loader_for_test

# --------------------------------------------------------

def plot_mp_prediction(A, MP, i_valid_spec, idx_select, name, y_hat):
    """plot prediction results
    """

    B, T, D, = A.shape
    x = np.arange(T)

    f = plt.figure(figsize=[16, 8])
    for ind in range(2*(B//2)):
        plt.subplot(2, B//2, ind+1)
        plt.plot(x, MP[ind], 'b.')
        plt.plot(x, MP[ind], 'b-')
        plt.plot(x[idx_select[ind]==1], MP[ind][idx_select[ind]==1], 'r.')
        plt.plot(x[i_valid_spec[ind]==0], MP[ind][i_valid_spec[ind]==0], 'k.')
        plt.plot(x, y_hat[ind], 'g--')
        plt.plot(x[idx_select[ind]==1], y_hat[ind][idx_select[ind]==1], 'g.')
        plt.xlabel('time')
        plt.ylabel('MP')
        plt.title(name[ind])
    
    return f
    
# --------------------------------------------------------
if __name__ == "__main__":
    
    train_dir = os.path.join("J:\MembranePotential\experiments")
    test_dir = os.path.join("J:\MembranePotential\experiments")

    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = set_up_dataset(train_dir, test_dir, batch_size=2, num_starts=10, chunk_length=4000, val_frac=0.1)

    # directly get one sample
    A, MP, i_valid_spec, idx_select, name = train_set[0]
    print("Get one sample ", A.shape, MP.shape, i_valid_spec.shape, idx_select.shape, name)

    # plot a batch
    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.imshow(A)
    plt.show()
    
    T, D = A.shape
    x = np.arange(T)
    f2 = plt.figure()
    for k in range(8):
        A, MP, i_valid_spec, idx_select, wav, name = train_set[k]
        
        plt.subplot(2, 4, k+1)
        plt.plot(x, MP, 'b.')
        plt.plot(x, MP, 'b-')
        plt.plot(x[idx_select==1], MP[idx_select==1], 'r.')
        plt.plot(x[i_valid_spec==0], MP[i_valid_spec==0], 'k.')
        plt.xlabel('time')
        plt.ylabel('MP')
    
    plt.show()
    plt.close(f2)
    