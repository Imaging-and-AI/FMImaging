"""
Data utilities for MRI data.
Provides the torch dataset class for traind and test and functions to load from multiple h5files
"""

import os
import sys
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

from noise_augmentation import *
from mri_data_utils import *

# -------------------------------------------------------------------------------------------------
# train dataset class

class MRIDenoisingDatasetTrain(torch.utils.data.Dataset):
    """
    Dataset for MRI denoising.
    The extracted patch maintains the strict temporal consistency
    This dataset is for 2D+T training, where the temporal redundancy is strong
    """
    def __init__(self, h5file, keys, data_type, max_load=-1,
                    time_cutout=30, cutout_shape=[64, 64], 
                    ignore_gmap=False,
                    use_complex=True, 
                    min_noise_level=1.0, max_noise_level=6.0,
                    matrix_size_adjust_ratio=[0.5, 0.75, 1.0, 1.25, 1.5],
                    kspace_filter_sigma=[0.8, 1.0, 1.5, 2.0, 2.25],
                    kspace_filter_T_sigma=[0.25, 0.5, 0.65, 0.85, 1.0, 1.5, 2.0, 2.25],
                    pf_filter_ratio=[1.0, 0.875, 0.75, 0.625],
                    phase_resolution_ratio=[1.0, 0.75, 0.65, 0.55],
                    readout_resolution_ratio=[1.0, 0.75, 0.65, 0.55],
                    cutout_jitter=[-1, 0.5, 0.75, 1.0],
                    snr_perturb_prob=0.0, 
                    snr_perturb=0.15,
                    only_white_noise=False,
                    cutout_shuffle_time=True,
                    num_patches_cutout=8,
                    patches_shuffle=False,
                    with_data_degrading=False,
                    add_noise=True,
                    load_2x_resolution=False):
        """
        Initilize the denoising dataset
        Loads and store all images and gmaps
        h5files should have the following strucutre
        file --> <key> --> "image"+"gmap"
        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for each h5file
            - data_type ("2d"|"2dt"|"3d"): types of mri data
            - max_load (int): number of loaded samples when instantiating the dataset
            - time_cutout (int): cutout size in time dimension
            - cutout_shape (int list): 2 values for patch cutout shape
            - ignore_gmap (bool): whether to ignore gmap in training
            - use_complex (bool): whether to return complex image
            - min_noise_level (float): minimal noise sigma to add
            - max_noise_level (float): maximal noise sigma to add
            - matrix_size_adjust_ratio (float list): down/upsample the image, keeping the fov
            - kspace_filter_sigma (float list): kspace filter sigma
            - pf_filter_ratio (float list): partial fourier filter
            - phase_resolution_ratio (float list): phase resolution ratio
            - readout_resolution_ratio (float list): readout resolution ratio
            - cutout_jitter (float list): for 3D, cutout jitter range along time dimenstion
            - cutout_shuffle_time (bool): for 3D, shuffle time dimension to break redundancy
            - num_patches_cutout (int): for 2D, number of patches per frame
            - patches_shuffle (bool) for 2D, shuffle patches 
            - with_data_degrading (bool): if True, train with resolution reduction, time filtering etc. to degrade data a bit
            - add_noise (bool): if False, not add noise - for testing the resolution improvement
            - load_2x_resolution (bool): whether to load 2x resolution data
        """
        assert data_type.lower()=="2d" or data_type.lower()=="2dt" or data_type.lower()=="3d",\
            f"Data type not implemented: {data_type}"
        self.data_type = data_type

        self.h5file = h5file
        self.keys = keys
        self.max_load = max_load
        
        self.time_cutout = time_cutout
        if self.data_type=="2d": self.time_cutout = 1
        self.cutout_shape = cutout_shape

        self.ignore_gmap = ignore_gmap
        self.use_complex = use_complex

        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.matrix_size_adjust_ratio = matrix_size_adjust_ratio
        self.kspace_filter_sigma = kspace_filter_sigma
        self.kspace_filter_T_sigma = kspace_filter_T_sigma
        self.pf_filter_ratio = pf_filter_ratio
        self.phase_resolution_ratio = phase_resolution_ratio
        self.readout_resolution_ratio = readout_resolution_ratio

        self.snr_perturb_prob = snr_perturb_prob
        self.snr_perturb = snr_perturb

        self.only_white_noise = only_white_noise

        self.cutout_jitter = cutout_jitter
        self.cutout_shuffle_time = cutout_shuffle_time
        self.num_patches_cutout = num_patches_cutout
        self.patches_shuffle = patches_shuffle

        self.with_data_degrading = with_data_degrading
        
        self.images = load_images_from_h5file(h5file, keys, max_load=self.max_load)

        self.rng = np.random.Generator(np.random.PCG64(85442365))

        self.add_noise = add_noise

        self.load_2x_resolution = load_2x_resolution

    def load_one_sample(self, i):
        """
        Loads one sample from the saved images
        @args:
            - i (int): index of the file to load
        @rets:
            - noisy_im (5D torch.Tensor): noisy data, in the shape of [2, RO, E1] for image and gmap
                if it is complex, the shape is [3, RO, E1] for real, imag and gmap
            - clean_im (5D torch.Tensor) : clean data, [1, RO, E1] for magnitude and [2, RO, E1] for complex
            - gmap_median (0D torch.Tensor): median value for the gmap patches
            - noise_sigma (0D torch.Tensor): noise sigma added to the image patch
        """
        # get the image
        data = self.images[i][0]
        gmaps = self.images[i][1]
        if self.load_2x_resolution: data_2x = self.images[i][2]

        if not isinstance(data, np.ndarray):
            ind = self.images[i][3]
            key_image = self.images[i][0]
            key_gmap = self.images[i][1]
            key_image_2x = self.images[i][2]
            data = np.array(self.h5file[ind][key_image])
            gmaps = np.array(self.h5file[ind][key_gmap])
            if self.load_2x_resolution: data_2x = np.array(self.h5file[ind][key_image_2x])

        if not self.load_2x_resolution:
            data_2x = None

        if self.data_type != '3d':
            gmap = self.load_gmap(gmaps, i, random_factor=-1)
        else:
            gmap = gmaps

            if data.shape[0] < self.time_cutout:
                data_t = np.zeros((self.time_cutout, data.shape[1], data.shape[2]))
                data_t[0:data.shape[0]] = data
                data_t[data.shape[0]:] = data[-1]
                data = data_t

                gmap_t = np.zeros((self.time_cutout, data.shape[1], data.shape[2]))
                gmap_t[0:gmap.shape[0]] = gmap
                gmap_t[gmap.shape[0]:] = gmap[-1]
                gmap = gmap_t

        if self.ignore_gmap:
            gmap.fill(1.0)

        if data.ndim == 2: data = data[np.newaxis,:,:]

        # if data.shape[1] != gmap.shape[0] and data.shape[0] == gmap.shape[0]:
        #     data = np.transpose(data, (2, 1, 0))

        data = data.astype(np.complex64)
        gmap = gmap.astype(np.float32)
        if self.load_2x_resolution: 
            data_2x = data_2x.astype(np.complex64)

        # pad symmetrically if not enough images in the time dimension
        if data.shape[0] < self.time_cutout:
            data = np.pad(data, ((0,self.time_cutout - data.shape[0]),(0,0),(0,0)), 'symmetric')
            if self.load_2x_resolution: data_2x = np.pad(data_2x, ((0,self.time_cutout - data_2x.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        data, gmap, data_2x = self.random_flip(data, gmap, data_2x)

        if self.data_type != '3d':
            assert data.shape[1] == gmap.shape[0] and data.shape[2] == gmap.shape[1]
        else:
            assert data.shape[1] == gmap.shape[1] and data.shape[2] == gmap.shape[2]
            if data.shape[0] > gmap.shape[0]:
                data = data[:gmap.shape[0]]
            if data.shape[0] < gmap.shape[0]:
                gmap = gmap[:data.shape[0]]

            assert data.shape[0] == gmap.shape[0]

        # random increase matrix size or reduce matrix size
        if(not self.only_white_noise) and (np.random.random()<0.5):
            matrix_size_adjust_ratio = self.matrix_size_adjust_ratio[np.random.randint(0, len(self.matrix_size_adjust_ratio))]
            data_adjusted = np.array([adjust_matrix_size(img, matrix_size_adjust_ratio) for img in data])
            if self.data_type == '3d':
                gmap_adjusted = np.zeros((gmap.shape[0], data_adjusted.shape[2], data_adjusted.shape[1]))
                for k in range(gmap.shape[0]):
                    gmap_adjusted[k] = cv2.resize(gmap[k], dsize=(data_adjusted.shape[2], data_adjusted.shape[1]), interpolation=cv2.INTER_LINEAR)
                assert data_adjusted.shape[1] == gmap_adjusted.shape[1] and data_adjusted.shape[2] == gmap_adjusted.shape[2]
            else:
                gmap_adjusted = cv2.resize(gmap, dsize=(data_adjusted.shape[2], data_adjusted.shape[1]), interpolation=cv2.INTER_LINEAR)
                assert data_adjusted.shape[1] == gmap_adjusted.shape[0] and data_adjusted.shape[2] == gmap_adjusted.shape[1]

            data = data_adjusted
            gmap = gmap_adjusted

            if self.load_2x_resolution: 
                data_2x_adjusted = np.array([adjust_matrix_size(img, matrix_size_adjust_ratio) for img in data_2x])
                data_2x = data_2x_adjusted

        if data.shape[1] < self.cutout_shape[0]:
            data = np.pad(data, ((0, 0), (0,self.cutout_shape[0] - data.shape[1]),(0,0)), 'symmetric')
            if self.data_type == '3d':
                gmap = np.pad(gmap, ((0,0), (0,self.cutout_shape[0] - gmap.shape[0]),(0,0)), 'symmetric')
            else:
                gmap = np.pad(gmap, ((0,self.cutout_shape[0] - gmap.shape[0]),(0,0)), 'symmetric')
            if self.load_2x_resolution: data_2x = np.pad(data_2x, ((0, 0), (0, 2*(2*self.cutout_shape[0] - data_2x.shape[1])),(0,0)), 'symmetric')

        if data.shape[2] < self.cutout_shape[1]:
            data = np.pad(data, ((0,0), (0,0), (0,self.cutout_shape[1] - data.shape[2])), 'symmetric')
            if self.data_type == '3d':
                gmap = np.pad(gmap, ((0,0), (0,0), (0,self.cutout_shape[1] - gmap.shape[1])), 'symmetric')
            else:
                gmap = np.pad(gmap, ((0,0), (0,self.cutout_shape[1] - gmap.shape[1])), 'symmetric')
            if self.load_2x_resolution: data_2x = np.pad(data_2x, ((0,0), (0,0), (0, 2*(2*self.cutout_shape[1] - data_2x.shape[2]))), 'symmetric')

        T, RO, E1 = data.shape

        if(self.with_data_degrading):
            ratio_RO = self.readout_resolution_ratio[self.rng.integers(0, len(self.readout_resolution_ratio))]
            ratio_E1 = self.phase_resolution_ratio[self.rng.integers(0, len(self.phase_resolution_ratio))]

            data_used = np.transpose(data, (1, 2, 0))

            data_reduced_resolution, fRO, fE1 = apply_resolution_reduction_2D(im=data_used, 
                                                                            ratio_RO=ratio_RO, 
                                                                            ratio_E1=ratio_E1, 
                                                                            snr_scaling=False, 
                                                                            norm='backward')

            ro_filter_sigma = self.kspace_filter_sigma[self.rng.integers(0, len(self.kspace_filter_sigma))]
            e1_filter_sigma = self.kspace_filter_sigma[self.rng.integers(0, len(self.kspace_filter_sigma))]
            data_reduced_resolution_filtered, _, _ = apply_image_filter(data_reduced_resolution, sigma_RO=ro_filter_sigma, sigma_E1=e1_filter_sigma)

            if self.data_type=="2dt" and np.random.uniform()<0.25:
                T_filter_sigma = self.kspace_filter_T_sigma[self.rng.integers(0, len(self.kspace_filter_T_sigma))]
                data_degraded, _ = apply_image_filter_T(data_reduced_resolution_filtered, sigma_T=T_filter_sigma)
            else:
                data_degraded = data_reduced_resolution_filtered

            data_degraded = np.transpose(data_degraded, (2, 0, 1))

        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # create noise
            if self.add_noise:
                ratio_RO = self.readout_resolution_ratio[np.random.randint(0, len(self.readout_resolution_ratio))]
                ratio_E1 = self.phase_resolution_ratio[np.random.randint(0, len(self.phase_resolution_ratio))]
                nn, noise_sigma = generate_3D_MR_correlated_noise(T=T, RO=RO, E1=E1, REP=1,
                                                                    min_noise_level=self.min_noise_level,
                                                                    max_noise_level=self.max_noise_level,
                                                                    kspace_filter_sigma=self.kspace_filter_sigma,
                                                                    kspace_filter_T_sigma=self.kspace_filter_T_sigma,
                                                                    pf_filter_ratio=self.pf_filter_ratio,
                                                                    phase_resolution_ratio=[ratio_E1],
                                                                    readout_resolution_ratio=[ratio_RO],
                                                                    rng=self.rng,
                                                                    only_white_noise=self.only_white_noise,
                                                                    verbose=False)
                # apply gmap
                nn *= gmap
            else:
                noise_sigma = 0.0
                nn = 0.0

            # give it a bit perturbation for signal level
            if np.random.uniform(0, 1) < self.snr_perturb_prob:
                signal_level_delta = np.random.uniform(1.0, self.snr_perturb)

                if(self.with_data_degrading):
                    data_degraded *= signal_level_delta

                data *= signal_level_delta
                if self.load_2x_resolution: data_2x *= signal_level_delta

            # add noise to complex image and scale
            if(self.with_data_degrading):
                noisy_data = data_degraded + nn
            else:
                noisy_data = data + nn
                #data_degraded = np.copy(data)

            # scale the data
            if noise_sigma > 0:
                data /= noise_sigma
                if self.load_2x_resolution: data_2x /= noise_sigma
                noisy_data /= noise_sigma
                if(self.with_data_degrading): data_degraded /= noise_sigma

            if self.data_type != '3d':
                gmap = np.repeat(gmap[None,:,:], T, axis=0)

            # cut out the patch on the original grid
            s_x, s_y, s_t = self.get_cutout_range(data)

            if self.load_2x_resolution:
                s_x_2x = [2*p for p in s_x]
                s_y_2x = [2*p for p in s_y]

                for ind, sx in enumerate(s_x_2x):
                    while sx+2*self.cutout_shape[0] >= data_2x.shape[1] and sx>=0:
                        s_x_2x[ind] -= 2 
                        s_x[ind] -= 1
                        sx = s_x_2x[ind]
                        
                for ind, sy in enumerate(s_y_2x):
                    while sy+2*self.cutout_shape[1] >= data_2x.shape[2] and sy>=0:
                        s_y_2x[ind] -= 2 
                        s_y[ind] -= 1
                        sy = s_y_2x[ind]
            else:
                s_x_2x = s_x
                s_y_2x = s_y

            if(self.use_complex):
                patch_data = self.do_cutout(data, s_x, s_y, s_t)[np.newaxis,:,:,:]
                patch_data_with_noise = self.do_cutout(noisy_data, s_x, s_y, s_t)[np.newaxis,:,:,:]
                if(self.with_data_degrading): patch_data_degraded = self.do_cutout(data_degraded, s_x, s_y, s_t)[np.newaxis,:,:,:]
                if self.load_2x_resolution: patch_data_2x = self.do_cutout(data_2x, s_x_2x, s_y_2x, s_t, is_resized=True)[np.newaxis,:,:,:]

                cutout = np.concatenate((patch_data.real, patch_data.imag),axis=0)
                cutout_train = np.concatenate((patch_data_with_noise.real, patch_data_with_noise.imag),axis=0)
                if(self.with_data_degrading): cutout_degraded = np.concatenate((patch_data_degraded.real, patch_data_degraded.imag),axis=0)
                if self.load_2x_resolution: cutout_2x = np.concatenate((patch_data_2x.real, patch_data_2x.imag),axis=0)
            else:
                cutout = np.abs(self.do_cutout(data, s_x, s_y, s_t))[np.newaxis,:,:,:]
                cutout_train = np.abs(self.do_cutout(noisy_data, s_x, s_y, s_t))[np.newaxis,:,:,:]
                if(self.with_data_degrading): cutout_degraded = np.abs(self.do_cutout(data_degraded, s_x, s_y, s_t))[np.newaxis,:,:,:]
                if self.load_2x_resolution: cutout_2x = np.abs(self.do_cutout(data_2x, s_x_2x, s_y_2x, s_t, is_resized=True))[np.newaxis,:,:,:]

            gmap_cutout = self.do_cutout(gmap, s_x, s_y, s_t)[np.newaxis,:,:,:]

            if self.data_type=="2d":
                C = cutout.shape[1]
                pad_H = (-1*cutout_train.shape[2])%self.cutout_shape[0]
                pad_W = (-1*cutout_train.shape[3])%self.cutout_shape[1]

                cutout = np.pad(cutout, ((0,0),(0, 0), (0,pad_H),(0,pad_W)), 'symmetric')
                cutout_train = np.pad(cutout_train, ((0,0),(0, 0), (0,pad_H),(0,pad_W)), 'symmetric')
                if(self.with_data_degrading): cutout_degraded = np.pad(cutout_degraded, ((0,0),(0, 0), (0,pad_H),(0,pad_W)), 'symmetric')
                gmap_cutout = np.pad(gmap_cutout, ((0,0),(0, 0), (0,pad_H),(0,pad_W)), 'symmetric')
                if self.load_2x_resolution: cutout_2x = np.pad(cutout_2x, ((0,0),(0, 0), (0,2*pad_H),(0,2*pad_W)), 'symmetric')

                cutout = view_as_blocks(cutout, (1,C,*self.cutout_shape))
                cutout = cutout.reshape(-1,*cutout.shape[-3:])
                cutout_train = view_as_blocks(cutout_train, (1,C,*self.cutout_shape))
                cutout_train = cutout_train.reshape(-1,*cutout_train.shape[-3:])
                if(self.with_data_degrading): cutout_degraded = view_as_blocks(cutout_degraded, (1,C,*self.cutout_shape))
                if(self.with_data_degrading): cutout_degraded = cutout_degraded.reshape(-1,*cutout_degraded.shape[-3:])
                gmap_cutout = view_as_blocks(gmap_cutout, (1,1,*self.cutout_shape))
                gmap_cutout = gmap_cutout.reshape(-1,*gmap_cutout.shape[-3:])
                if self.load_2x_resolution: 
                    cutout_2x = view_as_blocks(cutout_2x, (1,C, 2*self.cutout_shape[0], 2*self.cutout_shape[1]))
                    cutout_2x = cutout_2x.reshape(-1,*cutout_2x.shape[-3:])

                if self.patches_shuffle:
                    t_indexes = np.arange(cutout.shape[0])
                    np.random.shuffle(t_indexes)

                    cutout = np.take(cutout, t_indexes, axis=0)[:self.num_patches_cutout]
                    cutout_train = np.take(cutout_train, t_indexes, axis=0)[:self.num_patches_cutout]
                    if(self.with_data_degrading): cutout_degraded = np.take(cutout_degraded, t_indexes, axis=0)[:self.num_patches_cutout]
                    gmap_cutout = np.take(gmap_cutout, t_indexes, axis=0)[:self.num_patches_cutout]
                    if self.load_2x_resolution: cutout_2x = np.take(cutout_2x, t_indexes, axis=0)[:self.num_patches_cutout]
                else:
                    start_t = np.random.randint(0,max(cutout.shape[0] - self.num_patches_cutout, 1))

                    cutout = cutout[start_t:start_t+self.num_patches_cutout]
                    cutout_train = cutout_train[start_t:start_t+self.num_patches_cutout]
                    if(self.with_data_degrading): cutout_degraded = cutout_degraded[start_t:start_t+self.num_patches_cutout]
                    gmap_cutout = gmap_cutout[start_t:start_t+self.num_patches_cutout]
                    if self.load_2x_resolution: cutout_2x = cutout_2x[start_t:start_t+self.num_patches_cutout]

                pad_T = (-1*cutout_train.shape[0])%self.num_patches_cutout
                cutout = np.pad(cutout, ((0,pad_T),(0,0),(0,0),(0,0)), 'symmetric')
                cutout_train = np.pad(cutout_train, ((0,pad_T),(0,0),(0,0),(0,0)), 'symmetric')
                if(self.with_data_degrading): cutout_degraded = np.pad(cutout_degraded, ((0,pad_T),(0,0),(0,0),(0,0)), 'symmetric')
                gmap_cutout = np.pad(gmap_cutout, ((0,pad_T),(0,0),(0,0),(0,0)), 'symmetric')
                if self.load_2x_resolution: cutout_2x = np.pad(cutout_2x, ((0,pad_T),(0,0),(0,0),(0,0)), 'symmetric')

            if(self.data_type=="3d" and self.cutout_shuffle_time):
                # perform shuffle along time
                t_indexes = np.arange(cutout.shape[1])
                np.random.shuffle(t_indexes)

                np.take(cutout, t_indexes, axis=1, out=cutout)
                np.take(cutout_train, t_indexes, axis=1, out=cutout_train)
                if(self.with_data_degrading): np.take(cutout_degraded, t_indexes, axis=1, out=cutout_degraded)
                np.take(gmap_cutout, t_indexes, axis=1, out=gmap_cutout)
                if self.load_2x_resolution: np.take(cutout_2x, t_indexes, axis=1, out=cutout_2x)

            train_noise = np.concatenate([cutout_train, gmap_cutout], axis=0)

            noisy_im = torch.from_numpy(train_noise.astype(np.float32))
            clean_im = torch.from_numpy(cutout.astype(np.float32))
            if(self.with_data_degrading): 
                clean_im_degraded = torch.from_numpy(cutout_degraded.astype(np.float32))
            else:
                clean_im_degraded = clean_im
            gmaps_median = torch.tensor(np.median(gmap_cutout))
            noise_sigmas = torch.tensor(noise_sigma)

            if self.load_2x_resolution:
                clean_im_2x = torch.from_numpy(cutout_2x.astype(np.float32))
            else:
                clean_im_2x = torch.clone(clean_im)

            if self.data_type=="2d":
                noisy_im = torch.permute(noisy_im, (1, 0, 2, 3))
                clean_im = torch.permute(clean_im, (1, 0, 2, 3))
                clean_im_degraded = torch.permute(clean_im_degraded, (1, 0, 2, 3))
                clean_im_2x = torch.permute(clean_im_2x, (1, 0, 2, 3))

        return noisy_im, clean_im, clean_im_degraded, clean_im_2x, gmaps_median, noise_sigmas

    def get_cutout_range(self, data):

        t, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        initial_s_t = np.random.randint(0, t - ct + 1)
        initial_s_t = np.random.randint(0, t) if self.data_type=="2d" else initial_s_t
        initial_s_x = np.random.randint(0, x - cx + 1)
        initial_s_y = np.random.randint(0, y - cy + 1)

        cutout_jitter_used = self.cutout_jitter[np.random.randint(0, len(self.cutout_jitter))] \
                                if self.data_type=="3d" else -1

        s_t = np.zeros(ct, dtype=np.int16) + initial_s_t # no randomness along time
        s_x = np.zeros(ct, dtype=np.int16)
        s_y = np.zeros(ct, dtype=np.int16)

        if(cutout_jitter_used<0):
            s_x += initial_s_x
            s_y += initial_s_y
        else: # jitter along 3D
            jitter_s_x = max(0, initial_s_x - np.floor(cutout_jitter_used*cx*0.5))
            jitter_s_y = max(0, initial_s_y - np.floor(cutout_jitter_used*cy*0.5))

            for t in range(ct):
                s_x_t = np.random.randint(jitter_s_x, jitter_s_x+cx)
                s_x_t = np.clip(s_x_t, 0, x-cx)
                s_x[t] = s_x_t

                s_y_t = np.random.randint(jitter_s_y, jitter_s_y+cy)
                s_y_t = np.clip(s_y_t, 0, y-cy)
                s_y[t] = s_y_t

        return s_x, s_y, s_t

    def do_cutout(self, data, s_x, s_y, s_t, is_resized=False):
        """
        Cuts out the jittered patches across a random time interval
        """
        t, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape
        if is_resized:
            cx *= 2
            cy *= 2

        if t < ct or y < cy or x < cx:
            #raise RuntimeError(f"File is borken because {t} is less than {ct} or {x} is less than {cx} or {y} is less than {cy}")
            print(f"Warning - {t} is less than {ct} or {x} is less than {cx} or {y} is less than {cy}")

        result = np.zeros((ct, cx, cy), dtype=data.dtype)

        end_x = s_x[0]+cx
        if end_x>x: 
            diff = end_x - x
            end_x = x
            s_x[0] -= diff

        end_y = s_y[0]+cy
        if end_y>y: 
            diff = end_y - y
            end_y = y
            s_y[0] -= diff

        if s_x[0] < 0:
            print(f"Warning - input data size {data.shape}, cutout shape {cx}, {cy}")
            s_x[0] = 0

        if s_y[0] < 0:
            print(f"Warning - input data size {data.shape}, cutout shape {cx}, {cy}")
            s_y[0] = 0

        if end_x - s_x[0] > x:
            end_x = x

        if end_y - s_y[0] > y:
            end_y = y

        result[:, 0:end_x-s_x[0], 0:end_y-s_y[0]] = data[s_t[0]:s_t[0]+ct, s_x[0]:end_x, s_y[0]:end_y]

        if self.data_type=="2d":
            result = np.zeros((1, x, y), dtype=data.dtype)
            result[0] = data[s_t[0]]

        if self.data_type=="3d":
            for t in range(ct):
                result[t, :, :] = data[s_t[t]+t, s_x[t]:end_x, s_y[t]:end_y]

        return result

    def load_gmap(self, gmaps, i, random_factor=-1):
        """
        Loads a random gmap for current index
        """
        if(gmaps.ndim==2):
            #gmaps = np.expand_dims(gmaps, axis=0)
            return gmaps

        factors = gmaps.shape[0]
        if factors < 16:
            if(random_factor<0):
                random_factor = np.random.randint(0, factors+1)
            if random_factor < factors:
                return gmaps[random_factor, :,:]
            else:
                return np.ones(gmaps.shape[-2:])
        else:
            if(random_factor<0):
                random_factor = np.random.randint(0, gmaps.shape[2]+1)
            if random_factor < gmaps.shape[2]:
                return gmaps[:, :, random_factor]
            else:
                return np.ones(gmaps.shape[:2])


    def random_flip(self, data, gmap, data_2x):
        """
        Randomly flips the input image and gmap
        """
        flip1 = np.random.randint(0, 2) > 0
        flip2 = np.random.randint(0, 2) > 0

        def flip(image):
            if image.ndim == 2:
                if flip1:
                    image = image[::-1,:].copy()
                if flip2:
                    image = image[:,::-1].copy()
            else:
                if flip1:
                    image = image[:,::-1,:].copy()
                if flip2:
                    image = image[:,:,::-1].copy()
            return image

        res_data_2x = None
        if self.load_2x_resolution:
            res_data_2x = flip(data_2x)

        return flip(data), flip(gmap), res_data_2x

    def get_stat(self):
        stat = load_images_for_statistics(self.h5file, self.keys)
        return stat

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Gets the item given index
        """
        return self.load_one_sample(index)

# -------------------------------------------------------------------------------------------------
# test dataset class

class MRIDenoisingDatasetTest(torch.utils.data.Dataset):
    """
    Dataset for MRI denoising testing.
    Returns full images. No cutouts.
    """
    def __init__(self, h5file, keys, ignore_gmap=True, use_complex=True):
        """
        Initilize the denoising dataset
        Loads and stores everything
        h5files should have the following strucutre
        file --> <key> --> "noisy"+"clean"+"gmap"+"noise_sigma"
        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for every h5file
            - ignore_gmap (bool): whether to ignore gmap in training
            - use_complex (bool): whether to return complex image
        """
        self.ignore_gmap = ignore_gmap
        self.use_complex = use_complex
        self.h5file = h5file
        self.keys = keys

        self.images = load_test_images_from_h5file(h5file, keys)

    def load_one_sample(self, i):
        """
        Loads one sample from the saved images
        @args:
            - i (int): index of retreive
        @rets:
            - noisy_im (5D torch.Tensor): noisy data, in the shape of [2, RO, E1] for image and gmap
                if it is complex, the shape is [3, RO, E1] for real, imag and gmap
            - clean_im (5D torch.Tensor) : clean data, [1, RO, E1] for magnitude and [2, RO, E1] for complex
            - gmap_median (0D torch.Tensor): median value for the gmap patches
            - noise_sigma (0D torch.Tensor): noise sigma added to the image patch
        """
        # get the image
        ind = self.images[i][5]
        noisy = np.array(self.h5file[ind][self.images[i][0]]).squeeze()
        clean = np.array(self.h5file[ind][self.images[i][1]]).squeeze()
        clean_resized = np.array(self.h5file[ind][self.images[i][2]]).squeeze()
        gmap = np.array(self.h5file[ind][self.images[i][3]]).squeeze()
        noise_sigma = np.array(self.h5file[ind][self.images[i][4]])

        if noisy.ndim==2:
            noisy = noisy[np.newaxis,np.newaxis,:,:]
            clean = clean[np.newaxis,np.newaxis,:,:]
            clean_resized = clean_resized[np.newaxis,np.newaxis,:,:]
        else: # ndim==3
            noisy = noisy[np.newaxis,:,:,:]
            clean = clean[np.newaxis,:,:,:]
            clean_resized = clean_resized[np.newaxis,:,:,:]

        assert gmap.ndim==2 or gmap.shape[0] == 1, f"gmap for testing should only be 2 dimensional"

        if(self.use_complex):
            noisy = np.concatenate((noisy.real, noisy.imag),axis=0)
            clean = np.concatenate((clean.real, clean.imag),axis=0)
            clean_resized = np.concatenate((clean_resized.real, clean_resized.imag),axis=0)
        else:
            noisy = np.abs(noisy)
            clean = np.abs(clean)
            clean_resized = np.abs(clean_resized)

        gmap = np.repeat(gmap[None,:,:], noisy.shape[1], axis=0)[np.newaxis,:,:,:]
        noisy = np.concatenate([noisy, gmap], axis=0)

        noisy_im = torch.from_numpy(noisy.astype(np.float32))
        clean_im = torch.from_numpy(clean.astype(np.float32))
        clean_resized_im = torch.from_numpy(clean_resized.astype(np.float32))
        gmaps_median = torch.tensor(np.median(gmap))
        noise_sigmas = torch.tensor(noise_sigma)

        # noisy_im = torch.permute(noisy_im, (1, 0, 2, 3))
        # clean_im = torch.permute(clean_im, (1, 0, 2, 3))
        # clean_resized_im = torch.permute(clean_resized_im, (1, 0, 2, 3))

        return noisy_im, clean_im, clean_im, clean_resized_im, gmaps_median, noise_sigmas

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Gets the item given index
        """
        return self.load_one_sample(index)

# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# main loading function

def load_mri_data(config):
    """
    File loader for MRI h5files
    prepares the multiple train sets as well as val and test sets
    if test case is given val set is created using 5 samples from it
    @args:
        - config (Namespace): runtime namespace for setup
    @args (from config):
        - ratio (int list): ratio to divide the given train dataset
            3 integers for ratio between train, val and test. Can be [100,0,0]
        - data_dir (str): main folder of the data
        - train_files (str list): names of h5files in dataroot for training
        - test_files (str list): names of h5files in dataroot for testing
        - train_data_types ("2d"|"2dt"|"3d" list): type of each train data file
        - test_data_types ("2d"|"2dt"|"3d" list): type of each test data file
        - time (int): cutout size in time dimension
        - mri_height (int list): different height cutouts
        - mri_width (int list): different width cutouts
        - complex_i (bool): whether to use complex image
        - min_noise_level (float): minimal noise sigma to add. Defaults to 1.0
        - max_noise_level (float): maximal noise sigma to add. Defaults to 6.0
        - matrix_size_adjust_ratio (float list): down/upsample the image, keeping the fov
        - kspace_filter_sigma (float list): kspace filter sigma
        - pf_filter_ratio (float list): partial fourier filter
        - phase_resolution_ratio (float list): phase resolution ratio
        - readout_resolution_ratio (float list): readout resolution ratio
    """
    c = config # shortening due to numerous uses

    ratio = [x/100 for x in c.ratio]
    logging.info(f"--> loading data with ratio {ratio} ...")

    h5files = []
    train_keys = []
    val_keys = []
    test_keys = []

    train_paths = []
    for path_x in c.train_files:
        train_paths.append(os.path.join(c.data_dir, path_x))

    # check file
    for file in train_paths:
        if not os.path.exists(file):
            raise RuntimeError(f"File not found: {file}")
        print(f"file exist - {file}")

    for file in train_paths:
        if not os.path.exists(file):
            raise RuntimeError(f"File not found: {file}")

        logging.info(f"reading from file: {file}")
        h5file = h5py.File(file, libver='earliest', mode='r')
        keys = list(h5file.keys())

        n = len(keys)

        tra = int(ratio[0]*n)
        tra = 1 if tra == 0 else tra

        val = int((ratio[0]+ratio[1])*n)
        val = tra + 1 if val<=tra else val
        val = n if val>n else val

        h5files.append(h5file)
        train_keys.append(keys[:tra])
        val_keys.append(keys[tra:val])
        test_keys.append(keys[val:])

        # make sure there is no empty testing
        if len(val_keys[-1])==0:
            val_keys[-1] = keys[-1:]
        if len(test_keys[-1])==0:
            test_keys[-1] = keys[-1:]

        logging.info(f"Done - reading from file: {file}, \
                     tra {sum([len(v) for v in train_keys])},\
                     val {sum([len(v) for v in val_keys])}, \
                     test {sum([len(v) for v in test_keys])}")

    # common kwargs
    kwargs = {
        "time_cutout" : c.time,
        "use_complex" : c.complex_i,
        "min_noise_level" : c.min_noise_level,
        "max_noise_level" : c.max_noise_level,
        "matrix_size_adjust_ratio" : c.matrix_size_adjust_ratio,
        "kspace_filter_sigma" : c.kspace_filter_sigma,
        "pf_filter_ratio" : c.pf_filter_ratio,
        "phase_resolution_ratio" : c.phase_resolution_ratio,
        "readout_resolution_ratio" : c.readout_resolution_ratio,
        "cutout_jitter" : c.threeD_cutout_jitter,
        "cutout_shuffle_time" : c.threeD_cutout_shuffle_time,
        "snr_perturb_prob" : c.snr_perturb_prob,
        "snr_perturb" : c.snr_perturb,
        "with_data_degrading" : c.with_data_degrading,
        "add_noise": c.not_add_noise==False,
        "load_2x_resolution": c.super_resolution,
        "only_white_noise": c.only_white_noise,
        "ignore_gmap": c.ignore_gmap
    }

    train_set = []

    for (i, h_file) in enumerate(h5files):
        logging.info(f"--> loading data from file: {h_file} for {len(train_keys[i])} entries ...")
        images = load_images_from_h5file([h_file], [train_keys[i]], max_load=c.max_load)
        for hw in zip(c.mri_height, c.mri_width):        
            train_set.append(MRIDenoisingDatasetTrain(h5file=[h_file], keys=[train_keys[i]], max_load=-1, data_type=c.train_data_types[i], cutout_shape=hw, **kwargs))
            train_set[-1].images = images

    kwargs["snr_perturb_prob"] = 0
    val_set = [MRIDenoisingDatasetTrain(h5file=[h_file], keys=[val_keys[i]], max_load=c.max_load, 
                                        data_type=c.train_data_types[i], cutout_shape=[c.mri_height[-1], c.mri_width[-1]], **kwargs)
                                            for (i,h_file) in enumerate(h5files)]

    if c.test_files is None or c.test_files[0] is None: # no test case given so use some from train data
        test_set = [MRIDenoisingDatasetTrain(h5file=[h_file], keys=[test_keys[i]], max_load=c.max_load, 
                                             data_type=c.train_data_types[i], cutout_shape=[c.mri_height[-1], c.mri_width[-1]], **kwargs)
                                                for (i,h_file) in enumerate(h5files)]
    else: # test case is given. take part of it as val set
        test_set, test_h5files = load_mri_test_data(config, ratio_test=ratio[2])

    total_tra = sum([len(d) for d in train_set])
    total_val = sum([len(d) for d in val_set])
    total_test = sum([len(d) for d in test_set])

    logging.info(f"--->{Fore.YELLOW}Number of samples for tra/val/test are {total_tra}/{total_val}/{total_test}{Style.RESET_ALL}")

    return train_set, val_set, test_set

# -------------------------------------------------------------------------------------------------

def load_mri_test_data(config, ratio_test=1.0):
    c = config
    test_h5files = []
    test_paths = [os.path.join(c.data_dir, path_x) for path_x in c.test_files]

    cutout_shape=[c.mri_height[-1], c.mri_width[-1]]

    for i, file in enumerate(test_paths):
        if not os.path.exists(file):
            raise RuntimeError(f"File not found: {file}")

        logging.info(f"reading from file: {file}")
        h5file = h5py.File(file, libver='earliest', mode='r')
        keys = list(h5file.keys())

        if ratio_test>0:
            keys = keys[:int(len(keys)*ratio_test)]

        test_h5files.append((h5file,keys))

    logging.info(f"loading in test data ...")
    test_set = [MRIDenoisingDatasetTest([h_file], keys=[t_keys], use_complex=c.complex_i) for (h_file,t_keys) in test_h5files]
    logging.info(f"loading in test data --- completed")

    return test_set, test_h5files

# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    import nibabel as nib

    saved_path = "/export/Lab-Xue/projects/mri/results/loader_test"
    os.makedirs(saved_path, exist_ok=True)

    # -----------------------------------------------------------------

    file = "/data1/mri/data/train_3D_3T_retro_cine_2018.h5"
    h5file = h5py.File(file, libver='earliest', mode='r')
    keys = list(h5file.keys())

    images = load_images_from_h5file([h5file], [keys], max_load=-1)

    tra_data = MRIDenoisingDatasetTrain([h5file], [keys], data_type='2DT', load_2x_resolution=False, ignore_gmap=False, only_white_noise=False)

    for k in range(len(tra_data)):
        print(f"{k} out of {len(tra_data)}")
        noisy_im, clean_im, clean_im_degraded, clean_im_2x, gmaps_median, noise_sigmas = tra_data[k]

    for k in range(10):
        noisy_im, clean_im, clean_im_degraded, clean_im_2x, gmaps_median, noise_sigmas = tra_data[np.random.randint(len(tra_data))]

        noisy_im = np.transpose(noisy_im.numpy(), (2, 3, 0, 1)) # H, W, C, T
        clean_im = np.transpose(clean_im.numpy(), (2, 3, 0, 1))
        clean_im_degraded = np.transpose(clean_im_degraded.numpy(), (2, 3, 0, 1))
        clean_im_2x = np.transpose(clean_im_2x.numpy(), (2, 3, 0, 1))

        gmap = noisy_im[:,:,2,:]
        noisy_im = noisy_im[:,:,0,:] + 1j * noisy_im[:,:,1,:]
        clean_im = clean_im[:,:,0,:] + 1j * clean_im[:,:,1,:]
        clean_im_degraded = clean_im_degraded[:,:,0,:] + 1j * clean_im_degraded[:,:,1,:]
        clean_im_2x = clean_im_2x[:,:,0,:] + 1j * clean_im_2x[:,:,1,:]

        nib.save(nib.Nifti1Image(np.abs(noisy_im), affine=np.eye(4)), os.path.join(saved_path, f"noisy_im_{k}.nii"))
        nib.save(nib.Nifti1Image(np.abs(clean_im), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_{k}.nii"))
        nib.save(nib.Nifti1Image(np.abs(clean_im_degraded), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_degraded_{k}.nii"))
        nib.save(nib.Nifti1Image(np.abs(clean_im_2x), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_2x_{k}.nii"))
        nib.save(nib.Nifti1Image(gmap, affine=np.eye(4)), os.path.join(saved_path, f"gmap_{k}.nii"))
        print(gmaps_median, noise_sigmas)

# -----------------------------------------------------------------

    file = "/data/FM_data_repo/mri/VIDA_train_clean_0430.h5"
    h5file = h5py.File(file, libver='earliest', mode='r')
    keys = list(h5file.keys())

    images = load_images_from_h5file([h5file], [keys], max_load=-1)

    tra_data = MRIDenoisingDatasetTrain([h5file], [keys], time_cutout=12, data_type='3d', load_2x_resolution=False, ignore_gmap=False, only_white_noise=True)

    for k in range(len(tra_data)):
        print(f"{k} out of {len(tra_data)}")
        noisy_im, clean_im, clean_im_degraded, clean_im_2x, gmaps_median, noise_sigmas = tra_data[k]

    for k in range(10):
        noisy_im, clean_im, clean_im_degraded, clean_im_2x, gmaps_median, noise_sigmas = tra_data[np.random.randint(len(tra_data))]

        gmap = noisy_im[2]
        noisy_im = noisy_im[0] + 1j * noisy_im[1]
        clean_im = clean_im[0] + 1j * clean_im[1]
        clean_im_degraded = clean_im_degraded[0] + 1j * clean_im_degraded[1]
        clean_im_2x = clean_im_2x[0] + 1j * clean_im_2x[1]

        gmap = gmap.numpy()
        noisy_im = noisy_im.numpy()
        clean_im = clean_im.numpy()
        clean_im_degraded = clean_im_degraded.numpy()
        clean_im_2x = clean_im_2x.numpy()

        nib.save(nib.Nifti1Image(np.abs(noisy_im), affine=np.eye(4)), os.path.join(saved_path, f"noisy_im_{k}.nii"))
        nib.save(nib.Nifti1Image(np.abs(clean_im), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_{k}.nii"))
        nib.save(nib.Nifti1Image(np.abs(clean_im_degraded), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_degraded_{k}.nii"))
        nib.save(nib.Nifti1Image(np.abs(clean_im_2x), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_2x_{k}.nii"))
        nib.save(nib.Nifti1Image(gmap, affine=np.eye(4)), os.path.join(saved_path, f"gmap_{k}.nii"))
        print(gmaps_median, noise_sigmas)

    # ---------------------------------------------------------

    file = "/data/mri/data/train_3D_3T_retro_cine_2020_500_samples.h5"
    h5file = h5py.File(file, libver='earliest', mode='r')
    keys = list(h5file.keys())

    test_data = MRIDenoisingDatasetTest([h5file], [keys])

    noisy_im, clean_im, clean_im_degraded, clean_im_2x, gmaps_median, noise_sigmas = test_data[22]

    noisy_im = np.transpose(noisy_im.numpy(), (2, 3, 1, 0))
    clean_im = np.transpose(clean_im.numpy(), (2, 3, 1, 0))
    clean_im_2x = np.transpose(clean_im_2x.numpy(), (2, 3, 1, 0))

    gmap = noisy_im[:,:,12,2]
    noisy_im = noisy_im[:,:,:,0] + 1j * noisy_im[:,:,:,1]
    clean_im = clean_im[:,:,:,0] + 1j * clean_im[:,:,:,1]
    clean_im_2x = clean_im_2x[:,:,:,0] + 1j * clean_im_2x[:,:,:,1]

    nib.save(nib.Nifti1Image(np.abs(noisy_im), affine=np.eye(4)), os.path.join(saved_path, f"noisy_im.nii"))
    nib.save(nib.Nifti1Image(np.abs(clean_im), affine=np.eye(4)), os.path.join(saved_path, f"clean_im.nii"))
    nib.save(nib.Nifti1Image(np.abs(clean_im_2x), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_2x.nii"))
    nib.save(nib.Nifti1Image(gmap, affine=np.eye(4)), os.path.join(saved_path, f"gmap.nii"))
    print(gmaps_median, noise_sigmas)

    # ---------------------------------------------------------

    file = "/data/mri/data/train_3D_3T_retro_cine_2020_500_samples_with_2x_resized.h5"
    h5file = h5py.File(file, libver='earliest', mode='r')
    keys = list(h5file.keys())

    test_data = MRIDenoisingDatasetTest([h5file], [keys])

    noisy_im, clean_im, clean_im_degraded, clean_im_2x, gmaps_median, noise_sigmas = test_data[22]

    noisy_im = np.transpose(noisy_im.numpy(), (2, 3, 1, 0))
    clean_im = np.transpose(clean_im.numpy(), (2, 3, 1, 0))
    clean_im_2x = np.transpose(clean_im_2x.numpy(), (2, 3, 1, 0))

    gmap = noisy_im[:,:,12,2]
    noisy_im = noisy_im[:,:,:,0] + 1j * noisy_im[:,:,:,1]
    clean_im = clean_im[:,:,:,0] + 1j * clean_im[:,:,:,1]
    clean_im_2x = clean_im_2x[:,:,:,0] + 1j * clean_im_2x[:,:,:,1]

    nib.save(nib.Nifti1Image(np.abs(noisy_im), affine=np.eye(4)), os.path.join(saved_path, f"noisy_im_test.nii"))
    nib.save(nib.Nifti1Image(np.abs(clean_im), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_test.nii"))
    nib.save(nib.Nifti1Image(np.abs(clean_im_2x), affine=np.eye(4)), os.path.join(saved_path, f"clean_im_2x_test.nii"))
    nib.save(nib.Nifti1Image(gmap, affine=np.eye(4)), os.path.join(saved_path, f"gmap_test.nii"))
    print(gmaps_median, noise_sigmas)