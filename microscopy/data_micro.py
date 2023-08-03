"""
Data utilities for Microscopy data.
Provides the torch dataset class for train and test
And a function to load the said classes with multiple h5files

#TODO: correct this expected h5file. Also fix comments
Expected train h5file:
<file> ---> <key> ---> "/image"
                   |-> "/gmap"

Expected test h5file:
<file> ---> <key> ---> "/noisy"
                   |-> "/clean"
                   |-> "/gmap"
                   |-> "/noise_sigma"
"""

import os
import sys
import cv2
import h5py
import torch
import logging
from time import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
from skimage.util import view_as_blocks
from colorama import Fore, Style

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *

class MicroscopyDatasetTrain():
    """
    Dataset for Microscopy.
    """
    def __init__(self, h5file, keys,
                    time_cutout=30, 
                    cutout_shape=[64, 64],
                    num_samples_per_file=8,
                    rng = None,
                    transform = None,
                    test = False,
                    per_scaling = False,
                    im_value_scale = 4096,
                    time_scale = 0):
        """Initilize the denoising dataset

        Args:
            h5file (list): list of h5files to load images from
            keys ([ [list1], [list2], ... ]): List of list of keys. For every h5file, a list of keys are provided
            cutout_shape (list, optional): patch size. Defaults to [64, 64].
            use_gmap (bool, optional): Whether to load and return gmap. Defaults to True.
            use_complex (bool, optional): Whether to return complex image. Defaults to True.
            min_noise_level (float, optional): Minimal noise sigma to add. Defaults to 1.0. If <0, no noise is added.
            max_noise_level (float, optional): Maximal noise sigma to add. Defaults to 6.0. If <0 or <min_noise_level, no noise is added.
            matrix_size_adjust_ratio: down/upsample the image, keeping the fov
            kspace_filter_sigma: kspace filter sigma
            pf_filter_ratio: partial fourier filter
            phase_resolution_ratio: phase resolution ratio
            readout_resolution_ratio: readout resolution ratio
            train_for_super_resolution: if True, train with super-resolution with reduced spatial resolution
            upsampling_ratio: if train_for_super_resolution is True, the upsampling ratio for images (e.g. 2 means matrix size is doubled)
            rng (np.random): preset the seeds for the deterministic results
        """

        self.keys = keys
        self.N_files = len(self.keys)

        self.time_cutout = time_cutout
        self.cutout_shape = cutout_shape
        
        self.num_samples_per_file = num_samples_per_file
        self.transform = transform

        self.test = test
        self.per_scaling = per_scaling
        self.im_value_scale = im_value_scale
        self.time_scale = time_scale

        # ------------------------------------------------

        self.start_samples = np.zeros(self.N_files)
        self.end_samples = np.zeros(self.N_files)
        
        self.start_samples[0] = 0
        self.end_samples[0] = len(self.keys[0]) * num_samples_per_file
        
        if(rng is None):
            self.rng = np.random.default_rng(seed=np.random.randint(0, np.iinfo(np.int32).max))
        else:
            self.rng = rng
        
        for i in range(1, self.N_files):            
            self.start_samples[i] = self.end_samples[i-1]
            self.end_samples[i] = num_samples_per_file*len(self.keys[i]) + self.start_samples[i]

        self.tiff_dict = {}
        for i, hfile in enumerate(h5file):
            self.tiff_dict[i] = {}

            for key in keys[i]:
                self.tiff_dict[i][key] = {}

                noisy_data = np.array(hfile[key+"/noisy_im"]).astype(np.float32)
                clean_data = np.array(hfile[key+"/clean_im"]).astype(np.float32)

                if per_scaling:
                    noisy_data = normalize_image(noisy_data, percentiles=(1.5, 99.5), clip=True)
                    clean_data = normalize_image(clean_data, percentiles=(1.5, 99.5), clip=True)
                else:
                    noisy_data = normalize_image(noisy_data, values=(0, im_value_scale), clip=True)
                    clean_data = normalize_image(clean_data, values=(0, im_value_scale), clip=True)

                self.tiff_dict[i][key]["noisy_im"] = noisy_data
                self.tiff_dict[i][key]["clean_im"] = clean_data

            print(f"--> finish loading {hfile}")

    def load_one_sample(self, h5file, key):
        """Load one sample from the h5file and key pair

        Args:
            h5file (h5file handle): opened h5file handle
            key (str): key for the image and gmap
        
        Returns:
            noisy_im (list) : list of noisy data, every item is in the shape of [num_images_picked, num_patch_cutout, 2, RO, E1] for image and gmap;
                                if it is complex , the shape is [num_images_picked, num_patch_cutout, 3, RO, E1] for real, imag and gmap
            clean_im (list) : clearn images
            clean_im_low_res (list) : clearn images with reduced spatial resolution
            gmap_median (list of values): median value for the gmap patches
            noise_sigma (list of values): noise sigma added to the image patch
        """
        noisy_im = []
        clean_im = []
        clean_im_low_res = []
        gmaps_median = []
        noise_sigmas = []
        
        # get the image
        noisy_data = self.tiff_dict[h5file][key]["noisy_im"]
        clean_data = self.tiff_dict[h5file][key]["clean_im"]

        if self.test:
            noisy_cutout = noisy_data[:,np.newaxis,:,:]
            clean_cutout = clean_data[:,np.newaxis,:,:]

            train_noise = noisy_cutout

            noisy_im.append(torch.from_numpy(train_noise.astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout.astype(np.float32)))

            # appending empties to keep the code same across datatypes
            clean_im_low_res.append(np.array([]))
            gmaps_median.append(np.array([]))
            noise_sigmas.append(np.array([]))

            return noisy_im, clean_im, clean_im_low_res, gmaps_median, noise_sigmas

        # pad symmetrically if not enough images in the time dimension
        if noisy_data.shape[0] < self.time_cutout:
            noisy_data = np.pad(noisy_data, ((0,self.time_cutout - noisy_data.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,self.time_cutout - clean_data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_data, clean_data = self.random_flip(noisy_data, clean_data, rng = self.rng)

        if noisy_data.shape[1] < self.cutout_shape[0]:
            noisy_data = np.pad(noisy_data, ((0, 0), (0,self.cutout_shape[0] - noisy_data.shape[1]),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0, 0), (0,self.cutout_shape[0] - clean_data.shape[1]),(0,0)), 'symmetric')

        if noisy_data.shape[2] < self.cutout_shape[1]:
            noisy_data = np.pad(noisy_data, ((0,0), (0,0), (0,self.cutout_shape[1] - noisy_data.shape[2])), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0), (0,0), (0,self.cutout_shape[1] - clean_data.shape[2])), 'symmetric')

        T, RO, E1 = noisy_data.shape
        
        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # define a set of cut range
            s_x, s_y, s_t = self.get_cutout_range(noisy_data, rng=self.rng)
                
            noisy_cutout = self.do_cutout(noisy_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
            clean_cutout = self.do_cutout(clean_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
                
            train_noise = noisy_cutout

            noisy_im.append(torch.from_numpy(train_noise.astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout.astype(np.float32)))

            # appending empties to keep the code same across datatypes
            clean_im_low_res.append(np.array([]))
            gmaps_median.append(np.array([]))
            noise_sigmas.append(np.array([]))

        if(len(noisy_im)==0):
            print("noisy_im is empty ...")

        return noisy_im, clean_im
      
    def load_one_sample_timed(self, h5file, key):
        """Load one sample from the h5file and key pair

        Args:
            h5file (h5file handle): opened h5file handle
            key (str): key for the image and gmap
        
        Returns:
            noisy_im (list) : list of noisy data, every item is in the shape of [num_images_picked, num_patch_cutout, 2, RO, E1] for image and gmap;
                                if it is complex , the shape is [num_images_picked, num_patch_cutout, 3, RO, E1] for real, imag and gmap
            clean_im (list) : clearn images
            clean_im_low_res (list) : clearn images with reduced spatial resolution
            gmap_median (list of values): median value for the gmap patches
            noise_sigma (list of values): noise sigma added to the image patch
        """
        noisy_im = []
        clean_im = []
        clean_im_low_res = []
        gmaps_median = []
        noise_sigmas = []
        
        # get the image
        noisy_data = self.tiff_dict[h5file][key]["noisy_im"]
        clean_data = self.tiff_dict[h5file][key]["clean_im"]

        if self.time_scale > 0:
            noisy_data = np.average(noisy_data[:self.time_scale], axis=0)
        if self.time_scale < 0:
            scale = np.random.randint(1,noisy_data.shape[0]+1)
            noisy_data = np.average(noisy_data[:scale], axis=0)

        if self.test:
            noisy_cutout = noisy_data[:,np.newaxis,:,:]
            clean_cutout = clean_data[:,np.newaxis,:,:]

            train_noise = noisy_cutout

            noisy_im.append(torch.from_numpy(train_noise.astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout.astype(np.float32)))

            # appending empties to keep the code same across datatypes
            clean_im_low_res.append(np.array([]))
            gmaps_median.append(np.array([]))
            noise_sigmas.append(np.array([]))

            return noisy_im, clean_im, clean_im_low_res, gmaps_median, noise_sigmas

        # pad symmetrically if not enough images in the time dimension
        if noisy_data.shape[0] < self.time_cutout:
            noisy_data = np.pad(noisy_data, ((0,self.time_cutout - noisy_data.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,self.time_cutout - clean_data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_data, clean_data = self.random_flip(noisy_data, clean_data, rng = self.rng)

        if noisy_data.shape[1] < self.cutout_shape[0]:
            noisy_data = np.pad(noisy_data, ((0, 0), (0,self.cutout_shape[0] - noisy_data.shape[1]),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0, 0), (0,self.cutout_shape[0] - clean_data.shape[1]),(0,0)), 'symmetric')

        if noisy_data.shape[2] < self.cutout_shape[1]:
            noisy_data = np.pad(noisy_data, ((0,0), (0,0), (0,self.cutout_shape[1] - noisy_data.shape[2])), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0), (0,0), (0,self.cutout_shape[1] - clean_data.shape[2])), 'symmetric')

        T, RO, E1 = noisy_data.shape
        
        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # define a set of cut range
            s_x, s_y, s_t = self.get_cutout_range(noisy_data, rng=self.rng)
                
            noisy_cutout = self.do_cutout(noisy_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
            clean_cutout = self.do_cutout(clean_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
                
            train_noise = noisy_cutout

            noisy_im.append(torch.from_numpy(train_noise.astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout.astype(np.float32)))

            # appending empties to keep the code same across datatypes
            clean_im_low_res.append(np.array([]))
            gmaps_median.append(np.array([]))
            noise_sigmas.append(np.array([]))

        if(len(noisy_im)==0):
            print("noisy_im is empty ...")

        return noisy_im, clean_im

    def get_cutout_range(self, data, rng):
        """
        Return s_x, s_y and s_t
        s_x, s_y stores the starting location of cutouts for every frame
        s_t stores which frames to cut
        """    
        t, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape
        
        # initial_s_t = rng.integers(0, t - ct) if t>ct else 0
        # initial_s_x = rng.integers(0, x - cx) if x>cx else 0
        # initial_s_y = rng.integers(0, y - cy) if y>cy else 0

        initial_s_t = np.random.randint(0, t - ct) if t>ct else 0
        initial_s_x = np.random.randint(0, x - cx) if x>cx else 0
        initial_s_y = np.random.randint(0, y - cy) if y>cy else 0
        
        s_x = np.zeros(ct, dtype=np.int16) + initial_s_x
        s_y = np.zeros(ct, dtype=np.int16) + initial_s_y
        s_t = np.zeros(ct, dtype=np.int16) + initial_s_t
        
        return s_x, s_y, s_t
          
    def do_cutout(self, data, s_x, s_y, s_t):
        """
        Cuts out the patches
        
        if on_upsampled_grid is True, data is on the upsampled grid (self.upsampling_ratio)
        """
        T, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        if T < ct or y < cy or x < cx:
            raise RuntimeError(f"File is borken because {t} is less than {ct} or {x} is less than {cx} or {y} is less than {cy}")

        result = np.zeros((ct, cx, cy), dtype=data.dtype)

        for t in range(ct):
            result[t, :, :] = data[s_t[t]+t, s_x[t]:s_x[t]+cx, s_y[t]:s_y[t]+cy]

        return result
        
    def random_flip(self, noisy, clean, rng = np.random):
        
        flip1 = rng.integers(0,2) > 0
        flip2 = rng.integers(0,2) > 0

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

        return flip(noisy), flip(clean)
            
    def find_sample(self, index):
        ind_file = 0

        for i in range(self.N_files):
            if(index>= self.start_samples[i] and index<self.end_samples[i]):
                ind_file = i
                ind_in_file = int(index - self.start_samples[i])//self.num_samples_per_file
                break

        return ind_file, ind_in_file

    def content_over_time(self, clean_im):

        T, C, H, W = clean_im.shape

        content_ratio = np.zeros(T)
        threshold = 0.05 if self.per_scaling else 0.02 if self.im_value_scale > 2000 else 0.1

        for t in range(T):

            image_t = clean_im[t, 0]
            content_ratio[t] = max(torch.count_nonzero(image_t > threshold) / torch.prod(torch.as_tensor(image_t.shape)), 0.01)

        return content_ratio
    
    def __len__(self):
        total_num_samples = 0
        for key in self.keys:
            total_num_samples += len(key)*self.num_samples_per_file
        return total_num_samples
            
    def __getitem__(self, idx):
        
        #print(f"{idx}")
        sample_list = []
        count_list = []
        found = False
        threshold = 0.05 if self.per_scaling else 0.02
        for i in range(10):
            ind_file, ind_in_file = self.find_sample(idx)
            if self.time_scale == 0:
                noisy_im, clean_im = self.load_one_sample(ind_file, self.keys[ind_file][ind_in_file])
            else:
                noisy_im, clean_im = self.load_one_sample_timed(ind_file, self.keys[ind_file][ind_in_file])

            sample = (noisy_im[0], clean_im[0])

            area_threshold = 0.25 * torch.prod(torch.as_tensor(sample[1].shape))
            # different value check for percent vs constant scaling
            if (self.per_scaling and
                torch.count_nonzero(sample[1] > threshold) >= area_threshold):
                found = True
                break

            if (torch.count_nonzero(sample[1] > threshold) >= area_threshold):
                found = True
                break

            sample_list.append(sample)
            count_list.append(torch.count_nonzero(sample[1] > threshold) if self.per_scaling else torch.count_nonzero(sample[1] > threshold))

        if not found:
            sample = sample_list[count_list.index(max(count_list))]

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class MicroscopyDatasetTest():
    """
    Dataset for Microscopy.
    """
    def __init__(self, h5file, keys,
                    per_scaling = False,
                    im_value_scale = 4096):
        """Initilize the denoising dataset

        Args:
            h5file (list): list of h5files to load images from
            keys ([ [list1], [list2], ... ]): List of list of keys. For every h5file, a list of keys are provided
            cutout_shape (list, optional): patch size. Defaults to [64, 64].
            use_gmap (bool, optional): Whether to load and return gmap. Defaults to True.
            use_complex (bool, optional): Whether to return complex image. Defaults to True.
            min_noise_level (float, optional): Minimal noise sigma to add. Defaults to 1.0. If <0, no noise is added.
            max_noise_level (float, optional): Maximal noise sigma to add. Defaults to 6.0. If <0 or <min_noise_level, no noise is added.
            matrix_size_adjust_ratio: down/upsample the image, keeping the fov
            kspace_filter_sigma: kspace filter sigma
            pf_filter_ratio: partial fourier filter
            phase_resolution_ratio: phase resolution ratio
            readout_resolution_ratio: readout resolution ratio
            train_for_super_resolution: if True, train with super-resolution with reduced spatial resolution
            upsampling_ratio: if train_for_super_resolution is True, the upsampling ratio for images (e.g. 2 means matrix size is doubled)
            rng (np.random): preset the seeds for the deterministic results
        """

        self.keys = keys
        self.N_files = len(self.keys)
        self.per_scaling = per_scaling
        self.im_value_scale = im_value_scale

        # ------------------------------------------------

        self.start_samples = np.zeros(self.N_files)
        self.end_samples = np.zeros(self.N_files)
        
        self.start_samples[0] = 0
        self.end_samples[0] = len(self.keys[0]) 
        
        for i in range(1, self.N_files):            
            self.start_samples[i] = self.end_samples[i-1]
            self.end_samples[i] = len(self.keys[i]) + self.start_samples[i]

        self.tiff_dict = {}
        for i, hfile in enumerate(h5file):
            self.tiff_dict[i] = {}

            for key in keys[i]:
                self.tiff_dict[i][key] = {}

                noisy_data = np.array(hfile[key+"/noisy_im"]).astype(np.float32)
                clean_data = np.array(hfile[key+"/clean_im"]).astype(np.float32)

                if per_scaling:
                    noisy_data = normalize_image(noisy_data, percentiles=(1.5, 99.5), clip=True)
                    clean_data = normalize_image(clean_data, percentiles=(1.5, 99.5), clip=True)
                else:
                    noisy_data = normalize_image(noisy_data, values=(0, im_value_scale), clip=True)
                    clean_data = normalize_image(clean_data, values=(0, im_value_scale), clip=True)

                self.tiff_dict[i][key]["noisy_im"] = noisy_data
                self.tiff_dict[i][key]["clean_im"] = clean_data

            print(f"--> finish loading {hfile}")

    def load_one_sample(self, h5file, key):
        """Load one sample from the h5file and key pair

        Args:
            h5file (h5file handle): opened h5file handle
            key (str): key for the image and gmap
        
        Returns:
            noisy_im (list) : list of noisy data, every item is in the shape of [num_images_picked, num_patch_cutout, 2, RO, E1] for image and gmap;
                                if it is complex , the shape is [num_images_picked, num_patch_cutout, 3, RO, E1] for real, imag and gmap
            clean_im (list) : clearn images
            clean_im_low_res (list) : clearn images with reduced spatial resolution
            gmap_median (list of values): median value for the gmap patches
            noise_sigma (list of values): noise sigma added to the image patch
        """
        noisy_im = []
        clean_im = []
        clean_im_low_res = []
        gmaps_median = []
        noise_sigmas = []
        
        # get the image
        noisy_data = self.tiff_dict[h5file][key]["noisy_im"]
        clean_data = self.tiff_dict[h5file][key]["clean_im"]

        if self.test:
            noisy_cutout = noisy_data[:,np.newaxis,:,:]
            clean_cutout = clean_data[:,np.newaxis,:,:]

            train_noise = noisy_cutout

            noisy_im.append(torch.from_numpy(train_noise.astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout.astype(np.float32)))

            # appending empties to keep the code same across datatypes
            clean_im_low_res.append(np.array([]))
            gmaps_median.append(np.array([]))
            noise_sigmas.append(np.array([]))

            return noisy_im, clean_im, clean_im_low_res, gmaps_median, noise_sigmas

        # pad symmetrically if not enough images in the time dimension
        if noisy_data.shape[0] < self.time_cutout:
            noisy_data = np.pad(noisy_data, ((0,self.time_cutout - noisy_data.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,self.time_cutout - clean_data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_data, clean_data = self.random_flip(noisy_data, clean_data, rng = self.rng)

        if noisy_data.shape[1] < self.cutout_shape[0]:
            noisy_data = np.pad(noisy_data, ((0, 0), (0,self.cutout_shape[0] - noisy_data.shape[1]),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0, 0), (0,self.cutout_shape[0] - clean_data.shape[1]),(0,0)), 'symmetric')

        if noisy_data.shape[2] < self.cutout_shape[1]:
            noisy_data = np.pad(noisy_data, ((0,0), (0,0), (0,self.cutout_shape[1] - noisy_data.shape[2])), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0), (0,0), (0,self.cutout_shape[1] - clean_data.shape[2])), 'symmetric')

        T, RO, E1 = noisy_data.shape
        
        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # define a set of cut range
            s_x, s_y, s_t = self.get_cutout_range(noisy_data, rng=self.rng)
                
            noisy_cutout = self.do_cutout(noisy_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
            clean_cutout = self.do_cutout(clean_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
                
            train_noise = noisy_cutout

            noisy_im.append(torch.from_numpy(train_noise.astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout.astype(np.float32)))

            # appending empties to keep the code same across datatypes
            clean_im_low_res.append(np.array([]))
            gmaps_median.append(np.array([]))
            noise_sigmas.append(np.array([]))

        if(len(noisy_im)==0):
            print("noisy_im is empty ...")

        return noisy_im, clean_im, clean_im_low_res, gmaps_median, noise_sigmas
      
    def load_one_sample_timed(self, h5file, key):
        """Load one sample from the h5file and key pair

        Args:
            h5file (h5file handle): opened h5file handle
            key (str): key for the image and gmap
        
        Returns:
            noisy_im (list) : list of noisy data, every item is in the shape of [num_images_picked, num_patch_cutout, 2, RO, E1] for image and gmap;
                                if it is complex , the shape is [num_images_picked, num_patch_cutout, 3, RO, E1] for real, imag and gmap
            clean_im (list) : clearn images
            clean_im_low_res (list) : clearn images with reduced spatial resolution
            gmap_median (list of values): median value for the gmap patches
            noise_sigma (list of values): noise sigma added to the image patch
        """
        noisy_im = []
        clean_im = []
        clean_im_low_res = []
        gmaps_median = []
        noise_sigmas = []
        
        # get the image
        noisy_data = self.tiff_dict[h5file][key]["noisy_im"]
        clean_data = self.tiff_dict[h5file][key]["clean_im"]

        if self.time_scale > 0:
            noisy_data = np.average(noisy_data[:self.time_scale], axis=0)
        if self.time_scale < 0:
            scale = np.random.randint(1,noisy_data.shape[0]+1)
            noisy_data = np.average(noisy_data[:scale], axis=0)

        if self.test:
            noisy_cutout = noisy_data[:,np.newaxis,:,:]
            clean_cutout = clean_data[:,np.newaxis,:,:]

            train_noise = noisy_cutout

            noisy_im.append(torch.from_numpy(train_noise.astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout.astype(np.float32)))

            # appending empties to keep the code same across datatypes
            clean_im_low_res.append(np.array([]))
            gmaps_median.append(np.array([]))
            noise_sigmas.append(np.array([]))

            return noisy_im, clean_im, clean_im_low_res, gmaps_median, noise_sigmas

        # pad symmetrically if not enough images in the time dimension
        if noisy_data.shape[0] < self.time_cutout:
            noisy_data = np.pad(noisy_data, ((0,self.time_cutout - noisy_data.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,self.time_cutout - clean_data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_data, clean_data = self.random_flip(noisy_data, clean_data, rng = self.rng)

        if noisy_data.shape[1] < self.cutout_shape[0]:
            noisy_data = np.pad(noisy_data, ((0, 0), (0,self.cutout_shape[0] - noisy_data.shape[1]),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0, 0), (0,self.cutout_shape[0] - clean_data.shape[1]),(0,0)), 'symmetric')

        if noisy_data.shape[2] < self.cutout_shape[1]:
            noisy_data = np.pad(noisy_data, ((0,0), (0,0), (0,self.cutout_shape[1] - noisy_data.shape[2])), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0), (0,0), (0,self.cutout_shape[1] - clean_data.shape[2])), 'symmetric')

        T, RO, E1 = noisy_data.shape
        
        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # define a set of cut range
            s_x, s_y, s_t = self.get_cutout_range(noisy_data, rng=self.rng)
                
            noisy_cutout = self.do_cutout(noisy_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
            clean_cutout = self.do_cutout(clean_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
                
            train_noise = noisy_cutout

            noisy_im.append(torch.from_numpy(train_noise.astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout.astype(np.float32)))

            # appending empties to keep the code same across datatypes
            clean_im_low_res.append(np.array([]))
            gmaps_median.append(np.array([]))
            noise_sigmas.append(np.array([]))

        if(len(noisy_im)==0):
            print("noisy_im is empty ...")

        return noisy_im, clean_im, clean_im_low_res, gmaps_median, noise_sigmas

    def get_cutout_range(self, data, rng):
        """
        Return s_x, s_y and s_t
        s_x, s_y stores the starting location of cutouts for every frame
        s_t stores which frames to cut
        """    
        t, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape
        
        # initial_s_t = rng.integers(0, t - ct) if t>ct else 0
        # initial_s_x = rng.integers(0, x - cx) if x>cx else 0
        # initial_s_y = rng.integers(0, y - cy) if y>cy else 0

        initial_s_t = np.random.randint(0, t - ct) if t>ct else 0
        initial_s_x = np.random.randint(0, x - cx) if x>cx else 0
        initial_s_y = np.random.randint(0, y - cy) if y>cy else 0
        
        s_x = np.zeros(ct, dtype=np.int16) + initial_s_x
        s_y = np.zeros(ct, dtype=np.int16) + initial_s_y
        s_t = np.zeros(ct, dtype=np.int16) + initial_s_t
        
        return s_x, s_y, s_t
          
    def do_cutout(self, data, s_x, s_y, s_t):
        """
        Cuts out the patches
        
        if on_upsampled_grid is True, data is on the upsampled grid (self.upsampling_ratio)
        """
        T, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        if T < ct or y < cy or x < cx:
            raise RuntimeError(f"File is borken because {t} is less than {ct} or {x} is less than {cx} or {y} is less than {cy}")

        result = np.zeros((ct, cx, cy), dtype=data.dtype)

        for t in range(ct):
            result[t, :, :] = data[s_t[t]+t, s_x[t]:s_x[t]+cx, s_y[t]:s_y[t]+cy]

        return result
        
    def random_flip(self, noisy, clean, rng = np.random):
        
        flip1 = rng.integers(0,2) > 0
        flip2 = rng.integers(0,2) > 0

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

        return flip(noisy), flip(clean)
            
    def find_sample(self, index):
        ind_file = 0

        for i in range(self.N_files):
            if(index>= self.start_samples[i] and index<self.end_samples[i]):
                ind_file = i
                ind_in_file = int(index - self.start_samples[i])//self.num_samples_per_file
                break

        return ind_file, ind_in_file

    def content_over_time(self, clean_im):

        T, C, H, W = clean_im.shape

        content_ratio = np.zeros(T)
        threshold = 0.05 if self.per_scaling else 0.02 if self.im_value_scale > 2000 else 0.1

        for t in range(T):

            image_t = clean_im[t, 0]
            content_ratio[t] = max(torch.count_nonzero(image_t > threshold) / torch.prod(torch.as_tensor(image_t.shape)), 0.01)

        return content_ratio
    
    def __len__(self):
        total_num_samples = 0
        for key in self.keys:
            total_num_samples += len(key)*self.num_samples_per_file
        return total_num_samples
            
    def __getitem__(self, idx):
        
        #print(f"{idx}")
        sample_list = []
        count_list = []
        found = False
        threshold = 0.05 if self.per_scaling else 0.02
        for i in range(10):
            ind_file, ind_in_file = self.find_sample(idx)
            if self.time_scale == 0:
                noisy_im, clean_im, clean_im_low_res, gmaps_median, noise_sigmas = self.load_one_sample(ind_file, self.keys[ind_file][ind_in_file])
            else:
                noisy_im, clean_im, clean_im_low_res, gmaps_median, noise_sigmas = self.load_one_sample_timed(ind_file, self.keys[ind_file][ind_in_file])

            sample = (noisy_im[0], clean_im[0], clean_im_low_res[0], 0, self.keys[ind_file][ind_in_file])

            area_threshold = 0.25 * torch.prod(torch.as_tensor(sample[1].shape))
            # different value check for percent vs constant scaling
            if (self.per_scaling and
                torch.count_nonzero(sample[1] > threshold) >= area_threshold):
                found = True
                break

            if (torch.count_nonzero(sample[1] > threshold) >= area_threshold):
                found = True
                break

            sample_list.append(sample)
            count_list.append(torch.count_nonzero(sample[1] > threshold) if self.per_scaling else torch.count_nonzero(sample[1] > threshold))

        if not found:
            sample = sample_list[count_list.index(max(count_list))]

        sample = (sample[0], sample[1], sample[2], self.content_over_time(sample[1]), sample[4])

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
def load_micro_data(config):
    """
    Defines how to load micro data

    @input:
        config: config file from main
    """

    ratio = [x/100 for x in config.ratio]

    h5files = []
    train_keys = []
    val_keys = []
    test_keys = []

    if len(config.train_files)==0:
        train_files = sorted(os.listdir(config.data_root))
    else:
        train_files = config.train_files

    for file in train_files:
        file_p = os.path.join(config.data_root, file)
        if not os.path.exists(file_p):
            logging.info(f"File not found: {file_p}")
            exit(-1)

        try:
            logging.info(f"reading from file: {file_p}")
            h5file = h5py.File(file_p,libver='latest',mode='r')
            keys = list(h5file.keys())
        except:
            logging.info(f"Error reading file: {file_p}")
            exit(-1)

        n = len(keys)
        if config.fine_samples > 0:
            assert(len(config.h5files) == 1), f"Can only finetune with one train dataset"
            h5files.append(h5file)
            train_keys.append(keys[:config.fine_samples])
            val_keys.append(keys[-5:])
            test_keys.append(keys[-5:])

            break   # since only one file no need for rest
        random.shuffle((keys))

        h5files.append(h5file)
        train_keys.append(keys[:int(ratio[0]*n)])
        val_keys.append(keys[int(ratio[0]*n):int((ratio[0]+ratio[1])*n)])
        test_keys.append(keys[int((ratio[0]+ratio[1])*n):int((ratio[0]+ratio[1]+ratio[2])*n)])

        # make sure there is no empty testing
        if len(test_keys[-1])==0:
            test_keys[-1] = keys[-1:]
        if len(val_keys[-1])==0:
            val_keys[-1] = test_keys[-1]

    cutout_shape=[config.height[-1], config.width[-1]]

    train_set = []

    for hw in zip(config.height, config.width):
        train_set.append(MicroscopyDatasetTrain(h5file=h5files, keys=train_keys,
                                            time_cutout=config.time,
                                            cutout_shape=hw,
                                            num_samples_per_file=8,
                                            rng = None,
                                            transform = None,
                                            per_scaling = False,
                                            im_value_scale = 4096,)
        )

    if config.train_only:
        val_set = []
        test_set = []
    else:
        val_set = [MicroscopyDatasetTrain(h5file=h5files, keys=val_keys,
                                        time_cutout=config.time,
                                        cutout_shape=cutout_shape,
                                        rng = None,
                                        transform = None,)]

        test_set = [MicroscopyDatasetTrain(h5file=h5files, keys=test_keys,
                                        time_cutout=config.time,
                                        cutout_shape=cutout_shape,
                                        rng = None,
                                        transform = None,)]

    if len(config.test_files)!=0:

        h5files = []
        keyss = []
        for file in config.test_files:

            try:
                logging.info(f"reading from file (test_case): {file}")
                h5file = h5py.File(file,libver='latest',mode='r')
                keys = list(h5file.keys())
            except:
                logging.info(f"Error reading file (test_case): {file}")
                exit(-1)

            h5files.append(h5file)
            keyss.append(keys)

        noisy_im = h5file[keys[0]]["noisy_im"][:]

        test_set = [MicroscopyDatasetTrain(h5file=h5files, keys=keyss,
                                    time_cutout=noisy_im.shape[0],
                                    cutout_shape=noisy_im.shape[1:],
                                    rng = None,
                                    transform = None,)]

    return train_set, val_set, test_set
