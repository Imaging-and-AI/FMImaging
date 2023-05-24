"""
Data utilities for MRI data.
Provides the torch dataset class for traind and test
And a function to load the said classes with multiple h5files

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

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from noise_augmentation import *

# -------------------------------------------------------------------------------------------------
# train dataset class

def load_images_from_h5file(h5file, keys, max_load=100000):
        """
        Load images from h5 file objects
        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for each h5file
            
        @outputs:
            - images : list of image and gmap pairs as a list
        """
        images = []

        num_loaded = 0
        for i in range(len(h5file)):
            # with tqdm(total=len(keys[i])) as pbar:
            #     for n, key in enumerate(keys[i]):
            #         if num_loaded < max_load:
            #             images.append([np.array(h5file[i][key+"/image"]), np.array(h5file[i][key+"/gmap"]), i])
            #         else:
            #             images.append([key+"/image", key+"/gmap", i])
                        
            #         pbar.update(1)
            #         pbar.set_description_str(f"{h5file}, {n} in {len(keys[i])}, total {len(images)}")

            #     pbar.close()
            
            if max_load<=0:
                logging.info(f"{h5file[i]}, data will not be pre-read ...")
            
            with tqdm(total=len(keys[i])) as pbar:
                for n, key in enumerate(keys[i]):
                    if num_loaded < max_load:
                        images.append([np.array(h5file[i][key+"/image"]), np.array(h5file[i][key+"/gmap"]), i])
                        num_loaded += 1
                    else:
                        images.append([key+"/image", key+"/gmap", i])
                        
                    if n>0 and n%100 == 0:
                        pbar.update(100)
                        pbar.set_description_str(f"{h5file}, {n} in {len(keys[i])}, total {len(images)}")

        return images
    
    
class MRIDenoisingDatasetTrain():
    """
    Dataset for MRI denoising.
    The extracted patch maintains the strict temporal consistency
    This dataset is for 2D+T training, where the temporal redundancy is strong
    """
    def __init__(self, h5file, keys, data_type, max_load=10000,
                    time_cutout=30, cutout_shape=[64, 64], use_gmap=True,
                    use_complex=True, min_noise_level=1.0, max_noise_level=6.0,
                    matrix_size_adjust_ratio=[0.5, 0.75, 1.0, 1.25, 1.5],
                    kspace_filter_sigma=[0.8, 1.0, 1.5, 2.0, 2.25],
                    pf_filter_ratio=[1.0, 0.875, 0.75, 0.625],
                    phase_resolution_ratio=[1.0, 0.75, 0.65, 0.55],
                    readout_resolution_ratio=[1.0, 0.75, 0.65, 0.55],
                    cutout_jitter=[-1, 0.5, 0.75, 1.0],
                    cutout_shuffle_time=True,
                    num_patches_cutout=8,
                    patches_shuffle=False):
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
            - use_gmap (bool): whether to load and return gmap
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
        """
        assert data_type=="2d" or data_type=="2dt" or data_type=="3d",\
            f"Data type not implemented: {data_type}"
        self.data_type = data_type

        self.h5file = h5file
        self.keys = keys
        self.max_load = max_load
        
        self.time_cutout = time_cutout
        if self.data_type=="2d": self.time_cutout = 1
        self.cutout_shape = cutout_shape

        self.use_gmap = use_gmap
        self.use_complex = use_complex

        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.matrix_size_adjust_ratio = matrix_size_adjust_ratio
        self.kspace_filter_sigma = kspace_filter_sigma
        self.pf_filter_ratio = pf_filter_ratio
        self.phase_resolution_ratio = phase_resolution_ratio
        self.readout_resolution_ratio = readout_resolution_ratio

        self.cutout_jitter = cutout_jitter
        self.cutout_shuffle_time = cutout_shuffle_time
        self.num_patches_cutout = num_patches_cutout
        self.patches_shuffle = patches_shuffle

        self.images = load_images_from_h5file(h5file, keys, max_load=self.max_load)
                
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
        
        if not isinstance(data, np.ndarray):
            ind = self.images[i][2]
            key_image = self.images[i][0]
            key_gmap = self.images[i][1]
            data = np.array(self.h5file[ind][key_image])
            gmaps = np.array(self.h5file[ind][key_gmap])
        
        if data.ndim == 2: data = data[np.newaxis,:,:]
        gmap = self.load_gmap(gmaps, i, random_factor=-1)
        
        # pad symmetrically if not enough images in the time dimension
        if data.shape[0] < self.time_cutout:
            data = np.pad(data, ((0,self.time_cutout - data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        data, gmap = self.random_flip(data, gmap)

        assert data.shape[1] == gmap.shape[0] and data.shape[2] == gmap.shape[1]

        # random increase matrix size or reduce matrix size
        if(np.random.random()<0.5):
            matrix_size_adjust_ratio = self.matrix_size_adjust_ratio[np.random.randint(0, len(self.matrix_size_adjust_ratio))]
            data_adjusted = np.array([adjust_matrix_size(img, matrix_size_adjust_ratio) for img in data])
            gmap_adjusted = cv2.resize(gmap, dsize=(data_adjusted.shape[2], data_adjusted.shape[1]), interpolation=cv2.INTER_LINEAR)
            assert data_adjusted.shape[1] == gmap_adjusted.shape[0] and data_adjusted.shape[2] == gmap_adjusted.shape[1]
            data = data_adjusted
            gmap = gmap_adjusted

        if data.shape[1] < self.cutout_shape[0]:
            data = np.pad(data, ((0, 0), (0,self.cutout_shape[0] - data.shape[1]),(0,0)), 'symmetric')
            gmap = np.pad(gmap, ((0,self.cutout_shape[0] - gmap.shape[0]),(0,0)), 'symmetric')

        if data.shape[2] < self.cutout_shape[1]:
            data = np.pad(data, ((0,0), (0,0), (0,self.cutout_shape[1] - data.shape[2])), 'symmetric')
            gmap = np.pad(gmap, ((0,0), (0,self.cutout_shape[1] - gmap.shape[1])), 'symmetric')

        T, RO, E1 = data.shape

        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # create noise
            ratio_RO = self.readout_resolution_ratio[np.random.randint(0, len(self.readout_resolution_ratio))]
            ratio_E1 = self.phase_resolution_ratio[np.random.randint(0, len(self.phase_resolution_ratio))]
            nn, noise_sigma = generate_3D_MR_correlated_noise(T=T, RO=RO, E1=E1, REP=1,
                                                                min_noise_level=self.min_noise_level,
                                                                max_noise_level=self.max_noise_level,
                                                                kspace_filter_sigma=self.kspace_filter_sigma,
                                                                pf_filter_ratio=self.pf_filter_ratio,
                                                                phase_resolution_ratio=[ratio_E1],
                                                                readout_resolution_ratio=[ratio_RO],
                                                                verbose=False)
            # apply gmap
            nn *= gmap

            # add noise to complex image and scale
            noisy_data = data + nn

            # scale the data
            data /= noise_sigma
            noisy_data /= noise_sigma

            gmap = np.repeat(gmap[None,:,:], T, axis=0)

            # cut out the patch on the original grid
            s_x, s_y, s_t = self.get_cutout_range(data)

            if(self.use_complex):
                patch_data = self.do_cutout(data, s_x, s_y, s_t)[:,np.newaxis,:,:]
                patch_data_with_noise = self.do_cutout(noisy_data, s_x, s_y, s_t)[:,np.newaxis,:,:]

                cutout = np.concatenate((patch_data.real, patch_data.imag),axis=1)
                cutout_train = np.concatenate((patch_data_with_noise.real, patch_data_with_noise.imag),axis=1)
            else:
                cutout = np.abs(self.do_cutout(data, s_x, s_y, s_t))[:,np.newaxis,:,:]
                cutout_train = np.abs(self.do_cutout(noisy_data, s_x, s_y, s_t))[:,np.newaxis,:,:]

            gmap_cutout = self.do_cutout(gmap, s_x, s_y, s_t)[:,np.newaxis,:,:]

            if self.data_type=="2d":
                C = cutout.shape[1]
                pad_H = (-1*cutout_train.shape[2])%self.cutout_shape[0]
                pad_W = (-1*cutout_train.shape[3])%self.cutout_shape[1]

                cutout = np.pad(cutout, ((0,0),(0, 0), (0,pad_H),(0,pad_W)), 'symmetric')
                cutout_train = np.pad(cutout_train, ((0,0),(0, 0), (0,pad_H),(0,pad_W)), 'symmetric')
                gmap_cutout = np.pad(gmap_cutout, ((0,0),(0, 0), (0,pad_H),(0,pad_W)), 'symmetric')

                cutout = view_as_blocks(cutout, (1,C,*self.cutout_shape))
                cutout = cutout.reshape(-1,*cutout.shape[-3:])
                cutout_train = view_as_blocks(cutout_train, (1,C,*self.cutout_shape))
                cutout_train = cutout_train.reshape(-1,*cutout_train.shape[-3:])
                gmap_cutout = view_as_blocks(gmap_cutout, (1,1,*self.cutout_shape))
                gmap_cutout = gmap_cutout.reshape(-1,*gmap_cutout.shape[-3:])


                if self.patches_shuffle:
                    t_indexes = np.arange(cutout.shape[0])
                    np.random.shuffle(t_indexes)

                    cutout = np.take(cutout, t_indexes, axis=0)[:self.num_patches_cutout]
                    cutout_train = np.take(cutout_train, t_indexes, axis=0)[:self.num_patches_cutout]
                    gmap_cutout = np.take(gmap_cutout, t_indexes, axis=0)[:self.num_patches_cutout]
                else:
                    start_t = np.random.randint(0,max(cutout.shape[0] - self.num_patches_cutout, 1))

                    cutout = cutout[start_t:start_t+self.num_patches_cutout]
                    cutout_train = cutout_train[start_t:start_t+self.num_patches_cutout]
                    gmap_cutout = gmap_cutout[start_t:start_t+self.num_patches_cutout]

                pad_T = (-1*cutout_train.shape[0])%self.num_patches_cutout
                cutout = np.pad(cutout, ((0,pad_T),(0,0),(0,0),(0,0)), 'symmetric')
                cutout_train = np.pad(cutout_train, ((0,pad_T),(0,0),(0,0),(0,0)), 'symmetric')
                gmap_cutout = np.pad(gmap_cutout, ((0,pad_T),(0,0),(0,0),(0,0)), 'symmetric')

            if(self.data_type=="3d" and self.cutout_shuffle_time):
                # perform shuffle along time
                t_indexes = np.arange(cutout.shape[0])
                np.random.shuffle(t_indexes)

                np.take(cutout, t_indexes, axis=0, out=cutout)
                np.take(cutout_train, t_indexes, axis=0, out=cutout_train)
                np.take(gmap_cutout, t_indexes, axis=0, out=gmap_cutout)

            train_noise = np.concatenate([cutout_train, gmap_cutout], axis=1)

            noisy_im = torch.from_numpy(train_noise.astype(np.float32))
            clean_im = torch.from_numpy(cutout.astype(np.float32))
            gmaps_median = torch.tensor(np.median(gmap_cutout))
            noise_sigmas = torch.tensor(noise_sigma)

        return noisy_im, clean_im, gmaps_median, noise_sigmas

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

    def do_cutout(self, data, s_x, s_y, s_t):
        """
        Cuts out the jittered patches across a random time interval
        """
        t, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        if t < ct or y < cy or x < cx:
            raise RuntimeError(f"File is borken because {t} is less than {ct} or {x} is less than {cx} or {y} is less than {cy}")

        result = np.zeros((ct, cx, cy), dtype=data.dtype)
        result = data[s_t[0]:s_t[0]+ct, s_x[0]:s_x[0]+cx, s_y[0]:s_y[0]+cy]

        if self.data_type=="2d":
            result = np.zeros((1, x, y), dtype=data.dtype)
            result[0] = data[s_t[0]]

        if self.data_type=="3d":
            for t in range(ct):
                result[t, :, :] = data[s_t[t]+t, s_x[t]:s_x[t]+cx, s_y[t]:s_y[t]+cy]

        return result

    def load_gmap(self, gmaps, i, random_factor=-1):
        """
        Loads a random gmap for current index
        """           
        if(gmaps.ndim==2):
            gmaps = np.expand_dims(gmaps, axis=0)

        factors = gmaps.shape[0]
        if(random_factor<0):
            random_factor = np.random.randint(0, factors)

        return gmaps[random_factor, :,:]
        
    def random_flip(self, data, gmap):
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

        return flip(data), flip(gmap)

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

class MRIDenoisingDatasetTest():
    """
    Dataset for MRI denoising testing.
    Returns full images. No cutouts.
    """
    def __init__(self, h5file, keys, use_gmap=True, use_complex=True):
        """
        Initilize the denoising dataset
        Loads and stores everything
        h5files should have the following strucutre
        file --> <key> --> "noisy"+"clean"+"gmap"+"noise_sigma"
        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for every h5file
            - use_gmap (bool): whether to load and return gmap
            - use_complex (bool): whether to return complex image
        """
        self.use_gmap = use_gmap
        self.use_complex = use_complex

        self.images = []

        for i in range(len(h5file)):
            self.images.extend([(np.array(h5file[i][key+"/noisy"]),
                                    np.array(h5file[i][key+"/clean"]),
                                    np.array(h5file[i][key+"/gmap"]),
                                    np.array(h5file[i][key+"/noise_sigma"])) for key in keys[i]])

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
        noisy = (self.images[i][0])
        clean = (self.images[i][1])
        if noisy.ndim==2:
            noisy = noisy[np.newaxis,np.newaxis,:,:]
            clean = clean[np.newaxis,np.newaxis,:,:]
        else: # ndim==3
            noisy = noisy[:,np.newaxis,:,:]
            clean = clean[:,np.newaxis,:,:]
        gmap = self.images[i][2]
        noise_sigma = self.images[i][3]

        assert gmap.ndim==2, f"gmap for testing should only be 2 dimensional"

        if(self.use_complex):
            noisy = np.concatenate((noisy.real, noisy.imag),axis=1)
            clean = np.concatenate((clean.real, clean.imag),axis=1)
        else:
            noisy = np.abs(noisy)
            clean = np.abs(clean)

        gmap = np.repeat(gmap[None,:,:], noisy.shape[0], axis=0)[:,np.newaxis,:,:]
        noisy = np.concatenate([noisy, gmap], axis=1)

        noisy_im = torch.from_numpy(noisy.astype(np.float32))
        clean_im = torch.from_numpy(clean.astype(np.float32))
        gmaps_median = torch.tensor(np.median(gmap))
        noise_sigmas = torch.tensor(noise_sigma)

        return noisy_im, clean_im, gmaps_median, noise_sigmas

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
        - data_root (str): main folder of the data
        - train_files (str list): names of h5files in dataroot for training
        - test_files (str list): names of h5files in dataroot for testing
        - train_data_types ("2d"|"2dt"|"3d" list): type of each train data file
        - test_data_types ("2d"|"2dt"|"3d" list): type of each test data file
        - time (int): cutout size in time dimension
        - height (int list): different height cutouts
        - width (int list): different width cutouts
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

    train_paths = [os.path.join(c.data_root, path_x) for path_x in c.train_files]

    for file in train_paths:
        if not os.path.exists(file):
            raise RuntimeError(f"File not found: {file}")

        logging.info(f"reading from file: {file}")
        h5file = h5py.File(file, libver='earliest', mode='r')
        keys = list(h5file.keys())

        n = len(keys)

        h5files.append(h5file)
        train_keys.append(keys[:int(ratio[0]*n)])
        val_keys.append(keys[int(ratio[0]*n):int((ratio[0]+ratio[1])*n)])
        test_keys.append(keys[int((ratio[0]+ratio[1])*n):int((ratio[0]+ratio[1]+ratio[2])*n)])

        # make sure there is no empty testing
        if len(val_keys[-1])==0:
            val_keys[-1] = keys[-1:]
        if len(test_keys[-1])==0:
            test_keys[-1] = keys[-1:]

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
    }

    train_set = []
    
    for (i, h_file) in enumerate(h5files):
        logging.info(f"--> loading data from file: {h_file} for {len(train_keys[i])} entries ...")
        images = load_images_from_h5file([h_file], [train_keys[i]], max_load=c.max_load)
        for hw in zip(c.height, c.width):        
            train_set.append(MRIDenoisingDatasetTrain(h5file=[h_file], keys=[train_keys[i]], max_load=-1, data_type=c.train_data_types[i], cutout_shape=hw, **kwargs))
            train_set[-1].images = images
        
    if c.test_files is None or c.test_files[0] is None: # no test case given so use some from train data
        val_set = [MRIDenoisingDatasetTrain(h5file=[h_file], keys=[val_keys[i]], max_load=c.max_load, 
                                            data_type=c.train_data_types[i], cutout_shape=[c.height[-1], c.width[-1]], **kwargs)
                                                for (i,h_file) in enumerate(h5files)]

        test_set = [MRIDenoisingDatasetTrain(h5file=[h_file], keys=[test_keys[i]], max_load=c.max_load, 
                                             data_type=c.train_data_types[i], cutout_shape=[c.height[-1], c.width[-1]], **kwargs)
                                                for (i,h_file) in enumerate(h5files)]
    else: # test case is given. take part of it as val set
        test_h5files = []
        test_paths = [os.path.join(c.data_root, path_x) for path_x in c.test_files]

        cutout_shape=[c.height[-1], c.width[-1]]

        for i, file in enumerate(test_paths):
            if not os.path.exists(file):
                raise RuntimeError(f"File not found: {file}")

            logging.info(f"reading from file: {file}")
            h5file = h5py.File(file, libver='earliest', mode='r')
            keys = list(h5file.keys())

            test_h5files.append((h5file,keys))

        test_set = [MRIDenoisingDatasetTest([h_file], keys=[t_keys], use_complex=c.complex_i)\
                        for (h_file,t_keys) in test_h5files]

        val_set = []
        val_len = 0
        val_len_lim = 8
        per_file = 1 if len(test_h5files)>val_len_lim else val_len_lim//len(test_h5files)
        # take 8 samples through all files for val set
        for h_file,t_keys in test_h5files:
            val_set.append(MRIDenoisingDatasetTest([h_file], keys=[t_keys[:per_file]], use_complex=c.complex_i))
            val_len += per_file
            if val_len > val_len_lim:
                break

    return train_set, val_set, test_set
