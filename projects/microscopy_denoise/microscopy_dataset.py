"""
Data utilities for Microscopy data.
Provides the torch dataset class for train and test
And a function to load the said classes with multiple h5files

Expected train h5file:
<file> ---> <key> ---> "/noisy_im"
                   |-> "/clean_im"

Expected test h5file:
<file> ---> <key> ---> "/noisy_im"
                   |-> "/clean_im"
"""

import os
import sys
import h5py
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *

# -------------------------------------------------------------------------------------------------
# train dataset class

def load_images_from_h5file(h5file, keys, scaling_type="per", scaling_vals=[0,100], max_load=100000):
    """
    Load images from microscopy h5 file objects.
    Either the complete image or the path to it.
    @args:
        - h5file (h5File list): list of h5files to load images from
        - keys (key list list): list of list of keys. One for each h5file
        - scaling_type ("val" or "per"): "val" for scaling with a static value or "per" for scaling with a percentile
        - scaling_vals (int 2-tuple): min max values to scale with respect to the scaling type
        - max_load (int): max number of images to load
    @rets:
        - images (4-tuple list): list of images
            - 1st entry: noisy image nd.array or address in h5file
            - 2nd entry: clean image nd.array or address in h5file
            - 3rd entry: int index of h5file
            - 4th entry: str name of the image
    """
    images = []

    num_loaded = 0
    for i in range(len(h5file)):

        with tqdm(total=len(keys[i])) as pbar:
            for n, key in enumerate(keys[i]):
                if num_loaded < max_load:
                    noisy_im = np.array(h5file[i][key+"/noisy_im"])
                    clean_im = np.array(h5file[i][key+"/clean_im"])
                    if scaling_type=="per":
                        noisy_im = normalize_image(noisy_im, percentiles=scaling_vals)
                        clean_im = normalize_image(clean_im, percentiles=scaling_vals)
                    else:
                        noisy_im = normalize_image(noisy_im, values=scaling_vals)
                        clean_im = normalize_image(clean_im, values=scaling_vals)

                    images.append([np.array(h5file[i][key+"/noisy_im"]), np.array(h5file[i][key+"/clean_im"]), i, key])
                    num_loaded += 1
                else:
                    images.append([key+"/noisy_im", key+"/clean_im", i, key])

                if n and n%10 == 0:
                    pbar.update(10)
                    pbar.set_description_str(f"{h5file}, {n} in {len(keys[i])}, total {len(images)}")

    return images

class MicroscopyDatasetTrain():
    """
    Train dataset for Microscopy.
    Makes a cutout of original image pair to be used during training cycle.
    Since the content is sparse, "samples_per_image" number of samples are taken from same image every epoch.
    """
    def __init__(self, h5file, keys, max_load=10000,
                    time_cutout=30, cutout_shape=[64, 64], samples_per_image=8,
                    scaling_type="val", scaling_vals=[0,65536],
                    valu_thres=0.002, area_thres=0.25):
        """
        Initilize the dataset

        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for each h5file
            - max_load (int): max number of images to load during init
            - time_cutout (int): cutout size in time dimension
            - cutout_shape (int list): 2 values for patch cutout shape
            - samples_per_image (int): samples to take from a single image per epoch
            - scaling_type ("val" or "per"): "val" for scaling with a static value or "per" for scaling with a percentile
            - scaling_vals (int 2-tuple): min max values to scale with respect to the scaling type
            - valu_thres (float): threshold of pixel value between background and foreground
            - area_thres (float): percentage threshold of area that needs to be foreground
        """
        self.h5file = h5file
        self.keys = keys
        self.N_files = len(self.keys)

        self.time_cutout = time_cutout
        self.cutout_shape = cutout_shape

        self.samples_per_image = samples_per_image
        self.scaling_type = scaling_type
        self.scaling_vals = scaling_vals
        self.valu_thres = valu_thres
        self.area_thres = area_thres

        self.images = load_images_from_h5file(h5file, keys, scaling_type=scaling_type, scaling_vals=scaling_vals,max_load=max_load)

    def load_one_sample(self, i):
        """
        Load one sample from the images

        @args:
            - i (int): index of the image to load
        @rets:
            - noisy_cutout, clean_cutout (5D torch.Tensors): the pair of images cutouts
            - noisy_im_name (str): name of the image
        """
        noisy_im = self.images[i][0]
        clean_im = self.images[i][1]
        noisy_im_name = self.images[i][3]

        if not isinstance(noisy_im, np.ndarray):
            ind = self.images[i][2]
            key_noisy = self.images[i][0]
            key_clean = self.images[i][1]
            noisy_im = np.array(self.h5file[ind][key_noisy])
            clean_im = np.array(self.h5file[ind][key_clean])
            if self.scaling_type=="per":
                noisy_im = normalize_image(noisy_im, percentiles=self.scaling_vals)
                clean_im = normalize_image(clean_im, percentiles=self.scaling_vals)
            else:
                noisy_im = normalize_image(noisy_im, values=self.scaling_vals)
                clean_im = normalize_image(clean_im, values=self.scaling_vals)

        if noisy_im.ndim == 2: noisy_im = noisy_im[np.newaxis,:,:]
        if clean_im.ndim == 2: clean_im = clean_im[np.newaxis,:,:]

        # pad symmetrically if not enough images in the time dimension
        if noisy_im.shape[0] < self.time_cutout:
            noisy_im = np.pad(noisy_im, ((0,self.time_cutout - noisy_im.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_im = np.pad(clean_im, ((0,self.time_cutout - clean_im.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_im, clean_im = self.random_flip(noisy_im, clean_im)

        if noisy_im.shape[1] < self.cutout_shape[0]:
            noisy_im = np.pad(noisy_im, ((0, 0), (0,self.cutout_shape[0] - noisy_im.shape[1]),(0,0)), 'symmetric')
            clean_im = np.pad(clean_im, ((0, 0), (0,self.cutout_shape[0] - clean_im.shape[1]),(0,0)), 'symmetric')

        if noisy_im.shape[2] < self.cutout_shape[1]:
            noisy_im = np.pad(noisy_im, ((0,0), (0,0), (0,self.cutout_shape[1] - noisy_im.shape[2])), 'symmetric')
            clean_im = np.pad(clean_im, ((0,0), (0,0), (0,self.cutout_shape[1] - clean_im.shape[2])), 'symmetric')

        # define a set of cut range
        s_x, s_y, s_t = self.get_cutout_range(noisy_im)

        noisy_cutout = self.do_cutout(noisy_im, s_x, s_y, s_t)[np.newaxis,:,:,:]
        clean_cutout = self.do_cutout(clean_im, s_x, s_y, s_t)[np.newaxis,:,:,:]

        noisy_cutout = torch.from_numpy(noisy_cutout.astype(np.float32))
        clean_cutout = torch.from_numpy(clean_cutout.astype(np.float32))

        return noisy_cutout, clean_cutout, noisy_im_name

    def get_cutout_range(self, data):
        """
        Return s_x, s_y and s_t
        The starting location of the patch
        """
        t, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        s_t = np.random.randint(0, t - ct + 1)
        s_x = np.random.randint(0, x - cx + 1)
        s_y = np.random.randint(0, y - cy + 1)

        return s_x, s_y, s_t

    def do_cutout(self, data, s_x, s_y, s_t):
        """
        Cuts out the patches
        """
        T, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        if T < ct or y < cy or x < cx:
            raise RuntimeError(f"File is borken because {T} is less than {ct} or {x} is less than {cx} or {y} is less than {cy}")

        return data[s_t:s_t+ct, s_x:s_x+cx, s_y:s_y+cy]

    def random_flip(self, noisy, clean):
        """
        Randomly flips the noisy and clean image
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

        return flip(noisy), flip(clean)

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.images)*self.samples_per_image

    def __getitem__(self, idx):
        """
        Given index(idx) retreive the noisy clean image pair.
        For the given image tries 10 times to find a suitable patch wrt area and value thresholds.
        If found, returns that patch otherwise returns the one with the highest foreground content.
        @args:
            - idx (int): the index in the dataset
        @rets:
            - noisy_im, clean_im (5D torch.Tensors): the noisy and clean pair
            - noisy_im_name (str): name of the image
        """

        sample_list = []
        counts_list = []
        found = False

        # 10 tries to find a suitable sample
        for i in range(10):

            noisy_im, clean_im, noisy_im_name = self.load_one_sample(idx//self.samples_per_image) # the actual index of the image

            # The foreground content check
            valu_score = torch.count_nonzero(clean_im > self.valu_thres)
            area_score = self.area_thres * clean_im.numel()
            if (valu_score >= area_score):
                found = True
                break

            sample_list.append((noisy_im, clean_im, noisy_im_name))
            counts_list.append(valu_score)

        # if failed, find the one with the highest foreground counts
        if not found:
            noisy_im, clean_im, noisy_im_name = sample_list[counts_list.index(max(counts_list))]

        return noisy_im, clean_im, noisy_im_name

class MicroscopyDatasetTest():
    """
    Dataset for testing Microscopy.
    Returns the complete images with proper scaling for inference.
    """
    def __init__(self, h5file, keys, max_load=10000,
                    scaling_type="val", scaling_vals=[0,65536]):
        """
        Initilize the dataset

        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for each h5file
            - max_load (int): max number of images to load during init
            - scaling_type ("val" or "per"): "val" for scaling with a static value or "per" for scaling with a percentile
            - scaling_vals (int 2-tuple): min max values to scale with respect to the scaling type
        """
        self.h5file = h5file
        self.keys = keys
        self.N_files = len(self.keys)

        self.scaling_type = scaling_type
        self.scaling_vals = scaling_vals

        self.images = load_images_from_h5file(h5file, keys, scaling_type=scaling_type, scaling_vals=scaling_vals,max_load=max_load)

    def load_one_sample(self, i):
        """
        Load one sample from the images

        @args:
            - i (int): index of the image to load
        @rets:
            - noisy_cutout, clean_cutout (5D torch.Tensors): the pair of images cutouts
            - noisy_im_name (str): name of the image
        """
        noisy_im = self.images[i][0]
        clean_im = self.images[i][1]
        noisy_im_name = self.images[i][3]

        if not isinstance(noisy_im, np.ndarray):
            ind = self.images[i][2]
            key_noisy = self.images[i][0]
            key_clean = self.images[i][1]
            noisy_im = np.array(self.h5file[ind][key_noisy])
            clean_im = np.array(self.h5file[ind][key_clean])
            if self.scaling_type=="per":
                noisy_im = normalize_image(noisy_im, percentiles=self.scaling_vals)
                clean_im = normalize_image(clean_im, percentiles=self.scaling_vals)
            else:
                noisy_im = normalize_image(noisy_im, values=self.scaling_vals)
                clean_im = normalize_image(clean_im, values=self.scaling_vals)

        if noisy_im.ndim == 2: noisy_im = noisy_im[np.newaxis,:,:]
        if clean_im.ndim == 2: clean_im = clean_im[np.newaxis,:,:]

        noisy_im = noisy_im[np.newaxis,:,:,:]
        clean_im = clean_im[np.newaxis,:,:,:]

        noisy_im = torch.from_numpy(noisy_im.astype(np.float32))
        clean_im = torch.from_numpy(clean_im.astype(np.float32))

        return noisy_im, clean_im, noisy_im_name

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Given index(idx) retreive the noisy clean image pair.
        @args:
            - idx (int): the index in the dataset
        @rets:
            - noisy_im, clean_im (5D torch.Tensors): the noisy and clean pair
            - noisy_im_name (str): name of the image
        """
        noisy_im, clean_im, noisy_im_name = self.load_one_sample(idx)

        return noisy_im, clean_im, noisy_im_name

def load_microscopy_data_all(config):
    """
    Defines how to load microscopy data
    Loads the given ratio of the given h5files

    @args:
        - config (Namespace): runtime namespace for setup
    @args (from config):
        - ratio (int list): ratio to divide the given train dataset
            3 integers for ratio between train, val and test. Can be [100,0,0]
        - data_root (str): main folder of the data
        - train_files (str list): names of h5files in dataroot for training
            if empty entire dataroot is used for training
        - test_files (str list): names of h5files in dataroot for testing
            if empty test cases are created using ratio from trainset
        - micro_time (int): cutout size in time dimension
        - micro_height (int list): different height cutouts
        - micro_width (int list): different width cutouts
    """

    c = config # shortening due to numerous uses

    ratio = [x/100 for x in c.ratio]

    h5files = []
    train_keys = []
    val_keys = []
    test_keys = []

    if len(c.train_files)==0:
        logging.info(f"No train files specified, loading entire dataroot dir")
        train_files = sorted(os.listdir(c.data_root))
    else:
        train_files = c.train_files

    for file in train_files:
        file_p = os.path.join(c.data_root, file)
        if not os.path.exists(file_p):
            raise RuntimeError(f"File not found: {file_p}")

        logging.info(f"reading from file: {file_p}")
        h5file = h5py.File(file_p,libver='earliest',mode='r')
        keys = list(h5file.keys())

        n = len(keys)
        random.shuffle((keys))

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
        "time_cutout" : c.micro_time,
        "samples_per_image" : c.samples_per_image,
        "scaling_type" : c.scaling_type,
        "scaling_vals" : c.scaling_vals,
        "valu_thres" : c.valu_thres,
        "area_thres" : c.area_thres
    }

    train_set = []

    if c.max_load<=0:
        logging.info(f"Data will not be pre-read ...")

    for (i, h_file) in enumerate(h5files):
        logging.info(f"--> loading data from file: {h_file} for {len(train_keys[i])} entries ...")
        images = load_images_from_h5file([h_file], [train_keys[i]], max_load=c.max_load)
        for hw in zip(c.micro_height, c.micro_width):
            train_set.append(MicroscopyDatasetTrain(h5file=[h_file], keys=[train_keys[i]], max_load=-1,
                                                    cutout_shape=hw, **kwargs))
            train_set[-1].images = images[:c.train_samples] if c.train_samples>0 else images

    # kwargs for val set
    kwargs["cutout_shape"] = (c.micro_height[-1], c.micro_width[-1])
    kwargs["max_load"] = c.max_load

    if c.train_only:
        val_set, test_set = [], []

    elif len(c.test_files)!=0:
        # Test case given so use that for test and val
        h5files = []
        val_keys = []
        test_keys = []

        for file in c.test_files:
            file_p = file

            if not os.path.exists(file_p):
                file_p = os.path.join(c.data_root, file)
                if not os.path.exists(file_p):
                    raise RuntimeError(f"File not found: {file} OR {file_p}")

            logging.info(f"reading from file (test_case): {file_p}")
            h5file = h5py.File(file_p,libver='earliest',mode='r')
            keys = list(h5file.keys())

            h5files.append(h5file)
            test_keys.append(keys)

            val_keys.append(keys[:int(ratio[1]*n)])

            if len(val_keys[-1])==0:
                val_keys[-1] = keys[-1:]

        # val set is cutouts of parts of test set + two complete images of test set
        val_set = [MicroscopyDatasetTrain(h5file=h5files, keys=val_keys, **kwargs)]

        test_set = [MicroscopyDatasetTest(h5file=h5files, keys=test_keys, max_load=c.max_load,
                                            scaling_type=c.scaling_type, scaling_vals=c.scaling_vals)]

    else:
        # No test case given, use some of the train set
        val_set = [MicroscopyDatasetTrain(h5file=h5files, keys=val_keys, **kwargs)]

        test_set = [MicroscopyDatasetTest(h5file=h5files, keys=test_keys, max_load=c.max_load,
                                            scaling_type=c.scaling_type, scaling_vals=c.scaling_vals)]

    return train_set, val_set, test_set
