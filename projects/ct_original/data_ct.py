"""
Data utilities for CT data.
Provides the torch dataset class for train and test
And a function to load the said classes with multiple h5files

Expected train h5file:
<file> ---> <key> ---> "{scan_level}_{scan_type}"
Required:
<file> ---> <key> ---> "SOC_AiCE"

Expected test h5file:
<file> ---> <key> ---> "{scan_level}_{scan_type}"
Required:
<file> ---> <key> ---> "SOC_AiCE"
"""

import os
import sys
import h5py
import torch
import random
import logging
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *

# Hardcoded naming convention
scan_level = ["SOC", "Ag_140mA", "Ag_080mA", "Ag_040mA", "Ag_0SD40", "Cu_020mA"]
scan_types = ["AiCE", "FBP"]
name_prefs = [f"{x}_{y}" for x,y in itertools.product(scan_level, scan_types)]

clean_name = "SOC_AiCE"
noisy_names = name_prefs[1:]

# -------------------------------------------------------------------------------------------------
# some helper(s)

def load_images_from_h5file(h5files, keys, max_load=100000):
    """
    Load images from ct h5 file objects.
    Either the complete image or the path to it.
    @args:
        - h5file (h5File list): list of h5files to load images from
        - keys (key list list): list of list of keys. One for each h5file
        - max_load (int): max number of images to load
    @rets:
        - images (2-tuple dict dict list): list of dict (per h5file) of dict (per key_1) of 2-tuples
            - 1st entry: noisy image nd.array or address in h5file
            - 2nd entry: index of h5file
    """
    images = []

    num_loaded = 0
    for i in range(len(h5files)):

        with tqdm(total=len(keys[i]), bar_format=get_bar_format()) as pbar:
            for n, key_1 in enumerate(keys[i]):

                noisy_im_list = []

                found_clean = False
                for key_2 in h5files[i][key_1]:

                    complete_key = f"{key_1}/{key_2}"
                    image_2_save = [complete_key, complete_key]
                    if num_loaded < max_load:
                        image_2_save = [np.array(h5files[i][complete_key]), complete_key]
                        num_loaded += 1

                    if key_2 == clean_name:
                        clean_im = image_2_save
                        found_clean = True
                    else:
                        noisy_im_list.append(image_2_save)

                assert found_clean, f"clean image not found in {h5files[i]}/{key_1}"
                images.append((noisy_im_list, clean_im, i))

                pbar.update(1)
                pbar.set_description_str(f"{h5files}, {n} in {len(keys[i])}, total {len(images)}")

    return images

# -------------------------------------------------------------------------------------------------
# train dataset class

class CtDatasetTrain():
    """
    Train dataset for ct.
    Makes a cutout of original image pair to be used during training cycle.
    Since the image size is big, "samples_per_image" number of samples are taken from same image every epoch.
    Randomly select a noisy image each iteration.
    """
    def __init__(self, h5file, keys, max_load=10000,
                    time_cutout=30, cutout_shape=[64, 64], samples_per_image=32):
        """
        Initilize the dataset

        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for each h5file
            - max_load (int): max number of images to load during init
            - time_cutout (int): cutout size in time dimension
            - cutout_shape (int list): 2 values for patch cutout shape
            - samples_per_image (int): samples to take from a single image per epoch
        """
        self.h5file = h5file
        self.keys = keys

        self.time_cutout = time_cutout
        self.cutout_shape = cutout_shape

        self.samples_per_image = samples_per_image

        self.images = load_images_from_h5file(h5file, keys, max_load=max_load)

    def load_one_sample(self, i):
        """
        Load one sample from the images

        @args:
            - i (int): index of the image to load
        @rets:
            - noisy_cutout, clean_cutout (5D torch.Tensors): the pair of images cutouts
        """
        noisy_im, noisy_im_name = self.select_random_noisy(self.images[i][0])
        clean_im = self.images[i][1][0]

        if not isinstance(noisy_im, np.ndarray):
            ind = self.images[i][2]
            noisy_im = np.array(self.h5file[ind][noisy_im])
            clean_im = np.array(self.h5file[ind][clean_im])

        if noisy_im.ndim == 2: noisy_im = noisy_im[np.newaxis,:,:]
        if clean_im.ndim == 2: clean_im = clean_im[np.newaxis,:,:]

        min_t = min(noisy_im.shape[0], clean_im.shape[0])

        noisy_im = noisy_im[:min_t,:,:]
        clean_im = clean_im[:min_t,:,:]

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

        noisy_cutout = self.do_cutout(noisy_im, s_x, s_y, s_t)[:,np.newaxis,:,:]
        clean_cutout = self.do_cutout(clean_im, s_x, s_y, s_t)[:,np.newaxis,:,:]

        noisy_cutout = torch.from_numpy(noisy_cutout.astype(np.float32))
        clean_cutout = torch.from_numpy(clean_cutout.astype(np.float32))
        noisy_im_name = noisy_im_name.replace("/","_")

        return noisy_cutout, clean_cutout, noisy_im_name

    def select_random_noisy(self, noisy_im_list):
        """
        Randomly select a noisy image from the noisy image list
        """
        random.shuffle(noisy_im_list)
        return noisy_im_list[0][0], noisy_im_list[0][1]

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
            - noisy_im_name (str): the name of the noisy image for id purpose
        """

        sample_list = []
        counts_list = []
        found = False

        # 10 tries to find a suitable sample
        for i in range(10):

            noisy_im, clean_im, noisy_im_name = self.load_one_sample(idx//self.samples_per_image) # the actual index of the image

            # The foreground content check
            # hardcoded values because image is always normalized to [0,1] with background == 0
            # require >= half of image being foreground
            valu_score = torch.count_nonzero(clean_im)
            area_score = 0.5 * clean_im.numel()
            if (valu_score >= area_score):
                found = True
                break

            sample_list.append((noisy_im, clean_im, noisy_im_name))
            counts_list.append(valu_score)

        # if failed, find the one with the highest foreground ratio
        if not found:
            noisy_im, clean_im, noisy_im_name = sample_list[counts_list.index(max(counts_list))]

        return noisy_im, clean_im, noisy_im_name

class CTDatasetTest():
    """
    Dataset for testing CT.
    Returns the complete images with proper scaling for inference.
    Iterates over all the images as noisy besides the clean image.
    """
    def __init__(self, h5file, keys, max_load=10000):
        """
        Initilize the dataset

        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for each h5file
            - max_load (int): max number of images to load during init
        """
        self.h5file = h5file
        self.keys = keys

        self.images = load_images_from_h5file(h5file, keys, max_load=max_load)

    def load_one_sample(self, i, noisy_im_idx):
        """
        Load one sample from the images

        @args:
            - i (int): index of the image to load
        @rets:
            - noisy_cutout, clean_cutout (5D torch.Tensors): the pair of images
            - noisy_im_name (str): the name of the noisy image for id purpose
        """
        noisy_im = self.images[i][0][noisy_im_idx][0]
        noisy_im_name = self.images[i][0][noisy_im_idx][1]
        clean_im = self.images[i][1][0]

        if not isinstance(noisy_im, np.ndarray):
            ind = self.images[i][2]
            noisy_im = np.array(self.h5file[ind][noisy_im])
            clean_im = np.array(self.h5file[ind][clean_im])

        if noisy_im.ndim == 2: noisy_im = noisy_im[np.newaxis,:,:]
        if clean_im.ndim == 2: clean_im = clean_im[np.newaxis,:,:]

        min_t = min(noisy_im.shape[0], clean_im.shape[0])

        noisy_im = noisy_im[:min_t,np.newaxis,:,:]
        clean_im = clean_im[:min_t,np.newaxis,:,:]

        noisy_im = torch.from_numpy(noisy_im.astype(np.float32))
        clean_im = torch.from_numpy(clean_im.astype(np.float32))
        noisy_im_name = noisy_im_name.replace("/","_")

        return noisy_im, clean_im, noisy_im_name

    def __len__(self):
        """
        Length of dataset
        """
        return sum([len(noisy_im_list) for noisy_im_list, _, _ in self.images])

    def __getitem__(self, idx):
        """
        Given index(idx) retreive the noisy clean image pair.
        (TODO: not O(1) in getting the correct index. could be done better)

        @args:
            - idx (int): the index in the dataset
        @rets:
            - noisy_im, clean_im (5D torch.Tensors): the noisy and clean pair
            - noisy_im_name (str): the name of the noisy image for id purpose
        """
        cumulative_i = 0
        noisy_im_idx = -1
        for i, (noisy_im_list,_,_) in enumerate(self.images):
            start_sample = cumulative_i
            cumulative_i += len(noisy_im_list)
            if start_sample <= idx < cumulative_i:
                noisy_im_idx = idx - start_sample
                break

        noisy_im, clean_im, noisy_im_name = self.load_one_sample(i, noisy_im_idx)

        return noisy_im, clean_im, noisy_im_name

def load_ct_data(config):
    """
    Defines how to load ct data
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
        - time (int): cutout size in time dimension
        - height (int list): different height cutouts
        - width (int list): different width cutouts
    @rets:
        - train_set, val_set, test_set (custom dataloader list): the datasets
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
        "max_load" : c.max_load,
        "time_cutout" : c.time,
        "samples_per_image" : c.samples_per_image
    }

    train_set = []

    if c.max_load<=0:
        logging.info(f"Data will not be pre-read ...")

    for (i, h_file) in enumerate(h5files):
        logging.info(f"--> loading data from file: {h_file} for {len(train_keys[i])} entries ...")
        for hw in zip(c.height, c.width):
            train_set.append(CtDatasetTrain(h5file=[h_file], keys=[train_keys[i]], cutout_shape=hw, **kwargs))

    # kwargs for val set
    kwargs["cutout_shape"] = (c.height[-1], c.width[-1])

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
                    raise RuntimeError(f"File not found: {file}")

            logging.info(f"reading from file (test_case): {file_p}")
            h5file = h5py.File(file_p,libver='earliest',mode='r')
            keys = list(h5file.keys())

            h5files.append(h5file)
            test_keys.append(keys)

            val_keys.append(keys[:int(ratio[1]*n)])

            if len(val_keys[-1])==0:
                val_keys[-1] = keys[-1:]

        # val set is cutouts of parts of test set + two complete images of test set
        val_set = [CtDatasetTrain(h5file=h5files, keys=val_keys, **kwargs)]

        test_set = [CTDatasetTest(h5file=h5files, keys=test_keys, max_load=c.max_load)]
    else:
        # No test case given, use some of the train set
        val_set = [CtDatasetTrain(h5file=h5files, keys=val_keys, **kwargs)]

        test_set = [CTDatasetTest(h5file=h5files, keys=test_keys, max_load=c.max_load)]

    return train_set, val_set, test_set
