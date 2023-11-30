"""
Data utilities for MRI data.
Provides the torch dataset class for traind and test and functions to load from multiple h5files
"""

import sys
import logging

from tqdm import tqdm
import numpy as np
from pathlib import Path
from colorama import Fore, Style

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from trainer import get_bar_format

# -------------------------------------------------------------------------------------------------

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
            
            with tqdm(total=len(keys[i]), bar_format=get_bar_format()) as pbar:
                for n, key in enumerate(keys[i]):
                    if num_loaded < max_load:
                        images.append([np.array(h5file[i][key+"/image"]), np.array(h5file[i][key+"/gmap"]), np.array(h5file[i][key+"/image_resized"]), i])
                        num_loaded += 1
                    else:
                        images.append([key+"/image", key+"/gmap", key+"/image_resized", i])
                        
                    if n>0 and n%100 == 0:
                        pbar.update(100)
                        pbar.set_description_str(f"{h5file}, {n} in {len(keys[i])}, total {len(images)}")

        return images

# -------------------------------------------------------------------------------------------------

def load_images_for_statistics(h5file, keys, max_loaded=30):
        """
        Load images from h5 file objects to count image statistics
        @args:
            - h5file (h5File list): list of h5files to load images from
            - keys (key list list): list of list of keys. One for each h5file
            
        @outputs:
            - stat : dict for statistics, keys 'mean', 'median', 'percentile'
        """
        num_loaded = 0
        stat = dict()
        stat['mean'] = list()
        stat['median'] = list()
        stat['gmap_median'] = list()
        for i in range(len(h5file)):
            with tqdm(total=len(keys[i]), bar_format=get_bar_format()) as pbar:
                case_num_loaded = 0
                for n, key in enumerate(keys[i]):
                    im = np.array(h5file[i][key+"/image"])
                    gmap = np.array(h5file[i][key+"/gmap"])

                    stat['mean'].extend(np.mean(im, axis=(0, 1)).flatten())
                    stat['median'].extend(np.median(im, axis=(0, 1)).flatten())
                    stat['gmap_median'].extend(np.median(gmap, axis=(0, 1)).flatten())
                    num_loaded += 1
                    pbar.update(1)

                    case_num_loaded += 1
                    if case_num_loaded > max_loaded:
                        break

        stat['num_loaded'] = num_loaded
        return stat

# -------------------------------------------------------------------------------------------------
# test dataset class

def load_test_images_from_h5file(h5file, keys, data_types):
    """
    Load images from h5 file objects
    @args:
        - h5file (h5File list): list of h5files to load images from
        - keys (key list list): list of list of keys. One for each h5file
        - data_types (data type list): list of data types for each h5file

    @outputs:
        - images : list of image and gmap pairs as a list
    """
    images = []

    num_loaded = 0
    for i in range(len(h5file)):
        with tqdm(total=len(keys[i]), bar_format=get_bar_format()) as pbar:
            for n, key in enumerate(keys[i]):
                if 'clean' in h5file[i][key].keys():
                    images.append([key+"/noisy", key+"/clean", key+"/clean", key+"/gmap", key+"/noise_sigma", i, data_types[i]])
                else:
                    images.append([key+"/noisy", key+"/image", key+"/image_resized", key+"/gmap", key+"/noise_sigma", i, data_types[i]])
                num_loaded += 1

                if n>0 and n%100 == 0:
                    pbar.update(100)
                    pbar.set_description_str(f"{h5file}, {n} in {len(keys[i])}, total {len(images)}")

    return images

# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    pass