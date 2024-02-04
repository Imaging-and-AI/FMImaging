"""
Utility functions
"""
import os
import h5py
import torch
import argparse
import tifffile
import numpy as np

from tqdm import tqdm
from time import time

import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

MRI_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(MRI_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[3].resolve()
sys.path.append(str(REPO_DIR))

from utils import *

# -------------------------------------------------------------------------------------------------
# load and normalize data

def load_tiff(input_dir, tifffile_name):

    a = tifffile.imread(os.path.join(input_dir, tifffile_name))
    return [a, None, f"{tifffile_name}"]

def load_numpy(input_dir, file_name):

    return [np.load(os.path.join(input_dir, file_name)), None, f"{file_name}"]

def load_h5(input_dir, h5file_name):

    images = []

    h5file = h5py.File(os.path.join(input_dir, h5file_name), mode='r')
    keys = list(h5file.keys())

    for key in keys:
        noisy_im = np.array(h5file[key+"/noisy_im"])
        clean_im = np.array(h5file[key+"/clean_im"])

        images.append([noisy_im, clean_im, f"{h5file_name}_{key}"])

    return images

def load_all(input_dir, file_names):

    images = []

    print(file_names)
    for file_name in tqdm(file_names):

        if os.path.isdir(os.path.join(input_dir, file_name)):
            continue

        if file_name.endswith(".tif") or file_name.endswith(".tiff") or file_name.endswith(".TIFF"):
            images.append(load_tiff(input_dir, file_name))
        elif file_name.endswith(".npy"):
            images.append(load_numpy(input_dir, file_name))
        elif file_name.endswith(".h5") or file_name.endswith(".h5py"):
            images.extend(load_h5(input_dir, file_name))
        else:
            print(f"Only supported formats: '.tif', '.tiff', '.TIFF', '.h5', '.h5py'. Given format:{file_name}")

    return images

def data_all(args, config):

    c = config

    input_dir = args.input_dir
    file_names = args.input_file_s if args.input_file_s is not None else os.listdir(input_dir)

    images = load_all(input_dir, file_names)

    for image in tqdm(images):

        if args.image_order=="HWT":
            image[0] = np.transpose(image[0], (2,0,1))
            if image[1] is not None:
                image[1] = np.transpose(image[1], (2,0,1))

        if c.scaling_type=="per":
            image[0] = normalize_image(image[0], percentiles=c.scaling_vals, clip=not args.no_clip_data).astype(np.float32)[np.newaxis,:,np.newaxis]
            if image[1] is not None:
                image[1] = normalize_image(image[1], percentiles=c.scaling_vals).astype(np.float32)[np.newaxis,:,np.newaxis]
        else:
            image[0] = normalize_image(image[0], values=c.scaling_vals, clip=not args.no_clip_data).astype(np.float32)[np.newaxis,:,np.newaxis]
            if image[1] is not None:
                image[1] = normalize_image(image[1], values=c.scaling_vals).astype(np.float32)[np.newaxis,:,np.newaxis]

    return images

# -------------------------------------------------------------------------------------------------
# save images

def save_one(saved_path, fname, x, output, y=None):

        if isinstance(x, np.ndarray):
            noisy_im = x
        else:
            noisy_im = x.numpy(force=True)

        if isinstance(output, np.ndarray):
            predi_im = output.numpy(force=True)
        else:
            predi_im = output

        if y is not None:
            if isinstance(y, np.ndarray):
                clean_im = y
            else:
                clean_im = y.numpy(force=True)
        else:
            clean_im = None

        np.save(os.path.join(saved_path, f"{fname}_x.npy"), noisy_im)
        np.save(os.path.join(saved_path, f"{fname}_output.npy"), predi_im)
        if clean_im is not None: np.save(os.path.join(saved_path, f"{fname}_y.npy"), clean_im)

        save_x = noisy_im[0,0,0]
        save_p = predi_im[0,0,0]
        save_y = clean_im[0,0,0] if clean_im is not None else []

        if clean_im is None:
            composed_channel_wise = np.transpose(np.array([save_x, save_p]), (1,0,2,3))
        else:
            composed_channel_wise = np.transpose(np.array([save_x, save_p, save_y]), (1,0,2,3))

        tifffile.imwrite(os.path.join(saved_path, f"{fname}_combined.tiff"),\
                            composed_channel_wise, imagej=True)

# -------------------------------------------------------------------------------------------------

if __name__=="__main__":
    pass
