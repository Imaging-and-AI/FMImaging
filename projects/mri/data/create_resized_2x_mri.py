"""
Create test set for STCNNT MRI
"""
import os
import tqdm
import h5py
import random
import numpy as np

import torch
import interpol
import nibabel as nib

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *

device = get_device()

def arg_parser():
    parser = argparse.ArgumentParser("Argument parser for 2x resizing images")

    parser.add_argument("--data_root", default=None, help="folder to load the data")
    parser.add_argument("--input_fname", type=str, default="train_3D_3T_retro_cine_2020", help='input file name')

    return parser.parse_args()


def create_resized_data(write_path, h5_file, keys):

    h5_file_2d = h5py.File(write_path, mode="w", libver="earliest")

    for ind in tqdm(range(len(keys))):
        key = keys[ind]
        try:
            data = np.array(h5_file[key+"/image"])
            is_test = False
        except:
            data = np.array(h5_file[key+"/clean"])
            noisy = np.array(h5_file[key+"/noisy"])
            noise_sigma = np.array(h5_file[key+"/noise_sigma"])
            is_test = True
        gmap = np.array(h5_file[key+"/gmap"])

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        T, RO, E1 = data.shape

        if gmap.ndim == 2:
            gmap = np.expand_dims(gmap, axis=0)
        N = gmap.shape[0]

        opt = dict(shape=[T, 2*RO, 2*E1], anchor='first', bound='replicate')
        x = torch.from_numpy(np.real(data)).to(device=device, dtype=torch.float32)
        x_2x = interpol.resize(x, **opt, interpolation=5)
        y = torch.from_numpy(np.imag(data)).to(device, dtype=torch.float32)
        y_2x = interpol.resize(y, **opt, interpolation=5)
        data_resized = torch.complex(x_2x.to(dtype=torch.float32), y_2x.to(dtype=torch.float32)).cpu().numpy()

        opt = dict(shape=[N, 2*RO, 2*E1], anchor='first', bound='replicate')
        x = torch.from_numpy(gmap).to(device)
        x_2x = interpol.resize(x, **opt, interpolation=5)
        gmap_resized = x_2x.cpu().numpy().astype(np.float32)

        # -------------------------------------------------------
        if RO>64 and E1>64 and data_resized.shape[1]>128 and data_resized.shape[2]>128:
            data_folder = h5_file_2d.create_group(key)

            data_folder["image"] = data.squeeze().astype(np.csingle)
            data_folder["gmap"] = gmap.squeeze().astype(np.float16)
            data_folder["image_resized"] = data_resized.squeeze()
            data_folder["gmap_resized"] = gmap_resized.squeeze()

            if is_test:
                data_folder["noisy"] = noisy
                data_folder["noise_sigma"] = noise_sigma
        else:
            print(f"Warning - {key} - data resized is too small - {data_resized.shape}")

        # saved_path = "/export/Lab-Xue/projects/mri/test"
        # nib.save(nib.Nifti1Image(np.real(np.transpose(data, (1, 2, 0))), affine=np.eye(4)), os.path.join(saved_path, f"{key}_x_real.nii"))
        # nib.save(nib.Nifti1Image(np.imag(np.transpose(data, (1, 2, 0))), affine=np.eye(4)), os.path.join(saved_path, f"{key}_x_imag.nii"))
        # nib.save(nib.Nifti1Image(np.abs(np.transpose(data, (1, 2, 0))), affine=np.eye(4)), os.path.join(saved_path, f"{key}_x_mag.nii"))
        # nib.save(nib.Nifti1Image(np.real(np.transpose(gmap, (1, 2, 0))), affine=np.eye(4)), os.path.join(saved_path, f"{key}_gmap.nii"))
        # nib.save(nib.Nifti1Image(np.abs(np.transpose(data_resized, (1, 2, 0))), affine=np.eye(4)), os.path.join(saved_path, f"{key}_data_resized.nii"))
        # nib.save(nib.Nifti1Image(np.abs(np.transpose(gmap_resized, (1, 2, 0))), affine=np.eye(4)), os.path.join(saved_path, f"{key}_gmap_resized.nii"))

def main():

    args = arg_parser()

    base_file_path = args.data_root
    base_file_name = args.input_fname

    h5_file = h5py.File(os.path.join(base_file_path, base_file_name+'.h5'))
    keys = list(h5_file.keys())
    n = len(keys)

    print(f"--> resizeing {base_file_path} - {base_file_name} - {n} keys")

    write_path = f"{base_file_path}/{base_file_name}_with_2x_resized.h5"
    create_resized_data(write_path, h5_file, keys)

    print(f"{write_path} - All done")

if __name__=="__main__":
    main()
