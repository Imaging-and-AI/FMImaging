"""
Create mri set for STCNNT MRI
"""
import sys
import os
import argparse
from tqdm import tqdm
import h5py
import random
import numpy as np
import glob
import pickle

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[3].resolve()
sys.path.append(str(REPO_DIR))


def create_new_data(h5_file, write_path):

    keys = list(h5_file.keys())
    N = len(keys)

    if N == 0:
        return

    if "image" not in h5_file[keys[0]]:
        return
    
    if "gmap" not in h5_file[keys[0]]:
        return

    h5_file_3d = h5py.File(write_path, mode="w", libver="earliest")

    data_info = dict()

    with tqdm(total=N) as pbar:
        for i in range(N):

            data = np.array(h5_file[keys[i]+"/image"])
            gmap = np.array(h5_file[keys[i]+"/gmap"])

            grp_name = f"Case_{i}"
            data_folder = h5_file_3d.create_group(grp_name)

            data_folder["image"] = data
            data_folder["gmap"] = gmap

            data_info[keys[i]] = grp_name

            pbar.update(1)
            pbar.set_description_str(f"{keys[i]} --> {grp_name} -- {data.shape} -- {gmap.shape}")

    return data_info

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT MRI test evaluation")

    parser.add_argument("--input_dir", default="/isilon/lab-xue/data/mri", help="folder to load the data")
    parser.add_argument("--output_dir", default="/isilon/lab-xue/data/mri_anonymized", help="folder to save the data")

    return parser.parse_args()

def main():

    args = arg_parser()

    h5file_lists = [f for f in glob.glob(os.path.join(args.input_dir, "*.h5"))]
    print(f"h5 files are {h5file_lists}")

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in h5file_lists:
        fpath, fname = os.path.split(fname)
        fname_stem = os.path.splitext(fname)[0]
        
        rec_file = os.path.join(args.output_dir, fname_stem+".record")
        if os.path.exists(rec_file):
            print(f"--> ignore : {fname_stem}")
            continue
        
        print(f"--> process {os.path.join(args.input_dir, fname)} <--")
        try:
            h5_file = h5py.File(os.path.join(args.input_dir, fname))
            data_info = create_new_data(h5_file, os.path.join(args.output_dir, fname))
            with open(os.path.join(args.output_dir, fname_stem+".record"), "wb") as f:
                pickle.dump(data_info, f)
        except:
                continue
        
        print(f"-----------------------------------------")

if __name__=="__main__":
    main()
