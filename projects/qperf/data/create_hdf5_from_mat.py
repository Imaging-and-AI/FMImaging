import os
import argparse
import h5py
import scipy.io
import numpy as np
from pathlib import Path
import tqdm
import skimage.restoration as skr
from colorama import Fore, Back, Style
import glob 
from functools import reduce
np.seterr(all='raise')

from tqdm import tqdm

import scipy.io as sio

import sys
from pathlib import Path
Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

total_samples = 0

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--output")
    parser.add_argument("--test_output", default="test.h5")
    parser.add_argument("--test_fraction",default=0.1, type=float)
    parser.add_argument("folders",nargs="+")

    parser.add_argument("--max_sample_loaded", type=int, default=-1, help="if > 0, max number of samples loaded")

    args = parser.parse_args()
    print(args)

    all_data_files = []

    for f in args.folders:
        date_dirs = os.listdir(f)
        for d in date_dirs:
            a_case = os.path.join(f, d)
            if a_case.endswith('mat'):
                all_data_files.append(os.path.join(f, d, a_case))


    if(args.test_fraction>0):
        split_index = int(len(all_data_files)*args.test_fraction)

        test_cases  = all_data_files[:split_index]
        train_cases  = all_data_files[split_index:]
    else:
        train_cases  = all_data_files

    print(f"Number of train cases: {len(train_cases)}")
    if(args.test_fraction>0): print(f"Number of test cases: {len(test_cases)}")

    # ----------------------------------------------------------
    
    def save_cases_to_h5(args, filename, case_files):
        global total_samples
        grp_num = 20
        total_samples = 0
        num_cases = len(case_files)
        h5file = None
        pbar = tqdm(total=num_cases)
        for ii, a_case in enumerate(case_files):

            if ii % 100 == 0:
                if h5file is not None: 
                    h5file.close()
                fname = f"{filename}_{ii}.h5"
                print(f"--> create file {fname}, grp_num {grp_num}")
                h5file = h5py.File(f"{filename}_{ii}.h5" , mode="w", libver='latest')

            if args.max_sample_loaded > 0 and total_samples >= args.max_sample_loaded:
                break

            #print(f"---> {ii} out of {num_cases}, {a_case} <---")
            
            if os.path.isfile(a_case):

                a_case_fname, mat_ext = os.path.splitext(a_case)
                aif_fname = f"{a_case_fname}_aif.npy"
                myo_fname = f"{a_case_fname}_myo.npy"
                params_fname = f"{a_case_fname}_params.npy"

                if not os.path.exists(aif_fname):
                    continue

                aif = np.load(aif_fname)
                myo = np.load(myo_fname)
                params = np.load(params_fname)
                params = params.astype(np.float32)

                head, tail = os.path.split(a_case_fname)

                N = aif.shape[0]
                case_folder = h5file.create_group(tail)
                for ind in range(0, N, grp_num):
                    if ind+grp_num >= N:
                        break

                    a_params = params[ind:ind+grp_num, :]
                    a_aif = aif[ind:ind+grp_num, :]
                    a_myo = myo[ind:ind+grp_num, :]

                    key = f"{ind}"
                    data_folder = case_folder.create_group(key)
                    data_folder["aif"] = a_aif
                    data_folder["myo"] = a_myo
                    data_folder["params"] = a_params

                    assert a_aif.shape[0] == grp_num

            pbar.update(1)
            pbar.set_description_str(f"{a_case_fname}")

        if h5file is not None: 
            h5file.close()

        print("----" * 20)
        print(f"--> total number of samples {total_samples}")

    # ----------------------------------------------------------

    save_cases_to_h5(args, args.output, train_cases)

    # ----------------------------------------------------------
    
    if(args.test_fraction>0):
        save_cases_to_h5(args.test_output, test_cases)

    # ----------------------------------------------------------
    
if __name__ == "__main__":
    main()
