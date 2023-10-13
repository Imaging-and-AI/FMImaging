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
        total_samples = 0
        N = len(case_files)
        with h5py.File(filename , mode="w",libver='earliest') as h5file:
            for ii, a_case in enumerate(case_files):

                if args.max_sample_loaded > 0 and total_samples >= args.max_sample_loaded:
                    break

                print(f"---> {ii} out of {N}, {a_case} <---")
                
                if os.path.isfile(a_case):
                    # load in the matlab file
                    
                    mat = scipy.io.loadmat(a_case)
                    
                    N = mat['out'].shape[1]
                    
                    pbar = tqdm(total=N)
                    
                    for ind in range(mat['out'].shape[1]):
                        params = mat['out'][0,ind]['params'][0,0].flatten()
                        aif = mat['out'][0,ind]['aif'][0,0].flatten()
                        myo = mat['out'][0,ind]['myo'][0,0].flatten()
                        
                        key = f"{a_case}_{ind}"
                        data_folder = h5file.create_group(key)
                        data_folder["aif"] = aif.astype(np.float32)
                        data_folder["myo"] = myo.astype(np.float32)
                        data_folder["params"] = params.astype(np.float32)
                        
                        pbar.update(1)
                        pbar.set_description_str(f"{key}, {aif.shape[0]}, {params[0]}")
                        

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
