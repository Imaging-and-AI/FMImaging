import os
import argparse
import h5py
import numpy as np
from pathlib import Path
import tqdm
import skimage.restoration as skr
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Back, Style
import glob 
from functools import reduce
np.seterr(all='raise')

import scipy.io as sio

import sys
from pathlib import Path
Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

total_samples = 0

def add_image_to_h5group(args, folder, h5_group):
    
    global total_samples
    
    if(is_valid_folder(folder) is False):
        return

    possible_filenames = [args.input_fname]

    image = None 
    for filename in possible_filenames:
        complex_fname = os.path.join(folder, f"{filename}_real.npy")
        has_complex = False
        if(os.path.exists(complex_fname)):
            has_complex = True

        if has_complex:
            image = np.load(os.path.join(folder, f"{filename}_real.npy")) + np.load(os.path.join(folder, f"{filename}_imag.npy")) * 1j
            break
        elif (os.path.exists(os.path.join(folder, f"{filename}.npy"))):
            image = np.load(os.path.join(folder, f"{filename}.npy"))
            break

    if image is None: 
        print(f"{folder} does not contain images")
        return

    base_path, base_name = os.path.split(folder)
    # base_name = folder.stem

    if(image.ndim==2):
        return

    if len(image.shape) == 3:
        image = image[:,:,np.newaxis,:]

    x, y, slices, frames = image.shape

    print(f"{Fore.GREEN}{folder}, images - {image.shape}{Style.RESET_ALL}")

    if(x<64 or y<64):
        print(f"{Fore.RED}{folder} -- pass over - x<64 or y<64 ...{Style.RESET_ALL}")
        return
    
    if(frames<20):
        print(f"{Fore.RED}{folder} -- pass over - frames {frames} ...{Style.RESET_ALL}")
        return
        
    if(slices>20):
        print(f"{Fore.RED}{folder} -- pass over - slices {slices} ...{Style.RESET_ALL}")
        return
    
    # remote the default scaling
    image /= args.im_scaling
    
    # check all gmaps
    for s in range(slices):
        gmap_file = f"{folder}/gmap_slc_{s+1}.npy"
        if(os.path.exists(gmap_file) == False):
            print(f"{folder} -- pass over - cannot find gmap {gmap_file} ...")
            return

    for s in range(slices):

        gmap = np.load(f"{folder}/gmap_slc_{s+1}.npy")
        # gmap /= args.gmap_scaling

        image_slice = image[:, :, s, :]
        if np.sum(np.abs(image_slice)) < 1:
            continue

        assert gmap.shape[0]==image_slice.shape[0] and gmap.shape[1]==image_slice.shape[1] 

        # for gslice in gmap:
        #     avg = np.mean(gslice)
        #     if avg <1e-6 or np.isnan(avg):
        #         raise RuntimeError(f"Something is up with {base_name}. Gmap: {gmap.shape}, image_slice: {image_slice.shape}")

        data_folder = h5_group.create_group(f"{base_name}_slc_{s+1}")
        data_folder["image"] = image_slice.astype(np.complex64)
        data_folder["gmap"] = gmap.astype(np.float32)
        
        total_samples += 1

def is_valid_folder(folder):
    return os.path.exists(os.path.join(folder, "gmap_slc_1.npy"))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--output")
    parser.add_argument("--test_output", default="test.h5")
    parser.add_argument("--test_fraction",default=0.0, type=float)
    parser.add_argument("folders",nargs="+")
    
    parser.add_argument("--im_scaling", type=float, default=10.0, help="extra scaling applied to image")
    parser.add_argument("--gmap_scaling", type=float, default=100.0, help="extra scaling applied to gmap")

    parser.add_argument(
        "--only_3T",
        action="store_true",
        help="If true, only include 3T data for training",
    )
    
    parser.add_argument(
        "--no_3T",
        action="store_true",
        help="If true, only include non-3T data for training",
    )

    parser.add_argument("--input_fname", type=str, default="im", help='input file name')

    args = parser.parse_args()
    print(args)
    
    folders = []

    for f in args.folders:
        date_dirs = os.listdir(f)
        for d in date_dirs:
            scan_dirs = os.listdir(os.path.join(f, d))
            for s in scan_dirs:
                folders.append(os.path.join(f, d, s))

    rng = np.random.Generator(np.random.PCG64(424242))
    rng.shuffle(folders)

    if(args.test_fraction>0):
        split_index = int(len(folders)*args.test_fraction)

        test_folders  = folders[:split_index]
        train_folders  = folders[split_index:]
    else:
        train_folders  = folders
    
    print(f"Number of folders: {len(folders)}")

    def save_folders_to_h5(args, filename, datafolders, only_3T=True):
        global total_samples
        total_samples = 0
        N = len(datafolders)
        with h5py.File(filename , mode="w",libver='earliest') as h5file:
            #with ThreadPoolExecutor() as executor:  
            #    list(tqdm.tqdm(executor.map(lambda   f: add_image_to_h5group(f,h5file), datafolders),total=len(datafolders)))
            for ii, folder in enumerate(datafolders):
                # if(folder.find("66016")>=0):
                #     print("here")

                # get the field strength
                ismrmrd_file = os.path.join(folder, f"ismrmrd_hdr.mat")
                try:
                    if os.path.isfile(ismrmrd_file):
                        a = h5py.File(ismrmrd_file, 'r')
                        systemFieldStrength_T = np.array(a['hdr']['acquisitionSystemInformation']['systemFieldStrength_T'])[0, 0]
                    else:
                        continue
                except:
                    a = sio.loadmat(ismrmrd_file)
                    systemFieldStrength_T = a['systemFieldStrength_T'][0, 0]

                print(f"---> {ii} out of {N}, {systemFieldStrength_T}, {folder} <---")
                if only_3T and systemFieldStrength_T<2.0:
                    continue

                if args.no_3T and systemFieldStrength_T>2.0:
                    continue
                
                add_image_to_h5group(args, folder, h5file)
                
        print("----" * 20)
        print(f"--> total number of samples {total_samples}")

    save_folders_to_h5(args, args.output, train_folders, args.only_3T)

    if(args.test_fraction>0):
        save_folders_to_h5(args.test_output,test_folders, args.only_3T)

if __name__ == "__main__":
    main()
