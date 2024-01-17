"""
Run MRI inference data
"""
import os
import argparse
import copy
from time import time

import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

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

from setup import *
from utils import *
from inference import apply_model, load_model, apply_model_3D, load_model_pre_backbone_post, compute_uncertainty
from mri_data import MRIDenoisingDatasetTrain, load_mri_data
from mri_parser import mri_parser

from max_entropy_utils import calc_max_entropy_dist_params
from moments_calculations import get_eigvecs, calc_moments

# -------------------------------------------------------------------------------------------------
# setup for testing from cmd

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT MRI test evaluation")

    parser.add_argument("--input_dir", default=None, help="folder to load the data")
    parser.add_argument("--output_dir", default=None, help="folder to save the data")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="scaling factor to adjust model strength; higher scaling means lower strength")
    parser.add_argument("--im_scaling", type=float, default=1.0, help="extra scaling applied to image")
    parser.add_argument("--gmap_scaling", type=float, default=1.0, help="extra scaling applied to gmap")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    #parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    parser.add_argument("--overlap", nargs='+', type=int, default=None, help='overlap for (T, H, W), e.g. (2, 8, 8), (0, 0, 0) means no overlap')

    parser.add_argument("--input_fname", type=str, default="im", help='input file name')
    parser.add_argument("--gmap_fname", type=str, default="gfactor", help='gmap input file name')

    parser.add_argument('-a', '--amount', default=2, type=int,
                        help='Amount of sigmas to multiply the ev by (recommended 1-3)')
    parser.add_argument('-c', '--const', type=float, default=1e-6, help='Normalizing const for the power iterations')
    parser.add_argument('-e', '--n_ev', default=3, type=int, help='Number of eigenvectors to compute')
    parser.add_argument('-g', '--gpu_num', default=0, type=int, help='GPU device to use. -1 for cpu')
    parser.add_argument('-i', '--input', help='path to input file or input folder of files')
    parser.add_argument('-m', '--manual', nargs=4, default=None, type=int,
                        help='Choose a patch for uncertainty quantification in advanced, instead of choosing '
                             'interactively. Format: x1 x2 y1 y2.')
    parser.add_argument('-n', '--noise_std', type=float, default=None, help='Noise level to add to images')
    parser.add_argument('-o', '--outpath', default='Outputs', help='path to dump results')
    parser.add_argument('-p', '--padding', default=None, type=int,
                        help='The size of margin around the patch to insert to the model.')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Set seed number')
    parser.add_argument('-t', '--iters', type=int, default=50, help='Amount of power iterations')
    parser.add_argument('-v', '--marginal_dist', action='store_true',
                        help='Calc the marginal distribution along the evs (v\\mu_i)')
    parser.add_argument('--var_c', type=float, default=1e-6, help='Normalizing constant for 2rd moment approximation')
    parser.add_argument('--skew_c', type=float, default=1e-5, help='Normalizing constant for 3rd moment approximation')
    parser.add_argument('--kurt_c', type=float, default=1e-5, help='Normalizing constant for 4th moment approximation')
    parser.add_argument('--model_zoo', default='./KAIR/model_zoo', help='Directory of the models\' weights')
    parser.add_argument('--force_grayscale', action='store_true', help='Convert the image to gray scale')
    parser.add_argument('--low_acc', dest='double_precision', action='store_false',
                        help='Recomended when calculating only PCs (and not higher-order moments)')
    parser.add_argument('--use_poly', action='store_true',
                        help='Use a polynomial fit before calculating the derivatives for moments calculation')
    parser.add_argument('--poly_deg', type=int, default=6, help='The degree for the polynomial fit')
    parser.add_argument('--poly_bound', type=float, default=1, help='The bound around the MMSE for the polynomial fit')
    parser.add_argument('--poly_pts', type=int, default=30, help='The amount of points to use for the polynomial fit')
    parser.add_argument('--mnist_break_at', type=int, default=None, help='Stop iterating over MNIST at this index')
    parser.add_argument('--mnist_choose_one', type=int, default=None, help='Stop iterating over MNIST at this index')
    parser.add_argument('--fmd_choose_one', type=int, default=None, help='Choose a specific FOV from the FMD.')
    parser.add_argument('--old_noise_selection', action='store_true',
                        help='Deprecated. Only here to reproduce the paper\'s figures')

    parser.add_argument("--frame", type=int, default=-1, help='which frame picked to compute PCA; if <0, pick the middle frame')

    #parser.add_argument("--model_type", type=str, default=None, help="if set, overwrite the config setting, STCNNT_MRI or MRI_hrnet, MRI_double_net")

    args, unknown_args = parser.parse_known_args(namespace=Nestedspace())

    return args, unknown_args

def check_args(args):
    """
    checks the cmd args to make sure they are correct
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - args (Namespace): the checked and updated argparse for MRI
    """
    # get the args path
    fname = os.path.splitext(args.saved_model_path)[0]
    args.saved_model_config  = fname + '.yaml'

    return args

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def pca_one_slice(slc, model, args, full_config, image, gmap, device):

    H, W, T = image.shape

    x = np.transpose(image, [2, 0, 1]).reshape([1, T, 1, H, W])
    g = np.repeat(gmap[np.newaxis, np.newaxis, np.newaxis, :, :], T, axis=1)

    #x -= np.mean(x)

    nim = np.concatenate((x.real, x.imag), axis=2)
    nim = np.transpose(nim, [0, 2, 1, 3, 4])
    nim = np.squeeze(nim)

    input = np.concatenate((x.real, x.imag, g), axis=2)

    input = np.transpose(input, [0, 2, 1, 3, 4])

    mask = np.zeros(nim.shape)
    if args.frame >= 0 and args.frame < T:
        print(f"--> pca, picke frame {args.frame}")
        mask[:,args.frame,:,:] = 1
    else:
        mask[:,T//2,:,:] = 1

    input = torch.from_numpy(input)
    mask = torch.from_numpy(mask)
    nim = torch.from_numpy(nim)

    sigma = np.mean(g)

    if args.double_precision:
        model.to(torch.double)
        nim = nim.to(torch.double)
        input = input.to(torch.double)
    else:
        model.to(torch.float32)
        nim = nim.to(torch.float32)
        input = input.to(torch.float32)

    model = model.to(device)
    input = input.to(device)
    nim = nim.to(device)
    mask = mask.to(device)

    with torch.inference_mode():
        res = model(input)

    rpatch = res.clone()

    res = res.cpu().numpy()
    res = np.transpose(res, [3, 4, 2, 1, 0])
    res_name = os.path.join(full_config.output_dir, f'res_{slc}.npy')
    print(res_name)
    np.save(res_name, res)
        
    eigvecs, eigvals, mmse, sigma, subspace_corr = get_eigvecs(model,
                                                               input,
                                                               nim,
                                                                mask,
                                                                args.n_ev,
                                                                sigma,
                                                                device,
                                                                c=args.const, iters=args.iters,
                                                                double_precision=args.double_precision)

    moments = calc_moments(model, input, nim, mask, sigma, device,
                                   mmse, eigvecs, eigvals,
                                   var_c=args.var_c, skew_c=args.skew_c, kurt_c=args.kurt_c,
                                   use_poly=args.use_poly, poly_deg=args.poly_deg,
                                   poly_bound=args.poly_bound, poly_pts=args.poly_pts,
                                   double_precision=args.double_precision)

    V = eigvecs.cpu().numpy()
    V = np.transpose(V, [3, 4, 2, 1, 0])
    res_name = os.path.join(full_config.output_dir, f"eigvecs_{slc}.npy")
    print(res_name)
    np.save(res_name, V)

    nim = nim.cpu().numpy()
    a = np.transpose(nim, [2, 3, 1, 0])
    res_name = os.path.join(full_config.output_dir, f'nim_{slc}.npy')
    print(res_name)
    np.save(res_name, a)

    sd_map = np.sqrt(np.sum(moments.vmu2_per_pixel * moments.vmu2_per_pixel, axis=0))
    a = np.transpose(sd_map, [2, 3, 1, 0])
    res_name = os.path.join(full_config.output_dir, f'sd_map_{slc}.npy')
    print(res_name)
    np.save(res_name, a)

    steps = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    perturb_im = torch.zeros([len(steps), args.n_ev, *nim.shape], dtype=eigvecs.dtype)
    for row in range(args.n_ev):
        for i, step in enumerate(steps):
            evup = (rpatch.cpu() + (step * eigvals[row].cpu().sqrt() * eigvecs[row].cpu()))
            perturb_im[i, row] = evup

    a = np.transpose(perturb_im, [4, 5, 3, 2, 1, 0])
    res_name = os.path.join(full_config.output_dir, f'perturb_im_{slc}.npy')
    print(res_name)
    np.save(res_name, a)

    eigvals = eigvals.cpu().numpy()
    print(eigvals)

    res_name = os.path.join(full_config.output_dir, f'eigvals_{slc}.npy')
    print(res_name)
    np.save(res_name, eigvals)

def main():

    args, unknown_args = arg_parser()
    print(args)

    project_config, unknown_args_project = mri_parser().parser.parse_known_args(namespace=Nestedspace())

    full_config = Nestedspace(**vars(args),
                              **vars(project_config))

    print(f"---> support bfloat16 is {support_bfloat16(device=get_device())}")

    print(f"{Fore.YELLOW}Load in model file - {full_config.saved_model_path}{Style.RESET_ALL}")
    if os.path.exists(full_config.saved_model_path):
        model, config = load_model(full_config.saved_model_path)
    else:
        model, config = load_model_pre_backbone_post(full_config.saved_model_path)

    # train_loader, val_loader
    print(f"{Fore.YELLOW}Load in data - {full_config.train_files[0]}{Style.RESET_ALL}")

    # -----------------------------------------------------------------------
    config.height = config.mri_height
    config.width = config.mri_width

    patch_size_inference = full_config.patch_size_inference
    config.pad_time = full_config.pad_time
    config.ddp = False

    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference

    #setup_run(config, dirs=["log_path"])

    # load the data
    image = np.load(os.path.join(full_config.input_dir, f"{full_config.input_fname}_real.npy")) + np.load(os.path.join(full_config.input_dir, f"{full_config.input_fname}_imag.npy")) * 1j
    image /= full_config.im_scaling
    image = np.squeeze(image)

    gmap_file = f"{full_config.input_dir}/{full_config.gmap_fname}.npy"
    if os.path.exists(gmap_file):
        gmap = np.load(f"{full_config.input_dir}/{full_config.gmap_fname}.npy")
        gmap /= full_config.gmap_scaling
    else:
        gmap = np.ones((image.shape[:2]))

    if len(image.shape) == 2:
        image = image[:,:,np.newaxis,np.newaxis]

    if len(image.shape) == 3:
        image = image[:,:,:,np.newaxis]

    if(image.shape[3]>20):
        image = np.transpose(image, (0, 1, 3, 2))

    RO, E1, frames, slices = image.shape

    if(gmap.ndim==2):
        gmap = np.expand_dims(gmap, axis=2)

    if gmap.shape[2] >= slices and gmap.shape[2] == frames:
        image = np.transpose(image, (0, 1, 3, 2))
        RO, E1, frames, slices = image.shape

    if full_config.overlap:
        overlap_used = tuple(full_config.overlap)
    else:
        overlap_used = None

    #output = apply_model(image, model, gmap, config=config, scaling_factor=full_config.scaling_factor, device=get_device(), overlap=overlap_used, verbose=True)
    device = get_device()

    print(f"2DT mode, {full_config.input_dir}, images - {image.shape}, gmap - {gmap.shape}, median gmap {np.median(gmap)}")

    os.makedirs(full_config.output_dir, exist_ok=True)

    for slc in range(slices):
        print(f"{Fore.YELLOW}--> processing slice {slc}{Style.RESET_ALL}")
        pca_one_slice(slc, model, args, full_config, image[:,:,:,slc], gmap[:,:,slc], device)

    # -------------------------------------------    

if __name__=="__main__":
    main()
