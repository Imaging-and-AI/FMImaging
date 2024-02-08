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

from setup import *
from utils import *
from utils import calc_max_entropy_dist_params, get_eigvecs, calc_moments

from microscopy_utils import *
from temp_utils.inference_utils import apply_model, load_model

import torch.multiprocessing as mp

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

    parser.add_argument("--input_dir", type=str, default=None, help="folder to load the data")
    parser.add_argument("--input_file_s", nargs='+', type=str, default=None, help="specific file(s) to load. .tiff or .h5. If None load the entire input_dir")
    parser.add_argument("--output_dir", type=str, default=None, help="folder to save the data")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    parser.add_argument("--overlap", nargs='+', type=int, default=None, help='overlap for (T, H, W), e.g. (2, 8, 8), (0, 0, 0) means no overlap')
    parser.add_argument("--no_clip_data", action="store_true", help="whether to not clip the data to [0,1] after scaling. default: do clip")
    parser.add_argument("--image_order", type=str, default="THW", help='the order of axis in the input image: THW or WHT')
    parser.add_argument("--device", type=str, default="cuda", help='the device to run on')
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size for running inference')

    parser.add_argument("--scaling_vals", type=float, nargs='+', default=[0, 4096], help='min max values to scale with respect to the scaling type')

    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help='devices for inference')

    # -----------------------------------------

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

    parser.add_argument("--sigma", type=float, default=-1, help='sigma for the data; if < 0, estimate from the data')

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

def run_a_case(i, images, rank, config, args):

    if rank >=0:
        config.device = torch.device(f'cuda:{rank}')

    print(f"{Fore.RED}--> Processing on device {config.device}{Style.RESET_ALL}")

    model, _ = load_model(args.saved_model_path)

    for ind, v in enumerate(images):
        noisy_im, clean_im, f_name = v
        print(f"--> Processing {f_name}, {noisy_im.shape}, {ind} out of {len(images)}")

        pca_one_case(f_name, model, args, config, noisy_im, config.device)

    print(f"{Fore.RED}--> Processing on device {config.device} -- completed {Style.RESET_ALL}")

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def pca_one_case(f_name, model, args, full_config, image, device):

    B, T, C, H, W = image.shape
    print(f"--> pca_one_case, image {image.shape}")

    max_H = 256
    max_W = 256
    x = np.zeros([B, T, C, max_H, max_W], dtype=image.dtype)

    if H<=max_H and W<=max_W:
        x[:,:,:,:H,:W] = image
    elif H<max_H and W>max_W:
        sW = int(W/2 - max_W/2)
        x[:,:,:,:H, :] = image[:,:,:,:,sW:sW+max_W]
    elif H>max_H and W>max_W:
        sH = int(H/2 - max_H/2)
        sW = int(W/2 - max_W/2)
        x = image[:,:,:,sH:sH+max_H,sW:sW+max_W]
    elif H>max_H and W<max_W:
        sH = int(H/2 - max_H/2)
        x[:,:,:,:,:W] = image[:,:,:,sH:sH+max_H,:]

    print(f"--> pca_one_case, x {x.shape}")

    output_dir = os.path.join(args.output_dir, f_name)
    print(f"---> {f_name}, output is at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    mask = np.zeros((B, T, x.shape[-2], x.shape[-1]))
    if args.frame >= 0 and args.frame < T:
        mask[:,args.frame,:,:] = 1
        frame = args.frame
    else:
        # compute mean along T
        mean_signal = np.mean(image, axis=(0,2,3,4)).squeeze()
        frame = np.argmax(mean_signal)
        mask[:,frame,:,:] = 1

    print(f"--> pca, pick frame {frame}")

    sf = frame - 3
    if sf < 0:
        sf = 0
    ef = sf+8

    x = x[:,sf:ef]
    mask = mask[:,sf:ef]
    frame = 3

    print(f"--> pca_one_case, x {x.shape}")

    input = torch.from_numpy(x)
    mask = torch.from_numpy(mask)
    nim = torch.from_numpy(x)

    sigma = None
    if args.sigma > 0:
        sigma = args.sigma

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

    input = torch.permute(input, [0, 2, 1, 3, 4])
    nim = torch.permute(nim, [0, 2, 1, 3, 4])

    with torch.inference_mode():
        res = model(input)

    rpatch = res.clone()

    x= np.transpose(x, [3, 4, 2, 1, 0]).squeeze()
    res_name = os.path.join(output_dir, f'x.npy')
    print(res_name)
    np.save(res_name, x)

    res = torch.permute(res, [0, 2, 1, 3, 4])
    res = res.cpu().numpy()
    res = np.transpose(res, [3, 4, 2, 1, 0])
    res_name = os.path.join(output_dir, f'res.npy')
    print(res_name)
    np.save(res_name, res.squeeze())

    eigvecs, eigvals, mmse, sigma, subspace_corr = get_eigvecs(model,
                                                               input,
                                                               nim[0],
                                                                mask,
                                                                args.n_ev,
                                                                sigma,
                                                                device,
                                                                c=args.const, iters=args.iters,
                                                                double_precision=args.double_precision)

    moments = calc_moments(model, input, nim[0], mask, sigma, device,
                                   mmse, eigvecs, eigvals,
                                   var_c=args.var_c, skew_c=args.skew_c, kurt_c=args.kurt_c,
                                   use_poly=args.use_poly, poly_deg=args.poly_deg,
                                   poly_bound=args.poly_bound, poly_pts=args.poly_pts,
                                   double_precision=args.double_precision)

    V = eigvecs.cpu().numpy()
    V = np.transpose(V, [3, 4, 2, 1, 0])
    res_name = os.path.join(output_dir, f"eigvecs.npy")
    print(res_name)
    np.save(res_name, V)

    nim = nim[0].cpu().numpy()
    a = np.transpose(nim, [2, 3, 1, 0])
    res_name = os.path.join(output_dir, f'nim.npy')
    print(res_name)
    np.save(res_name, a)

    sd_map = np.sqrt(np.sum(moments.vmu2_per_pixel * moments.vmu2_per_pixel, axis=0))
    a = np.transpose(sd_map, [2, 3, 1, 0])
    res_name = os.path.join(output_dir, f'sd_map.npy')
    print(res_name)
    np.save(res_name, a)

    steps = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    perturb_im = torch.zeros([len(steps), args.n_ev, *nim.shape], dtype=eigvecs.dtype)
    for row in range(args.n_ev):
        for i, step in enumerate(steps):
            evup = (rpatch.cpu() + (step * eigvals[row].cpu().sqrt() * eigvecs[row].cpu()))
            perturb_im[i, row] = evup

    a = np.transpose(perturb_im, [4, 5, 3, 2, 1, 0])
    res_name = os.path.join(output_dir, f'perturb_im.npy')
    print(res_name)
    np.save(res_name, a)

    eigvals = eigvals.cpu().numpy()
    print(eigvals)

    res_name = os.path.join(output_dir, f'eigvals.npy')
    print(res_name)
    np.save(res_name, eigvals)

    frame = np.argmax(np.mean(np.squeeze(mask.cpu().numpy()), axis=(1,2)))
    res_name = os.path.join(output_dir, f'frame.npy')
    print(res_name)
    np.save(res_name, np.array(frame))

def main():

    args, _ = arg_parser()
    print(args)

    print(f"---> support bfloat16 is {support_bfloat16(device=get_device())}")

    print(f"{Fore.YELLOW}Load in model file - {args.saved_model_path}")
    model, config = load_model(args.saved_model_path)

    patch_size_inference = args.patch_size_inference
    config.pad_time = args.pad_time
    config.ddp = False
    config.device = args.device

    config.height = config.micro_height
    config.width = config.micro_width
    config.time = config.micro_time

    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference

    print("Start loading data")

    images = data_all(args, config)

    # Each image is 3 tuple:
    # noisy_im, clean_im (optional), im_name

    print("End loading data")

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()

    print("Start inference and saving")
    print(f"---> support bfloat16 is {support_bfloat16(device=get_device())}")

    if args.cuda_devices is None:
        rank = -1
        run_a_case(0, images, rank, config, args)
    else:
        print(f"Perform inference on devices {args.cuda_devices}")

        num_devices = len(args.cuda_devices) if len(args.cuda_devices) < len(images) else len(images)

        def chunkify(lst,n):
            return [lst[i::n] for i in range(n)]

        tasks = chunkify(images, num_devices)

        for ind, a_task in enumerate(tasks):
            mp.spawn(run_a_case, args=(a_task, args.cuda_devices[ind], config, args), nprocs=1, join=False, daemon=False, start_method='spawn')
            print(f"{Fore.RED}--> Spawn task {ind}{Style.RESET_ALL}")

        print(f"Perform inference on devices {args.cuda_devices} for {len(images)} cases -- completed")

    print("End pesudo-replica test and saving")

    # -------------------------------------------    

if __name__=="__main__":
    main()
