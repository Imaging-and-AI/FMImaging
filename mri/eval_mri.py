"""
File for testing the MRI models
Can be run from command line to load and test a model
Provides functionality to also be called during runtime while/after training:
    - eval_test
"""
import json
import wandb
import logging
import argparse
import tifffile

import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *
from model_base.losses import *
from model_mri import STCNNT_MRI
from utils.running_inference import running_inference

# -------------------------------------------------------------------------------------------------
# main test function

def eval_test(model, config, test_set=None, device="cpu", id=""):
    """
    The test evaluation.
    @args:
        - model (torch model): model to be tested
        - config (Namespace): runtime namespace for setup
        - test_set (torch Dataset): the data to test on
        - device (torch.device): the device to run the test on
        - id (str): unique id to log and save results with
    @rets:
        - test_loss_avg (float): the average test loss
        - test_mse_loss_avg (float): the average test mse loss
        - test_l1_loss_avg (float): the average test l1 loss
        - test_ssim_loss_avg (float): the average test ssim loss
        - test_ssim3D_loss_avg (float): the average test ssim3D loss
        - test_psnr_avg (float): the average test psnr
    """
    c = config # shortening due to numerous uses

    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, sampler=None,
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0)

    loss_f = model.loss_f

    test_results_dir = os.path.join(c.results_path, f"{c.run_name}_{c.date}_{id}_results")
    os.makedirs(test_results_dir, exist_ok=True)

    test_loss_meter = AverageMeter()
    test_mse_meter = AverageMeter()
    test_l1_meter = AverageMeter()
    test_ssim_meter = AverageMeter()
    test_ssim3D_meter = AverageMeter()
    test_psnr_meter = AverageMeter()

    mse_loss_func = MSE_Loss(complex_i=c.complex_i)
    l1_loss_func = L1_Loss(complex_i=c.complex_i)
    ssim_loss_func = SSIM_Loss(complex_i=c.complex_i, device=device)
    ssim3D_loss_func = SSIM3D_Loss(complex_i=c.complex_i, device=device)
    psnr_func = PSNR()

    model.eval()
    model.to(device)

    cutout = (c.time, c.height[-1], c.width[-1])
    overlap = (c.time//4, c.height[-1]//4, c.width[-1]//4)

    test_loader_iter = iter(test_loader)
    total_iters = len(test_loader) if not c.debug else 5
    
    with torch.no_grad():
        with tqdm(total=total_iters) as pbar:

            for idx in range(total_iters):

                x, y, gmaps_median, noise_sigmas = next(test_loader_iter)
                x = x.to(device)
                y = y.to(device)

                try:
                    _, output = running_inference(model, x, cutout=cutout, overlap=overlap, device=device)
                except:
                    _, output = running_inference(model, x, cutout=cutout, overlap=overlap, device="cpu")
                    y = y.to("cpu")

                loss = loss_f(output, y)

                mse_loss = mse_loss_func(output, y).item()
                l1_loss = l1_loss_func(output, y).item()
                ssim_loss = ssim_loss_func(output, y).item()
                ssim3D_loss = ssim3D_loss_func(output, y).item()
                psnr = psnr_func(output, y).item()

                total = x.shape[0]

                test_loss_meter.update(loss.item(), n=total)
                test_mse_meter.update(mse_loss, n=total)
                test_l1_meter.update(l1_loss, n=total)
                test_ssim_meter.update(ssim_loss, n=total)
                test_ssim3D_meter.update(ssim3D_loss, n=total)
                test_psnr_meter.update(psnr, n=total)

                wandb.log({f"running_test_loss_{id}": loss.item(),
                            f"running_test_mse_loss_{id}": mse_loss,
                            f"running_test_l1_loss_{id}": l1_loss,
                            f"running_test_ssim_loss_{id}": ssim_loss,
                            f"running_test_ssim3D_loss_{id}": ssim3D_loss,
                            f"running_test_psnr_{id}": psnr})

                pbar.update(1)
                pbar.set_description(f"Test, {x.shape}, "+
                                        f"{loss.item():.4f}, {mse_loss:.4f}, {l1_loss:.4f}, "+
                                        f"{ssim_loss:.4f}, {ssim3D_loss:.4f}, {psnr:.4f},")

                save_image(test_results_dir, c.complex_i, idx,\
                            x.cpu().detach().numpy(),\
                            output.cpu().detach().numpy(),\
                            y.cpu().detach().numpy())

            pbar.set_description(f"Test, {x.shape}, {test_loss_meter.avg:.4f}, "+
                                    f"{test_mse_meter.avg:.4f}, {test_l1_meter.avg:.4f}, {test_ssim_meter.avg:.4f}, "+
                                    f"{test_ssim3D_meter.avg:.4f}, {test_psnr_meter.avg:.4f}")

    logging.getLogger("file_only").info(f"Test, {x.shape}, {test_loss_meter.avg:.4f}, "+
                                        f"{test_mse_meter.avg:.4f}, {test_l1_meter.avg:.4f}, {test_ssim_meter.avg:.4f}, "+
                                        f"{test_ssim3D_meter.avg:.4f}, {test_psnr_meter.avg:.4f}")

    wandb.log({f"test_loss_{id}": test_loss_meter.avg,
                f"test_mse_loss_{id}": test_mse_meter.avg,
                f"test_l1_loss_{id}": test_l1_meter.avg,
                f"test_ssim_loss_{id}": test_ssim_meter.avg,
                f"test_ssim3D_loss_{id}": test_ssim3D_meter.avg,
                f"test_psnr_{id}": test_psnr_meter.avg})

    losses = test_loss_meter.avg, test_mse_meter.avg, test_l1_meter.avg, test_ssim_meter.avg, test_ssim3D_meter.avg, test_psnr_meter.avg

    save_results(config, losses, id=id)

    return losses

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

    parser.add_argument("--data_root", type=str, default=None, help='root folder for the data')
    parser.add_argument("--test_files", type=str, nargs='+', default=["train_3D_3T_retro_cine_2020_small_test.h5"], help='list of test h5files')
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--saved_model_config", type=str, default=None, help='the config of the model. endswith ".json"')

    parser = add_shared_STCNNT_args(parser=parser)

    return parser.parse_args()

def check_args(config):
    """
    checks the cmd args to make sure they are correct
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - config (Namespace): the checked and updated argparse for MRI
    """
    assert config.run_name is not None, f"Please provide a \"--run_name\" for wandb"
    assert config.data_root is not None, f"Please provide a \"--data_root\" to load the data"
    assert config.test_files is not None, f"Please provide a \"--test_files\" to load the data"
    assert config.results_path is not None, f"Please provide a \"--results_path\" to save the results in"
    assert config.saved_model_path is not None, f"Please provide a \"--saved_model_path\" for loading a checkpoint"
    assert config.saved_model_config is not None, f"Please provide a \"--saved_model_config\" for loading a config"

    assert config.saved_model_path.endswith(".pt") or config.saved_model_path.endswith(".pts"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\""
    assert config.saved_model_config.endswith(".json"),\
            f"Config should be \"*.json\""

    config.C_in = 3 if config.complex_i else 2
    config.C_out = 2 if config.complex_i else 1
    config.load_path = config.saved_model_path
    if config.log_path is None: config.log_path = config.results_path

    return config

# -------------------------------------------------------------------------------------------------
# load model

def load_model(config):
    """
    load a ".pt" or ".pts" model
    ".pt" models require ".json" to create the model
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - model (torch model): the model ready for inference
    """
    if config.saved_model_path.endswith(".pt"):
        config.load_path = config.saved_model_path
        model = STCNNT_MRI(config=config)
    else:
        model = torch.jit.load(config.saved_model_path)

    return model

# -------------------------------------------------------------------------------------------------
# save results

def save_results(config, losses, id=""):
    """
    save the results
    @args:
        - config (Namespace): runtime namespace for setup
        - losses (float list): test losses:
            - model loss
            - mse loss
            - l1 loss
            - ssim loss
            - ssim3D loss
            - psnr
        - id (str): unique id to save results with
    """
    file_name = f"{config.run_name}_{config.date}_{id}_results"
    results_file_name = os.path.join(config.results_path, file_name)

    result_dict = {f"test_loss_{id}": losses[0],
                    f"test_mse_loss_{id}": losses[1],
                    f"test_l1_loss_{id}": losses[2],
                    f"test_ssim_loss_{id}": losses[3],
                    f"test_ssim3D_loss_{id}": losses[4],
                    f"test_psnr_{id}": losses[5]}

    with open(f"{results_file_name}.json", "w") as file:
        json.dump(result_dict, file)

def save_image(path, complex_i, i, noisy, predi, clean):
    """
    Saves the image locally as a 4D tiff [T,C,H,W]
    3 channels: noisy, predicted, clean
    If complex image then save the magnitude
    @args:
        - path (str): the directory to save the images in
        - complex_i (bool): complex images or not
        - i (int): index of the image
        - noisy (5D numpy array): the noisy image
        - predi (5D numpy array): the predicted image
        - clean (5D numpy array): the clean image
    """

    if complex_i:
        save_x = np.sqrt(np.square(noisy[0,:,0]) + np.square(noisy[0,:,1]))
        save_p = np.sqrt(np.square(predi[0,:,0]) + np.square(predi[0,:,1]))
        save_y = np.sqrt(np.square(clean[0,:,0]) + np.square(clean[0,:,1]))
    else:
        save_x = noisy[0,:,0]
        save_p = predi[0,:,0]
        save_y = clean[0,:,0]

    composed_channel_wise = np.transpose(np.array([save_x, save_p, save_y]), (1,0,2,3))

    tifffile.imwrite(os.path.join(path, f"Image_{i:03d}_{save_x.shape}.tif"),\
                        composed_channel_wise, imagej=True)

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results

def main():

    config = check_args(arg_parser())
    setup_run(config, dirs=["log_path"])

    model = load_model(config)
    wandb.init(project=config.project, entity=config.wandb_entity, config=config,
                        name=config.run_name, notes=config.run_notes)

    eval_test(model, config, test_set=None, device="cuda")

if __name__=="__main__":
    main()
