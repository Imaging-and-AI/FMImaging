"""
MRI run
"""


import copy
import numpy as np
from time import time

import os
import sys
import logging

from colorama import Fore, Back, Style
import nibabel as nib

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from trainer import *
from utils.status import model_info, start_timer, end_timer, support_bfloat16
from setup import setup_logger, Nestedspace
from optim.optim_utils import compute_total_steps

# Default functions
from setup import parse_config_and_setup_run, config_to_yaml
from optim.optim_base import OptimManager
from utils.status import get_device

# Custom functions
from mri_parser import mri_parser
from mri_data import MRIDenoisingDatasetTrain, load_mri_data
from mri_loss import mri_loss 
from mri_model import STCNNT_MRI, MRI_hrnet, MRI_double_net, omnivore_MRI, create_model
from LSUV import LSUVinit
from mri_metrics import MriMetricManager
from mri_trainer import MRITrainManager, get_rank_str
from inference import apply_model, load_model, apply_model_3D, load_model_pre_backbone_post

# -------------------------------------------------------------------------------------------------
def main():

    # -----------------------------------------------

    config = parse_config_and_setup_run(mri_parser) 

    if config.complex_i:
        config.no_in_channel = 3
    else:
        config.no_in_channel = 2

    # -----------------------------------------------

    if config.ddp:
        rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        device = torch.device(f"cuda:{rank}")
        config.device = device
    else:
        rank = -1
        global_rank = -1
        device = get_device()

    rank_str = get_rank_str(rank)

    # -----------------------------------------------

    # Save config to yaml file
    if rank<=0:
        yaml_file = config_to_yaml(config,os.path.join(config.log_dir, config.run_name))
        config.yaml_file = yaml_file

    # -----------------------------------------------

    logging.info(f"{rank_str}, {Fore.YELLOW}config, min noise level - {config.min_noise_level}, max noise level - {config.max_noise_level} {Style.RESET_ALL}")

    start = time()
    train_set, val_set, test_set = load_mri_data(config=config)
    logging.info(f"load_mri_data took {time() - start} seconds ...")

    if not config.disable_LSUV:
        if (config.pre_model_load_path is None and config.backbone_model_load_path is None and config.post_model_load_path is None) or (not config.continued_training):
            t0 = time()
            num_samples = len(train_set[-1])
            sampled_picked = np.random.randint(0, num_samples, size=32)
            input_data  = torch.stack([train_set[-1][i][0] for i in sampled_picked])
            logging.info(f"{rank_str}, prepared data {input_data.shape}, LSUV prep data took {time()-t0 : .2f} seconds ...")

    # -----------------------------------------------

    loss_f = mri_loss(config=config)

    # -----------------------------------------------
    # arguments to be replaced if loading a saved pth

    num_epochs = config.num_epochs
    batch_size = config.batch_size
    global_lr = config.optim.global_lr
    lr = config.optim.lr
    optim = config.optim
    scheduler_type = config.scheduler_type
    scheduler = config.scheduler
    losses = config.losses
    loss_weights = config.loss_weights
    weighted_loss_snr = config.weighted_loss_snr
    weighted_loss_temporal = config.weighted_loss_temporal
    weighted_loss_added_noise = config.weighted_loss_added_noise
    save_train_samples = config.save_train_samples
    save_val_samples = config.save_val_samples
    save_test_samples = config.save_test_samples
    num_saved_samples = config.num_saved_samples
    height = config.mri_height
    width = config.mri_width
    c_time = config.time
    use_amp = config.use_amp
    num_workers = config.num_workers
    continued_training = config.continued_training
    freeze_pre = config.freeze_pre
    freeze_backbone = config.freeze_backbone
    freeze_post = config.freeze_post
    model_type = config.model_type
    run_name = config.run_name
    run_notes = config.run_notes
    disable_LSUV = config.disable_LSUV
    super_resolution = config.super_resolution
    with_data_degrading = config.with_data_degrading

    post_backbone = config.post_backbone

    post_hrnet_block_str = config.post_hrnet.block_str
    post_hrnet_separable_conv = config.post_hrnet.separable_conv

    post_mixed_unetr_num_resolution_levels = config.post_mixed_unetr.num_resolution_levels
    post_mixed_unetr_block_str = config.post_mixed_unetr.block_str
    post_mixed_unetr_use_unet_attention = config.post_mixed_unetr.use_unet_attention
    post_mixed_unetr_transformer_for_upsampling = config.post_mixed_unetr.transformer_for_upsampling
    post_mixed_unetr_n_heads = config.post_mixed_unetr.n_heads
    post_mixed_unetr_use_conv_3d = config.post_mixed_unetr.use_conv_3d
    post_mixed_unetr_use_window_partition = config.post_mixed_unetr.use_window_partition
    post_mixed_unetr_separable_conv = config.post_mixed_unetr.separable_conv

    pre_model_load_path = config.pre_model_load_path
    backbone_model_load_path = config.backbone_model_load_path
    post_model_load_path = config.post_model_load_path
    post_model_of_1st_net = config.post_model_of_1st_net

    train_model = config.train_model

    min_noise_level = config.min_noise_level
    max_noise_level = config.max_noise_level

    ratio_to_eval = config.ratio_to_eval
    eval_train_set = config.eval_train_set
    eval_val_set = config.eval_val_set

    ddp = config.ddp

    # -----------------------------------------------        

    if config.pre_model_load_path is not None:
        status = torch.load(config.pre_model_load_path, map_location=device)
        config = status['config']

        # overwrite the config parameters with current settings
        config.device = device
        config.losses = losses
        config.loss_weights = loss_weights
        config.optim = optim
        config.scheduler_type = scheduler_type
        config.scheduler = scheduler
        config.optim.lr = lr
        config.optim.global_lr = global_lr
        config.num_epochs = num_epochs
        config.batch_size = batch_size
        config.weighted_loss_snr = weighted_loss_snr
        config.weighted_loss_temporal = weighted_loss_temporal
        config.weighted_loss_added_noise = weighted_loss_added_noise
        config.save_train_samples = save_train_samples
        config.save_val_samples = save_val_samples
        config.save_test_samples = save_test_samples
        config.num_saved_samples = num_saved_samples
        config.mri_height = height
        config.mri_width = width
        config.time = c_time
        config.use_amp = use_amp
        config.num_workers = num_workers
        config.freeze_pre = freeze_pre
        config.freeze_backbone = freeze_backbone
        config.freeze_post = freeze_post
        config.model_type = model_type
        config.run_name = run_name
        config.run_notes = run_notes
        config.disable_LSUV = disable_LSUV
        config.super_resolution = super_resolution
        config.with_data_degrading = with_data_degrading

        config.post_backbone = post_backbone

        config.post_hrnet = Nestedspace()
        config.post_hrnet.block_str = post_hrnet_block_str
        config.post_hrnet.separable_conv = post_hrnet_separable_conv

        config.post_mixed_unetr = Nestedspace()
        config.post_mixed_unetr.num_resolution_levels = post_mixed_unetr_num_resolution_levels
        config.post_mixed_unetr.block_str = post_mixed_unetr_block_str
        config.post_mixed_unetr.use_unet_attention = post_mixed_unetr_use_unet_attention
        config.post_mixed_unetr.transformer_for_upsampling = post_mixed_unetr_transformer_for_upsampling
        config.post_mixed_unetr.n_heads = post_mixed_unetr_n_heads
        config.post_mixed_unetr.use_conv_3d = post_mixed_unetr_use_conv_3d
        config.post_mixed_unetr.use_window_partition = post_mixed_unetr_use_window_partition
        config.post_mixed_unetr.separable_conv = post_mixed_unetr_separable_conv

        config.pre_model_load_path = pre_model_load_path
        config.backbone_model_load_path = backbone_model_load_path
        config.post_model_load_path = post_model_load_path
        config.post_model_of_1st_net = post_model_of_1st_net

        config.train_model = train_model

        config.min_noise_level = min_noise_level
        config.max_noise_level = max_noise_level

        config.ddp = ddp

        config.ratio_to_eval = ratio_to_eval

        config.eval_train_set = eval_train_set
        config.eval_val_set = eval_val_set

        logging.info(f"{rank_str}, {Fore.WHITE}=============================================================={Style.RESET_ALL}")

    # -----------------------------------------------

    if continued_training:
        config.load_optim_and_sched = True
    else:
        config.load_optim_and_sched = False

    # -----------------------------------------------

    model = create_model(config=config, model_type=config.model_type) 

    # -----------------------------------------------
    logging.info(f"{rank_str}, load saved model, continued_training - {continued_training}")
    if continued_training:
        config.device = device

        model.load_pre(config.pre_model_load_path, device=device)

        model.load_backbone(config.backbone_model_load_path, device=device)

        model.load_post(config.post_model_load_path, device=device)

    else: # new stage training
        model = model.to(device)

        if not config.disable_LSUV:
            t0 = time()
            logging.info(f"{rank_str}, LSUVinit starts ...")
            LSUVinit(model, input_data.to(device=device), verbose=False, cuda=True)
            logging.info(f"{rank_str}, LSUVinit took {time()-t0 : .2f} seconds ...")

        # ------------------------------
        if config.pre_model_load_path is not None:
            logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, pre_state{Style.RESET_ALL}")
            model.load_pre(config.pre_model_load_path, device=device)
        else:
            logging.info(f"{rank_str}, {Fore.RED}load saved model, WITHOUT pre_state{Style.RESET_ALL}")

        # if config.freeze_pre:
        #     logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, pre requires_grad_(False){Style.RESET_ALL}")
        #     model.freeze_pre()
        # else:
        #     logging.info(f"{rank_str}, {Fore.RED}load saved model, pre requires_grad_(True){Style.RESET_ALL}")
        # ------------------------------
        if config.backbone_model_load_path is not None:
            logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, backbone_state{Style.RESET_ALL}")
            model.load_backbone(config.backbone_model_load_path, device=device)
        else:
            logging.info(f"{rank_str}, {Fore.RED}load saved model, WITHOUT backbone_state{Style.RESET_ALL}")

        if config.post_model_of_1st_net is not None and config.model_type == "MRI_double_net":
            logging.info(f"{rank_str}, {Fore.YELLOW}load post module of the 1st net{Style.RESET_ALL}")
            model.load_post_1st_net(config.post_model_of_1st_net, device=device)

        # if config.freeze_backbone:
        #     logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, backbone requires_grad_(False){Style.RESET_ALL}")
        #     model.freeze_backbone()
        # else:
        #     logging.info(f"{rank_str}, {Fore.RED}load saved model, backbone requires_grad_(True){Style.RESET_ALL}")
        # ------------------------------
        if config.post_model_load_path is not None:
            logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, post_state{Style.RESET_ALL}")
            model.load_post(config.post_model_load_path, device=device)
        else:
            logging.info(f"{rank_str}, {Fore.RED}load saved model, WITHOUT post_state{Style.RESET_ALL}")

        # if config.freeze_post:
        #     logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, post requires_grad_(False){Style.RESET_ALL}")
        #     model.freeze_post()
        # else:
        #     logging.info(f"{rank_str}, {Fore.RED}load saved model, post requires_grad_(True){Style.RESET_ALL}")

    # ---------------------------------------------------

    if config.freeze_pre:
        logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, pre requires_grad_(False){Style.RESET_ALL}")
        model.freeze_pre()
    else:
        logging.info(f"{rank_str}, {Fore.RED}load saved model, pre requires_grad_(True){Style.RESET_ALL}")

    if config.freeze_backbone:
        logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, backbone requires_grad_(False){Style.RESET_ALL}")
        model.freeze_backbone()
    else:
        logging.info(f"{rank_str}, {Fore.RED}load saved model, backbone requires_grad_(True){Style.RESET_ALL}")

    if config.freeze_post:
        logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, post requires_grad_(False){Style.RESET_ALL}")
        model.freeze_post()
    else:
        logging.info(f"{rank_str}, {Fore.RED}load saved model, post requires_grad_(True){Style.RESET_ALL}")

    # ---------------------------------------------------

    model = model.to(device)

    # model.eval()
    # with torch.inference_mode():
    #     save_path, save_file_name, config_yaml_file = model.save_entire_model(epoch=0)
    #     model_loaded, config_loaded = load_model(os.path.join(save_path, save_file_name+'.pth'))

    # ---------------------------------------------------

    optim_manager = OptimManager(config=config, model_manager=model, train_set=train_set)

    logging.info(f"{rank_str}, after initializing model, the config for running - {config}")
    logging.info(f"{rank_str}, after initializing model, config.use_amp for running - {config.use_amp}")
    logging.info(f"{rank_str}, after initializing model, config.optim for running - {config.optim}")
    logging.info(f"{rank_str}, after initializing model, config.scheduler_type for running - {config.scheduler_type}")
    logging.info(f"{rank_str}, after initializing model, config.weighted_loss_snr for running - {config.weighted_loss_snr}")
    logging.info(f"{rank_str}, after initializing model, config.weighted_loss_temporal for running - {config.weighted_loss_temporal}")
    logging.info(f"{rank_str}, after initializing model, config.weighted_loss_added_noise for running - {config.weighted_loss_added_noise}")
    logging.info(f"{rank_str}, after initializing model, config.num_workers for running - {config.num_workers}")
    logging.info(f"{rank_str}, after initializing model, config.super_resolution for running - {config.super_resolution}")
    logging.info(f"{rank_str}, after initializing model, config.post_backbone for running - {config.post_backbone}")
    logging.info(f"{rank_str}, after initializing model, config.post_hrnet for running - {config.post_hrnet}")
    logging.info(f"{rank_str}, after initializing model, config.post_mixed_unetr for running - {config.post_mixed_unetr}")

    logging.info(f"{rank_str}, after initializing model, optim_manager.curr_epoch for running - {optim_manager.curr_epoch}")
    logging.info(f"{rank_str}, {Fore.GREEN}after initializing model, model type - {config.model_type}{Style.RESET_ALL}")
    logging.info(f"{rank_str}, {Fore.RED}after initializing model, model.device - {model.device}{Style.RESET_ALL}")
    logging.info(f"{rank_str}, {Fore.WHITE}=============================================================={Style.RESET_ALL}")

    if config.ddp:
        dist.barrier()

    # -----------------------------------------------

    metric_manager = MriMetricManager(config=config)

    # ---------------------------------------------------

    trainer = MRITrainManager(config=config,
                            train_sets=train_set,
                            val_sets=val_set,
                            test_sets=test_set,
                            loss_f=loss_f,
                            model_manager=model,
                            optim_manager=optim_manager,
                            metric_manager=metric_manager)

    trainer.train()

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()
