"""
Trainer for CT denoising.
Provides the mian function to call for training:
    - trainer
"""
import copy
import wandb
import numpy as np
import torch
import torch.distributed as dist

from time import time
from tqdm import tqdm
from colorama import Fore, Back, Style
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *
from model_base.losses import *
from utils.save_model import save_final_model
from utils.running_inference import running_inference

from ct.model_ct import STCNNT_CT, STCNNT_double_net
from ct.data_ct import CtDatasetTrain, load_ct_data

# Helpers
# -------------------------------------------------------------------------------------------------

class ct_trainer_meters(object):
    """
    A helper class to organize meters for training
    """
    def __init__(self, config, device):
        super().__init__()

        self.config = config

        self.mse_meter = AverageMeter()
        self.l1_meter = AverageMeter()
        self.ssim_meter = AverageMeter()
        self.ssim3D_meter = AverageMeter()
        self.psnr_meter = AverageMeter()
        self.psnr_loss_meter = AverageMeter()
        self.gaussian_meter = AverageMeter()
        self.gaussian3D_meter = AverageMeter()

        self.mse_loss_func = MSE_Loss()
        self.l1_loss_func = L1_Loss()
        self.ssim_loss_func = SSIM_Loss(device=device)
        self.ssim3D_loss_func = SSIM3D_Loss(device=device)
        self.psnr_func = PSNR(range=1.0)
        self.psnr_loss_func = PSNR_Loss(range=1.0)
        self.gaussian_func = GaussianDeriv_Loss(sigmas=[0.5, 1.0, 1.5], device=device)
        self.gaussian3D_func = GaussianDeriv3D_Loss(sigmas=[0.5, 1.0, 1.5], sigmas_T=[0.5, 0.5, 0.5], device=device)

    def reset(self):

        self.mse_meter.reset()
        self.l1_meter.reset()
        self.ssim_meter.reset()
        self.ssim3D_meter.reset()
        self.psnr_meter.reset()
        self.psnr_loss_meter.reset()
        self.gaussian_meter.reset()
        self.gaussian3D_meter.reset()

    def update(self, output, y):

        total = y.shape[0]

        mse_loss = self.mse_loss_func(output, y).item()
        l1_loss = self.l1_loss_func(output, y).item()
        ssim_loss = self.ssim_loss_func(output, y).item()
        ssim3D_loss = self.ssim3D_loss_func(output, y).item()
        psnr_loss = self.psnr_loss_func(output, y).item()
        psnr = self.psnr_func(output, y).item()
        gauss_loss = self.gaussian_func(output, y).item()
        gauss3D_loss = self.gaussian3D_func(output, y).item()

        self.mse_meter.update(mse_loss, n=total)
        self.l1_meter.update(l1_loss, n=total)
        self.ssim_meter.update(ssim_loss, n=total)
        self.ssim3D_meter.update(ssim3D_loss, n=total)
        self.psnr_loss_meter.update(psnr_loss, n=total)
        self.psnr_meter.update(psnr, n=total)
        self.gaussian_meter.update(gauss_loss, n=total)
        self.gaussian3D_meter.update(gauss3D_loss, n=total)

    def get_loss(self):
        # mse, l1, ssim, ssim3D, psnr_loss, psnr, gaussian, gaussian3D
        return self.mse_meter.avg, self.l1_meter.avg, self.ssim_meter.avg, self.ssim3D_meter.avg,\
            self.psnr_loss_meter.avg, self.psnr_meter.avg, self.gaussian_meter.avg, self.gaussian3D_meter.avg

# -------------------------------------------------------------------------------------------------

def create_log_str(config, epoch, rank, data_shape, loss, loss_meters, curr_lr, role):

    if data_shape is not None:
        data_shape_str = f"{data_shape[-1]:3d}, "
    else:
        data_shape_str = ""

    if curr_lr >=0:
        lr_str = f", lr {curr_lr:.8f}"
    else:
        lr_str = ""

    if role == 'tra':
        C = Fore.YELLOW
    else:
        C = Fore.GREEN

    mse, l1, ssim, ssim3D, psnr_loss, psnr, gaussian, gaussian3D = loss_meters.get_loss()

    str= f"{Fore.GREEN}Epoch {epoch}/{config.num_epochs}, "\
            f"{C}{role}, {Style.RESET_ALL}{rank}, " + data_shape_str + \
            f"{Fore.BLUE}{Back.WHITE}{Style.BRIGHT}loss {loss:.4f},"\
            f"{Style.RESET_ALL} {Fore.WHITE}{Back.LIGHTBLUE_EX}{Style.NORMAL}{Style.RESET_ALL}"\
            f"{C}mse {mse:.4f}, l1 {l1:.4f}, ssim {ssim:.4f}, ssim3D {ssim3D:.4f}, "\
            f"gaussian {gaussian:.4f}, gaussian3D {gaussian3D:.4f}, psnr loss {psnr_loss:.4f}, psnr {psnr:.4f}{Style.RESET_ALL}{lr_str} "

    return str

# -------------------------------------------------------------------------------------------------

def distribute_learning_rates(rank, optim, src=0):
    """
    For ddp.
    distributes learning rates across processes
    """

    N = len(optim.param_groups)
    new_lr = torch.zeros(N).to(rank)
    for ind in range(N):
        new_lr[ind] = optim.param_groups[ind]["lr"]

    dist.broadcast(new_lr, src=src)

    if rank != src:
        for ind in range(N):
            optim.param_groups[ind]["lr"] = new_lr[ind].item()

# -------------------------------------------------------------------------------------------------

def get_rank_str(rank):
    if rank == 0:
        return f"{Fore.BLUE}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 1:
        return f"{Fore.GREEN}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 2:
        return f"{Fore.YELLOW}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 3:
        return f"{Fore.MAGENTA}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 4:
        return f"{Fore.LIGHTYELLOW_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 5:
        return f"{Fore.LIGHTBLUE_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 6:
        return f"{Fore.LIGHTRED_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 7:
        return f"{Fore.LIGHTCYAN_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"

    return f"{Fore.WHITE}{Style.BRIGHT}rank {rank} {Style.RESET_ALL}"

# -------------------------------------------------------------------------------------------------

def create_model(config, model_type, total_steps=-1):

    if model_type == "STCNNT_CT":
        model = STCNNT_CT(config=config, total_steps=total_steps)
    elif model_type == "STCNNT_double":
        model = STCNNT_double_net(config=config, total_steps=total_steps)
    else:
        raise NotImplementedError(f"model type not supported:{model_type}")

    return model

# -------------------------------------------------------------------------------------------------

def create_wandb_log_vid(noisy, predi, clean, complex_i=False):
    """
    Create the log video for wandb as a 5D gif [B,T,C,H,W]
    If complex image then save the magnitude using first 2 channels
    Else use just the first channel
    @args:
        - complex_i (bool): complex images or not
        - noisy (5D numpy array): the noisy image [B, T, C+1, H, W]
        - predi (5D numpy array): the predicted image [B, T, C, H, W]
        - clean (5D numpy array): the clean image [B, T, C, H, W]
    """

    if noisy.ndim == 4:
        noisy = np.expand_dims(noisy, axis=0)
        predi = np.expand_dims(predi, axis=0)
        clean = np.expand_dims(clean, axis=0)

    if complex_i:
        save_x = np.sqrt(np.square(noisy[:,:,0,:,:]) + np.square(noisy[:,:,1,:,:]))
        save_p = np.sqrt(np.square(predi[:,:,0,:,:]) + np.square(predi[:,:,1,:,:]))
        save_y = np.sqrt(np.square(clean[:,:,0,:,:]) + np.square(clean[:,:,1,:,:]))
    else:
        save_x = noisy[:,:,0,:,:]
        save_p = predi[:,:,0,:,:]
        save_y = clean[:,:,0,:,:]

    save_x = normalize_image(save_x, percentiles=[0,100])
    save_p = normalize_image(save_p, percentiles=[0,100])
    save_y = normalize_image(save_y, percentiles=[0,100])

    # scale down the images before logging
    while save_x.shape[2] > 500 or save_x.shape[3] > 500:
        save_x = save_x[:,:,::2,::2]
        save_p = save_p[:,:,::2,::2]
        save_y = save_y[:,:,::2,::2]

    B, T, H, W = save_x.shape

    max_col = 16
    if B>max_col:
        num_row = B//max_col
        if max_col*num_row < B:
            num_row += 1
        composed_res = np.zeros((T, 3*H*num_row, max_col*W))
        for b in range(B):
            r = b//max_col
            c = b - r*max_col
            for t in range(T):
                composed_res[t, 3*r*H:(3*r+1)*H, c*W:(c+1)*W] = save_x[b,t,:,:].squeeze()
                composed_res[t, (3*r+1)*H:(3*r+2)*H, c*W:(c+1)*W] = save_p[b,t,:,:].squeeze()
                composed_res[t, (3*r+2)*H:(3*r+3)*H, c*W:(c+1)*W] = save_y[b,t,:,:].squeeze()
    elif B>2:
        composed_res = np.zeros((T, 3*H, B*W))
        for b in range(B):
            for t in range(T):
                composed_res[t, :H, b*W:(b+1)*W] = save_x[b,t,:,:].squeeze()
                composed_res[t, H:2*H, b*W:(b+1)*W] = save_p[b,t,:,:].squeeze()
                composed_res[t, 2*H:3*H, b*W:(b+1)*W] = save_y[b,t,:,:].squeeze()
    else:
        composed_res = np.zeros((T, B*H, 3*W))
        for b in range(B):
            for t in range(T):
                composed_res[t, b*H:(b+1)*H, :W] = save_x[b,t,:,:].squeeze()
                composed_res[t, b*H:(b+1)*H, W:2*W] = save_p[b,t,:,:].squeeze()
                composed_res[t, b*H:(b+1)*H, 2*W:3*W] = save_y[b,t,:,:].squeeze()

    # composed_res = np.clip(composed_res, a_min=0, a_max=np.percentile(composed_res, 99))

    temp = np.zeros_like(composed_res)
    composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

    return np.repeat(composed_res[:,np.newaxis,:,:].astype('uint8'), 3, axis=1)

# -------------------------------------------------------------------------------------------------
# trainer

def trainer(rank, global_rank, config, wandb_run):
    """
    The trainer cycle. Allows training on cpu/single gpu/multiple gpu(ddp)
    @args:
        - rank (int): for distributed data parallel (ddp)
            -1 if running on cpu or only one gpu
        - global_rank (int): for distributed data parallel (ddp)
            global rank of current process
        - config (Namespace): runtime namespace for setup
        - wandb_run (wandb.Run): the run object for loggin to wandb
    """
    c = config # shortening due to numerous uses
    rank_str = get_rank_str(rank)

    # -----------------------------------------------

    start = time()
    train_set, val_set, test_set = load_ct_data(config=c)
    logging.info(f"{rank_str}, load_ct_data took {time() - start} seconds ...")

    total_num_samples = sum([len(s) for s in train_set])

    total_steps = compute_total_steps(c, total_num_samples)
    logging.info(f"{rank_str}, total_steps for this run: {total_steps}, len(train_set) {[len(s) for s in train_set]}, batch {c.batch_size}")

    # -----------------------------------------------
    if not c.disable_LSUV:
        if (c.load_path is None) or (not c.continued_training):
            t0 = time()
            num_samples = len(train_set[-1])
            sampled_picked = np.random.randint(0, num_samples, size=32)
            input_data  = torch.stack([train_set[-1][i][0] for i in sampled_picked])
            logging.info(f"{rank_str}, LSUV prep data took {time()-t0 : .2f} seconds ...")

    # -----------------------------------------------

    if c.ddp:
        device = torch.device(f"cuda:{rank}")
        c.device = device
    else:
        device = c.device

    # -----------------------------------------------

    logging.info(f"{rank_str}, {Style.BRIGHT}{Fore.RED}{Back.LIGHTWHITE_EX}RUN NAME - {c.run_name}{Style.RESET_ALL}")

    # -----------------------------------------------

    num_epochs = c.num_epochs
    batch_size = c.batch_size
    lr = c.global_lr
    optim = c.optim
    scheduler_type = c.scheduler_type
    losses = c.losses
    loss_weights = c.loss_weights
    save_samples = c.save_samples
    num_saved_samples = c.num_saved_samples
    height = c.height
    width = c.width
    c_time = c.time
    use_amp = c.use_amp
    num_workers = c.num_workers
    lr_pre = c.lr_pre
    lr_backbone = c.lr_backbone
    lr_post = c.lr_post
    continued_training = c.continued_training
    disable_pre = c.disable_pre
    disable_backbone = c.disable_backbone
    disable_post = c.disable_post
    model_type = c.model_type
    not_load_pre = c.not_load_pre
    not_load_backbone = c.not_load_backbone
    not_load_post = c.not_load_post
    run_name = c.run_name
    run_notes = c.run_notes
    disable_LSUV = c.disable_LSUV
    post_backbone = c.post_backbone
    training_step = c.training_step
    device_type = "cpu" if c.device == torch.device("cpu") else "cuda"

    ddp = c.ddp

    if c.load_path is not None:

        status = torch.load(c.load_path)
        c = status['config']

        # overwrite the config parameters with current settings
        c.device = device
        c.losses = losses
        c.loss_weights = loss_weights
        c.optim = optim
        c.scheduler_type = scheduler_type
        c.global_lr = lr
        c.num_epochs = num_epochs
        c.batch_size = batch_size
        c.save_samples = save_samples
        c.num_saved_samples = num_saved_samples
        c.height = height
        c.width = width
        c.time = c_time
        c.use_amp = use_amp
        c.num_workers = num_workers
        c.lr_pre = lr_pre
        c.lr_backbone = lr_backbone
        c.lr_post = lr_post
        c.disable_pre = disable_pre
        c.disable_backbone = disable_backbone
        c.disable_post = disable_post
        c.not_load_pre = not_load_pre
        c.not_load_backbone = not_load_backbone
        c.not_load_post = not_load_post
        c.model_type = model_type
        c.run_name = run_name
        c.run_notes = run_notes
        c.disable_LSUV = disable_LSUV
        c.post_backbone = post_backbone
        c.training_step = training_step
        # c.load_path = load_path

        logging.info(f"{rank_str}, {Fore.WHITE}=============================================================={Style.RESET_ALL}")

        model = create_model(c, model_type, total_steps)

        if 'backbone_state' in status:
            logging.info(f"{rank_str}, load saved model, continued_training - {continued_training}")
            if continued_training:
                model.load_from_status(status=status, device=device, load_others=continued_training)
            else: # new stage training
                model = model.to(device)
                if not c.disable_LSUV:
                    t0 = time()
                    LSUVinit(model, input_data.to(device=device), verbose=True, cuda=True)
                    logging.info(f"{rank_str}, LSUVinit took {time()-t0 : .2f} seconds ...")

                # ------------------------------
                if not not_load_pre:
                    logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, pre_state{Style.RESET_ALL}")
                    model.pre.load_state_dict(status['pre_state'])
                else:
                    logging.info(f"{rank_str}, {Fore.RED}load saved model, WITHOUT pre_state{Style.RESET_ALL}")

                if disable_pre:
                    logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, pre requires_grad_(False){Style.RESET_ALL}")
                    model.pre.requires_grad_(False)
                    for param in model.pre.parameters():
                        param.requires_grad = False
                else:
                    logging.info(f"{rank_str}, {Fore.RED}load saved model, pre requires_grad_(True){Style.RESET_ALL}")
                # ------------------------------
                if not not_load_backbone:
                    logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, backbone_state{Style.RESET_ALL}")
                    model.backbone.load_state_dict(status['backbone_state'])
                else:
                    logging.info(f"{rank_str}, {Fore.RED}load saved model, WITHOUT backbone_state{Style.RESET_ALL}")

                if disable_backbone:
                    logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, backbone requires_grad_(False){Style.RESET_ALL}")
                    model.backbone.requires_grad_(False)
                    for param in model.backbone.parameters():
                        param.requires_grad = False
                else:
                    logging.info(f"{rank_str}, {Fore.RED}load saved model, backbone requires_grad_(True){Style.RESET_ALL}")
                # ------------------------------
                if not not_load_post:
                    logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, post_state{Style.RESET_ALL}")
                    model.post.load_state_dict(status['post_state'])
                else:
                    logging.info(f"{rank_str}, {Fore.RED}load saved model, WITHOUT post_state{Style.RESET_ALL}")

                if disable_post:
                    logging.info(f"{rank_str}, {Fore.YELLOW}load saved model, post requires_grad_(False){Style.RESET_ALL}")
                    model.post.requires_grad_(False)
                    for param in model.post.parameters():
                        param.requires_grad = False
                else:
                    logging.info(f"{rank_str}, {Fore.RED}load saved model, post requires_grad_(True){Style.RESET_ALL}")

                # ---------------------------------------------------

        model = model.to(device)

        c.ddp = ddp

        logging.info(f"{rank_str}, after load saved model, the config for running - {c}")
        logging.info(f"{rank_str}, after load saved model, config.use_amp for running - {c.use_amp}")
        logging.info(f"{rank_str}, after load saved model, config.optim for running - {c.optim}")
        logging.info(f"{rank_str}, after load saved model, config.scheduler_type for running - {c.scheduler_type}")
        logging.info(f"{rank_str}, after load saved model, config.num_workers for running - {c.num_workers}")
        logging.info(f"{rank_str}, after load saved model, model.curr_epoch for running - {model.curr_epoch}")
        logging.info(f"{rank_str}, {Fore.GREEN}after load saved model, model type - {c.model_type}{Style.RESET_ALL}")
        logging.info(f"{rank_str}, {Fore.RED}after load saved model, model.device - {model.device}{Style.RESET_ALL}")
        logging.info(f"{rank_str}, {Fore.WHITE}=============================================================={Style.RESET_ALL}")
    else:
        model = create_model(c, c.model_type, total_steps)

        model = model.to(device)
        if not c.disable_LSUV:
            t0 = time()
            LSUVinit(model, input_data.to(device=device), verbose=True, cuda=device_type=="cuda")
            logging.info(f"{rank_str}, LSUVinit took {time()-t0 : .2f} seconds ...")

    if c.ddp:
        dist.barrier()

    if rank<=0:

        # model summary
        model_summary = model_info(model, c)
        logging.info(f"Configuration for this run:\n{c}")
        logging.info(f"Model Summary:\n{str(model_summary)}")

        if wandb_run is not None:
            logging.info(f"Wandb name: {wandb_run.name}")
            wandb_run.watch(model, log="parameters")
            wandb_run.log_code(".")

    # -----------------------------------------------

    if c.ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optim = model.module.optim
        sched = model.module.sched
        stype = model.module.stype
        loss_f = model.module.loss_f
        curr_epoch = model.module.curr_epoch
        samplers = [DistributedSampler(train_set_x, shuffle=True) for train_set_x in train_set]
        shuffle = False
    else:
        optim = model.optim
        sched = model.sched
        stype = model.stype
        loss_f = model.loss_f
        curr_epoch = model.curr_epoch
        samplers = [None for _ in train_set]
        shuffle = True

    model_str = ""
    if c.backbone == 'hrnet':
        model_str = f"C {c.backbone_hrnet.C}, {c.n_head} heads, {c.backbone_hrnet.block_str}"
    elif c.backbone == 'unet':
        model_str = f"C {c.backbone_unet.C}, {c.n_head} heads, {c.backbone_unet.block_str}"

    logging.info(f"{rank_str}, {Fore.RED}Local Rank:{rank}, global rank: {global_rank}, {c.backbone}, {c.a_type}, {c.cell_type}, {c.optim}, {c.global_lr}, {c.scheduler_type}, {c.losses}, {c.loss_weights}, {c.norm_mode}, amp {c.use_amp}, {model_str}{Style.RESET_ALL}")

    # -----------------------------------------------

    train_loader = [DataLoader(dataset=train_set_x, batch_size=c.batch_size, shuffle=shuffle, sampler=samplers[i],
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, drop_last=False,
                                persistent_workers=c.num_workers>0) for i, train_set_x in enumerate(train_set)]

    # -----------------------------------------------

    if rank<=0: # main or master process
        if c.ddp: setup_logger(c) # setup master process logging

        if wandb_run is not None:
            wandb_run.summary["trainable_params"] = c.trainable_params
            wandb_run.summary["total_params"] = c.total_params
            wandb_run.summary["total_mult_adds"] = c.total_mult_adds

            wandb_run.define_metric("epoch")

            wandb_run.define_metric("train_loss_avg", step_metric='epoch')
            wandb_run.define_metric("train_mse_loss", step_metric='epoch')
            wandb_run.define_metric("train_l1_loss", step_metric='epoch')
            wandb_run.define_metric("train_ssim_loss", step_metric='epoch')
            wandb_run.define_metric("train_ssim3D_loss", step_metric='epoch')
            wandb_run.define_metric("train_psnr_loss", step_metric='epoch')
            wandb_run.define_metric("train_psnr", step_metric='epoch')
            wandb_run.define_metric("train_gaussian_deriv", step_metric='epoch')
            wandb_run.define_metric("train_gaussian3D_deriv", step_metric='epoch')

            wandb_run.define_metric("val_loss_avg", step_metric='epoch')
            wandb_run.define_metric("val_mse_loss", step_metric='epoch')
            wandb_run.define_metric("val_l1_loss", step_metric='epoch')
            wandb_run.define_metric("val_ssim_loss", step_metric='epoch')
            wandb_run.define_metric("val_ssim3D_loss", step_metric='epoch')
            wandb_run.define_metric("val_psnr", step_metric='epoch')
            wandb_run.define_metric("val_gaussian_deriv", step_metric='epoch')
            wandb_run.define_metric("val_gaussian3D_deriv", step_metric='epoch')

            # log a few training examples
            for i, train_set_x in enumerate(train_set):
                if i > 8 or wandb_run is None: break
                ind = np.random.randint(0, len(train_set_x), 4)
                x, y, _ = train_set_x[ind[0]]
                x = np.expand_dims(x, axis=0)
                y = np.expand_dims(y, axis=0)
                for ii in range(1, len(ind)):
                    a_x, a_y, _ = train_set_x[ind[ii]]
                    x = np.concatenate((x, np.expand_dims(a_x, axis=0)), axis=0)
                    y = np.concatenate((y, np.expand_dims(a_y, axis=0)), axis=0)

                title = f"Tra_samples_{i}_Noisy_Noisy_GT_{x.shape}"
                vid = create_wandb_log_vid(noisy=x, predi=x, clean=y)
                wandb_run.log({title:wandb.Video(vid, caption=f"Tra sample {i}", fps=1, format='gif')})
                logging.info(f"{Fore.YELLOW}---> Upload tra sample - {title}")

    # -----------------------------------------------
    # save best model to be saved at the end
    best_val_loss = np.inf
    best_model_wts = copy.deepcopy(model.module.state_dict() if c.ddp else model.state_dict())

    train_loss = AverageMeter()
    loss_meters = ct_trainer_meters(config=c, device=device)

    # -----------------------------------------------

    total_iters = sum([len(loader_x) for loader_x in train_loader])
    total_iters = total_iters if not c.debug else min(10, total_iters)

    # mix precision training
    scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp)

    optim.zero_grad(set_to_none=True)

    # -----------------------------------------------
    logging.info(f"{rank_str}, {Fore.GREEN}----------> Start training loop <----------{Style.RESET_ALL}")

    if c.ddp:
        model.module.check_model_learnable_status(rank_str)
    else:
        model.check_model_learnable_status(rank_str)

    for epoch in range(curr_epoch, c.num_epochs):
        logging.info(f"{Fore.GREEN}{'-'*20} Epoch:{epoch}/{c.num_epochs}, {rank_str}, global rank {global_rank} {'-'*20}{Style.RESET_ALL}")

        if c.save_samples:
            saved_path = os.path.join(c.log_path, c.run_name, f"tra_{epoch}")
            os.makedirs(saved_path, exist_ok=True)
            logging.info(f"{Fore.GREEN}saved_path - {saved_path}{Style.RESET_ALL}")

        train_loss.reset()
        loss_meters.reset()

        model.train()
        if c.ddp: [loader_x.sampler.set_epoch(epoch) for loader_x in train_loader]

        images_saved = 0
        images_logged = 0

        train_loader_iter = [iter(loader_x) for loader_x in train_loader]

        image_save_step_size = int(total_iters // c.num_saved_samples)
        if image_save_step_size == 0: image_save_step_size = 1

        curr_lr = 0

        all_lrs = [pg['lr'] for pg in optim.param_groups]
        logging.info(f"{rank_str}, {Fore.WHITE}{Style.BRIGHT}learning rate for epoch {epoch} - {all_lrs}{Style.RESET_ALL}")

        with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

            for idx in range(total_iters):

                loader_ind = idx % len(train_loader_iter)

                tm = start_timer(enable=c.with_timer)
                batch = next(train_loader_iter[loader_ind], None)
                while batch is None:
                    del train_loader_iter[loader_ind]
                    loader_ind = idx % len(train_loader_iter)
                    batch = next(train_loader_iter[loader_ind], None)

                x, y, _ = batch
                end_timer(enable=c.with_timer, t=tm, msg="---> load batch took ")

                tm = start_timer(enable=c.with_timer)
                x = x.to(device)
                y = y.to(device)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=c.use_amp):
                    output = model(x)
                    loss = loss_f(output, y)
                    loss = loss / c.iters_to_accumulate

                end_timer(enable=c.with_timer, t=tm, msg="---> forward pass took ")

                if torch.isnan(loss):
                    logging.info(f"Warning - loss is nan. Skipping to next iter")
                    optim.zero_grad()
                    continue

                tm = start_timer(enable=c.with_timer)
                scaler.scale(loss).backward()
                end_timer(enable=c.with_timer, t=tm, msg="---> backward pass took ")


                tm = start_timer(enable=c.with_timer)
                if (idx + 1) % c.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                    if(c.clip_grad_norm>0):
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip_grad_norm)

                    scaler.step(optim)
                    optim.zero_grad()
                    scaler.update()
                    if stype == "OneCycleLR": sched.step()

                end_timer(enable=c.with_timer, t=tm, msg="---> other steps took ")

                tm = start_timer(enable=c.with_timer)
                curr_lr = optim.param_groups[0]['lr']

                total=x.shape[0]
                train_loss.update(loss.item(), n=total)

                output = output.to(x.dtype)

                if rank<=0 and idx%image_save_step_size==0 and images_saved < c.num_saved_samples and c.save_samples:
                    save_image_local(saved_path, False, f"tra_epoch_{epoch}_{images_saved}", x.numpy(force=True), output.numpy(force=True), y.numpy(force=True))
                    images_saved += 1
                if rank<=0 and images_logged < config.num_uploaded and wandb_run is not None:
                    title = f"tra_{images_logged}_{x.shape}"
                    vid = create_wandb_log_vid(noisy=x.numpy(force=True), predi=output.numpy(force=True), clean=y.numpy(force=True))
                    wandb_run.log({title: wandb.Video(vid,
                                                      caption=f"epoch {epoch}, mse {loss_meters.mse_meter.val:.2f}, ssim {loss_meters.ssim_meter.val:.2f}, psnr {loss_meters.psnr_meter.val:.2f}",
                                                      fps=1, format="gif")})
                    images_logged += 1

                loss_meters.update(output, y)

                pbar.update(1)
                log_str = create_log_str(c, epoch, rank,
                                         x.shape,
                                         train_loss.avg,
                                         loss_meters,
                                         curr_lr,
                                         "tra")

                pbar.set_description_str(log_str)

                if wandb_run is not None:
                    wandb_run.log({"running_train_loss": loss.item()})
                    wandb_run.log({"lr": curr_lr})

                end_timer(enable=c.with_timer, t=tm, msg="---> logging and measuring took ")

            # ---------------------------------------
            log_str = create_log_str(c, epoch, rank,
                                         None,
                                         train_loss.avg,
                                         loss_meters,
                                         curr_lr,
                                         "tra")

            pbar.set_description_str(log_str)
        # -------------------------------------------------------

        val_losses = eval_val(rank, model, c, val_set, epoch, device, wandb_run)

        # -------------------------------------------------------
        if rank<=0: # main or master process
            model_e = model.module if c.ddp else model
            saved_model = model_e.save(epoch)

            if val_losses[0] < best_val_loss:
                best_val_loss = val_losses[0]
                best_model_wts = copy.deepcopy(model_e.state_dict())
                run_name = c.run_name.replace(" ", "_")
                saved_model = model_e.save(epoch, only_paras=True, save_file_name=f"{run_name}_epoch-{epoch}_best.pth")
                if wandb_run is not None:
                    wandb_run.log({"epoch": epoch, "best_val_loss":best_val_loss})

            if wandb_run is not None:
                wandb_run.log(
                                {
                                    "epoch": epoch,
                                    "train_loss_avg": train_loss.avg,
                                    "train_mse_loss": loss_meters.mse_meter.avg,
                                    "train_l1_loss": loss_meters.l1_meter.avg,
                                    "train_ssim_loss": loss_meters.ssim_meter.avg,
                                    "train_ssim3D_loss": loss_meters.ssim3D_meter.avg,
                                    "train_psnr_loss": loss_meters.psnr_loss_meter.avg,
                                    "train_psnr": loss_meters.psnr_meter.avg,
                                    "train_gaussian_deriv": loss_meters.gaussian_meter.avg,
                                    "train_gaussian3D_deriv": loss_meters.gaussian3D_meter.avg,
                                    "val_loss_avg": val_losses[0],
                                    "val_mse_loss": val_losses[1],
                                    "val_l1_loss": val_losses[2],
                                    "val_ssim_loss": val_losses[3],
                                    "val_ssim3D_loss": val_losses[4],
                                    "val_psnr": val_losses[5],
                                    "val_gaussian_deriv": val_losses[6],
                                    "val_gaussian3D_deriv": val_losses[7]
                                }
                              )

            if stype == "ReduceLROnPlateau":
                sched.step(val_losses[0])
            else: # stype == "StepLR"
                sched.step()

            if c.ddp:
                distribute_learning_rates(rank=rank, optim=optim, src=0)

        else: # child processes
            distribute_learning_rates(rank=rank, optim=optim, src=0)


    # test last model
    test_losses = eval_val(rank, model, c, test_set, epoch, device, wandb_run, id="tes_last")
    if rank<=0:
        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = best_val_loss
            wandb_run.summary["last_val_loss"] = val_losses[0]

            wandb_run.summary["test_loss_last"] = test_losses[0]
            wandb_run.summary["test_mse_last"] = test_losses[1]
            wandb_run.summary["test_l1_last"] = test_losses[2]
            wandb_run.summary["test_ssim_last"] = test_losses[3]
            wandb_run.summary["test_ssim3D_last"] = test_losses[4]
            wandb_run.summary["test_psnr_last"] = test_losses[5]
            wandb_run.summary["test_gaussian_deriv_last"] = test_losses[6]
            wandb_run.summary["test_gaussian3D_deriv_last"] = test_losses[7]

            model = model.module if c.ddp else model
            model.save(epoch)

            # save both models
            fname_last, fname_best = save_final_model(model, c, best_model_wts, only_pt=True)

            logging.info(f"--> {Fore.YELLOW}Save last model at {fname_last}{Style.RESET_ALL}")
            logging.info(f"--> {Fore.YELLOW}Save best model at {fname_best}{Style.RESET_ALL}")

    # test best model, reload the weights
    model = create_model(c, c.model_type, total_steps)

    model.load_state_dict(best_model_wts)
    model = model.to(device)

    if c.ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    test_losses = eval_val(rank, model, c, test_set, epoch, device, wandb_run, id="tes_best")
    if rank<=0:
        if wandb_run is not None:
            wandb_run.summary["test_loss_best"] = test_losses[0]
            wandb_run.summary["test_mse_best"] = test_losses[1]
            wandb_run.summary["test_l1_best"] = test_losses[2]
            wandb_run.summary["test_ssim_best"] = test_losses[3]
            wandb_run.summary["test_ssim3D_best"] = test_losses[4]
            wandb_run.summary["test_psnr_best"] = test_losses[5]
            wandb_run.summary["test_gaussian_deriv_best"] = test_losses[6]
            wandb_run.summary["test_gaussian3D_deriv_best"] = test_losses[7]

            wandb_run.save(fname_last+'.pt')
            # wandb_run.save(fname_last+'.pts')
            # wandb_run.save(fname_last+'.onnx')

            wandb_run.save(fname_best+'.pt')
            # wandb_run.save(fname_best+'.pts')
            # wandb_run.save(fname_best+'.onnx')

    if c.ddp:
        dist.barrier()
    logging.info(f"--> run finished ...")

# -------------------------------------------------------------------------------------------------
# evaluate the val set

def eval_val(rank, model, config, val_set, epoch, device, wandb_run, id="val", scaling_factor=-1):
    """
    The validation evaluation.
    @args:
        - rank (int): for distributed data parallel (ddp)
            -1 if running on cpu or only one gpu
        - model (torch model): model to be validated
        - config (Namespace): runtime namespace for setup
        - val_set (torch Dataset list): the data to validate on
        - epoch (int): the current epoch
        - device (torch.device): the device to run eval on
        - wandb_run (wandb.Run): the run object for loggin to wandb
        - id (str): the extra id name to save with
        - scaling_factor (int): factor to scale the input image with
    @rets:
        - val_loss (float): the average val loss
        - val_mse (float): the average val mse loss
        - val_l1 (float): the average val l1 loss
        - val_ssim (float): the average val ssim loss
        - val_ssim3D (float): the average val ssim3D loss
        - val_psnr (float): the average val psnr
        - val_gaussian (float): the average gaussian loss
        - val_gaussian3D (float): the average guassian 3D loss
    """
    c = config

    try:
        if c.ddp:
            loss_f = model.module.loss_f
        else:
            loss_f = model.loss_f
    except:
        loss_f = None

    if c.ddp:
        sampler = [DistributedSampler(val_set_x, shuffle=False) for val_set_x in val_set]
    else:
        sampler = [None for _ in val_set]

    batch_size = [c.batch_size if isinstance(val_set_x, CtDatasetTrain) else 1 for val_set_x in val_set]

    val_loader = [DataLoader(dataset=val_set_x, batch_size=batch_size[i], shuffle=False, sampler=sampler[i],
                                num_workers=c.num_workers, prefetch_factor=c.prefetch_factor,
                                persistent_workers=c.num_workers>0) for i, val_set_x in enumerate(val_set)]

    val_loss_meter = AverageMeter()
    loss_meters = ct_trainer_meters(config=c, device=device)

    model.eval()
    model.to(device)

    if rank <= 0 and epoch < 1:
        logging.info(f"Eval height and width is {c.height[-1]}, {c.width[-1]}")

    cutout = (c.time, c.height[-1], c.width[-1])
    overlap = (c.time//2, c.height[-1]//4, c.width[-1]//4)

    val_loader_iter = [iter(val_loader_x) for val_loader_x in val_loader]
    total_iters = sum([len(loader_x) for loader_x in val_loader])
    total_iters = total_iters if not c.debug else min(2, total_iters)

    images_logged = 0
    images_saved = 0
    if config.save_samples:
        if epoch >= 0:
            saved_path = os.path.join(config.log_path, config.run_name, id)
        else:
            saved_path = os.path.join(config.log_path, config.run_name, f"{id}_{epoch}")
        os.makedirs(saved_path, exist_ok=True)
        logging.info(f"save path is {saved_path}")

    with torch.inference_mode():
        with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

            for idx in range(total_iters):

                loader_ind = idx % len(val_loader_iter)
                batch = next(val_loader_iter[loader_ind], None)
                while batch is None:
                    del val_loader_iter[loader_ind]
                    loader_ind = idx % len(val_loader_iter)
                    batch = next(val_loader_iter[loader_ind], None)
                x, y, name = batch
                name = name[0]

                if scaling_factor > 0:
                    x *= scaling_factor

                if x.shape[0] > 1 and x.shape[-2] == cutout[-2] and x.shape[-1] == cutout[-1]:
                    # run normal inference
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)
                else:
                    cutout_in = cutout
                    overlap_in = overlap

                    x = x.to(device)
                    y = y.to(device)

                    B, T, C, H, W = x.shape

                    if not config.pad_time:
                        cutout_in = (T, c.height[-1], c.width[-1])
                        overlap_in = (0, c.height[-1]//2, c.width[-1]//2)

                    try:
                        _, output = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device=device)
                    except:
                        logging.info(f"{Fore.YELLOW}---> call inference on cpu ...")
                        _, output = running_inference(model, x, cutout=cutout_in, overlap=overlap_in, device=torch.device("cpu"))
                        y = y.to("cpu")

                output = output * (x > 0)

                if scaling_factor > 0:
                    output /= scaling_factor

                if loss_f:
                    loss = loss_f(output, y)
                    val_loss_meter.update(loss.item(), n=x.shape[0])

                loss_meters.update(output, y)

                if rank<=0 and images_logged < config.num_uploaded and wandb_run is not None:
                    images_logged += 1
                    name_wandb = name if x.shape[0]==1 else "random_level"
                    title = f"{id.upper()}_{images_logged}_{name_wandb}_{x.shape}"
                    vid = create_wandb_log_vid(noisy=x.numpy(force=True), predi=output.numpy(force=True), clean=y.numpy(force=True))
                    wandb_run.log({title: wandb.Video(vid,
                                                      caption=f"epoch {epoch}, mse {loss_meters.mse_meter.val:.2f}, ssim {loss_meters.ssim_meter.val:.2f}, psnr {loss_meters.psnr_meter.val:.2f}",
                                                      fps=1, format="gif")})

                if (rank<=0 or "tes" in id) and images_saved < config.num_saved_samples and config.save_samples:
                    save_image_local(saved_path, False, f"{id}_epoch_{epoch}_{name}_{images_saved}", x.numpy(force=True), output.numpy(force=True), y.numpy(force=True))
                    images_saved += 1

                pbar.update(1)
                log_str = create_log_str(c, epoch, rank,
                                         x.shape,
                                         val_loss_meter.avg,
                                         loss_meters,
                                         -1,
                                         id)

                pbar.set_description_str(log_str)

            # -----------------------------------
            log_str = create_log_str(c, epoch, rank,
                                         None,
                                         val_loss_meter.avg,
                                         loss_meters,
                                         -1,
                                         id)

            pbar.set_description_str(log_str)

    if c.ddp:
        val_loss = torch.tensor(val_loss_meter.avg).to(device=device)
        dist.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)

        val_mse = torch.tensor(loss_meters.mse_meter.avg).to(device=device)
        dist.all_reduce(val_mse, op=torch.distributed.ReduceOp.AVG)

        val_l1 = torch.tensor(loss_meters.l1_meter.avg).to(device=device)
        dist.all_reduce(val_l1, op=torch.distributed.ReduceOp.AVG)

        val_ssim = torch.tensor(loss_meters.ssim_meter.avg).to(device=device)
        dist.all_reduce(val_ssim, op=torch.distributed.ReduceOp.AVG)

        val_ssim3D = torch.tensor(loss_meters.ssim3D_meter.avg).to(device=device)
        dist.all_reduce(val_ssim3D, op=torch.distributed.ReduceOp.AVG)

        val_psnr_loss = torch.tensor(loss_meters.psnr_loss_meter.avg).to(device=device)
        dist.all_reduce(val_psnr_loss, op=torch.distributed.ReduceOp.AVG)

        val_psnr = torch.tensor(loss_meters.psnr_meter.avg).to(device=device)
        dist.all_reduce(val_psnr, op=torch.distributed.ReduceOp.AVG)

        val_gaussian = torch.tensor(loss_meters.gaussian_meter.avg).to(device=device)
        dist.all_reduce(val_gaussian, op=torch.distributed.ReduceOp.AVG)

        val_gaussian3D = torch.tensor(loss_meters.gaussian3D_meter.avg).to(device=device)
        dist.all_reduce(val_gaussian3D, op=torch.distributed.ReduceOp.AVG)
    else:
        val_loss = val_loss_meter.avg
        val_mse = loss_meters.mse_meter.avg
        val_l1 = loss_meters.l1_meter.avg
        val_ssim = loss_meters.ssim_meter.avg
        val_ssim3D = loss_meters.ssim3D_meter.avg
        val_psnr_loss = loss_meters.psnr_loss_meter.avg
        val_psnr = loss_meters.psnr_meter.avg
        val_gaussian = loss_meters.gaussian_meter.avg
        val_gaussian3D = loss_meters.gaussian3D_meter.avg

    if rank<=0:

        log_str = create_log_str(c, epoch, rank,
                                None,
                                val_loss,
                                loss_meters,
                                -1,
                                id)

        logging.info(log_str)

    return val_loss, val_mse, val_l1, val_ssim, val_ssim3D, val_psnr, val_gaussian, val_gaussian3D
