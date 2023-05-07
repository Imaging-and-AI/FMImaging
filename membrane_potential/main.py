"""Membrane potential prediction

    A transformer model is developed to predict the membrane potential from waveform.
    
    Input: 
        Waveform: [B, T, D]
        
"""

import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms

from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import *

import os
import sys
import math
import numpy as np
import argparse
import time
from time import gmtime, strftime
from tqdm import tqdm 
import wandb
import matplotlib.pyplot as plt

from model import *
import dataset

if "FMIMAGING_PROJECT_BASE" in os.environ:
    project_base_dir = os.environ['FMIMAGING_PROJECT_BASE']
else:
    project_base_dir = '/export/Lab-Xue/projects'
    
# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch transformer model for membrane prediction")

    parser.add_argument("--data_root", type=str, default='./data', help='root folder for the data')
    
    # parameters for training
    parser.add_argument('--epoch_to_load', type=int, default=-1, help='if >=0, load this check point')
        
    # parameter for models
    parser.add_argument('--seq_length', type=int, default=2048, help='length of sequence')
    parser.add_argument('--num_starts', type=int, default=10, help='number of sampled sequences for one experiments')
    parser.add_argument('--n_layers', type=int, default=8, help='number of transformer layers')
    parser.add_argument('--n_embd', type=int, default=1024, help='the embedding dimension of transformer layer')
    
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    
    parser.add_argument(
        "--training_record",
        type=str,
        default="membrane_potential",
        help='String to record this training')
    
    parser.add_argument('--dropout_p', type=float, default=0.1, help='drop out for cell output')
    parser.add_argument('--attn_dropout_p', type=float, default=0.0, help='drop out for attention matrix')
    parser.add_argument('--residual_dropout_p', type=float, default=0.1, help='drop out for the cell output')

    parser = add_shared_args(parser)
       
    return parser
# ----------------------------------
def check_args(config):
    """
    checks the cmd args to make sure they are correct
    @args:
        - config (Namespace): runtime namespace for setup
    @rets:
        - config (Namespace): the checked and updated argparse for MRI
    """
    if config.run_name is None:
        config.run_name = "membrane_potential"
        
    if config.data_root is None:
        config.data_root = os.path.join(project_base_dir,  "membrane_potential", "data")
        
    if config.log_path is None:
        config.log_path = os.path.join(project_base_dir,  "membrane_potential", "log")
        
    if config.results_path is None:
        config.results_path = os.path.join(project_base_dir,  "membrane_potential", "res")
        
    if config.model_path is None:
        config.model_path = os.path.join(project_base_dir,  "membrane_potential", "model")
        
    if config.check_path is None:
        config.check_path = os.path.join(project_base_dir,  "membrane_potential", "checkpoints")
    
    return config

# ----------------------------------

# load parameters
config = check_args(add_args().parse_args())
setup_run(config)
    
print(config)

device = get_device()

# ----------------------------------
                          
def run_training():
    """Run the training

    Outputs:
        model : best model after training
        loss_train : loss for every epoch
    """

    # read in the data
    train_dir = os.path.join(config.data_root, "train")
    test_dir = os.path.join(config.data_root, "test")
    
    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = dataset.set_up_dataset(train_dir, test_dir, batch_size=config.batch_size, chunk_length=config.seq_length, num_starts=config.num_starts, val_frac=0.1)
    
    # for k in range(len(train_set)):
    #     A, MP, i_valid_spec, idx_select, name  = train_set[k]
    #     print(A.shape)
    #     print(MP.shape)
    #     print(i_valid_spec.shape)
    #     print(idx_select.shape)
    #     print(len(name))
        
    A, MP, i_valid_spec, idx_select, name  = train_set[0]
    T, D = A.shape
    
    assert MP.shape[0] == T
    
    # declare the model
    model = MPPredictor(n_layer=config.n_layers, 
                              input_size=D, 
                              output_size=1, 
                              T=T, 
                              is_causal=False, 
                              use_pos_embedding=True, 
                              n_embd=config.n_embd, 
                              n_head=config.n_head, 
                              dropout_p=config.dropout_p, 
                              attn_dropout_p=config.attn_dropout_p, 
                              residual_dropout_p=config.residual_dropout_p)
    print(model)
   
    # declare the loss function, loss_func
    loss_func = LossMPPrediction()
   
    # a warmup stage is used
    # the total length of sequence processed is recorded
    # if the length is les than warmup_length, learning rate is scaled down for a warm start
    warmup_length=20*64*T
    # then the learning rate is decayed with cosine function
    final_length=(config.num_epochs-1)*len(train_set)*T
        
    # declare the optimizer, check the config.optimizer and define optimizer, note to use config.learning_rate and config.reg
    if(config.optim=='sgd'):
        optimizer = optim.SGD(model.parameters(), lr=config.global_lr, momentum=0.9, weight_decay=config.reg)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.global_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.weight_decay, amsgrad=False)
           
    # load models if needed
    if(config.epoch_to_load>=0):
        ckpt_model = os.path.join(config.check_path, f"ckpt_{config.epoch_to_load}.pbt")
        print("Load check point: ", ckpt_model)
        if os.path.isfile(ckpt_model):
            state = torch.load(str(ckpt_model))
            model.load_state_dict(state['model'])
            print('Restored model, epoch {}, step {:,}'.format(state['epoch'], state['step']))

    # save model function
    save = lambda ep, step: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, os.path.join(config.check_path, f"ckpt_{ep}.pbt"))

    # set up training
    n_batches = len(train_set)//config.batch_size 
    
    loss_train = []
    loss_val = []
    
    best_model = None
    best_val_loss = 1e4
    
    seq_length_processed = 0
    
    # uncomment to use multiple GPUs
    if device != torch.device('cpu') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Train model on %d GPUs ... " % torch.cuda.device_count())
            
    model.to(device=device)
    
    mse_loss_meter = AverageMeter()
    mse_loss_func = MSELossMPPrediction()
    
    l1_loss_meter = AverageMeter()
    l1_loss_func = L1LossMPPrediction()
    
    for e in range(config.num_epochs):
               
        # set up the progress bar
        tq = tqdm(total=(n_batches * config.batch_size), desc ='Epoch {}, total {}'.format(e, config.num_epochs))

        model.train()

        t0 = time.time()
        count = 0
        running_loss_training = 0
        loss = 0
        for A, MP, i_valid_spec, idx_select, name in loader_for_train:

            x = A.to(device=device, dtype=torch.float32)
            y = MP.to(device=device, dtype=torch.float32)
            i_valid_spec = i_valid_spec.to(device=device)
            idx_select = idx_select.to(device=device)
            
            output = model(x)
            loss = loss_func(output, y, i_valid_spec, idx_select)
            
            optimizer.zero_grad()
            loss.backward()
            
            if config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                
            optimizer.step()
                         
            # decay the learning rate based on our progress
            lr_mult = 1
            seq_length_processed += y.shape[0]*y.shape[1]
            if seq_length_processed < warmup_length:
                # linear warmup
                lr_mult = float(seq_length_processed) / float(max(1, warmup_length))
            else:
                # cosine learning rate decay
                progress = float(seq_length_processed - warmup_length) / float(max(1, final_length - warmup_length))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = config.global_lr * lr_mult
            
            # the learning rate can be set
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                                                   
            # bookkeeping    
            wandb.log({"batch loss":loss.item()})
            wandb.log({"batch learning rate":lr})
                
            tq.update(config.batch_size)
            tq.set_description(f"{e} - loss:{loss.item():.8f}, learning rate:{lr:.6f}, lr_mult:{lr_mult:.6f}")
            
            running_loss_training += loss.item()
            
            count += 1

        t1 = time.time()

        tq.set_postfix({'loss':loss.item(), 'learning rate':lr, 'lr_mult':lr_mult})
        
        # save the check point
        save(e, count)
      
        # ----------------------------------------------------------------
        # process the validation set
        # ----------------------------------------------------------------
        model.eval()
        val_losses = []
        t0_val = time.time()
        with torch.no_grad():
            for batch_num, (A, MP, i_valid_spec, idx_select, name) in enumerate(loader_for_val):

                x = A.to(device=device, dtype=torch.float32)
                y = MP.to(device=device, dtype=torch.float32)
                i_valid_spec = i_valid_spec.to(device=device)
                idx_select = idx_select.to(device=device)
            
                output = model(x)
                val_loss = loss_func(output, y, i_valid_spec, idx_select)
                
                val_losses.append(val_loss.item())
            
                f = dataset.plot_mp_prediction(A.cpu().numpy(), MP.cpu().numpy(), i_valid_spec.cpu().numpy(), idx_select.cpu().numpy(), name, output.cpu().numpy())
                
                f.savefig(f"{config.results_path}/validation_epoch_{e}_batch_{batch_num}.png")
                wandb.log({"val plot": f})                                
                
                np.save(f"{config.results_path}/validation_epoch_{e}_batch_{batch_num}_A.npy", x.detach().cpu().numpy())
                np.save(f"{config.results_path}/validation_epoch_{e}_batch_{batch_num}_MP.npy", MP.detach().cpu().numpy())
                np.save(f"{config.results_path}/validation_epoch_{e}_batch_{batch_num}_i_valid_spec.npy", i_valid_spec.detach().cpu().numpy())
                np.save(f"{config.results_path}/validation_epoch_{e}_batch_{batch_num}_idx_select.npy", idx_select.detach().cpu().numpy())
                np.save(f"{config.results_path}/validation_epoch_{e}_batch_{batch_num}_output.npy", output.detach().cpu().numpy())
                
        t1_val = time.time()
            
        loss_train.append(running_loss_training/count)
        loss_val.append(np.mean(val_losses))
        
        # keep the best model, evaluated on the validation set
        if(best_val_loss>np.mean(val_losses)):
            best_val_loss = np.mean(val_losses)
            best_model = copy.deepcopy(model)
        
        wandb.log({"epoch":e, "train loss":loss_train[e], "val loss":loss_val[e]})
                       
        str_after_val = '%.2f/%.2f seconds for Training/Validation - Tra loss = %.4f, Val loss = %.4f, - learning rate = %.6f' % (t1-t0, t1_val-t0_val, loss_train[e], loss_val[e], lr)
        tq.set_postfix_str(str_after_val)
        tq.close() 
        
    # ----------------------------------------------------------------
    # apply the model on the test set
    # ----------------------------------------------------------------    
    test_x = []
    test_y = []
    test_y_hat = []

    if len(loader_for_test):
        test_bar = tqdm(enumerate(loader_for_test), total=len(loader_for_test))
        
        t0_test = time.time()
        best_model.eval()
        test_loss = 0
        with torch.no_grad():
            for it, (A, MP, i_valid_spec, idx_select, name) in test_bar:
                x = A.to(device=device, dtype=torch.float32)
                y = MP.to(device=device, dtype=torch.float32)
                i_valid_spec = i_valid_spec.to(device=device)
                idx_select = idx_select.to(device=device)
                
                output = best_model(x)
                loss = loss_func(output, y, i_valid_spec, idx_select)
                
                test_loss += loss.item()
                
                y_hat = output
                
                test_x.append(x.detach().cpu().numpy())
                test_y.append(y.detach().cpu().numpy())
                test_y_hat.append(y_hat.detach().cpu().numpy())
                
                t1_test = time.time()    
                test_bar.set_description(f"duration: {t1_test-t0_test:.1f}, batch: {it}, loss: {loss.item():.6f}")
            
                f = dataset.plot_mp_prediction(A.cpu().numpy(), MP.cpu().numpy(), i_valid_spec.cpu().numpy(), idx_select.cpu().numpy(), name, y_hat.cpu().numpy())                
                f.savefig(f"{config.results_path}/test_batch_{batch_num}.png")
                
                wandb.log({"test plot": f})
                
                np.save(f"{config.results_path}/test_batch_{it}_A.npy", x.detach().cpu().numpy())
                np.save(f"{config.results_path}/test_batch_{it}_MP.npy", MP.detach().cpu().numpy())
                np.save(f"{config.results_path}/validation_batch_{it}_i_valid_spec.npy", i_valid_spec.detach().cpu().numpy())
                np.save(f"{config.results_path}/validation_batch_{it}_idx_select.npy", idx_select.detach().cpu().numpy())
                np.save(f"{config.results_path}/validation_batch_{it}_output.npy", output.detach().cpu().numpy())
                
        test_loss /= len(loader_for_test)
    
    # ----------------------------------------------

    return best_model, loss_train, loss_val, (test_x, test_y, test_y_hat)

def main():
       
    moment = strftime("%Y%m%d_%H%M%S", gmtime())

    wandb.init(project=config.project, entity=config.wandb_entity, config=config, name=config.run_name, notes=config.run_notes)

    # perform training
    best_model, loss_train, loss_val, test_results = run_training()
    
    if isinstance(best_model, nn.DataParallel):
        best_model = best_model.module
    
    torch.save(best_model.cpu().state_dict(), os.path.join(config.model_path, "best_model.pbt"))
            
if __name__ == '__main__':
    main()