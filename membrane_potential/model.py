"""

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import sys
import math
import logging
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from model_base.transformer import * 

from utils.utils import *

__all__ = ['MPPredictor', 'LossMPPrediction', 'MSELossMPPrediction', 'L1LossMPPrediction']

# ----------------------------------------------------------           

class MPPredictor(nn.Module):
    """A transformer based membrane potential predictor

        This model uses positional embedding. 
        
        The architecture is quite straight-forward :
        
        x -> input_proj --> + --> drop_out --> attention layers one after another --> LayerNorm --> output_proj --> logits
                            |
        pos_embedding-------|            
    """

    def __init__(self, n_layer=8, input_size=725, output_size=1, T=1024, is_causal=False, use_pos_embedding=True, n_embd=1024, n_head=8, dropout_p=0.1, attn_dropout_p=0.0, residual_dropout_p=0.1):
        super().__init__()

        self.n_layer = n_layer
        self.input_size = input_size
        self.output_size = output_size
        self.T = T
        self.is_causal = is_causal
        self.use_pos_embedding = use_pos_embedding
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.residual_dropout_p = residual_dropout_p

        # input projection
        self.input_proj = nn.Linear(input_size, self.n_embd)
        
        # the positional embedding is used
        # this is learned through the training
        if self.use_pos_embedding:
            self.pos_emb = nn.Parameter(torch.zeros(1, T, self.n_embd))
                
        self.drop = nn.Dropout(dropout_p)
        
        # transformer modules
        # stack them for n_layers
        self.blocks = nn.Sequential(*[Cell(T=T, n_embd=n_embd, is_causal=is_causal, n_head=n_head, attn_dropout_p=attn_dropout_p, residual_dropout_p=residual_dropout_p) for _ in range(n_layer)])
        
        # decoder head
        self.layer_norm = nn.LayerNorm(n_embd)
        self.output_proj1 = nn.Linear(n_embd, n_embd//2, bias=True)
        self.output_proj2 = nn.Linear(n_embd//2, output_size, bias=True)

        self.apply(self._init_weights)

        # a good trick to count how many parameters
        logging.info("number of parameters: %d", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """Forward pass of detector

        Args:
            x ([B, T, C]]): Input membrane waveform with B batches and T time points with C length

            Due to the positional embedding is used, the input T is limited to less or equal to self.T

        Returns:
            logits: [B, T, output_size]
        """
        
        B, T, C = x.size()
        assert T <= self.T, "The positional embedding is used, so the maximal series length is %d" % self.T
                       
        # project input from C channels to n_embd channels
        x_proj = self.input_proj(x)
        
        if self.use_pos_embedding:
            x = x_proj + self.pos_emb[:, :T, :]
        else:
            x = x_proj + position_encoding(seq_len=T, dim_model=C, device=self.device)
            
        x = self.drop(x)
        
        # go through all layers of attentions
        x = self.blocks(x)
        
        # project outputs to output_size channel        
        x = self.layer_norm(x)
        x = F.relu(self.output_proj1(x))
        logits = self.output_proj2(x)
        
        return logits
    
# -------------------------------------------

class LossMPPrediction:
    """
    Loss for membrane potential prediction
    """

    def __init__(self):
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def __call__(self, y_hat, y, i_valid_spec, idx_select):
        """Compute prediction loss

        Args:
            y_hat ([B, T, 1]): logits from the model, predicted potential
            y ([B, T]): membrane potential
            i_valid_spec ([B, T]): time points for reliable waveform inputs
            idx_select ([B, T]): time points selected for reliable membrane potential computation

        Returns:
            loss (tensor): L2+L1 loss
        """
        
        idx_select = torch.flatten(idx_select)
        y_hat = torch.flatten(y_hat)
        y = torch.flatten(y)
        
        loss = torch.sqrt(self.mse_loss(y[idx_select==1], y_hat[idx_select==1])) + self.l1_loss(y[idx_select==1], y_hat[idx_select==1])
        
        return loss
                
class MSELossMPPrediction:
    """
    MSE loss for membrane potential prediction
    """

    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, y_hat, y, i_valid_spec, idx_select):
        """Compute prediction loss

        Args:
            y_hat ([B, T, 1]): logits from the model, predicted potential
            y ([B, T]): membrane potential
            i_valid_spec ([B, T]): time points for reliable waveform inputs
            idx_select ([B, T]): time points selected for reliable membrane potential computation

        Returns:
            loss (tensor): L2 loss
        """
        
        idx_select = torch.flatten(idx_select)
        y_hat = torch.flatten(y_hat)
        y = torch.flatten(y)
        
        loss = torch.sqrt(self.mse_loss(y[idx_select==1], y_hat[idx_select==1]))
        
        return loss
    
class L1LossMPPrediction:
    """
    Loss for membrane potential prediction
    """

    def __init__(self):
        self.l1_loss = nn.L1Loss()

    def __call__(self, y_hat, y, i_valid_spec, idx_select):
        """Compute prediction loss

        Args:
            y_hat ([B, T, 1]): logits from the model, predicted potential
            y ([B, T]): membrane potential
            i_valid_spec ([B, T]): time points for reliable waveform inputs
            idx_select ([B, T]): time points selected for reliable membrane potential computation

        Returns:
            loss (tensor): L1 loss
        """
        
        idx_select = torch.flatten(idx_select)
        y_hat = torch.flatten(y_hat)
        y = torch.flatten(y)
        
        loss = self.l1_loss(y[idx_select==1], y_hat[idx_select==1])
        
        return loss
                        
# ----------------------------------------------------------

def main():
    """Model testing code
    """
    pass

if __name__ == '__main__':
    main()