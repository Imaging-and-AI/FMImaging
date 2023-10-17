"""
Standard model for transformer

Input : [B, T, D]
Output : [B, T, D]

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import sys
import math
import logging
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

from utils import get_device, model_info, get_gpu_ram_usage
from setup import create_generic_class_str
from imaging_attention import create_activation_func

def position_encoding(
    seq_len: int,
    dim_model: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Standard positional encoding; this function is from https://github.com/fkodom/transformer-from-scratch/tree/main

    Args:
        seq_len (int): sequence length, or T
        dim_model (int): embedding dimension, D
        device (torch.device, optional): device to store the encoding. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: positional encoding matrix [seq_len, dim_model]
    """
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** torch.div(dim, dim_model, rounding_mode="floor"))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
  
    
class StandardAttention(nn.Module):
    """
    Multi-head attention model    
    Dropout is added on the attention matrix and output.    
    """

    def __init__(self, T=128, n_embd=128, is_causal=False, n_head=8, attn_dropout_p=0.0, residual_dropout_p=0.1):
        """Define the layers for a self-attention

            Input to the attention layer has the size [B, T, n_embd]
            Output has the size [B, T, n_embd]

        Args:
            T (int, optional): number of time points for attention layer. Defaults to 1024.
            is_causal (bool, optional): whether applying the masking to make the layer causal. Defaults to False.
            n_embd (int, optional): number of internal dimension. Defaults to 128.
            n_head (int, optional): number of heads. Defaults to 8.
        """
        super().__init__()            
        
        assert n_embd % n_head == 0
        
        self.is_causal = is_causal
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_dropout_p = attn_dropout_p
        self.residual_dropout_p = residual_dropout_p
        
        # key, query, value projections matrix
        # Wk, Wq, Wv
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
                
        self.output_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_dropout_p)
        self.resid_drop = nn.Dropout(residual_dropout_p)
    
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))
                
    def forward(self, x):
        """forward pass for the 

        Args:
            x ([B, T, n_embd]): Input of a batch of time series

        Returns:
            y: logits in the shape of [B, T, n_embd]
        """
        
        B, T, C = x.size()

        # apply the key, query and value matrix
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute attention matrix, use the matrix broadcasing 
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # if causality is needed, apply the mask
        if(self.is_causal):
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.output_proj(y))
        return y

    @property
    def device(self):
        return next(self.parameters()).device
    
    def __str__(self):
        res = create_generic_class_str(self)
        return res

class Cell(nn.Module):
    """ Transformer cell
    
    The Pre-LayerNorm implementation is used here:
    
    x-> LayerNorm -> attention -> + -> LayerNorm -> LinearLayers -> + -> logits
    |-----------------------------| |-------------------------------|

    """

    def __init__(self, T=128, n_embd=128, is_causal=False, n_head=8, attn_dropout_p=0.0, residual_dropout_p=0.1, activation_func="prelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = StandardAttention(T=T, n_embd=n_embd, is_causal=is_causal, n_head=n_head, attn_dropout_p=attn_dropout_p, residual_dropout_p=residual_dropout_p)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            create_activation_func(name=activation_func),
            #nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(residual_dropout_p),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    def __str__(self):
        res = create_generic_class_str(self)
        return res

# -------------------------------------------------------------------------------------------------

def tests():
    # tests

    B, T, D = 12, 512, 400
    x = torch.randn((B, T, D))
    
    print("Begin Testing")

    att = StandardAttention(T=T, n_embd=D, is_causal=False, n_head=8, attn_dropout_p=0.0, residual_dropout_p=0.1)
    test_out = att(x)

    Bo, To, Do = test_out.shape
    assert B==Bo and T==To and Do==D

    print("Passed StandardAttention")

    cell = Cell(T=T, n_embd=D, is_causal=False, n_head=8, attn_dropout_p=0.0, residual_dropout_p=0.1)
    test_out = cell(x)

    Bo, To, Do = test_out.shape
    assert B==Bo and T==To and Do==D
    
    print("Passed Cell")


    print("Passed all tests")

if __name__=="__main__":
    tests()
