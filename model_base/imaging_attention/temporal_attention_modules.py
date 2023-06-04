"""
Spatio-Temporal Convolutional Neural Net Transformer (STCNNT)

Implement the temporal cnn attention.

"""

import math
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

from attention_modules import *

# -------------------------------------------------------------------------------------------------
# Temporal attention layer. Attention is computed between images along dimension T.

class TemporalCnnAttention(CnnAttentionBase):
    """
    Multi-head cnn attention model for complete temporal attention
    """
    def __init__(self, C_in, C_out=16, H=128, W=128, is_causal=False, n_head=8, \
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), \
                    stride_t=(2,2), att_dropout_p=0.0, 
                    cosine_att=False, normalize_Q_K=False, att_with_output_proj=True):
        """
        Defines the layer for a cnn self-attention on temporal axis

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H', W']
        Usually used with conv definition such that H',W' = H,W

        Calculates attention using all the time points

        @args:
            - is_causal (bool): whether to mask attention to imply causality
            - stride_t (int, int): special stride for temporal attention k,q matrices
        """
        super().__init__(C_in=C_in, 
                         C_out=C_out, 
                         H=H, W=W,
                         n_head=n_head, 
                         kernel_size=kernel_size, 
                         stride=stride, 
                         padding=padding, 
                         att_dropout_p=att_dropout_p, 
                         cosine_att=cosine_att,
                         normalize_Q_K=normalize_Q_K, 
                         att_with_output_proj=att_with_output_proj)

        self.is_causal = is_causal
        self.stride_f = stride_t[0]
        
        assert self.C_out % self.n_head == 0, \
            f"Number of output channles {self.C_out} should be divisible by number of heads {self.n_head}"

        # key, query, value projections convolution
        # Wk, Wq, Wv
        self.key = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_t, padding=padding, bias=False)
        self.query = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_t, padding=padding, bias=False)
        self.value = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
           
        self.register_buffer("mask", torch.tril(torch.ones(1000, 1000, dtype=torch.bool)).view(1, 1, 1000, 1000))

    def forward(self, x):
        """
        @args:
            x ([B, T, C, H, W]): Input of a batch of time series

        @rets:
            y ([B, T, C_out, H', W']): logits
        """
        B, T, C, H, W = x.size()

        assert C == self.C_in, f"Input channel {C} does not match expected input channel {self.C_in}"

        # apply the key, query and value matrix
        k = self.key(x)
        q = self.query(x)
        
        _, _, C_prime, H_prime, W_prime = k.shape
        
        if self.normalize_Q_K:
            eps = torch.finfo(k.dtype).eps
            # add normalization for k and q, along [C_prime, H_prime, W_prime]
            k = (k - torch.mean(k, dim=(-3, -2, -1), keepdim=True)) / ( torch.sqrt(torch.var(k, dim=(-3, -2, -1), keepdim=True) + eps) )
            q = (q - torch.mean(q, dim=(-3, -2, -1), keepdim=True)) / ( torch.sqrt(torch.var(q, dim=(-3, -2, -1), keepdim=True) + eps) )
            
        k = k.view(B, T, self.n_head, torch.div(self.C_out, self.n_head, rounding_mode="floor"), H_prime, W_prime).transpose(1, 2)
        q = q.view(B, T, self.n_head, torch.div(self.C_out, self.n_head, rounding_mode="floor"), H_prime, W_prime).transpose(1, 2)            
        v = self.value(x).view(B, T, self.n_head, torch.div(self.C_out, self.n_head, rounding_mode="floor"), H_prime*self.stride_f, W_prime*self.stride_f).transpose(1, 2)

        # k, q, v are [B, nh, T, hc, H', W']

        B, nh, T, hc, H_prime, W_prime = k.shape

        # (B, nh, T, hc, H', W') x (B, nh, hc, H', W', T) -> (B, nh, T, T)
        if self.cosine_att:
            att = F.normalize(q.view(B, nh, T, hc*H_prime*W_prime), dim=-1) @ F.normalize(k.view(B, nh, T, hc*H_prime*W_prime), dim=-1).transpose(-2, -1)
        else:       
            att = (q.view(B, nh, T, hc*H_prime*W_prime) @ k.view(B, nh, T, hc*H_prime*W_prime).transpose(-2, -1))\
                    * torch.tensor(1.0 / math.sqrt(hc*H_prime*W_prime))

        # if causality is needed, apply the mask
        if(self.is_causal):
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # (B, nh, T, T) * (B, nh, T, hc, H', W')
        y = att @ v.view(B, nh, T, hc*H_prime*W_prime*self.stride_f*self.stride_f)
        y = y.transpose(1, 2).contiguous().view(B, T, self.C_out, H_prime*self.stride_f, W_prime*self.stride_f)
        
        y = self.output_proj(y)

        return y
    
# -------------------------------------------------------------------------------------------------

def tests():
    # tests

    B, T, C, H, W = 2, 4, 3, 64, 64
    C_out = 8
    test_in = torch.rand(B,T,C,H,W)

    print("Begin Testing")

    causals = [True, False]
    normalize_Q_Ks = [True, False]
    att_with_output_projs = [True, False]
    for causal in causals:
        for normalize_Q_K in normalize_Q_Ks:
            for att_with_output_proj in att_with_output_projs:

                temporal = TemporalCnnAttention(C, C_out=C_out, is_causal=causal, normalize_Q_K=normalize_Q_K, att_with_output_proj=att_with_output_proj)
                test_out = temporal(test_in)

                Bo, To, Co, Ho, Wo = test_out.shape
                assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed temporal")

    print("Passed all tests")

if __name__=="__main__":
    tests()
