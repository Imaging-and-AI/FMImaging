"""
Spatio-Temporal Convolutional Neural Net Transformer (STCNNT)

Implement the global patch spatial attention.

"""

import math
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

from attention_modules import *

# -------------------------------------------------------------------------------------------------
# CNN attention with the spatial global patching - an image is split into windows. A window is split into patches.
# Attention coefficients are computed among all corresponding patches in all windows.

class SpatialGlobalAttention(CnnAttentionBase):
    """
    Multi-head cnn attention model for the global patching. Number of pixels in a window are [wind_size, wind_size].
    Number of pixels in a patch are [patch_size, patch_size]
    """
    def __init__(self, C_in, C_out=16, 
                 wind_size=64, patch_size=8, 
                 a_type="conv", n_head=8,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                 att_dropout_p=0.0, dropout_p=0.1, 
                 normalize_Q_K=False, att_with_output_proj=True):
        """
        Defines the layer for a cnn attention on spatial dimension with local windows and patches.

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H, W]

        Shared parameters are defined in base class.

        @args:
            - wind_size (int): number of pixels in a window
            - patch_size(int): number of pixels in a patch
        """
        super().__init__(C_in=C_in, 
                         C_out=C_out, 
                         n_head=n_head, 
                         kernel_size=kernel_size, 
                         stride=stride, 
                         padding=padding, 
                         att_dropout_p=att_dropout_p, 
                         dropout_p=dropout_p, 
                         normalize_Q_K=normalize_Q_K, 
                         att_with_output_proj=att_with_output_proj)

        self.a_type = a_type
        self.wind_size = wind_size
        self.patch_size = patch_size

        assert self.C_out*self.patch_size*self.patch_size % self.n_head == 0, \
            f"Number of pixels in a window {self.C_out*self.patch_size*self.patch_size} should be divisible by number of heads {self.n_head}"

        if a_type=="conv":
            # key, query, value projections convolution
            # Wk, Wq, Wv
            self.key = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.query = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.value = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif a_type=="lin":
            # linear projections
            self.key = LinearGridExt(C_in*patch_size*patch_size, C_out*patch_size*patch_size, bias=False)
            self.query = LinearGridExt(C_in*patch_size*patch_size, C_out*patch_size*patch_size, bias=False)
            self.value = LinearGridExt(C_in*patch_size*patch_size, C_out*patch_size*patch_size, bias=False)
        else:
            raise NotImplementedError(f"Attention type not implemented: {a_type}")


    def forward(self, x):
        """
        @args:
            x ([B, T, C, H, W]): Input of a batch of time series

        @rets:
            y ([B, T, C_out, H', W']): output tensor
        """
        B, T, C, H, W = x.size()
        Ws = self.wind_size
        Ps = self.patch_size

        assert C == self.C_in, f"Input channel {C} does not match expected input channel {self.C_in}"
        assert H % Ws == 0, f"Height {H} should be divisible by window size {Ws}"
        assert W % Ws == 0, f"Width {W} should be divisible by window size {Ws}"
        assert Ws % Ps == 0, f"Ws {Ws} should be divisible by patch size {Ps}"

        if self.a_type=="conv":
            k = self.key(x) # (B, T, C, H_prime, W_prime)
            q = self.query(x)
            v = self.value(x)

            k = self.im2grid(k) # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, C, Ps, Ps)
            q = self.im2grid(q)
            v = self.im2grid(v)
        else:
            x = self.im2grid(x) # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, C_in, Ps, Ps)
            k = self.key(x) # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, C, Ps, Ps)
            q = self.query(x)
            v = self.value(x)
            
        B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, C, ph, pw = k.shape

        # format the window
        hc = torch.div(C*ph*pw, self.n_head, rounding_mode="floor")

        # k, q, v will be [B, T, num_patch_h_per_win*num_patch_w_per_win, self.n_head, num_win_h*num_win_w, hc]
        k = k.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc)).transpose(3, 4)         
        q = q.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc)).transpose(3, 4)
        v = v.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc)).transpose(3, 4)
        
        if self.normalize_Q_K:
            eps = torch.finfo(k.dtype).eps
            k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
            q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )
            
        # Compute attention matrix, use the matrix broadcasing 
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        # [B, T, num_patches, num_heads, num_windows, hc] x [B, T, num_patches, num_heads, hc, num_windows] -> (B, T, num_patches, num_heads, num_windows, num_windows)
        att = q @ k.transpose(-2, -1) * torch.tensor(1.0 / math.sqrt(hc))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # (B, T, num_patches, num_heads, num_windows, num_windows) * (B, T, num_patches, num_heads, num_windows, hc)
        y = att @ v # (B, T, num_patches, num_heads, num_windows, hc)
        y = y.transpose(3, 4) # (B, T, num_patches, num_windows, num_heads, hc)
        y = torch.reshape(y, (B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, C, ph, pw))
        
        y = self.grid2im(y)
        
        if self.att_with_output_proj:
            y = y + self.resid_drop(self.output_proj(y))
        else:
            y = self.resid_drop(y)

        return y

    def im2grid(self, x):
        """
        Reshape the input into windows of local areas        
        """
        b, t, c, h, w = x.shape
        Ws = self.wind_size
        Ps = self.patch_size

        wind_view = rearrange(x, 'b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b t num_win_h num_win_w num_patch_h num_patch_w c patch_size_h patch_size_w', 
                              num_win_h=h//Ws, num_patch_h=Ws//Ps, patch_size_h=Ps, 
                              num_win_w=w//Ws, num_patch_w=Ws//Ps, patch_size_w=Ps)
        
        wind_view = torch.permute(wind_view, (0, 1, 4, 5, 2, 3, 6, 7, 8))
        
        return wind_view

    def grid2im(self, x):
        """
        Reshape the windows back into the complete image
        """
        b, t, num_patch_h, num_patch_w, num_win_h, num_win_w, c, ph, pw = x.shape

        im_view = torch.permute(x, (0, 1, 4, 5, 2, 3, 6, 7, 8))
        
        im_view = rearrange(im_view, 'b t num_win_h num_win_w num_patch_h num_patch_w c patch_size_h patch_size_w -> b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)', 
                              num_win_h=num_win_h, num_patch_h=num_patch_h, patch_size_h=ph, 
                              num_win_w=num_win_w, num_patch_w=num_patch_w, patch_size_w=pw)
        return im_view
    
# -------------------------------------------------------------------------------------------------

def tests():
    
    print("Begin Testing")

    t = np.arange(256)
    t = np.reshape(t, (16,16))

    w = 8

    t = torch.from_numpy(t).to(dtype=torch.float32)
    t = torch.cat((t[None, :], t[None, :]), dim=0)

    B, T, C, H, W = 2, 4, 2, 16, 16
    C_out = 8
    test_in = t.repeat(B, T, 1, 1, 1)
    print(test_in.shape)
    
    spacial_vit = SpatialGlobalAttention(wind_size=w, patch_size=4, a_type="conv", C_in=C, C_out=C_out)
    
    a = spacial_vit.im2grid(test_in)  
    b = spacial_vit.grid2im(a)
    
    gt = torch.tensor([[[[ 64.,  65.,  66.,  67.],
          [ 80.,  81.,  82.,  83.],
          [ 96.,  97.,  98.,  99.],
          [112., 113., 114., 115.]],

         [[ 72.,  73.,  74.,  75.],
          [ 88.,  89.,  90.,  91.],
          [104., 105., 106., 107.],
          [120., 121., 122., 123.]]],


        [[[192., 193., 194., 195.],
          [208., 209., 210., 211.],
          [224., 225., 226., 227.],
          [240., 241., 242., 243.]],

         [[200., 201., 202., 203.],
          [216., 217., 218., 219.],
          [232., 233., 234., 235.],
          [248., 249., 250., 251.]]]])
    
    if torch.norm(a[0, 0, 1, 0, :, :, 0, :, :] - gt)>1e-3:
        raise "im2grid test failed"
    
    if torch.norm(b-test_in)<1e-3:   
        print("Passed im2grid test")
    else:
        raise "im2grid test failed"
       
    a_types = ["conv", "lin"]
    normalize_Q_Ks = [True, False]
    att_with_output_projs = [True, False]

    for a_type in a_types:
        for normalize_Q_K in normalize_Q_Ks:
            for att_with_output_proj in att_with_output_projs:

                spacial_vit = SpatialGlobalAttention(wind_size=8, patch_size=4, a_type=a_type, C_in=C, C_out=C_out, normalize_Q_K=normalize_Q_K, att_with_output_proj=att_with_output_proj)
                test_out = spacial_vit(test_in)

                Bo, To, Co, Ho, Wo = test_out.shape
                assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo
                
                loss = nn.MSELoss()
                mse = loss(test_in, test_out[:,:,:C,:,:])
                mse.backward()
                
    print("Passed SpatialGlobalAttention tests")
    
    print("Passed all tests")

if __name__=="__main__":
    tests()
