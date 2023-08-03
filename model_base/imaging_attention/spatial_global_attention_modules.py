"""
Spatio-Temporal Convolutional Neural Net Transformer (STCNNT)

Implement the global patch spatial attention.

"""

import math
import sys
import time
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
    Multi-head cnn attention model for the global patching. Number of pixels in a window are [window_size, window_size].
    Number of pixels in a patch are [patch_size, patch_size]
    """
    def __init__(self, C_in, C_out=16, H=128, W=128,
                 window_size=[32, 32], patch_size=[8, 8], 
                 num_wind=[4, 4], num_patch=[4, 4], 
                 a_type="conv", n_head=8,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), stride_qk=(1,1),
                 separable_conv=False, 
                 att_dropout_p=0.0, 
                 cosine_att=False, 
                 normalize_Q_K=False, 
                 att_with_relative_postion_bias=True,
                 att_with_output_proj=True,
                 shuffle_in_window=False,
                 use_einsum=False,
                 with_timer=False):
        """
        Defines the layer for a cnn attention on spatial dimension with local windows and patches.

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H, W]

        Shared parameters are defined in base class.

        @args:
            - window_size (int): number of pixels in a window
            - patch_size(int): number of pixels in a patch
            - shuffle_in_window (bool): If True, shuffle the order of patches in all windows; this will avoid the same set of patches are always inputted into the attention, but introducing randomness
        """
        super().__init__(C_in=C_in, 
                         C_out=C_out, 
                         H=H, W=W,
                         n_head=n_head, 
                         kernel_size=kernel_size, 
                         stride=stride, 
                         separable_conv=separable_conv,
                         padding=padding, 
                         stride_qk=stride_qk,
                         att_dropout_p=att_dropout_p, 
                         cosine_att=cosine_att,
                         normalize_Q_K=normalize_Q_K, 
                         att_with_relative_postion_bias=att_with_relative_postion_bias,
                         att_with_output_proj=att_with_output_proj, with_timer=with_timer)

        self.a_type = a_type
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_wind = num_wind
        self.num_patch = num_patch
        self.shuffle_in_window = shuffle_in_window
        self.use_einsum = use_einsum

        self.set_and_check_wind()
        self.set_and_check_patch()

        self.validate_window_patch()

        assert self.C_out*self.patch_size[0]*self.patch_size[1] % self.n_head == 0, \
            f"Number of pixels in a window {self.C_out*self.patch_size[0]*self.patch_size[1]} should be divisible by number of heads {self.n_head}"

        if a_type=="conv":
            # key, query, value projections convolution
            # Wk, Wq, Wv
            self.key = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_qk, padding=padding, bias=False, separable_conv=self.separable_conv)
            self.query = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_qk, padding=padding, bias=False, separable_conv=self.separable_conv)
            self.value = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, separable_conv=self.separable_conv)
        elif a_type=="lin":
            # linear projections
            num_pixel_patch = self.patch_size[0]*self.patch_size[1]
            self.key = LinearGridExt(C_in*num_pixel_patch, C_out*num_pixel_patch, bias=False)
            self.query = LinearGridExt(C_in*num_pixel_patch, C_out*num_pixel_patch, bias=False)
            self.value = LinearGridExt(C_in*num_pixel_patch, C_out*num_pixel_patch, bias=False)
        else:
            raise NotImplementedError(f"Attention type not implemented: {a_type}")

        if self.att_with_relative_postion_bias:
            self.define_relative_position_bias_table(num_win_h=self.num_wind[0], num_win_w=self.num_wind[1])
            self.define_relative_position_index(num_win_h=self.num_wind[0], num_win_w=self.num_wind[1])

    def attention(self, k, q, v):
        B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, ph, pw, C = k.shape
        ph_v, pw_v, _ = v.shape[-3:]

        # format the window
        hc = torch.div(C*ph*pw, self.n_head, rounding_mode="floor")
        hc_v = torch.div(C*ph_v*pw_v, self.n_head, rounding_mode="floor")

        tm = start_timer(enable=self.with_timer)
        # k, q, v will be [B, T, num_patch_h_per_win*num_patch_w_per_win, self.n_head, num_win_h*num_win_w, hc]
        k = k.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc)).transpose(3, 4)
        q = q.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc)).transpose(3, 4)
        v = v.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc_v)).transpose(3, 4)
        end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")
        
        if self.shuffle_in_window:

            tm = start_timer(enable=self.with_timer)
            
            # random permute within a window
            patch_indexes = torch.zeros([num_win_h*num_win_w, num_patch_h_per_win*num_patch_w_per_win], dtype=torch.long)
            for w in range(num_win_h*num_win_w):
                patch_indexes[w, :] = torch.randperm(num_patch_h_per_win*num_patch_w_per_win)

            reverse_patch_indexes = num_patch_h_per_win*num_patch_w_per_win - 1 - patch_indexes
            reverse_patch_indexes = torch.flip(reverse_patch_indexes, dims=(1,))

            k_shuffled = torch.clone(k)
            q_shuffled = torch.clone(q)
            v_shuffled = torch.clone(v)

            for w in range(num_win_h*num_win_w):
                k_shuffled[:, :, :, :, w] = k[:, :, patch_indexes[w, :], :, w]
                q_shuffled[:, :, :, :, w] = q[:, :, patch_indexes[w, :], :, w]
                v_shuffled[:, :, :, :, w] = v[:, :, patch_indexes[w, :], :, w]

            k = k_shuffled
            q = q_shuffled
            v = v_shuffled

            end_timer(enable=self.with_timer, t=tm, msg="shuffle_in_window")
            
        tm = start_timer(enable=self.with_timer)
        # [B, T, num_patches, num_heads, num_windows, hc] x [B, T, num_patches, num_heads, hc, num_windows] -> (B, T, num_patches, num_heads, num_windows, num_windows)
        if self.cosine_att:
            att = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        else:
            if self.normalize_Q_K:
                eps = torch.finfo(k.dtype).eps
                k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
                q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

            att = q @ k.transpose(-2, -1) * torch.tensor(1.0 / math.sqrt(hc))
        end_timer(enable=self.with_timer, t=tm, msg="att")
        
        tm = start_timer(enable=self.with_timer)
        att = F.softmax(att, dim=-1)
        end_timer(enable=self.with_timer, t=tm, msg="softmax")
        
        tm = start_timer(enable=self.with_timer)
        if self.att_with_relative_postion_bias:
            relative_position_bias = self.get_relative_position_bias(num_win_h, num_win_w)
            att = att + relative_position_bias
        end_timer(enable=self.with_timer, t=tm, msg="relative_position_bias")

        tm = start_timer(enable=self.with_timer)
        att = self.attn_drop(att)
        end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

        tm = start_timer(enable=self.with_timer)
        # (B, T, num_patches, num_heads, num_windows, num_windows) * (B, T, num_patches, num_heads, num_windows, hc_v)
        y = att @ v # (B, T, num_patches, num_heads, num_windows, hc_v)
        end_timer(enable=self.with_timer, t=tm, msg="att @ v")
        
        tm = start_timer(enable=self.with_timer)
        y = y.transpose(3, 4) # (B, T, num_patches, num_windows, num_heads, hc_v)
        end_timer(enable=self.with_timer, t=tm, msg="y.transpose")

        tm = start_timer(enable=self.with_timer)
        if self.shuffle_in_window:
            y_restored = torch.clone(y)
            for w in range(num_win_h*num_win_w):
                y_restored[:, :, :, w] = y[:, :, reverse_patch_indexes[w, :], w]

            y = torch.reshape(y_restored, (B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, ph_v, pw_v, C))
        else:
            y = torch.reshape(y, (B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, ph_v, pw_v, C))
        end_timer(enable=self.with_timer, t=tm, msg="y reshape")
        
        tm = start_timer(enable=self.with_timer)
        y = self.grid2im(y)
        end_timer(enable=self.with_timer, t=tm, msg="grid2im")
        
        return y

    def einsum_attention(self, k, q, v):
        B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, ph, pw, C = k.shape
        ph_v, pw_v, _ = v.shape[-3:]

        # format the window
        hc = torch.div(C*ph*pw, self.n_head, rounding_mode="floor")
        hc_v = torch.div(C*ph_v*pw_v, self.n_head, rounding_mode="floor")

        if self.has_flash_attention and hc <= 256 and (not self.att_with_relative_postion_bias) and (not self.shuffle_in_window) and (hc_v == hc):
            tm = start_timer(enable=self.with_timer)
            D = B*T*num_patch_h_per_win*num_patch_w_per_win

            k = k.reshape((D, num_win_h*num_win_w, self.n_head, hc))
            q = q.reshape((D, num_win_h*num_win_w, self.n_head, hc))
            v = v.reshape((D, num_win_h*num_win_w, self.n_head, hc_v))
            end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

            tm = start_timer(enable=self.with_timer)
            y = self.perform_flash_atten(k, q, v)
            end_timer(enable=self.with_timer, t=tm, msg="perform_flash_atten")


            tm = start_timer(enable=self.with_timer)
            y = y.reshape(B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, ph_v, pw_v, C)
            end_timer(enable=self.with_timer, t=tm, msg="y.reshape")
        else:
            tm = start_timer(enable=self.with_timer)
            # k, q, v will be [B, T, num_patch_h_per_win*num_patch_w_per_win, self.n_head, num_win_h*num_win_w, hc]
            k = k.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc))
            q = q.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc))
            v = v.reshape((B, T, num_patch_h_per_win*num_patch_w_per_win, num_win_h*num_win_w, self.n_head, hc_v))
            end_timer(enable=self.with_timer, t=tm, msg="k, q, v - reshape")

            if self.shuffle_in_window:

                tm = start_timer(enable=self.with_timer)
                
                # random permute within a window
                patch_indexes = torch.zeros([num_win_h*num_win_w, num_patch_h_per_win*num_patch_w_per_win], dtype=torch.long)
                for w in range(num_win_h*num_win_w):
                    patch_indexes[w, :] = torch.randperm(num_patch_h_per_win*num_patch_w_per_win)

                reverse_patch_indexes = num_patch_h_per_win*num_patch_w_per_win - 1 - patch_indexes
                reverse_patch_indexes = torch.flip(reverse_patch_indexes, dims=(1,))

                k_shuffled = torch.clone(k)
                q_shuffled = torch.clone(q)
                v_shuffled = torch.clone(v)

                for w in range(num_win_h*num_win_w):
                    k_shuffled[:, :, :, w] = k[:, :, patch_indexes[w, :], w]
                    q_shuffled[:, :, :, w] = q[:, :, patch_indexes[w, :], w]
                    v_shuffled[:, :, :, w] = v[:, :, patch_indexes[w, :], w]

                k = k_shuffled
                q = q_shuffled
                v = v_shuffled
                end_timer(enable=self.with_timer, t=tm, msg="shuffle_in_window")
                
            tm = start_timer(enable=self.with_timer)
            # [B, T, num_patches, num_windows, num_heads, hc] x [B, T, num_patches, num_windows, num_heads, hc] -> (B, T, num_patches, num_heads, num_windows, num_windows)
            if self.cosine_att:
                att = torch.einsum("BTPWND, BTPIND -> BTPNWI", F.normalize(q, dim=-1), F.normalize(k, dim=-1))
            else:
                if self.normalize_Q_K:
                    eps = torch.finfo(k.dtype).eps
                    k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
                    q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

                att = torch.einsum("BTPWND, BTPIND -> BTPNWI", q, k) * torch.tensor(1.0 / math.sqrt(hc))
            end_timer(enable=self.with_timer, t=tm, msg="BTPWND, BTPIND -> BTPNWI")
            
            tm = start_timer(enable=self.with_timer)
            att = F.softmax(att, dim=-1)
            end_timer(enable=self.with_timer, t=tm, msg="softmax")
            
            tm = start_timer(enable=self.with_timer)
            if self.att_with_relative_postion_bias:
                relative_position_bias = self.get_relative_position_bias(num_win_h, num_win_w)
                att = att + relative_position_bias
            end_timer(enable=self.with_timer, t=tm, msg="relative_position_bias")

            tm = start_timer(enable=self.with_timer)
            att = self.attn_drop(att)
            end_timer(enable=self.with_timer, t=tm, msg="attn_drop")

            tm = start_timer(enable=self.with_timer)
            # (B, T, num_patches, num_heads, num_windows, num_windows) * (B, T, num_patches, num_windows, num_heads, hc)
            y = torch.einsum("BTPNWI, BTPIND -> BTPWND", att, v)
            end_timer(enable=self.with_timer, t=tm, msg="BTPNWI, BTPIND -> BTPWND")

            tm = start_timer(enable=self.with_timer)
            if self.shuffle_in_window:                
                y_restored = torch.clone(y)
                for w in range(num_win_h*num_win_w):
                    y_restored[:, :, :, w] = y[:, :, reverse_patch_indexes[w, :], w]

                y = torch.reshape(y_restored, (B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, ph_v, pw_v, C))
            else:
                y = y.reshape(B, T, num_patch_h_per_win, num_patch_w_per_win, num_win_h, num_win_w, ph_v, pw_v, C)
            end_timer(enable=self.with_timer, t=tm, msg="y.reshape")
            
        tm = start_timer(enable=self.with_timer)
        y = self.grid2im(y)
        end_timer(enable=self.with_timer, t=tm, msg="grid2im")

        return y

    def forward(self, x):
        """
        @args:
            x ([B, T, C, H, W]): Input of a batch of time series

        @rets:
            y ([B, T, C_out, H', W']): output tensor
        """
        B, T, C, H, W = x.size()

        assert C == self.C_in, f"Input channel {C} does not match expected input channel {self.C_in}"

        if self.a_type=="conv":
            tm = start_timer(enable=self.with_timer)
            k = self.key(x) # (B, T, C, H_prime, W_prime)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")
            
            tm = start_timer(enable=self.with_timer)
            k = self.im2grid(k) # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, Ps, Ps, C)
            q = self.im2grid(q)
            v = self.im2grid(v)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")
        else:
            tm = start_timer(enable=self.with_timer)
            x = self.im2grid(x) # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, Ps, Ps, C_in)
            end_timer(enable=self.with_timer, t=tm, msg="im2grid")
            
            tm = start_timer(enable=self.with_timer)
            k = self.key(x) # (B, T, num_patch_per_win, num_patch_per_win, num_win_h, num_win_w, Ps, Ps, C)
            q = self.query(x)
            v = self.value(x)
            end_timer(enable=self.with_timer, t=tm, msg="compute k, q, v")
            
        if self.use_einsum:
            y = self.einsum_attention(k, q, v)
        else:
            y = self.attention(k, q, v)

        tm = start_timer(enable=self.with_timer)
        y = self.output_proj(y)
        end_timer(enable=self.with_timer, t=tm, msg="output_proj")

        return y

    def im2grid(self, x):
        """
        Reshape the input into windows of local areas
        """
        b, t, c, h, w = x.shape

        if self.a_type=="conv" and self.stride_qk[0]>1:
            wind_view = rearrange(x, 'b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b t num_patch_h num_patch_w num_win_h num_win_w patch_size_h patch_size_w c', 
                                num_win_h=self.num_wind[0], num_patch_h=self.num_patch[0], patch_size_h=h//(self.num_wind[0]*self.num_patch[0]), 
                                num_win_w=self.num_wind[1], num_patch_w=self.num_patch[1], patch_size_w=w//(self.num_wind[0]*self.num_patch[1]))
        else:
            wind_view = rearrange(x, 'b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w) -> b t num_patch_h num_patch_w num_win_h num_win_w patch_size_h patch_size_w c', 
                            num_win_h=self.num_wind[0], num_patch_h=h//(self.num_wind[0]*self.patch_size[0]), patch_size_h=self.patch_size[0], 
                            num_win_w=self.num_wind[1], num_patch_w=w//(self.num_wind[0]*self.patch_size[1]), patch_size_w=self.patch_size[1])

        return wind_view

    def grid2im(self, x):
        """
        Reshape the windows back into the complete image
        """
        b, t, num_patch_h, num_patch_w, num_win_h, num_win_w, ph, pw, c = x.shape

        # im_view = torch.permute(x, (0, 1, 8, 4, 2, 6, 5, 3, 7))

        # im_view = rearrange(im_view, 'b t c num_win_h num_patch_h patch_size_h num_win_w num_patch_w patch_size_w -> b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)', 
        #                       num_win_h=num_win_h, num_patch_h=num_patch_h, patch_size_h=ph, 
        #                       num_win_w=num_win_w, num_patch_w=num_patch_w, patch_size_w=pw)

        im_view = rearrange(x, 'b t num_patch_h num_patch_w num_win_h num_win_w patch_size_h patch_size_w c -> b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)', 
                              num_win_h=num_win_h, num_patch_h=num_patch_h, patch_size_h=ph, 
                              num_win_w=num_win_w, num_patch_w=num_patch_w, patch_size_w=pw)
        
        #im_view = torch.permute(x, (0, 1, 4, 5, 2, 3, 6, 7, 8))

        # im_view = rearrange(x, 'b t num_patch_h num_patch_w num_win_h num_win_w patch_size_h patch_size_w c -> b t c (num_win_h num_patch_h patch_size_h) (num_win_w num_patch_w patch_size_w)', 
        #                       num_win_h=num_win_h, num_patch_h=num_patch_h, patch_size_h=ph, 
        #                       num_win_w=num_win_w, num_patch_w=num_patch_w, patch_size_w=pw)
        return im_view

# -------------------------------------------------------------------------------------------------

def tests():
    import time
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

    spacial_vit = SpatialGlobalAttention(H=H, W=W, window_size=[8, 8], patch_size=[4, 4], stride_qk=(2,2), num_wind=None, num_patch=None, a_type="conv", C_in=C, C_out=C_out)

    a = spacial_vit.im2grid(test_in)  
    b = spacial_vit.grid2im(a)

    assert torch.allclose(test_in, b)

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

    if torch.norm(a[0, 0, 1, 0, :, :, :, :, 0] - gt)>1e-3:
        raise "im2grid test failed"

    if torch.norm(b-test_in)<1e-3:   
        print("Passed im2grid test")
    else:
        raise "im2grid test failed"

    a_types = ["conv", "lin"]
    normalize_Q_Ks = [True, False]
    cosine_atts = [True, False]
    att_with_relative_postion_biases = [True, False]
    att_with_output_projs = [True, False]
    #stride_qks = [[1, 1], [2, 2]]
    stride_qks = [[2, 2]]

    device = get_device()

    B, T, C, H1, W1 = 1, 3, 2, 32, 32
    C_out = 4
    test_in = torch.rand(B, T, C, H1, W1).to(device=device)
    print(test_in.shape)
    
    B, T, C, H2, W2 = 1, 3, 2, 128, 128
    C_out = 4
    test_in2 = torch.rand(B, T, C, H2, W2).to(device=device)
    print(test_in2.shape)

    for a_type in a_types:
        for normalize_Q_K in normalize_Q_Ks:
            for att_with_output_proj in att_with_output_projs:
                for cosine_att in cosine_atts:
                    for att_with_relative_postion_bias in att_with_relative_postion_biases:
                        for stride_qk in stride_qks:
                            t0 = time.time()
                            m = SpatialGlobalAttention(window_size=[16, 16], patch_size=[8, 8], 
                                                        num_wind=None, num_patch=None, 
                                                        a_type=a_type, 
                                                        C_in=C, C_out=C_out, 
                                                        H=H1, W=W1, 
                                                        stride_qk=stride_qk,
                                                        separable_conv=True,
                                                        cosine_att=cosine_att, 
                                                        normalize_Q_K=normalize_Q_K, 
                                                        att_with_relative_postion_bias=att_with_relative_postion_bias,
                                                        att_with_output_proj=att_with_output_proj)

                            m.to(device=device)
                            test_out = m(test_in)
                            t1 = time.time()
                            print(f"forward pass - {t1-t0} seconds")

                            Bo, To, Co, Ho, Wo = test_out.shape
                            assert B==Bo and T==To and Co==C_out and H1==Ho and W1==Wo

                            t0 = time.time()
                            loss = nn.MSELoss()
                            mse = loss(test_in, test_out[:,:,:C,:,:])
                            mse.backward()
                            t1 = time.time()
                            print(f"backward pass - {t1-t0} seconds")

                            test_out = m(test_in2)

                            Bo, To, Co, Ho, Wo = test_out.shape
                            assert B==Bo and T==To and Co==C_out and H2==Ho and W2==Wo

    print("Passed SpatialGlobalAttention tests")

    print("Passed all tests")

# -------------------------------------------------------------------------------------------------

def benchmark():

    from utils.benchmark import benchmark_all, benchmark_memory, pytorch_profiler
    from utils.setup_training import set_seed
    from colorama import Fore, Style

    set_seed(seed=53)

    device = get_device()

    B, T, C, H, W = 16, 12, 3, 256, 256    
    test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)

    min_run_time = 5

    C_out = 32
    n_head = 32
    
    window_size=[16, 16]
    patch_size=[2, 2] 
    num_wind=[8, 8]
    num_patch=[4, 4]

    att_dropout_p=0.1 
    cosine_att=True 
    normalize_Q_K=True 
    att_with_relative_postion_bias=True
    att_with_output_proj=True
    shuffle_in_window=False

    print(f"{Fore.YELLOW}==================================================================={Style.RESET_ALL}")

    stride_qk = (2, 2)
    separable_conv = True

    m = SpatialGlobalAttention(C_in=C, C_out=C_out, H=H, W=H,
                            window_size=window_size, patch_size=patch_size, 
                            num_wind=num_wind, num_patch=num_patch,  
                            a_type="conv", n_head=n_head,
                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                            stride_qk = stride_qk, separable_conv=separable_conv, 
                            att_dropout_p=att_dropout_p, 
                            cosine_att=cosine_att, 
                            normalize_Q_K=normalize_Q_K, 
                            att_with_relative_postion_bias=att_with_relative_postion_bias,
                            att_with_output_proj=att_with_output_proj,
                            shuffle_in_window=shuffle_in_window,
                            use_einsum=False)

    m.to(device=device)

    m.with_timer = True
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            t0 = start_timer(enable=True)
            y = m(test_in)
            end_timer(enable=True, t=t0, msg=f"{Fore.RED} --- > forward, {m.stride_qk}, {m.separable_conv} < --- {Style.RESET_ALL}")

            t0 = start_timer(enable=True)
            loss = nn.MSELoss()
            mse = loss(test_in, y[:,:,:C,:,:])
            mse.backward()
            end_timer(enable=True, t=t0, msg=f"{Fore.RED} --- > backward, {m.stride_qk}, {m.separable_conv} < --- {Style.RESET_ALL}")

    print(f"{Fore.YELLOW}==================================================================={Style.RESET_ALL}")

    stride_qk = (1, 1)
    separable_conv = False
    
    m = SpatialGlobalAttention(C_in=C, C_out=C_out, H=H, W=H,
                            window_size=window_size, patch_size=patch_size, 
                            num_wind=num_wind, num_patch=num_patch,  
                            a_type="conv", n_head=n_head,
                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                            stride_qk = stride_qk, separable_conv=separable_conv, 
                            att_dropout_p=att_dropout_p, 
                            cosine_att=cosine_att, 
                            normalize_Q_K=normalize_Q_K, 
                            att_with_relative_postion_bias=att_with_relative_postion_bias,
                            att_with_output_proj=att_with_output_proj,
                            shuffle_in_window=shuffle_in_window,
                            use_einsum=False)

    m.with_timer = True
    m.to(device=device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for _ in range(10):
            t0 = start_timer(enable=True)
            y = m(test_in)
            end_timer(enable=True, t=t0, msg=f"{Fore.RED} --- > forward, {m.stride_qk}, {m.separable_conv} < --- {Style.RESET_ALL}")

            t0 = start_timer(enable=True)
            loss = nn.MSELoss()
            mse = loss(test_in, y[:,:,:C,:,:])
            mse.backward()
            end_timer(enable=True, t=t0, msg=f"{Fore.RED} --- > backward, {m.stride_qk}, {m.separable_conv} < --- {Style.RESET_ALL}")

    time.sleep(3.0)
    
    print(f"{Fore.YELLOW}==================================================================={Style.RESET_ALL}")
    
    stride_qk = (2, 2)
    separable_conv = True
    
    print(f"{Fore.GREEN}-------------> SpatialGlobalAttention <----------------------{Style.RESET_ALL}")

    m = SpatialGlobalAttention(C_in=C, C_out=C_out, H=H, W=H,
                            window_size=window_size, patch_size=patch_size, 
                            num_wind=num_wind, num_patch=num_patch,  
                            a_type="conv", n_head=n_head,
                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                            stride_qk = stride_qk, separable_conv=separable_conv, 
                            att_dropout_p=att_dropout_p, 
                            cosine_att=cosine_att, 
                            normalize_Q_K=normalize_Q_K, 
                            att_with_relative_postion_bias=att_with_relative_postion_bias,
                            att_with_output_proj=att_with_output_proj,
                            shuffle_in_window=shuffle_in_window,
                            use_einsum=False)

    m.to(device=device)

    with torch.inference_mode():
        y = m(test_in)

    benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialGlobalAttention-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    benchmark_memory(m, test_in, desc='SpatialGlobalAttention-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    print(f"{Fore.YELLOW}==================================================================={Style.RESET_ALL}")
    
    m = SpatialGlobalAttention(C_in=C, C_out=C_out, H=H, W=H,
                            window_size=window_size, patch_size=patch_size, 
                            num_wind=num_wind, num_patch=num_patch,  
                            a_type="conv", n_head=n_head,
                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                            stride_qk = stride_qk, separable_conv=separable_conv, 
                            att_dropout_p=att_dropout_p, 
                            cosine_att=cosine_att, 
                            normalize_Q_K=normalize_Q_K, 
                            att_with_relative_postion_bias=att_with_relative_postion_bias,
                            att_with_output_proj=att_with_output_proj,
                            shuffle_in_window=shuffle_in_window,
                            use_einsum=True)

    m.to(device=device)

    with torch.inference_mode():
        y = m(test_in)

    benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialGlobalAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    benchmark_memory(m, test_in, desc='SpatialGlobalAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    # def loss(model, x):
    #     y = model(x)
    #     l = torch.sum(y)
    #     return l

    # pytorch_profiler(loss, m, test_in, trace_filename='/export/Lab-Xue/projects/mri/profiling/SpatialViTAttention.json', backward=True, amp=True, amp_dtype=torch.bfloat16, cpu=False, verbose=True)

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":
    tests()
    benchmark()