"""
Spatio-Temporal Convolutional Neural Net Transformer (STCNNT)

This file implements the cell structure in the model architecture. A cell is a 'transformer module' consisting 
of attention layers, normalization layers and mixers with non-linearities.

Two type of  cells are implemented here: 

- sequential norm first, transformer model
- Parallel cell, as in the Google 22B ViT

"""

import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

from pathlib import Path
Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from utils.utils import create_generic_class_str

from imaging_attention import *

__all__ = ['STCNNT_Cell', 'STCNNT_Parallel_Cell']

# -------------------------------------------------------------------------------------------------
# Complete transformer cell

class STCNNT_Cell(nn.Module):
    """
    CNN Transformer Cell with any attention type

    The Pre-Norm implementation is used here:

    x-> Norm -> attention -> + -> Norm -> mixer -> + -> logits
    |------------------------| |-----------------------|
    """
    def __init__(self, C_in, C_out=16, H=128, W=128, 
                 att_mode="temporal", 
                 a_type="conv",
                 mixer_type="conv",
                 window_size=None, patch_size=None, num_wind=[8, 8], num_patch=[4, 4], 
                 is_causal=False, n_head=8,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), stride_t=(2,2),
                 mixer_kernel_size=(5, 5), mixer_stride=(1, 1), mixer_padding=(2, 2),
                 normalize_Q_K=False, 
                 att_dropout_p=0.0, 
                 dropout_p=0.1, 
                 cosine_att=True, 
                 att_with_relative_postion_bias=True,
                 att_with_output_proj=True, 
                 scale_ratio_in_mixer=4.0, 
                 with_mixer=True, 
                 norm_mode="layer",
                 shuffle_in_window=True):
        """
        Complete transformer cell

        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - H (int): expected height of the input
            - W (int): expected width of the input
            - att_mode ("local", "global", "temporal", 'vit'):
                different methods of attention mechanism
            - a_type ("conv", "lin"): type of attention in spatial heads
            - mixer_type ("conv", "lin"): type of mixers; for temporal attention, only conv mixer is possible
            - window_size (int): size of window for local and global att
            - patch_size (int): size of patch for local and global att
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - stride_t (int, int): special stride for temporal attention k,q matrices
            - normalize_Q_K (bool): whether to use layernorm to normalize Q and K, as in 22B ViT paper
            - att_dropout_p (float): probability of dropout for attention coefficients
            - dropout_p (float): probability of dropout for attention output
            - att_with_output_proj (bool): whether to add output projection in the attention layer
            - with_mixer (bool): whether to add a conv2D mixer after attention
            - scale_ratio_in_mixer (float): channel scaling ratio in the mixer
            - norm_mode ("layer", "batch2d", "instance2d", "batch3d", "instance3d"):
                - layer: each C,H,W
                - batch2d: along B*T
                - instance2d: each H,W
                - batch3d: along B
                - instance3d: each T,H,W
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.att_mode = att_mode
        self.a_type = a_type
        self.mixer_type = mixer_type
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_wind = num_wind
        self.num_patch = num_patch
        self.is_causal = is_causal
        self.n_head = n_head
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.stride_t = stride_t
        
        self.mixer_kernel_size = mixer_kernel_size
        self.mixer_stride = mixer_stride
        self.mixer_padding = mixer_padding
        
        self.normalize_Q_K = normalize_Q_K
        self.cosine_att = cosine_att
        self.att_with_relative_postion_bias = att_with_relative_postion_bias
        self.att_dropout_p = att_dropout_p
        self.dropout_p = dropout_p
        self.att_with_output_proj = att_with_output_proj
        self.with_mixer = with_mixer
        self.scale_ratio_in_mixer = scale_ratio_in_mixer
        self.norm_mode = norm_mode
        self.shuffle_in_window = shuffle_in_window
        
        if(norm_mode=="layer"):
            self.n1 = nn.LayerNorm([C_in, H, W])
            self.n2 = nn.LayerNorm([C_out, H, W])
        elif(norm_mode=="batch2d"):
            self.n1 = BatchNorm2DExt(C_in)
            self.n2 = BatchNorm2DExt(C_out)
        elif(norm_mode=="instance2d"):
            self.n1 = InstanceNorm2DExt(C_in)
            self.n2 = InstanceNorm2DExt(C_out)
        elif(norm_mode=="batch3d"):
            self.n1 = BatchNorm3DExt(C_in)
            self.n2 = BatchNorm3DExt(C_out)
        elif(norm_mode=="instance3d"):
            self.n1 = InstanceNorm3DExt(C_in)
            self.n2 = InstanceNorm3DExt(C_out)
        else:
            raise NotImplementedError(f"Norm mode not implemented: {norm_mode}")

        if C_in!=C_out:
            self.input_proj = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            self.input_proj = nn.Identity()

        if(att_mode=="temporal"):
            self.attn = TemporalCnnAttention(C_in=C_in, C_out=C_out, H=self.H, W=self.W,
                                             is_causal=is_causal, n_head=n_head, 
                                             kernel_size=kernel_size, stride=stride, padding=padding, 
                                             stride_t=stride_t, 
                                             cosine_att=cosine_att, normalize_Q_K=normalize_Q_K, 
                                             att_dropout_p=att_dropout_p, 
                                             att_with_output_proj=att_with_output_proj)
        elif(att_mode=="local"):
            self.attn = SpatialLocalAttention(C_in=C_in, C_out=C_out, H=self.H, W=self.W, 
                                              window_size=window_size, patch_size=patch_size, 
                                              num_wind=num_wind, num_patch=num_patch,
                                              a_type=a_type, n_head=n_head, 
                                              kernel_size=kernel_size, stride=stride, padding=padding, 
                                              normalize_Q_K=normalize_Q_K, 
                                              cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias,
                                              att_dropout_p=att_dropout_p, 
                                              att_with_output_proj=att_with_output_proj)
        elif(att_mode=="global"):
            self.attn = SpatialGlobalAttention(C_in=C_in, C_out=C_out, H=self.H, W=self.W, 
                                               window_size=window_size, patch_size=patch_size, 
                                               num_wind=num_wind, num_patch=num_patch, 
                                               a_type=a_type, n_head=n_head, 
                                               kernel_size=kernel_size, stride=stride, padding=padding, 
                                               normalize_Q_K=normalize_Q_K, 
                                               cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias,
                                               att_dropout_p=att_dropout_p, 
                                               att_with_output_proj=att_with_output_proj,
                                               shuffle_in_window=shuffle_in_window)
        elif(att_mode=="vit"):
            self.attn = SpatialViTAttention(C_in=C_in, C_out=C_out, H=self.H, W=self.W, 
                                            window_size=window_size, num_wind=num_wind, 
                                            kernel_size=kernel_size, stride=stride, padding=padding, 
                                            normalize_Q_K=normalize_Q_K, 
                                            cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias,
                                            att_dropout_p=att_dropout_p, 
                                            att_with_output_proj=att_with_output_proj)
        else:
            raise NotImplementedError(f"Attention mode not implemented: {att_mode}")
                    
        self.stochastic_depth = torchvision.ops.StochasticDepth(p=self.dropout_p, mode="row")        

        self.with_mixer = with_mixer            
        if(self.with_mixer):
            if self.mixer_type == "conv" or att_mode=="temporal":
                mixer_cha = int(scale_ratio_in_mixer*C_out)
                
                self.mlp = nn.Sequential(
                    Conv2DExt(C_out, mixer_cha, kernel_size=mixer_kernel_size, stride=mixer_stride, padding=mixer_padding, bias=True),
                    NewGELU(),
                    Conv2DExt(mixer_cha, C_out, kernel_size=mixer_kernel_size, stride=mixer_stride, padding=mixer_padding, bias=True)
                )
            elif self.mixer_type == "lin":
                # apply mixer on every patch
                if att_mode == "local" or att_mode == "global":                    
                    D = C_out * self.attn.patch_size[0] * self.attn.patch_size[1]
                elif att_mode == "vit":
                    D = C_out * self.attn.window_size[0] * self.attn.window_size[1]
                    
                D_prime = int(scale_ratio_in_mixer*D)
                    
                self.mlp = nn.Sequential(
                    nn.Linear(D, D_prime, bias=True),
                    NewGELU(),
                    nn.Linear(D_prime, D, bias=True)
                )
            else:
                raise NotImplementedError(f"Mixer mode not implemented: {mixer_type}")    

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):

        x = self.input_proj(x) + self.stochastic_depth(self.attn(self.n1(x)))

        if(self.with_mixer):
            if self.mixer_type == "conv" or self.att_mode=="temporal":
                x = x + self.stochastic_depth(self.mlp(self.n2(x)))
            else:
                x = self.n2(x)
                x = self.attn.im2grid(x)
                *Dim, C, wh, ww = x.shape
                x = self.mlp(x.reshape((*Dim, -1)))
                x = self.stochastic_depth(self.attn.grid2im(x.reshape((*Dim, C, wh, ww))))
                
        return x

    def __str__(self):
        res = create_generic_class_str(self)
        return res

# -------------------------------------------------------------------------------------------------

class STCNNT_Parallel_Cell(nn.Module):
    """
    Parallel transformer cell
                   
    x -> Norm ----> attention --> + -> + -> logits
      |        |--> CNN mixer-----|    |
      |----------> input_proj ----> ---|
                    
    """
    def __init__(self, C_in, C_out=16, 
                 H=128, W=128, 
                 att_mode="temporal", a_type="conv",
                 mixer_type="conv",
                 window_size=None, patch_size=None, num_wind=[8, 8], num_patch=[4, 4], 
                 is_causal=False, n_head=8,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),stride_t=(2,2),
                 mixer_kernel_size=(5, 5), mixer_stride=(1, 1), mixer_padding=(2, 2),
                 normalize_Q_K=False, att_dropout_p=0.0, dropout_p=0.1, 
                 cosine_att=True, 
                 att_with_relative_postion_bias=True,
                 att_with_output_proj=True, 
                 scale_ratio_in_mixer=4.0, 
                 with_mixer=True, 
                 norm_mode="layer",
                 shuffle_in_window=True):
        """
        Complete transformer parallel cell
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.att_mode = att_mode
        self.a_type = a_type
        self.mixer_type = mixer_type
        self.window_size = window_size
        self.patch_size = patch_size
        self.is_causal = is_causal
        self.n_head = n_head
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.stride_t = stride_t
        
        self.mixer_kernel_size = mixer_kernel_size
        self.mixer_stride = mixer_stride
        self.mixer_padding = mixer_padding
        
        self.stride_t = stride_t
        self.normalize_Q_K = normalize_Q_K
        self.att_dropout_p = att_dropout_p
        self.dropout_p = dropout_p
        self.att_with_output_proj = att_with_output_proj
        self.with_mixer = with_mixer
        self.scale_ratio_in_mixer = scale_ratio_in_mixer
        self.norm_mode = norm_mode
        self.cosine_att = cosine_att
        self.att_with_relative_postion_bias = att_with_relative_postion_bias

        self.shuffle_in_window = shuffle_in_window

        if(norm_mode=="layer"):
            self.n1 = nn.LayerNorm([C_in, H, W])
        elif(norm_mode=="batch2d"):
            self.n1 = BatchNorm2DExt(C_in)
        elif(norm_mode=="instance2d"):
            self.n1 = InstanceNorm2DExt(C_in)
        elif(norm_mode=="batch3d"):
            self.n1 = BatchNorm3DExt(C_in)
        elif(norm_mode=="instance3d"):
            self.n1 = InstanceNorm3DExt(C_in)
        else:
            raise NotImplementedError(f"Norm mode not implemented: {norm_mode}")

        if C_in!=C_out:
            self.input_proj = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            self.input_proj = nn.Identity()

        if(att_mode=="temporal"):
            self.attn = TemporalCnnAttention(C_in=C_in, C_out=C_out, 
                                             H=H, W=W,
                                             is_causal=is_causal, n_head=n_head, 
                                             kernel_size=kernel_size, stride=stride, padding=padding, 
                                             stride_t=stride_t, 
                                             cosine_att=cosine_att, normalize_Q_K=normalize_Q_K, 
                                             att_dropout_p=att_dropout_p, 
                                             att_with_output_proj=att_with_output_proj)
        elif(att_mode=="local"):
            self.attn = SpatialLocalAttention(C_in=C_in, C_out=C_out, 
                                              H=H, W=W,
                                              window_size=window_size, patch_size=patch_size, 
                                              num_wind=num_wind, num_patch=num_patch, 
                                              a_type=a_type, n_head=n_head, 
                                              kernel_size=kernel_size, stride=stride, padding=padding, 
                                              normalize_Q_K=normalize_Q_K, 
                                              att_dropout_p=att_dropout_p, 
                                              cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias,
                                              att_with_output_proj=att_with_output_proj)
        elif(att_mode=="global"):
            self.attn = SpatialGlobalAttention(C_in=C_in, C_out=C_out, 
                                               H=H, W=W,
                                               window_size=window_size, patch_size=patch_size, 
                                               num_wind=num_wind, num_patch=num_patch, 
                                               a_type=a_type, n_head=n_head, 
                                               kernel_size=kernel_size, stride=stride, padding=padding, 
                                               normalize_Q_K=normalize_Q_K, 
                                               att_dropout_p=att_dropout_p, 
                                               cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias,
                                               att_with_output_proj=att_with_output_proj,
                                               shuffle_in_window=shuffle_in_window)
        elif(att_mode=="vit"):
            self.attn = SpatialViTAttention(C_in=C_in, C_out=C_out, 
                                            H=H, W=W,
                                            window_size=window_size, a_type=a_type, n_head=n_head, 
                                            kernel_size=kernel_size, stride=stride, padding=padding, 
                                            normalize_Q_K=normalize_Q_K, 
                                            att_dropout_p=att_dropout_p, 
                                            cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias,
                                            att_with_output_proj=att_with_output_proj)
        else:
            raise NotImplementedError(f"Attention mode not implemented: {att_mode}")

        self.stochastic_depth = torchvision.ops.StochasticDepth(p=self.dropout_p, mode="row")        
            
        self.with_mixer = with_mixer            
        if(self.with_mixer):
            if self.mixer_type == "conv" or att_mode=="temporal":        
                mixer_cha = int(scale_ratio_in_mixer*C_out)
                
                self.mlp = nn.Sequential(
                    Conv2DExt(C_in, mixer_cha, kernel_size=mixer_kernel_size, stride=mixer_stride, padding=mixer_padding, bias=True),
                    NewGELU(),
                    Conv2DExt(mixer_cha, C_out, kernel_size=mixer_kernel_size, stride=mixer_stride, padding=mixer_padding, bias=True)
                )
            elif self.mixer_type == "lin":
                # apply mixer on every patch
                if att_mode == "local" or att_mode == "global":                    
                    D = C_in * self.attn.patch_size[0] * self.attn.patch_size[1]                    
                elif att_mode == "vit":
                    D = C_in * self.attn.window_size[0] * self.attn.window_size[1]
                    
                D_out = int(D//C_in * C_out)
                
                D_prime = int(scale_ratio_in_mixer*D_out)
                    
                self.mlp = nn.Sequential(
                    nn.Linear(D, D_prime, bias=True),
                    NewGELU(),
                    nn.Linear(D_prime, D_out, bias=True)
                )
            else:
                raise NotImplementedError(f"Mixer mode not implemented: {mixer_type}") 
            
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):

        x_normed = self.n1(x)

        y = self.stochastic_depth(self.attn(x_normed))

        if(self.with_mixer):
            if self.mixer_type == "conv" or self.att_mode=="temporal":
                res_mixer = self.stochastic_depth(self.mlp(x_normed))
            else:
                res_mixer = self.attn.im2grid(x_normed)
                *Dim, C, wh, ww = res_mixer.shape
                res_mixer = self.mlp(res_mixer.reshape((*Dim, -1)))
                res_mixer = self.stochastic_depth(self.attn.grid2im(res_mixer.reshape((*Dim, self.C_out, wh, ww))))
            
            y += res_mixer        

        y += self.input_proj(x)

        return y

    def __str__(self):
        res = create_generic_class_str(self)
        return res

# -------------------------------------------------------------------------------------------------

def tests():
    # tests

    B, T, C, H, W = 2, 4, 3, 64, 64
    C_out = 8
    test_in = torch.rand(B,T,C,H,W)

    print("Begin Testing")

    att_types = [ "local", "global", "vit", "temporal"]
    norm_types = ["instance2d", "batch2d", "layer", "instance3d", "batch3d"]
    cosine_atts = ["True", "False"]
    att_with_relative_postion_biases = ["True", "False"]
    mixer_types = [ "conv", "lin"]
    
    for att_type in att_types:
        for norm_type in norm_types:
            for cosine_att in cosine_atts:
                for att_with_relative_postion_bias in att_with_relative_postion_biases:
                    for mixer_type in mixer_types:
                        print(norm_type, att_type, mixer_types, cosine_att, att_with_relative_postion_bias)
                        
                        CNNT_Cell = STCNNT_Cell(C_in=C, C_out=C_out, H=H, W=W, window_size=[H//8, W//8], patch_size=[H//16, W//16], 
                                                att_mode=att_type, mixer_type=mixer_type, norm_mode=norm_type,
                                                cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias)
                        test_out = CNNT_Cell(test_in)

                        Bo, To, Co, Ho, Wo = test_out.shape
                        assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed CNNT Cell")

    for att_type in att_types:
        for norm_type in norm_types:

            for cosine_att in cosine_atts:
                for att_with_relative_postion_bias in att_with_relative_postion_biases:
                    for mixer_type in mixer_types:
                        
                        print(norm_type, att_type, mixer_type, cosine_att, att_with_relative_postion_bias)
                
                        p_cell = STCNNT_Parallel_Cell(C_in=C, C_out=C_out, H=H, W=W, window_size=[H//8, W//8], patch_size=[H//16, W//16], 
                                                    att_mode=att_type, mixer_type=mixer_type, norm_mode=norm_type,
                                                    cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias)
                        test_out = p_cell(test_in)

                        Bo, To, Co, Ho, Wo = test_out.shape
                        assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed Parallel CNNT Cell")

    print("Passed all tests")

if __name__=="__main__":
    tests()
