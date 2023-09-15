"""
Spatio-Temporal Convolutional Neural Net Transformer (STCNNT)

Provide implementation of STCNNT_Block: A stack of STCNNT_Cell.

A block is a set of cells. Block structure is configurable by the 'block string'. 
For example, 'L1T1G1' means to configure with a local attention (L1) with mixer (1 after 'L'), followed by a temporal attention with mixer (T1)
and a global attention with mixer (G1).

"""

import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from pathlib import Path
Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import create_generic_class_str

from imaging_attention import *
from cells import *

__all__ = ['STCNNT_Block']

# -------------------------------------------------------------------------------------------------
# A block of multiple transformer cells stacked on top of each other
                   
class STCNNT_Block(nn.Module):
    """
    A stack of CNNT cells
    The first cell expands the channel dimension.
    Can use Conv2D mixer with all cells, last cell, or none at all.
    """
    def __init__(self, att_types, C_in, C_out=16, H=64, W=64,
                    a_type="conv", mixer_type="conv", cell_type="sequential",
                    window_size=None, patch_size=None, num_wind=[8, 8], num_patch=[4, 4], 
                    is_causal=False, n_head=8,
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                    stride_s=(1,1), 
                    stride_t=(2,2),
                    activation_func="prelu",
                    separable_conv = False,
                    mixer_kernel_size=(5, 5), mixer_stride=(1, 1), mixer_padding=(2, 2),
                    cosine_att=True,  
                    normalize_Q_K=False, 
                    att_dropout_p=0.0, 
                    dropout_p=0.1, 
                    att_with_relative_postion_bias=True,
                    att_with_output_proj=True, 
                    scale_ratio_in_mixer=4.0, 
                    norm_mode="layer",
                    interpolate="none", 
                    interp_align_c=False,
                    block_dense_connection=True,
                    shuffle_in_window=False,
                    use_einsum=False,
                    temporal_flash_attention=False):
        """
        Transformer block

        @args:
            - att_types (str): order of attention types and their following mlps
                format is XYXY...
                - X is "L", "G" or "T" for attention type
                - Y is "0" or "1" for with or without mixer
                - requires len(att_types) to be even
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - H (int): expected height of the input
            - W (int): expected width of the input
            - a_type ("conv", "lin"): type of attention in spatial heads
            - cell_type ("sequential" or "parallel"): type of cells
            - window_size (int): size of window for local and global att
            - patch_size (int): size of patch for local and global att
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - stride_s (int, int): stride for spatial attention k,q matrices
            - stride_t (int, int): stride for temporal attention k,q matrices
            - normalize_Q_K (bool): whether to use layernorm to normalize Q and K, as in 22B ViT paper
            - att_dropout_p (float): probability of dropout for attention coefficients
            - dropout (float): probability of dropout
            - att_with_output_proj (bool): whether to add output projection in the attention layer
            - scale_ratio_in_mixer (float): channel scaling ratio in the mixer
            - norm_mode ("layer", "batch", "instance"):
                layer - norm along C, H, W; batch - norm along B*T; or instance
            - interpolate ("none", "up", "down"):
                whether to interpolate and scale the image up or down by 2
            - interp_align_c (bool):
                whether to align corner or not when interpolating
            - block_dense_connection (bool): whether to add dense connection between cells
        """
        super().__init__()

        assert (len(att_types)>=1), f"At least one attention module is required to build the model"
        assert not (len(att_types)%2), f"require attention and mixer info for each cell"

        assert interpolate=="none" or interpolate=="up" or interpolate=="down", \
            f"Interpolate not implemented: {interpolate}"

        self.att_types = att_types
        self.C_in = C_in
        self.C_out =C_out
        self.H = H
        self.W = W
        self.a_type = a_type
        self.mixer_type = mixer_type
        self.cell_type = cell_type
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_wind = num_wind
        self.num_patch = num_patch
        self.is_causal = is_causal
        self.n_head = n_head

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.activation_func = activation_func

        self.stride_s = stride_s
        self.stride_t = stride_t

        self.separable_conv = separable_conv,

        self.mixer_kernel_size = mixer_kernel_size
        self.mixer_stride = mixer_stride
        self.mixer_padding = mixer_padding

        self.normalize_Q_K = normalize_Q_K
        self.cosine_att = cosine_att
        self.att_with_relative_postion_bias = att_with_relative_postion_bias
        self.att_dropout_p = att_dropout_p
        self.dropout_p = dropout_p
        self.att_with_output_proj = att_with_output_proj
        self.scale_ratio_in_mixer = scale_ratio_in_mixer
        self.norm_mode = norm_mode
        self.interpolate = interpolate
        self.interp_align_c = interp_align_c
        self.block_dense_connection = block_dense_connection

        self.shuffle_in_window = shuffle_in_window

        self.use_einsum = use_einsum
        self.temporal_flash_attention = temporal_flash_attention

        self.cells = []

        for i in range(len(att_types)//2):

            att_type = att_types[2*i]
            mixer = att_types[2*i+1]

            assert att_type=='L' or att_type=='G' or att_type=='T' or att_type=='V' or att_type=='C', \
                f"att_type not implemented: {att_type} at index {2*i} in {att_types}"
            assert mixer=='0' or mixer=='1' or mixer=='2' or mixer=='3', \
                f"mixer not implemented: {mixer} at index {2*i+1} in {att_types}"
            assert not att_type=='C' or mixer=='2' or mixer=='3', \
                f"mixer: {mixer} can not be used with att_type=='C'"

            if att_type=='L':
                att_type = "local"
            elif att_type=='G':
                att_type = "global"
            elif att_type=='T':
                att_type = "temporal"
            elif att_type=='V':
                att_type = "vit"
            elif att_type=='C' and mixer=='2':
                att_type = "conv2d"
            elif att_type=='C' and mixer=='3':
                att_type = "conv3d"
            else:
                raise f"Incorrect att_type: {att_type}, mixer: {mixer}"

            C = C_in if i==0 else C_out

            if self.cell_type.lower() == "sequential":
                self.cells.append((f"cell_{i}", STCNNT_Cell(C_in=C, C_out=C_out, H=H, W=W, 
                                                                  att_mode=att_type, a_type=a_type, mixer_type=mixer_type,
                                                                  window_size=window_size, patch_size=patch_size, 
                                                                  num_wind=num_wind, num_patch=num_patch, 
                                                                  is_causal=is_causal, n_head=n_head,
                                                                  kernel_size=kernel_size, stride=stride, padding=padding, stride_s=stride_s, stride_t=stride_t,
                                                                  activation_func=activation_func,
                                                                  separable_conv=self.separable_conv,
                                                                  mixer_kernel_size=mixer_kernel_size, mixer_stride=mixer_stride, mixer_padding=mixer_padding,
                                                                  normalize_Q_K=normalize_Q_K, att_dropout_p=att_dropout_p, dropout_p=dropout_p,
                                                                  cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias, 
                                                                  att_with_output_proj=att_with_output_proj, scale_ratio_in_mixer=scale_ratio_in_mixer, 
                                                                  with_mixer=(mixer=='1'), norm_mode=norm_mode, shuffle_in_window=shuffle_in_window, 
                                                                  use_einsum = self.use_einsum, temporal_flash_attention = self.temporal_flash_attention)))
            else:
                self.cells.append((f"cell_{i}", STCNNT_Parallel_Cell(C_in=C, C_out=C_out, H=H, W=W, 
                                                                           att_mode=att_type, a_type=a_type, mixer_type=mixer_type, 
                                                                           window_size=window_size, patch_size=patch_size, 
                                                                           num_wind=num_wind, num_patch=num_patch,
                                                                           is_causal=is_causal, n_head=n_head,
                                                                           kernel_size=kernel_size, stride=stride, padding=padding, stride_s=stride_s, stride_t=stride_t,
                                                                           activation_func=activation_func,
                                                                           separable_conv=self.separable_conv,
                                                                           mixer_kernel_size=mixer_kernel_size, mixer_stride=mixer_stride, mixer_padding=mixer_padding,
                                                                           normalize_Q_K=normalize_Q_K, att_dropout_p=att_dropout_p, dropout_p=dropout_p, 
                                                                           cosine_att=cosine_att, att_with_relative_postion_bias=att_with_relative_postion_bias, 
                                                                           att_with_output_proj=att_with_output_proj, scale_ratio_in_mixer=scale_ratio_in_mixer,
                                                                           with_mixer=(mixer=='1'), norm_mode=norm_mode, shuffle_in_window=shuffle_in_window,
                                                                           use_einsum = self.use_einsum, temporal_flash_attention = self.temporal_flash_attention)))

        self.make_block()

        self.interpolate = interpolate
        self.interp_align_c = interp_align_c

    @property
    def device(self):
        return next(self.parameters()).device

    def make_block(self):
        self.block = nn.ModuleDict(OrderedDict(self.cells))

    def forward(self, x):

        num_cells = len(self.block)

        if self.block_dense_connection:
            block_res = []

            for c in range(num_cells):
                if c ==0:
                    block_res.append(self.block[f"cell_{c}"](x))
                else:
                    input = 0
                    for k in block_res:
                        input = input + k
                    block_res.append(self.block[f"cell_{c}"](input))

            x = block_res[-1]
        else:
            for c in range(num_cells):
                x = self.block[f"cell_{c}"](x)

        B, T, C, H, W = x.shape
        interp = None

        if self.interpolate=="down":
            interp = F.interpolate(x, scale_factor=(1.0, 0.5, 0.5), mode="trilinear", align_corners=self.interp_align_c, recompute_scale_factor=False)
            interp = interp.view(B, T, C, torch.div(H, 2, rounding_mode="floor"), torch.div(W, 2, rounding_mode="floor"))

        elif self.interpolate=="up":
            interp = F.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode="trilinear", align_corners=self.interp_align_c, recompute_scale_factor=False)
            interp = interp.view(B, T, C, H*2, W*2)

        else: # self.interpolate=="none"
            pass

        # Returns both: "x" without interpolation and "interp" that is x interpolated
        return x, interp

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

    att_typess = ["C2", "C3", "L1C2T1C3G1", "L1", "G1", "T1", "L0", "L1", "G0", "G1", "T1", "T0", "V1", "V0", "L0G1T0V1", "T1L0G1V0"]

    for att_types in att_typess:
        CNNT_Block = STCNNT_Block(H=H, W=W, att_types=att_types, C_in=C, C_out=C_out, window_size=[H//8, W//8], patch_size=[H//16, W//16], separable_conv=True)
        test_out, _ = CNNT_Block(test_in)

        Bo, To, Co, Ho, Wo = test_out.shape
        assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed CNNT Block att_types and mixers")

    interpolates = ["up", "down", "none"]
    interp_align_cs = [True, False]

    for interpolate in interpolates:
        for interp_align_c in interp_align_cs:
            CNNT_Block = STCNNT_Block(H=H, W=W, att_types=att_types, C_in=C, C_out=C_out, window_size=[H//8, W//8], patch_size=[H//16, W//16], stride_s=(2,2), separable_conv=False,
                                   interpolate=interpolate, interp_align_c=interp_align_c)
            test_out_1, test_out_2 = CNNT_Block(test_in)

            Bo, To, Co, Ho, Wo = test_out_1.shape if interpolate=="none" else test_out_2.shape
            factor = 2 if interpolate=="up" else 0.5 if interpolate=="down" else 1
            assert B==Bo and T==To and Co==C_out and (H*factor)==Ho and (W*factor)==Wo

    print("Passed CNNT Block interpolation")

    print("Passed all tests")

if __name__=="__main__":
    tests()
