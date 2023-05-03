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

from utils.utils import create_generic_class_str

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
                    a_type="conv", cell_type="sequential",
                    window_size=64, patch_size=16, 
                    is_causal=False, n_head=8,
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), stride_t=(2,2), 
                    normalize_Q_K=False, att_dropout_p=0.0, dropout_p=0.1, 
                    att_with_output_proj=True, scale_ratio_in_mixer=4.0, 
                    norm_mode="layer",\
                    interpolate="none", interp_align_c=False):
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
            - stride_t (int, int): special stride for temporal attention k,q matrices
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
        self.cell_type = cell_type
        self.window_size = window_size
        self.patch_size = patch_size
        self.is_causal = is_causal
        self.n_head = n_head
        self.kernel_size = kernel_size
        self.stride = stride
        self.stride_t = stride_t
        self.padding = padding
        self.normalize_Q_K = normalize_Q_K
        self.att_dropout_p = att_dropout_p
        self.dropout_p = dropout_p
        self.att_with_output_proj = att_with_output_proj
        self.scale_ratio_in_mixer = scale_ratio_in_mixer
        self.norm_mode = norm_mode
        self.interpolate = interpolate
        self.interp_align_c = interp_align_c
                    
        self.cells = []

        for i in range(len(att_types)//2):

            att_type = att_types[2*i]
            mixer = att_types[2*i+1]

            assert att_type=='L' or att_type=='G' or att_type=='T' or att_type=='V', \
                f"att_type not implemented: {att_type} at index {2*i} in {att_types}"
            assert mixer=='0' or mixer=='1', \
                f"mixer not implemented: {mixer} at index {2*i+1} in {att_types}"

            if att_type=='L':
                att_type = "local"
            elif att_type=='G':
                att_type = "global"
            elif att_type=='T':
                att_type = "temporal"
            elif att_type=='V':
                att_type = "vit"
            else:
                raise f"Incorrect att_type: {att_type}"

            C = C_in if i==0 else C_out

            if self.cell_type.lower() == "sequential":
                self.cells.append((f"{att_type}_{i}", STCNNT_Cell(C_in=C, C_out=C_out, H=H, W=W, 
                                                                  att_mode=att_type, a_type=a_type,
                                                                  window_size=window_size, patch_size=patch_size, 
                                                                  is_causal=is_causal, n_head=n_head,
                                                                  kernel_size=kernel_size, stride=stride, padding=padding, stride_t=stride_t,
                                                                  normalize_Q_K=normalize_Q_K, att_dropout_p=att_dropout_p, dropout_p=dropout_p, 
                                                                  att_with_output_proj=att_with_output_proj, scale_ratio_in_mixer=scale_ratio_in_mixer, 
                                                                  with_mixer=(mixer=='1'), norm_mode=norm_mode)))
            else:
                self.cells.append((f"{att_type}_{i}", STCNNT_Parallel_Cell(C_in=C, C_out=C_out, H=H, W=W, 
                                                                           att_mode=att_type, a_type=a_type,
                                                                           window_size=window_size, patch_size=patch_size, is_causal=is_causal, n_head=n_head,
                                                                           kernel_size=kernel_size, stride=stride, padding=padding, stride_t=stride_t,
                                                                           normalize_Q_K=normalize_Q_K, att_dropout_p=att_dropout_p, dropout_p=dropout_p, 
                                                                           att_with_output_proj=att_with_output_proj, scale_ratio_in_mixer=scale_ratio_in_mixer, 
                                                                           with_mixer=(mixer=='1'), norm_mode=norm_mode)))

                
        self.make_block()

        self.interpolate = interpolate
        self.interp_align_c = interp_align_c

    @property
    def device(self):
        return next(self.parameters()).device

    def make_block(self):
        self.block = nn.Sequential(OrderedDict(self.cells))

    def forward(self, x):

        x = self.block(x)

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

    att_typess = ["L1", "G1", "T1", "L0", "L1", "G0", "G1", "T1", "T0", "V1", "V0", "L0G1T0V1", "T1L0G1V0"]

    for att_types in att_typess:
        CNNT_Block = STCNNT_Block(att_types=att_types, C_in=C, C_out=C_out, window_size=H//8, patch_size=H//16)
        test_out, _ = CNNT_Block(test_in)

        Bo, To, Co, Ho, Wo = test_out.shape
        assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed CNNT Block att_types and mixers")

    interpolates = ["up", "down", "none"]
    interp_align_cs = [True, False]

    for interpolate in interpolates:
        for interp_align_c in interp_align_cs:
            CNNT_Block = STCNNT_Block(att_types=att_types, C_in=C, C_out=C_out, window_size=H//8, patch_size=H//16, 
                                   interpolate=interpolate, interp_align_c=interp_align_c)
            test_out_1, test_out_2 = CNNT_Block(test_in)

            Bo, To, Co, Ho, Wo = test_out_1.shape if interpolate=="none" else test_out_2.shape
            factor = 2 if interpolate=="up" else 0.5 if interpolate=="down" else 1
            assert B==Bo and T==To and Co==C_out and (H*factor)==Ho and (W*factor)==Wo

    print("Passed CNNT Block interpolation")

    print("Passed all tests")

if __name__=="__main__":
    tests()
