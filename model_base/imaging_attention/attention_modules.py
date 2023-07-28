"""
Spatio-Temporal Convolutional Neural Net Transformer (STCNNT)

A novel structure that combines the ideas behind CNNs and Transformers.
STCNNT is able to utilize the spatial and temporal correlation 
while keeping the computations efficient.

Attends across complete temporal dimension and
across spatial dimension in restricted local and diluted global methods.

Provides implementation of following modules (in order of increasing complexity):
    - SpatialLocalAttention: Local windowed spatial attention
    - SpatialGlobalAttention: Global grided spatial attention
    - TemporalCnnAttention: Complete temporal attention

"""

import math
import sys
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

from pathlib import Path
Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import get_device, model_info, get_gpu_ram_usage, create_generic_class_str, start_timer, end_timer

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_ATTEN=True
except:
    FLASH_ATTEN=False

# -------------------------------------------------------------------------------------------------
# Extensions and helpers

def compute_conv_output_shape(h_w, kernel_size, stride, pad, dilation):
    """
    Utility function for computing output of convolutions given the setup
    @args:
        - h_w (int, int): 2-tuple of height, width of input
        - kernel_size, stride, pad (int, int): 2-tuple of conv parameters
        - dilation (int): dilation conv parameter
    @rets:
        - h, w (int, int): 2-tuple of height, width of image returned by the conv
    """
    h_0 = (h_w[0]+(2*pad[0])-(dilation*(kernel_size[0]-1))-1)
    w_0 = (h_w[1]+(2*pad[1])-(dilation*(kernel_size[1]-1))-1)

    h = torch.div( h_0, stride[0], rounding_mode="floor") + 1
    w = torch.div( w_0, stride[1], rounding_mode="floor") + 1

    return h, w

# class Conv2DExt(nn.Module):
#     # Extends torch 2D conv to support 5D inputs

#     def __init__(self, in_channels, out_channels, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, separable_conv=False):
#         super().__init__()
#         self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

#     def forward(self, input):
#         # requires input to have 5 dimensions
#         B, T, C, H, W = input.shape
#         y = self.conv2d(input.reshape((B*T, C, H, W)))
#         return y.reshape([B, T, *y.shape[1:]])

class Conv2DExt(nn.Module):
    # Extends torch 2D conv to support 5D inputs

    def __init__(self, in_channels, out_channels, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, separable_conv=False):
        super().__init__()
        self.separable_conv = separable_conv
        if separable_conv:
            self.convA = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=in_channels)
            self.convB = nn.Conv2d(in_channels, out_channels, kernel_size=[1,1], stride=[1,1], padding=[0,0], bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, input):
        # requires input to have 5 dimensions
        B, T, C, H, W = input.shape
        if self.separable_conv:
            y = self.convB(self.convA(input.reshape((B*T, C, H, W))))
        else:
            y = self.conv(input.reshape((B*T, C, H, W)))

        return y.reshape([B, T, *y.shape[1:]])

class Conv2DGridExt(nn.Module):
    # Extends torch 2D conv for grid attention with 7D inputs

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 7 dimensions
        B, T, Hg, Wg, Gh, Gw, C = input.shape
        input = input.permute(0,1,2,3,6,4,5)
        y = self.conv2d(input.reshape((-1, C, Gh, Gw)))
        y = y.reshape(B, T, Hg, Wg, *y.shape[-3:]) # B, T, Hg, Wg, C, Gh, Gw

        return y.permute(0,1,2,3,5,6,4)
  
class LinearGridExt(nn.Module):
    # Extends torch linear layer for grid attention with 7D inputs

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.linear = nn.Linear(*args,**kwargs)

    def forward(self, input):
        # requires input to have 7 dimensions
        *S, Gh, Gw, C = input.shape
        y = self.linear(input.reshape((-1, C*Gh*Gw)))
        y = y.reshape((*S, Gh, Gw, -1))

        return y

class Conv3DExt(nn.Module):
    # Extends troch 3D conv by permuting T and C

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.conv3d = nn.Conv3d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        return torch.permute(self.conv3d(torch.permute(input, (0, 2, 1, 3, 4))), (0, 2, 1, 3, 4))
    
class BatchNorm2DExt(nn.Module):
    # Extends BatchNorm2D to 5D inputs

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        B, T, C, H, W = input.shape
        norm_input = self.bn(input.reshape(B*T,C,H,W))
        return norm_input.reshape(input.shape)
    
class InstanceNorm2DExt(nn.Module):
    # Extends InstanceNorm2D to 5D inputs

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.inst = nn.InstanceNorm2d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        B, T, C, H, W = input.shape
        norm_input = self.inst(input.reshape(B*T,C,H,W))
        return norm_input.reshape(input.shape)
    
class BatchNorm3DExt(nn.Module):
    # Corrects BatchNorm3D, switching first and second dimension

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.bn = nn.BatchNorm3d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        norm_input = self.bn(input.permute(0,2,1,3,4))
        return norm_input.permute(0,2,1,3,4)
    
class InstanceNorm3DExt(nn.Module):
    # Corrects InstanceNorm3D, switching first and second dimension

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.inst = nn.InstanceNorm3d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        norm_input = self.inst(input.permute(0,2,1,3,4))
        return norm_input.permute(0,2,1,3,4)

class AvgPool2DExt(nn.Module):
    # Extends torch 2D averaging pooling to support 5D inputs

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.avg_pool_2d = nn.AvgPool2d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        B, T, C, H, W = input.shape
        y = self.avg_pool_2d(input.reshape((B*T, C, H, W)))
        return torch.reshape(y, [B, T, *y.shape[1:]])
    
def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: Tuple[int]
) -> torch.Tensor:
    """
    From pytorch source code
    """
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias

# -------------------------------------------------------------------------------------------------
class CnnAttentionBase(nn.Module):
    """
    Base class for cnn attention layers
    """
    def __init__(self, C_in, C_out=16, 
                    H=128, W=128,
                    n_head=8, 
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                    stride_qk = (1, 1),
                    separable_conv=False,
                    att_dropout_p=0.0, 
                    cosine_att=False, 
                    normalize_Q_K=False, 
                    att_with_relative_postion_bias=True,
                    att_with_output_proj=True,
                    with_timer=False):
        """
        Base class for the cnn attentions.

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H', W']
        Usually used with conv definition such that H',W' = H,W

        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - H, W (int): image height and width
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - stride_qk (int, int): stride to compute q and k
            - att_dropout_p (float): probability of dropout for the attention matrix
            - cosine_att (bool): whether to perform cosine attention; if True, normalize_Q_K will be ignored, as Q and K are already normalized
            - normalize_Q_K (bool): whether to add normalization for Q and K matrix
            - att_with_relative_postion_bias (bool): whether to add relative positional bias
            - att_with_output_proj (bool): whether to add output projection
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.n_head = n_head
        self.kernel_size = kernel_size, 
        self.stride = stride 
        self.stride_qk = stride_qk
        self.padding = padding
        self.separable_conv = separable_conv
        self.att_dropout_p = att_dropout_p
        self.cosine_att = cosine_att
        self.normalize_Q_K = normalize_Q_K
        self.att_with_relative_postion_bias = att_with_relative_postion_bias
        self.att_with_output_proj = att_with_output_proj
        self.with_timer = with_timer

        if att_with_output_proj:
            self.output_proj = Conv2DExt(C_out, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            self.output_proj = nn.Identity()
            
        if att_dropout_p>0:
            self.attn_drop = nn.Dropout(p=att_dropout_p)
        else:
            self.attn_drop = nn.Identity()

        self.has_flash_attention = FLASH_ATTEN

        self.flash_atten_type = torch.float32
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            if gpu_name.find("A100") >= 0 or gpu_name.find("H100") >= 0:
                self.flash_atten_type = torch.bfloat16

    def perform_flash_atten(self, k, q, v):
        softmax_scale = None
        if self.cosine_att:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            softmax_scale = 1
        elif self.normalize_Q_K:
            eps = torch.finfo(k.dtype).eps
            k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
            q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

        original_dtype = k.dtype
        if self.training:
            y = flash_attn_func(q.type(self.flash_atten_type), k.type(self.flash_atten_type), v.type(self.flash_atten_type), dropout_p=self.att_dropout_p, softmax_scale=softmax_scale, causal=False).type(original_dtype)
        else:
            y = flash_attn_func(q.type(self.flash_atten_type), k.type(self.flash_atten_type), v.type(self.flash_atten_type), dropout_p=0.0, softmax_scale=softmax_scale, causal=False).type(original_dtype)

        return y

    def define_relative_position_bias_table(self, num_win_h=100, num_win_w=100):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * num_win_h - 1) * (2 * num_win_w - 1), self.n_head)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def define_relative_position_index(self, num_win_h=100, num_win_w=100):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(num_win_h)
        coords_w = torch.arange(num_win_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += num_win_h - 1  # shift to start from 0
        relative_coords[:, :, 1] += num_win_w - 1
        relative_coords[:, :, 0] *= 2 * num_win_w - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self, num_win_h, num_win_w) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, (num_win_h, num_win_w)  # type: ignore[arg-type]
        )
        
    @property
    def device(self):
        return next(self.parameters()).device

    def set_and_check_wind(self):
        if self.num_wind is not None:
            self.window_size = [self.H//self.num_wind[0], self.W//self.num_wind[1]]
        else:
            self.num_wind = [self.H//self.window_size[0], self.W//self.window_size[1]]
            
        assert self.num_wind[0]*self.window_size[0] == self.H, \
            f"self.num_wind[0]*self.window_size[0] == self.H, num_wind {self.num_wind}, window_size {self.window_size}, H {self.H}"
            
        assert self.num_wind[1]*self.window_size[1] == self.W, \
            f"self.num_wind[1]*self.window_size[1] == self.W, num_wind {self.num_wind}, window_size {self.window_size}, W {self.W}"

    def set_and_check_patch(self):
        if self.num_patch is not None:
            self.patch_size = [self.window_size[0]//self.num_patch[0], self.window_size[1]//self.num_patch[1]]
        else:
            self.num_patch = [self.window_size[0]//self.patch_size[0], self.window_size[1]//self.patch_size[1]]
            
        assert (self.patch_size[0]*self.num_patch[0] == self.window_size[0]) and (self.patch_size[1]*self.num_patch[1] == self.window_size[1]), \
            f"self.patch_size*self.num_patch == self.window_size, patch_size {self.patch_size}, num_patch {self.num_patch}, window_size {self.window_size}"
            
    def validate_window_patch(self):
        assert self.window_size[0]*self.num_wind[0] == self.H, f"self.window_size[0]*self.num_wind[0] == self.H"
        assert self.window_size[1]*self.num_wind[1] == self.W, f"self.window_size[1]*self.num_wind[1] == self.W"
        assert self.patch_size[0]*self.num_patch[0] == self.window_size[0], f"self.patch_size[0]*self.num_patch[0] == self.window_size[0]"
        assert self.patch_size[0]*self.num_patch[1] == self.window_size[1], f"self.patch_size[1]*self.num_patch[1] == self.window_size[1]"
        
    def __str__(self):
        res = create_generic_class_str(self)
        return res
        
# -------------------------------------------------------------------------------------------------

def tests():
    # tests

    print("Begin Testing")

    print("Passed all tests")

if __name__=="__main__":
    tests()
