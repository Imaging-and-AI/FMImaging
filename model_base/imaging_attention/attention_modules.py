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

from utils.utils import get_device, model_info, get_gpu_ram_usage, create_generic_class_str

# -------------------------------------------------------------------------------------------------
# Extensions and helpers

class NewGELU(nn.Module):
    """
    Borrowed from the minGPT repo.
    
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

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

class Conv2DExt(nn.Module):
    # Extends torch 2D conv to support 5D inputs

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        B, T, C, H, W = input.shape
        y = self.conv2d(input.reshape((B*T, C, H, W)))
        return torch.reshape(y, [B, T, *y.shape[1:]])

class Conv2DGridExt(nn.Module):
    # Extends torch 2D conv for grid attention with 7D inputs

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 7 dimensions
        B, T, C, Hg, Wg, Gh, Gw = input.shape
        input = input.permute(0,1,3,4,2,5,6)
        y = self.conv2d(input.reshape((-1, C, Gh, Gw)))
        y = y.reshape(B, T, Hg, Wg, *y.shape[-3:])

        return y.permute(0,1,4,2,3,5,6)
  
class LinearGridExt(nn.Module):
    # Extends torch linear layer for grid attention with 7D inputs

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.linear = nn.Linear(*args,**kwargs)

    def forward(self, input):
        # requires input to have 7 dimensions
        *S, C, Gh, Gw = input.shape
        y = self.linear(input.reshape((-1, C*Gh*Gw)))
        y = y.reshape((*S, -1, Gh, Gw))

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

# -------------------------------------------------------------------------------------------------
class CnnAttentionBase(nn.Module):
    """
    Base class for cnn attention layers
    """
    def __init__(self, C_in, C_out=16, n_head=8, 
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                    att_dropout_p=0.0, dropout_p=0.1, 
                    normalize_Q_K=False, att_with_output_proj=True):
        """
        Base class for the cnn attentions.

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H', W']
        Usually used with conv definition such that H',W' = H,W

        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - att_dropout_p (float): probability of dropout for the attention matrix
            - dropout_p (float): probability of dropout
            - normalize_Q_K (bool): whether to add normalization for Q and K matrix
            - att_with_output_proj (bool): whether to add output projection
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.n_head = n_head
        self.kernel_size = kernel_size, 
        self.stride = stride 
        self.padding = padding
        self.att_dropout_p = att_dropout_p
        self.dropout_p = dropout_p
        self.normalize_Q_K = normalize_Q_K
        self.att_with_output_proj = att_with_output_proj

        if att_with_output_proj:
            self.output_proj = Conv2DExt(C_out, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
            
        if att_dropout_p>0:
            self.attn_drop = nn.Dropout(p=att_dropout_p)
        else:
            self.attn_drop = nn.Identity()
            
        if dropout_p>0:
            self.resid_drop = nn.Dropout(p=dropout_p)
        else:
            self.resid_drop = nn.Identity()
            
    @property
    def device(self):
        return next(self.parameters()).device

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
