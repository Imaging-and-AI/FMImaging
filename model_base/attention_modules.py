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
    - CnnTransformer: A CNNT cell that wraps above attention with norms and mixers
    - CNNTBlock: A stack of CnnTransformer cells

"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

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
        B, T, C, Hg, Wg, Gh, Gw = input.shape
        input = input.permute(0,1,3,4,2,5,6)
        y = self.linear(input.reshape((-1, C*Gh*Gw)))
        y = y.reshape(B, T, Hg, Wg, -1, Gh, Gw)

        return y.permute(0,1,4,2,3,5,6)

class Conv3DExt(nn.Module):
    # Extends troch 3D conv by permuting T and C

    def __init__(self,*args,**kwargs):
        super().__init__()
        self.conv3d = nn.Conv3d(*args,**kwargs)

    def forward(self, input):
        # requires input to have 5 dimensions
        y = self.conv3d(torch.permute(input, (0, 2, 1, 3, 4)))
        return torch.permute(y, (0, 2, 1, 3, 4))
    
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
# The CNN transformers to process the [B, T, C, H, W], a series of images
# Defines a class for local spatial, global spatial, and temporal attentions

class SpatialLocalAttention(nn.Module):
    """
    Multi-head cnn attention model for local spatial attention
    """
    def __init__(self, C_in, C_out=16, wind_size=8, a_type="conv", n_head=8,\
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_p=0.1):
        """
        Defines the layer for a cnn self-attention on spatial dimension with local windows

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H', W']
        Usually used with conv definition such that H',W' = H,W

        Breaks the input into windows of given size and each window attends to itself locally

        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - wind_size (int): window size for local attention
            - a_type ("conv", "lin"): defines what type of attention to use
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - dropout (float): probability of dropout
        """
        super().__init__()

        self.wind_size = wind_size
        self.C_in = C_in
        self.C_out = C_out
        self.n_head = n_head

        assert self.C_out % self.n_head == 0, \
            f"Number of output channles {self.C_out} should be divisible by number of heads {self.n_head}"

        if a_type=="conv":
            # key, query, value projections convolution
            # Wk, Wq, Wv
            self.key = Conv2DGridExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
            self.query = Conv2DGridExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
            self.value = Conv2DGridExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        elif a_type=="lin":
            # linear projections
            self.key = LinearGridExt(C_in*wind_size*wind_size, C_out*wind_size*wind_size, bias=True)
            self.query = LinearGridExt(C_in*wind_size*wind_size, C_out*wind_size*wind_size, bias=True)
            self.value = LinearGridExt(C_in*wind_size*wind_size, C_out*wind_size*wind_size, bias=True)
        else:
            raise NotImplementedError(f"Attention type not implemented: {a_type}")

        self.output_proj = Conv2DExt(C_out, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.attn_drop = nn.Dropout(dropout_p)
        self.resid_drop = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        @args:
            x ([B, T, C, H, W]): Input of a batch of time series

        @rets:
            y ([B, T, C_out, H', W']): logits
        """
        B, T, C, H, W = x.size()
        Ws = self.wind_size

        assert C == self.C_in, f"Input channel {C} does not match expected input channel {self.C_in}"
        assert H % Ws == 0, f"Height {H} should be divisible by window size {Ws}"
        assert W % Ws == 0, f"Width {W} should be divisible by window size {Ws}"

        Hg = torch.div(H, Ws, rounding_mode="floor")
        Wg = torch.div(W, Ws, rounding_mode="floor")

        nh = self.n_head
        hc = torch.div(self.C_out, nh, rounding_mode="floor")

        x_grid = self.im2grid(x)

        # apply the key, query and value matrix
        k = self.key(x_grid).reshape(B, T, self.n_head, hc, Hg*Wg, Ws, Ws).transpose(3, 4)
        q = self.query(x_grid).reshape(B, T, self.n_head, hc, Hg*Wg, Ws, Ws).transpose(3, 4)
        v = self.value(x_grid).reshape(B, T, self.n_head, hc, Hg*Wg, Ws, Ws).transpose(3, 4)

        # k, q, v are [B, T, nh, Hg * Wg, hc, Ws, Ws]
        B, T, nh, HWg, hc, Ws, Ws = k.shape

        # Compute attention matrix, use the matrix broadcasing 
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        # (B, T, nh, HWg, hc*Ws*Ws) x (B, T, nh, hc*Ws*Ws, HWg) -> (B, T, nh, HWg, HWg)
        att = (q.reshape(B,T,nh,HWg,hc*Ws*Ws) @ k.reshape(B,T,nh,HWg,hc*Ws*Ws).transpose(-2, -1))\
                * torch.tensor(1.0 / math.sqrt(hc*Ws*Ws))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # (B, T, nh, HWg, HWg) * (B, T, nh, HWg, hc*Gs*Gs)
        y = att @ v.reshape(B, T, nh, HWg, hc*Ws*Ws)
        y = y.reshape(B,T,nh,HWg,hc,Ws,Ws).transpose(3, 4).reshape(B, T, nh*hc, Hg, Wg, Ws, Ws)

        y = self.grid2im(y, H, W)
        y = y + self.resid_drop(self.output_proj(y))

        return y

    def im2grid(self, x):
        """
        Reshape the input into windows of local areas
        """
        b, t, c, h, w = x.shape
        Ws = self.wind_size

        wind_view = x.reshape(b, t, c, h//Ws, Ws, w//Ws, Ws)
        wind_view = wind_view.permute(0, 1, 2, 3, 5, 4, 6)
        wind_view = wind_view.reshape(b, t, c, h//Ws, w//Ws, Ws, Ws)

        return wind_view

    def grid2im(self, x, h, w):
        """
        Reshape the windows back into the complete image
        """
        b, t, c, _, _, _, _ = x.shape

        im_view = x.permute(0, 1, 2, 3, 5, 4, 6)
        im_view = im_view.reshape(b, t, c, h, w)

        return im_view

class SpatialGlobalAttention(nn.Module):
    """
    Multi-head cnn attention model for global spatial attention
    """
    def __init__(self, C_in, C_out=16, grid_size=8, a_type="conv", n_head=8,\
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_p=0.1):
        """
        Defines the layer for a cnn self-attention on spatial dimension with global grid

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H', W']
        Usually used with conv definition such that H',W' = H,W

        Breaks the input into grid of given size made up of dilated original image
        each grid attends to itself

        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - grid_size (int): grid size for global attention
            - a_type ("conv", "lin"): defines what type of attention to use
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - dropout (float): probability of dropout
        """
        super().__init__()

        self.grid_size = grid_size
        self.C_in = C_in
        self.C_out = C_out
        self.n_head = n_head

        assert self.C_out % self.n_head == 0, \
            f"Number of output channles {self.C_out} should be divisible by number of heads {self.n_head}"

        if a_type=="conv":
            # key, query, value projections convolution
            # Wk, Wq, Wv
            self.key = Conv2DGridExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
            self.query = Conv2DGridExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
            self.value = Conv2DGridExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        elif a_type=="lin":
            # linear projections
            self.key = LinearGridExt(C_in*grid_size*grid_size, C_out*grid_size*grid_size, bias=True)
            self.query = LinearGridExt(C_in*grid_size*grid_size, C_out*grid_size*grid_size, bias=True)
            self.value = LinearGridExt(C_in*grid_size*grid_size, C_out*grid_size*grid_size, bias=True)
        else:
            raise NotImplementedError(f"Attention type not implemented: {a_type}")

        self.output_proj = Conv2DExt(C_out, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.attn_drop = nn.Dropout(dropout_p)
        self.resid_drop = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        @args:
            x ([B, T, C, H, W]): Input of a batch of time series

        @rets:
            y ([B, T, C_out, H', W']): logits
        """
        B, T, C, H, W = x.size()
        Gs = self.grid_size

        assert C == self.C_in, f"Input channel {C} does not match expected input channel {self.C_in}"
        assert H % Gs == 0, f"Height {H} should be divisible by grid size {Gs}"
        assert W % Gs == 0, f"Width {W} should be divisible by grid size {Gs}"

        Hg = torch.div(H, Gs, rounding_mode="floor")
        Wg = torch.div(W, Gs, rounding_mode="floor")

        nh = self.n_head
        hc = torch.div(self.C_out, nh, rounding_mode="floor")

        x_grid = self.im2grid(x)

        # apply the key, query and value matrix
        k = self.key(x_grid).reshape(B, T, self.n_head, hc, Hg*Wg, Gs, Gs).transpose(3, 4)
        q = self.query(x_grid).reshape(B, T, self.n_head, hc, Hg*Wg, Gs, Gs).transpose(3, 4)
        v = self.value(x_grid).reshape(B, T, self.n_head, hc, Hg*Wg, Gs, Gs).transpose(3, 4)

        # k, q, v are [B, T, nh, Hg * Wg, hc, Gs, Gs]

        B, T, nh, HWg, hc, Gs, Gs = k.shape

        # Compute attention matrix, use the matrix broadcasing 
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        # (B, T, nh, HWg, hc*Gs*Gs) x (B, T, nh, hc*Gs*Gs, HWg) -> (B, T, nh, HWg, HWg)
        att = (q.reshape(B,T,nh,HWg,hc*Gs*Gs) @ k.reshape(B,T,nh,HWg,hc*Gs*Gs).transpose(-2, -1))\
                * torch.tensor(1.0 / math.sqrt(hc*Gs*Gs))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # (B, T, nh, HWg, HWg) * (B, T, nh, HWg, hc*Gs*Gs)
        y = att @ v.reshape(B, T, nh, HWg, hc*Gs*Gs)
        y = y.reshape(B,T,nh,HWg,hc,Gs,Gs).transpose(3, 4).reshape(B, T, nh*hc, Hg, Wg, Gs, Gs)

        y = self.grid2im(y, H, W)
        y = y + self.resid_drop(self.output_proj(y))

        return y

    def im2grid(self, x):
        """
        Reshape the input into sparse global grid of fixed size
        """
        b, t, c, h, w = x.shape
        gs = self.grid_size

        grid_view = x.reshape(b, t, c, gs, h//gs, gs, w//gs)
        grid_view = grid_view.permute(0, 1, 2, 4, 6, 3, 5)
        grid_view = grid_view.reshape(b, t, c, h//gs, w//gs, gs, gs)
        
        return grid_view

    def grid2im(self, x, h, w):
        """
        Reshape the sparse global grid back into complete image
        """
        b, t, c, _, _, _, _ = x.shape

        im_view = x.permute(0, 1, 2, 5, 3, 6, 4)
        im_view = im_view.reshape(b, t, c, h, w)

        return im_view

class TemporalCnnAttention(nn.Module):
    """
    Multi-head cnn attention model for complete temporal attention
    """
    def __init__(self, C_in, C_out=16, is_causal=False, n_head=8, \
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_p=0.1):
        """
        Defines the layer for a cnn self-attention on temporal axis

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H', W']
        Usually used with conv definition such that H',W' = H,W

        Calculates attention using all the time points

        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - dropout (float): probability of dropout
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.is_causal = is_causal
        self.n_head = n_head

        assert self.C_out % self.n_head == 0, \
            f"Number of output channles {self.C_out} should be divisible by number of heads {self.n_head}"

        # key, query, value projections convolution
        # Wk, Wq, Wv
        self.key = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.query = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.value = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

        self.output_proj = Conv2DExt(C_out, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.attn_drop = nn.Dropout(dropout_p)
        self.resid_drop = nn.Dropout(dropout_p)

        self.register_buffer("mask", torch.tril(torch.ones(1000, 1000)).view(1, 1, 1000, 1000))

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
        _,_,_,H_prime,W_prime = k.shape
        k = k.view(B, T, self.n_head, torch.div(self.C_out, self.n_head, rounding_mode="floor"), H_prime, W_prime).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, torch.div(self.C_out, self.n_head, rounding_mode="floor"), H_prime, W_prime).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, torch.div(self.C_out, self.n_head, rounding_mode="floor"), H_prime, W_prime).transpose(1, 2)

        # k, q, v are [B, nh, T, hc, H', W']

        B, nh, T, hc, H_prime, W_prime = k.shape

        # Compute attention matrix, use the matrix broadcasting 
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        # (B, nh, T, hc, H', W') x (B, nh, hc, H', W', T) -> (B, nh, T, T)
        att = (q.view(B, nh, T, hc*H_prime*W_prime) @ k.view(B, nh, T, hc*H_prime*W_prime).transpose(-2, -1))\
                * torch.tensor(1.0 / math.sqrt(hc*H_prime*W_prime))

        # if causality is needed, apply the mask
        if(self.is_causal):
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # (B, nh, T, T) * (B, nh, T, hc, H', W')
        y = att @ v.view(B, nh, T, hc*H_prime*W_prime)
        y = y.transpose(1, 2).contiguous().view(B, T, self.C_out, H_prime, W_prime)
        y = y + self.resid_drop(self.output_proj(y))

        return y

# -------------------------------------------------------------------------------------------------
# Complete transformer cell

class STCNNT_Cell(nn.Module):
    """
    CNN Transformer Cell with any attention type

    The Pre-LayerNorm implementation is used here:

    x-> LayerNorm -> attention -> + -> LayerNorm -> CNN mixer -> + -> logits
    |-----------------------------| |----------------------------|
    """
    def __init__(self, C_in, C_out=16, H=64, W=64, att_mode="temporal", a_type="conv",\
                    window_size=8, is_causal=False, n_head=8,\
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),\
                    dropout_p=0.1, with_mixer=True, norm_mode="layer"):
        """
        Complete transformer cell

        @args:
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - H (int): expected height of the input
            - W (int): expected width of the input
            - att_mode ("local", "global", "temporal"):
                different methods of attention mechanism
            - a_type ("conv", "lin"): type of attention in spatial heads
            - window_size (int): size of window for local and global att
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - dropout (float): probability of dropout
            - with_mixer (bool): whether to add a conv2D mixer after attention
            - norm_mode ("layer", "batch2d", "instance2d", "batch3d", "instance3d"):
                - layer: each C,H,W
                - batch2d: along B*T
                - instance2d: each H,W
                - batch3d: along B
                - instance3d: each T,H,W
        """
        super().__init__()

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
            self.attn = TemporalCnnAttention(C_in=C_in, C_out=C_out, is_causal=is_causal, n_head=n_head, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p)
        elif(att_mode=="local"):
            self.attn = SpatialLocalAttention(C_in=C_in, C_out=C_out, wind_size=window_size, a_type=a_type, n_head=n_head, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p)
        elif(att_mode=="global"):
            self.attn = SpatialGlobalAttention(C_in=C_in, C_out=C_out, grid_size=window_size, a_type=a_type, n_head=n_head, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p)
        else:
            raise NotImplementedError(f"Attention mode not implemented: {att_mode}")

        self.with_mixer = with_mixer
        if(self.with_mixer):
            self.mlp = nn.Sequential(
                Conv2DExt(C_out, 4*C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.GELU(),
                Conv2DExt(4*C_out, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.Dropout(dropout_p),
            )

    def forward(self, x):

        x = self.input_proj(x) + self.attn(self.n1(x))

        if(self.with_mixer):
            x = x + self.mlp(self.n2(x))

        return x

# -------------------------------------------------------------------------------------------------
# A block of multiple transformer cells stacked on top of each other

class STCNNT_Block(nn.Module):
    """
    A stack of CNNT cells
    The first cell expands the channel dimension.
    Can use Conv2D mixer with all cells, last cell, or none at all.
    """
    def __init__(self, att_types, C_in, C_out=16, H=64, W=64,\
                    a_type="conv", window_size=8, is_causal=False, n_head=8,\
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),\
                    dropout_p=0.1, norm_mode="layer",\
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
            - window_size (int): size of window for local and global att
            - is_causal (bool): whether to mask attention to imply causality
            - n_head (int): number of heads in self attention
            - kernel_size, stride, padding (int, int): convolution parameters
            - dropout (float): probability of dropout
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

        self.cells = []

        for i in range(len(att_types)//2):

            att_type = att_types[2*i]
            mixer = att_types[2*i+1]

            assert att_type=='L' or att_type=='G' or att_type=='T', \
                f"att_type not implemented: {att_type} at index {2*i} in {att_types}"
            assert mixer=='0' or mixer=='1', \
                f"mixer not implemented: {mixer} at index {2*i+1} in {att_types}"

            if att_type=='L':
                att_type = "local"
            elif att_type=='G':
                att_type = "global"
            else: #'T'
                att_type = "temporal"

            C = C_in if i==0 else C_out

            self.cells.append(STCNNT_Cell(C_in=C, C_out=C_out, H=H, W=W, att_mode=att_type, a_type=a_type,
                                            window_size=window_size, is_causal=is_causal, n_head=n_head,
                                            kernel_size=kernel_size, stride=stride, padding=padding,
                                            dropout_p=dropout_p, with_mixer=(mixer=='1'), norm_mode=norm_mode))

        self.make_block()

        self.interpolate = interpolate
        self.interp_align_c = interp_align_c

    def make_block(self):

        self.block = nn.Sequential(*self.cells)

    def forward(self, x):

        x = self.block(x)

        B, T, C, H, W = x.shape
        interp = x

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

# -------------------------------------------------------------------------------------------------

def tests():
    # tests

    B, T, C, H, W = 2, 4, 3, 64, 64
    C_out = 8
    test_in = torch.rand(B,T,C,H,W)

    print("Begin Testing")

    a_types = ["conv", "lin"]
    for a_type in a_types:

        spacial_local = SpatialLocalAttention(wind_size=8, a_type=a_type, C_in=C, C_out=C_out)
        test_out = spacial_local(test_in)

        Bo, To, Co, Ho, Wo = test_out.shape
        assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed spacial local")

    a_types = ["conv", "lin"]
    for a_type in a_types:

        spacial_local = SpatialGlobalAttention(grid_size=8, a_type=a_type, C_in=C, C_out=C_out)
        test_out = spacial_local(test_in)

        Bo, To, Co, Ho, Wo = test_out.shape
        assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed spacial global")

    causals = [True, False]
    for causal in causals:

        temporal = TemporalCnnAttention(C, C_out=C_out, is_causal=causal)
        test_out = temporal(test_in)

        Bo, To, Co, Ho, Wo = test_out.shape
        assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed temporal")

    att_types = ["temporal", "local", "global"]
    norm_types = ["instance2d", "batch2d", "layer", "instance3d", "batch3d"]
    for att_type in att_types:
        for norm_type in norm_types:

            CNNT_Cell = STCNNT_Cell(C_in=C, C_out=C_out, H=H, W=W, att_mode=att_type, norm_mode=norm_type)
            test_out = CNNT_Cell(test_in)

            Bo, To, Co, Ho, Wo = test_out.shape
            assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed CNNT Cell")

    att_typess = ["L1", "G1", "T1", "L0", "L1", "G0", "G1", "T1", "T0", "L0G1T0", "T1L0G1"]

    for att_types in att_typess:
        CNNT_Block = STCNNT_Block(att_types=att_types, C_in=C, C_out=C_out)
        test_out, _ = CNNT_Block(test_in)

        Bo, To, Co, Ho, Wo = test_out.shape
        assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed CNNT Block att_types and mixers")

    interpolates = ["up", "down", "none"]
    interp_align_cs = [True, False]

    for interpolate in interpolates:
        for interp_align_c in interp_align_cs:
            CNNT_Block = STCNNT_Block(att_types=att_types, C_in=C, C_out=C_out,\
                                   interpolate=interpolate, interp_align_c=interp_align_c)
            _, test_out = CNNT_Block(test_in)

            Bo, To, Co, Ho, Wo = test_out.shape
            factor = 2 if interpolate=="up" else 0.5 if interpolate=="down" else 1
            assert B==Bo and T==To and Co==C_out and (H*factor)==Ho and (W*factor)==Wo

    print("Passed CNNT Block interpolation")

    print("Passed all tests")

if __name__=="__main__":
    tests()
