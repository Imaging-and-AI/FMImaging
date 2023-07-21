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

class TemporalCnnStandardAttention(CnnAttentionBase):
    """
    Multi-head cnn attention model for complete temporal attention
    """
    def __init__(self, C_in, C_out=16, H=128, W=128, is_causal=False, n_head=8, \
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), \
                    stride_t=(2,2), att_dropout_p=0.0, 
                    cosine_att=False, normalize_Q_K=False, att_with_output_proj=True,
                    use_einsum=True):
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
        self.use_einsum = use_einsum

        assert self.C_out % self.n_head == 0, \
            f"Number of output channles {self.C_out} should be divisible by number of heads {self.n_head}"

        # key, query, value projections convolution
        # Wk, Wq, Wv
        self.key = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_t, padding=padding, bias=False)
        self.query = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_t, padding=padding, bias=False)
        self.value = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.register_buffer("mask", torch.tril(torch.ones(1000, 1000, dtype=torch.bool)).view(1, 1, 1000, 1000))

    def attention(self, k, q, v):
        B, T, C_prime, H_prime, W_prime = k.shape

        H = torch.div(self.C_out, self.n_head, rounding_mode="floor")
        D = H*H_prime*W_prime
        Hv, Wv = v.shape[-2:]

        k = k.view(B, T, self.n_head, H, H_prime, W_prime).transpose(1, 2)
        q = q.view(B, T, self.n_head, H, H_prime, W_prime).transpose(1, 2)
        v = v.view(B, T, self.n_head, H, H_prime*self.stride_f, W_prime*self.stride_f).transpose(1, 2)

        # k, q, v are [B, nh, T, hc, H', W']

        B, nh, T, hc, H_prime, W_prime = k.shape

        # (B, nh, T, hc, H', W') x (B, nh, hc, H', W', T) -> (B, nh, T, T)
        if self.cosine_att:
            att = F.normalize(q.view(B, nh, T, hc*H_prime*W_prime), dim=-1) @ F.normalize(k.view(B, nh, T, hc*H_prime*W_prime), dim=-1).transpose(-2, -1)
        else:
            if self.normalize_Q_K:
                eps = torch.finfo(k.dtype).eps
                k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
                q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

            att = (q.view(B, nh, T, hc*H_prime*W_prime) @ k.view(B, nh, T, hc*H_prime*W_prime).transpose(-2, -1))\
                    * torch.tensor(1.0 / math.sqrt(hc*H_prime*W_prime))

        if(self.is_causal):
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        y = att @ v.contiguous().view(B, nh, T, H*H_prime*self.stride_f*W_prime*self.stride_f)
        y = y.transpose(1, 2).contiguous().view(B, T, self.C_out, H_prime*self.stride_f, W_prime*self.stride_f)

        return y

    def einsum_attention(self, k, q, v):

        B, T, C_prime, H_prime, W_prime = k.shape

        H = torch.div(self.C_out, self.n_head, rounding_mode="floor")
        D = H*H_prime*W_prime
        Hv, Wv = v.shape[-2:]

        k = k.view(B, T, self.n_head, D)
        q = q.view(B, T, self.n_head, D)
        v = v.view(B, T, self.n_head, H*Hv*Wv)

        # (B, T, nh, D) x (B, K, nh, D) -> (B, nh, T, K)
        if self.cosine_att:
            att = torch.einsum("BTND, BKND -> BNTK", F.normalize(q, dim=-1), F.normalize(k, dim=-1))
        else:
            if self.normalize_Q_K:
                eps = torch.finfo(k.dtype).eps
                k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
                q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

            att = torch.einsum("BTND, BKND -> BNTK", q, k) * torch.tensor(1.0 / math.sqrt(D))

        if(self.is_causal):
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # (B, nh, T, K) * (B, K, nh, D)
        y = torch.einsum("BNTK, BKND -> BTND", att, v).contiguous().view(B, T, self.C_out, Hv, Wv)

        return y

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
        v = self.value(x)

        # if self.normalize_Q_K:
        #     eps = torch.finfo(k.dtype).eps
        #     # add normalization for k and q, along C, H, W
        #     k = (k - torch.mean(k, dim=(-3, -2, -1), keepdim=True)) / ( torch.sqrt(torch.var(k, dim=(-3, -2, -1), keepdim=True) + eps) )
        #     q = (q - torch.mean(q, dim=(-3, -2, -1), keepdim=True)) / ( torch.sqrt(torch.var(q, dim=(-3, -2, -1), keepdim=True) + eps) )

        if self.use_einsum:
            y = self.einsum_attention(k, q, v)
        else:
            y = self.attention(k, q, v)

        y = self.output_proj(y)

        return y

class TemporalCnnAttention(CnnAttentionBase):
    """
    Multi-head cnn attention model for complete temporal attention with flash attention implementation
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

        gpu_name = torch.cuda.get_device_name()
        if gpu_name.find("A100") >= 0 or gpu_name.find("H100") >= 0:
            self.flash_atten_type = torch.bfloat16
        else:
            self.flash_atten_type = torch.float32


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

        # if self.normalize_Q_K:
        #     eps = torch.finfo(k.dtype).eps
        #     # add normalization for k and q, along [C_prime, H_prime, W_prime]
        #     k = (k - torch.mean(k, dim=(-3, -2, -1), keepdim=True)) / ( torch.sqrt(torch.var(k, dim=(-3, -2, -1), keepdim=True) + eps) )
        #     q = (q - torch.mean(q, dim=(-3, -2, -1), keepdim=True)) / ( torch.sqrt(torch.var(q, dim=(-3, -2, -1), keepdim=True) + eps) )

        H = torch.div(self.C_out, self.n_head, rounding_mode="floor")

        k = k.view(B, T, self.n_head, H, H_prime, W_prime).transpose(1, 2)
        q = q.view(B, T, self.n_head, H, H_prime, W_prime).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, H, H_prime*self.stride_f, W_prime*self.stride_f).transpose(1, 2)

        B, nh, T, hc, H_prime, W_prime = k.shape

        ### START OF FLASH ATTENTION IMPLEMENTATION ###
        q = q.view(B, nh, T, hc*H_prime*W_prime)
        k = k.view(B, nh, T, hc*H_prime*W_prime)
        v = v.view(B, nh, T, hc*H_prime*W_prime*self.stride_f*self.stride_f)

        if self.cosine_att:
            q = F.normalize(q,dim=-1) / torch.tensor(1.0 / math.sqrt(hc*H_prime*W_prime))
            k = F.normalize(k,dim=-1)
        elif self.normalize_Q_K:
            eps = torch.finfo(k.dtype).eps
            # add normalization for k and q, along [C_prime, H_prime, W_prime]
            k = (k - torch.mean(k, dim=(-1), keepdim=True)) / ( torch.sqrt(torch.var(k, dim=(-1), keepdim=True) + eps) )
            q = (q - torch.mean(q, dim=(-1), keepdim=True)) / ( torch.sqrt(torch.var(q, dim=(-1), keepdim=True) + eps) )

        if k.is_cuda:
            # Leaving forced self-attention commented out so default behavior can kick in when flash attention isn't applicable (e.g., q, k, v are not the same size)
            # with torch.backends.cuda.sdp_kernel(
            #             enable_flash=True, enable_math=False, enable_mem_efficient=False
            #     ):
            original_dtype = k.dtype
            y = F.scaled_dot_product_attention(q.type(self.flash_atten_type), k.type(self.flash_atten_type), v.type(self.flash_atten_type), dropout_p=self.att_dropout_p,is_causal=self.is_causal).type(original_dtype)
        else:
            y = F.scaled_dot_product_attention(q,k,v,dropout_p=self.att_dropout_p,is_causal=self.is_causal)

        ### END OF FLASH ATTENTION IMPLEMENTATION ###

        y = y.transpose(1, 2).contiguous().view(B, T, self.C_out, H_prime*self.stride_f, W_prime*self.stride_f)

        y = self.output_proj(y)

        return y
    
# -------------------------------------------------------------------------------------------------

def tests():
    # tests
    
    B, T, C, H, W = 2, 4, 3, 64, 64
    C_out = 8    

    device = get_device()
    
    test_in = torch.rand(B,T,C,H,W, device=device)
    
    print("Begin Testing")

    causals = [True, False]
    normalize_Q_Ks = [True, False]
    att_with_output_projs = [True, False]
    for causal in causals:
        for normalize_Q_K in normalize_Q_Ks:
            for att_with_output_proj in att_with_output_projs:

                temporal = TemporalCnnAttention(C, C_out=C_out, is_causal=causal, normalize_Q_K=normalize_Q_K, att_with_output_proj=att_with_output_proj).to(device=device)
                test_out = temporal(test_in)

                Bo, To, Co, Ho, Wo = test_out.shape
                assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed temporal")

    print("Passed all tests")

def benchmark():
    
    from utils.benchmark import benchmark_all, benchmark_memory, pytorch_profiler
    from utils.setup_training import set_seed
    from colorama import Fore, Style
        
    set_seed(seed=53)
    
    device = get_device()
    
    B, T, C, H, W = 16, 12, 3, 128, 128
    C_out = 64
    test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)
       
    print(test_in[6:9,2:6, 2, 54, 34])
    print(test_in[11:,7:, 2, 54, 34])
       
    import torch.utils.benchmark as benchmark
    
    X1 = torch.randn(100, 534, 12, 256, dtype=torch.float32, device=device)    
    X2 = torch.randn(100, 534, 12, 256, dtype=torch.float32, device=device)
    
    R1 = torch.einsum("ntdg, ncdg -> ndtc", X1, X2)
    R2 = torch.einsum("ntdg, ncdg -> ntdc", X1, X2)

    def f1(X1, X2):
        a = torch.einsum("ntdg, ncdg -> ndtc", X1, X2)
        
    def f2(X1, X2):
        a = X1.transpose(1, 2)
        b = X2.permute((0, 2, 3, 1))
        c = a @ b

    t0 = benchmark.Timer(
        stmt='f1(X1, X2)',
        globals={'f1':f1, 'X1': X1, 'X2':X2})
    
    print(t0.timeit(100))
    
    t0 = benchmark.Timer(
        stmt='f2(X1, X2)',
        globals={'f2':f2, 'X1': X1, 'X2':X2})
    
    print(t0.timeit(100))

    print(f"{Fore.GREEN}-------------> Flash temporal attention <----------------------{Style.RESET_ALL}")

    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)

    temporal = TemporalCnnAttention(C_in=C, 
                                    C_out=C_out, 
                                    H=H, W=W,
                                    n_head=32,
                                    cosine_att=True,
                                    normalize_Q_K=True, 
                                    att_with_output_proj=0.1)

    temporal.to(device=device)
                
    with torch.inference_mode():
        y = temporal(test_in)
                    
    benchmark_all(temporal, test_in, grad=None, repeats=80, desc='TemporalCnnAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    
    benchmark_memory(temporal, test_in, desc='TemporalCnnAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)
    
    print(f"{Fore.YELLOW}-------------> Standard temporal attention <----------------------{Style.RESET_ALL}")
    temporal = TemporalCnnStandardAttention(C_in=C, 
                                    C_out=C_out, 
                                    H=H, W=W,
                                    n_head=32,
                                    cosine_att=True,
                                    normalize_Q_K=True, 
                                    att_with_output_proj=0.1,
                                    use_einsum=True)

    temporal.to(device=device)
                
    with torch.inference_mode():
        y = temporal(test_in)

    benchmark_all(temporal, test_in, grad=None, repeats=80, desc='TemporalCnnStandardAttention-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)
    
    benchmark_memory(temporal, test_in, desc='TemporalCnnStandardAttention-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    temporal = TemporalCnnStandardAttention(C_in=C, 
                                    C_out=C_out, 
                                    H=H, W=W,
                                    n_head=32,
                                    cosine_att=True,
                                    normalize_Q_K=True, 
                                    att_with_output_proj=0.1,
                                    use_einsum=False)

    temporal.to(device=device)
                
    with torch.inference_mode():
        y = temporal(test_in)

    benchmark_all(temporal, test_in, grad=None, repeats=80, desc='TemporalCnnStandardAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    benchmark_memory(temporal, test_in, desc='TemporalCnnStandardAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    def loss(model, x):
        y = model(x)
        l = torch.sum(y)
        return l

    pytorch_profiler(loss, temporal, test_in, trace_filename='/export/Lab-Xue/projects/mri/profiling/TemporalCnnAttention.json', backward=True, amp=True, amp_dtype=torch.bfloat16, cpu=False, verbose=True)

if __name__=="__main__":
    tests()
    benchmark()
