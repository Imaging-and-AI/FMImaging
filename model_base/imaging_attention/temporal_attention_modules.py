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
    def __init__(self, C_in, C_out=16, H=128, W=128, is_causal=False, n_head=8, 
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                    stride_qk=(2,2), separable_conv=False, att_dropout_p=0.0, 
                    cosine_att=False, normalize_Q_K=False, att_with_output_proj=True,
                    use_einsum=False):
        """
        Defines the layer for a cnn self-attention on temporal axis

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H', W']
        Usually used with conv definition such that H',W' = H,W

        Calculates attention using all the time points

        @args:
            - is_causal (bool): whether to mask attention to imply causality
            - stride_qk (int, int): special stride for temporal attention k,q matrices
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
                         att_with_output_proj=att_with_output_proj)

        self.is_causal = is_causal
        self.stride_f = stride_qk[0]
        self.use_einsum = use_einsum

        assert self.C_out*H*W % self.n_head == 0, \
            f"Number of output {self.C_out*H*W} should be divisible by number of heads {self.n_head}"

        # key, query, value projections convolution
        # Wk, Wq, Wv
        self.key = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_qk, padding=padding, bias=False, separable_conv=self.separable_conv)
        self.query = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_qk, padding=padding, bias=False, separable_conv=self.separable_conv)
        self.value = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, separable_conv=self.separable_conv)

        self.register_buffer("mask", torch.tril(torch.ones(1000, 1000, dtype=torch.bool)).view(1, 1, 1000, 1000))

    def attention(self, k, q, v):
        B, T, C_prime, H_prime, W_prime = k.shape

        H = torch.div(C_prime*H_prime*W_prime, self.n_head, rounding_mode="floor")
        Hv, Wv = v.shape[-2:]

        k = k.view(B, T, self.n_head, H).transpose(1, 2)
        q = q.view(B, T, self.n_head, H).transpose(1, 2)
        v = v.view(B, T, self.n_head, H*self.stride_f*self.stride_f).transpose(1, 2)

        # (B, nh, T, hc, H', W') x (B, nh, hc, H', W', T) -> (B, nh, T, T)
        if self.cosine_att:
            att = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        else:
            if self.normalize_Q_K:
                eps = torch.finfo(k.dtype).eps
                k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
                q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

            att = (q @ k.transpose(-2, -1)) * torch.tensor(1.0 / math.sqrt(H))

        if(self.is_causal):
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.C_out, Hv, Wv)

        return y

    def einsum_attention(self, k, q, v):

        B, T, C_prime, H_prime, W_prime = k.shape

        H = torch.div(C_prime*H_prime*W_prime, self.n_head, rounding_mode="floor")
        Hv, Wv = v.shape[-2:]

        k = k.view(B, T, self.n_head, H)
        q = q.view(B, T, self.n_head, H)
        v = v.view(B, T, self.n_head, H*self.stride_f*self.stride_f)

        if self.has_flash_attention and H <= 256 and (H == v.shape[-1]):
            y = self.perform_flash_atten(k, q, v).contiguous().view(B, T, self.C_out, Hv, Wv)
        else:
            # (B, T, nh, D) x (B, K, nh, D) -> (B, nh, T, K)
            if self.cosine_att:
                att = torch.einsum("BTND, BKND -> BNTK", F.normalize(q, dim=-1), F.normalize(k, dim=-1))
            else:
                if self.normalize_Q_K:
                    eps = torch.finfo(k.dtype).eps
                    k = (k - torch.mean(k, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(k, dim=-1, keepdim=True) + eps) )
                    q = (q - torch.mean(q, dim=-1, keepdim=True)) / ( torch.sqrt(torch.var(q, dim=-1, keepdim=True) + eps) )

                att = torch.einsum("BTND, BKND -> BNTK", q, k) * torch.tensor(1.0 / math.sqrt(H))

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

        #y1 = self.attention(torch.clone(k), torch.clone(q), torch.clone(v))
        if self.use_einsum:
            y = self.einsum_attention(k, q, v)
        else:
            y = self.attention(k, q, v)

        #assert torch.allclose(y1, y)

        y = self.output_proj(y)

        return y

class TemporalCnnAttention(CnnAttentionBase):
    """
    Multi-head cnn attention model for complete temporal attention with flash attention implementation
    """
    def __init__(self, C_in, C_out=16, H=128, W=128, is_causal=False, n_head=8, \
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), \
                    stride_qk=(2,2), att_dropout_p=0.0, 
                    cosine_att=False, normalize_Q_K=False, att_with_output_proj=True):
        """
        Defines the layer for a cnn self-attention on temporal axis

        Input to the attention layer has the size [B, T, C, H, W]
        Output has the size [B, T, output_channels, H', W']
        Usually used with conv definition such that H',W' = H,W

        Calculates attention using all the time points

        @args:
            - is_causal (bool): whether to mask attention to imply causality
            - stride_qk (int, int): special stride for temporal attention k,q matrices
        """
        super().__init__(C_in=C_in, 
                         C_out=C_out, 
                         H=H, W=W,
                         n_head=n_head, 
                         kernel_size=kernel_size, 
                         stride=stride, 
                         padding=padding, 
                         stride_qk=stride_qk,
                         att_dropout_p=att_dropout_p, 
                         cosine_att=cosine_att,
                         normalize_Q_K=normalize_Q_K, 
                         att_with_output_proj=att_with_output_proj)

        self.is_causal = is_causal
        self.stride_f = stride_qk[0]
        
        assert self.C_out % self.n_head == 0, \
            f"Number of output channles {self.C_out} should be divisible by number of heads {self.n_head}"

        # key, query, value projections convolution
        # Wk, Wq, Wv
        self.key = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_qk, padding=padding, bias=False, separable_conv=self.separable_conv)
        self.query = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride_qk, padding=padding, bias=False, separable_conv=self.separable_conv)
        self.value = Conv2DExt(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, separable_conv=self.separable_conv)

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

        _, _, C_prime, H_prime, W_prime = k.shape

        H = torch.div(C_prime*H_prime*W_prime, self.n_head, rounding_mode="floor")

        k = k.view(B, T, self.n_head, H).transpose(1, 2)
        q = q.view(B, T, self.n_head, H).transpose(1, 2)
        v = v.view(B, T, self.n_head, H*self.stride_f*self.stride_f).transpose(1, 2)

        ### START OF FLASH ATTENTION IMPLEMENTATION ###
        if self.cosine_att:
            q = F.normalize(q,dim=-1) / torch.tensor(1.0 / math.sqrt(H))
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

        y = y.reshape((B, T, self.C_out, H_prime*self.stride_f, W_prime*self.stride_f))

        y = self.output_proj(y)

        return y
    
# -------------------------------------------------------------------------------------------------

def tests():
    # tests
    import time

    B, T, C, H, W = 2, 16, 3, 16, 16
    C_out = 32

    device = get_device()
    
    test_in = torch.rand(B,T,C,H,W, device=device)
    
    print("Begin Testing")

    causals = [True, False]
    normalize_Q_Ks = [True, False]
    att_with_output_projs = [True, False]
    stride_qks = [[1,1], [2,2]]
    for causal in causals:
        for normalize_Q_K in normalize_Q_Ks:
            for stride_qk in stride_qks:
                for att_with_output_proj in att_with_output_projs:
                    t0 = time.time()
                    temporal = TemporalCnnAttention(C, C_out=C_out, n_head=32, is_causal=causal, normalize_Q_K=normalize_Q_K, att_with_output_proj=att_with_output_proj, stride_qk=stride_qk).to(device=device)
                    test_out = temporal(test_in)
                    t1 = time.time()
                    print(f"forward pass - {t1-t0} seconds")

                    t0 = time.time()
                    loss = nn.MSELoss()
                    mse = loss(test_in, test_out[:,:,:C,:,:])
                    mse.backward()
                    t1 = time.time()
                    print(f"backward pass - {t1-t0} seconds")

                    Bo, To, Co, Ho, Wo = test_out.shape
                    assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

                    # --------------------------------------
                    t0 = time.time()
                    temporal = TemporalCnnStandardAttention(C, C_out=C_out, is_causal=causal, normalize_Q_K=normalize_Q_K, att_with_output_proj=att_with_output_proj).to(device=device)
                    test_out = temporal(test_in)
                    t1 = time.time()
                    print(f"forward pass - {t1-t0} seconds")

                    t0 = time.time()
                    loss = nn.MSELoss()
                    mse = loss(test_in, test_out[:,:,:C,:,:])
                    mse.backward()
                    t1 = time.time()
                    print(f"backward pass - {t1-t0} seconds")

                    Bo, To, Co, Ho, Wo = test_out.shape
                    assert B==Bo and T==To and Co==C_out and H==Ho and W==Wo

    print("Passed temporal")

    # regression test
    set_seed(23564)
    torch.set_printoptions(precision=10)

    test_in = torch.rand(B,T,C,H,W, device=device)

    print(f"test_in - {test_in[0,2,0,4,:]}")

    test_in_GT = torch.tensor([0.1865558922, 0.4845264554, 0.2366391718, 0.7913835049, 0.4388458729,
        0.8051983118, 0.3325050771, 0.4242798388, 0.8450012207, 0.7058756351,
        0.2761471868, 0.4937677681, 0.5228261352, 0.5961654782, 0.6768726110,
        0.4204639494])

    assert torch.allclose(test_in[0,2,0,4,:].cpu(), test_in_GT)

    temporal = TemporalCnnAttention(C, C_out=C_out, 
                                    n_head=32, 
                                    is_causal=causal, 
                                    normalize_Q_K=normalize_Q_K, 
                                    att_with_output_proj=att_with_output_proj, 
                                    stride_qk=stride_qk,
                                    att_dropout_p=0.0, 
                                    cosine_att=False 
                                    ).to(device=device)
    test_out = temporal(test_in)
    print(f"test_out - {test_out[1,1,0,3,:]}")

    test_out_GT = torch.tensor([-0.1166992188, -0.1445312500, -0.0563964844, -0.0598144531,
        -0.0703125000, -0.1098632812, -0.0771484375, -0.0805664062,
        -0.0515136719, -0.0722656250, -0.1171875000, -0.1054687500,
        -0.0771484375, -0.1269531250, -0.0805664062,  0.2031250000])

    assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

    loss = nn.MSELoss()
    mse = loss(test_in, test_out[:,:,:C,:,:])
    mse.backward()

    # --------------------------------------

    print("Passed all tests")

def benchmark():

    from utils.benchmark import benchmark_all, benchmark_memory, pytorch_profiler
    from utils.setup_training import set_seed
    from colorama import Fore, Style

    set_seed(seed=53)

    device = get_device()

    B, T, C, H, W = 16, 64, 3, 64, 64
    C_out = 16
    test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)

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
                                    n_head=16,
                                    cosine_att=True,
                                    normalize_Q_K=True, 
                                    att_with_output_proj=0.1)

    temporal.to(device=device)

    with torch.inference_mode():
        y = temporal(test_in)

    f, b, all1 = benchmark_all(temporal, test_in, grad=None, repeats=80, desc='TemporalCnnAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    mem = benchmark_memory(temporal, test_in, desc='TemporalCnnAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    print(f"{Fore.YELLOW}-------------> Standard temporal attention <----------------------{Style.RESET_ALL}")
    temporal = TemporalCnnStandardAttention(C_in=C, 
                                    C_out=C_out, 
                                    H=H, W=W,
                                    n_head=16,
                                    cosine_att=True,
                                    normalize_Q_K=True, 
                                    att_with_output_proj=False,
                                    use_einsum=True)

    temporal.to(device=device)

    with torch.inference_mode():
        y = temporal(test_in)

    f, b, all2 = benchmark_all(temporal, test_in, grad=None, repeats=80, desc='TemporalCnnStandardAttention-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    benchmark_memory(temporal, test_in, desc='TemporalCnnStandardAttention-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    temporal = TemporalCnnStandardAttention(C_in=C, 
                                    C_out=C_out, 
                                    H=H, W=W,
                                    n_head=16,
                                    cosine_att=True,
                                    normalize_Q_K=True, 
                                    att_with_output_proj=False,
                                    use_einsum=False)

    temporal.to(device=device)

    with torch.inference_mode():
        y = temporal(test_in)

    f, b, all = benchmark_all(temporal, test_in, grad=None, repeats=80, desc='TemporalCnnStandardAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)

    benchmark_memory(temporal, test_in, desc='TemporalCnnStandardAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

    def loss(model, x):
        y = model(x)
        l = torch.sum(y)
        return l

    pytorch_profiler(loss, temporal, test_in, trace_filename='/export/Lab-Xue/projects/mri/profiling/TemporalCnnAttention.json', backward=True, amp=True, amp_dtype=torch.bfloat16, cpu=False, verbose=True)

if __name__=="__main__":
    tests()
    benchmark()
