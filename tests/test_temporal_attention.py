import time
import pytest
import os

import numpy as np
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

from utils import *

import torch
from model_base.losses import *
from model_base.imaging_attention import *

from utils.benchmark import benchmark_all, benchmark_memory, pytorch_profiler
from utils.setup_training import set_seed
from colorama import Fore, Style

class Test_Temporal_Attention(object):

    @classmethod
    def setup_class(cls):
        set_seed(23564)
        torch.set_printoptions(precision=10)

    @classmethod
    def teardown_class(cls):
        pass

    def test_TemporalCnnStandardAttention(self):

        device = get_device()

        B, T, C, H, W = 2, 16, 3, 16, 16
        C_out = 32

        cosine_att=True
        normalize_Q_K=True
        att_with_output_proj=True
        use_einsum=True

        test_in = torch.rand(B,T,C,H,W, device=device)

        print(f"test_in - {test_in[0,2,0,4,:]}")

        test_in_GT = torch.tensor([0.1865558922, 0.4845264554, 0.2366391718, 0.7913835049, 0.4388458729,
            0.8051983118, 0.3325050771, 0.4242798388, 0.8450012207, 0.7058756351,
            0.2761471868, 0.4937677681, 0.5228261352, 0.5961654782, 0.6768726110,
            0.4204639494])

        assert torch.allclose(test_in[0,2,0,4,:].cpu(), test_in_GT)

        # =======================================================

        temporal = TemporalCnnStandardAttention(C_in=C, C_out=C_out, H=H, W=W, is_causal=False, n_head=8, 
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                    stride_qk=(2,2), 
                    separable_conv=False, 
                    att_dropout_p=0.0, 
                    cosine_att=cosine_att, 
                    normalize_Q_K=normalize_Q_K, 
                    att_with_output_proj=att_with_output_proj,
                    use_einsum=use_einsum).to(device=device)

        test_out = temporal(test_in)
        print(f"test_out - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([-0.0360875800,  0.0534653366,  0.0975006297,  0.1131298319,
         0.1271313876,  0.1806149781,  0.1149735525,  0.1122790799,
         0.1107020900,  0.1498304605,  0.1251391768,  0.1338548809,
         0.1185180992,  0.1099569276,  0.1424572021,  0.0815283433])

        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # ------------------------------------------------

        temporal.use_einsum = True
        test_out = temporal(test_in)
        print(f"test_out, use_einsum - {test_out[1,1,0,3,:]}")
        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # =======================================================

        temporal = TemporalCnnStandardAttention(C_in=C, C_out=C_out, H=H, W=W, is_causal=False, n_head=8, 
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                    stride_qk=(2,2), 
                    separable_conv=True, 
                    att_dropout_p=0.0, 
                    cosine_att=cosine_att, 
                    normalize_Q_K=normalize_Q_K, 
                    att_with_output_proj=att_with_output_proj,
                    use_einsum=use_einsum).to(device=device)

        test_out = temporal(test_in)
        print(f"test_out, separable_conv - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([-0.0141698122,  0.1080663130,  0.0447995774,  0.0814836845,
         0.0290265903,  0.0839581937,  0.0388186648,  0.0710153878,
         0.0754534081,  0.0330269784,  0.0979471505,  0.0427290238,
         0.0517599620,  0.0690403059,  0.0495539606,  0.0146387015])

        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # ------------------------------------------------

        temporal.use_einsum = True
        test_out = temporal(test_in)
        print(f"test_out, separable_conv, use_einsum - {test_out[1,1,0,3,:]}")
        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # =======================================================

        temporal = TemporalCnnStandardAttention(C_in=C, C_out=C_out, H=H, W=W, is_causal=False, n_head=8, 
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                    stride_qk=(1,1), 
                    separable_conv=True, 
                    att_dropout_p=0.0, 
                    cosine_att=cosine_att, 
                    normalize_Q_K=normalize_Q_K, 
                    att_with_output_proj=att_with_output_proj,
                    use_einsum=use_einsum).to(device=device)

        test_out = temporal(test_in)
        print(f"test_out, stride_qk=(1,1), separable_conv=True - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([-0.0276717134,  0.0344329067,  0.0234597232,  0.0219585523,
         0.0291473921,  0.0394139178,  0.0342935137,  0.0305856038,
         0.0287156980,  0.0274330452,  0.0449837111,  0.0164905936,
         0.0344967619,  0.0375222191,  0.0210449249,  0.0986497775])

        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # ------------------------------------------------

        temporal.use_einsum = True
        test_out = temporal(test_in)
        print(f"test_out, stride_qk=(1,1), separable_conv=True - {test_out[1,1,0,3,:]}")
        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # =======================================================

    def test_TemporalCnnStandardAttention_benchmark(self):

        device = get_device()
        min_run_time = 4

        forward_time_limit = 50
        backward_time_limit = 100
        all_time_limit = 150
        mem_limit = 15

        B, T, C, H, W = 16, 12, 32, 256, 256
        C_out = 32

        test_in = torch.rand(B,T,C,H,W, device=device)

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

        f, b, all1 = benchmark_all(temporal, test_in, grad=None, min_run_time=min_run_time, desc='TemporalCnnStandardAttention-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)

        mem = benchmark_memory(temporal, test_in, desc='TemporalCnnStandardAttention-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

        assert f[1].mean*1e3 < forward_time_limit
        assert b[1].mean*1e3 < backward_time_limit
        assert all1[1].mean*1e3 < all_time_limit
        assert mem < mem_limit

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

        f, b, all2 = benchmark_all(temporal, test_in, grad=None, min_run_time=min_run_time, desc='TemporalCnnStandardAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)
        mem = benchmark_memory(temporal, test_in, desc='TemporalCnnStandardAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

        assert f[1].mean*1e3 < forward_time_limit
        assert b[1].mean*1e3 < backward_time_limit
        assert all1[1].mean*1e3 < all_time_limit
        assert mem < mem_limit

        # print(f"{Fore.GREEN}-------------> Flash temporal attention <----------------------{Style.RESET_ALL}")

        # torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)

        # temporal = TemporalCnnAttention(C_in=C, 
        #                                 C_out=C_out, 
        #                                 H=H, W=W,
        #                                 n_head=16,
        #                                 cosine_att=True,
        #                                 normalize_Q_K=True, 
        #                                 att_with_output_proj=0.1)

        # temporal.to(device=device)

        # with torch.inference_mode():
        #     y = temporal(test_in)

        # f, b, all3 = benchmark_all(temporal, test_in, grad=None, min_run_time=min_run_time, desc='TemporalCnnAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)

        # mem = benchmark_memory(temporal, test_in, desc='TemporalCnnAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)