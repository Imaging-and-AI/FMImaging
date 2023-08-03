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

class Test_Spatial_Vit_Attention(object):

    @classmethod
    def setup_class(cls):
        set_seed(23564)
        torch.set_printoptions(precision=10)

    @classmethod
    def teardown_class(cls):
        pass

    def test_attention(self):

        device = get_device()

        B, T, C, H, W = 2, 16, 3, 32, 32
        C_out = 32

        w = 8

        n_head = 32

        kernel_size = (3 ,3)
        stride = (1, 1)
        separable_conv = False
        padding = (1, 1)
        stride_qk = (1, 1)
        att_dropout_p = 0.0

        cosine_att=True
        normalize_Q_K=True
        att_with_output_proj=True

        att_with_relative_postion_bias = True
        with_timer = False

        test_in = torch.rand(B,T,C,H,W, device=device)

        print(f"test_in - {test_in[0,2,0,4,:]}")

        test_in_GT = torch.tensor([0.6839770675, 0.0236936603, 0.5925013423, 0.2092551887, 0.7638227344,
        0.0837925375, 0.8982807398, 0.1484473497, 0.0738541260, 0.5388930440,
        0.8055599332, 0.6586310863, 0.2615487874, 0.5661342144, 0.7855483294,
        0.2527054250, 0.1286259145, 0.9622011781, 0.6921065450, 0.6648769975,
        0.9932268262, 0.2785696387, 0.3337220550, 0.1158811226, 0.0733189881,
        0.9776614308, 0.4704501331, 0.7005786300, 0.3519243002, 0.1822092831,
        0.7065823078, 0.0820372477])

        assert torch.allclose(test_in[0,2,0,4,:].cpu(), test_in_GT)

        # =======================================================

        m = SpatialViTAttention(window_size=None, num_wind=[8, 8],
                                        a_type="conv", 
                                        C_in=C, C_out=C_out, 
                                        H=H, W=W, 
                                        stride_qk=(1,1),
                                        separable_conv=separable_conv,
                                        n_head=n_head,
                                        cosine_att=cosine_att, 
                                        normalize_Q_K=normalize_Q_K, 
                                        att_with_relative_postion_bias=att_with_relative_postion_bias,
                                        att_with_output_proj=att_with_output_proj,
                                        use_einsum=False).to(device=device)

        test_out = m(test_in)
        print(f"test_out - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([0.0190410614, 0.1682995111, 0.1273979992, 0.1562866122, 0.1436318457,
        0.1642317325, 0.1427072138, 0.1492585093, 0.1185442507, 0.1695247740,
        0.1490375698, 0.1559143513, 0.1045652777, 0.1526439786, 0.1260437965,
        0.1415534317, 0.1429069638, 0.1113877818, 0.1144868731, 0.1276132613,
        0.1298388243, 0.1244202927, 0.1335884035, 0.1239876449, 0.1242658347,
        0.1296669543, 0.1057593748, 0.1033983231, 0.1359413117, 0.0882831663,
        0.1587940753, 0.0445018895])

        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # ------------------------------------------------

        m.use_einsum = True
        test_out = m(test_in)
        print(f"test_out, use_einsum - {test_out[1,1,0,3,:]}")
        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # =======================================================

        separable_conv = True

        m = SpatialViTAttention(window_size=None, num_wind=[8, 8],
                                        a_type="conv", 
                                        C_in=C, C_out=C_out, 
                                        H=H, W=W, 
                                        stride_qk=(1,1),
                                        separable_conv=separable_conv,
                                        n_head=n_head,
                                        cosine_att=cosine_att, 
                                        normalize_Q_K=normalize_Q_K, 
                                        att_with_relative_postion_bias=att_with_relative_postion_bias,
                                        att_with_output_proj=att_with_output_proj,
                                        use_einsum=False).to(device=device)

        test_out = m(test_in)
        print(f"test_out, separable_conv - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([-0.0266224649, -0.0265606586, -0.0431437567, -0.0069671683,
        -0.0016356902, -0.0487447418, -0.0247402005, -0.0123609751,
        -0.0250353012, -0.0395653695, -0.0303690583, -0.0134745017,
        -0.0215543322, -0.0298888739, -0.0073195831, -0.0138597395,
        -0.0339221917, -0.0398234464, -0.0295350011, -0.0080666151,
        -0.0273884889, -0.0363376960,  0.0037998413, -0.0167316124,
        -0.0318259001, -0.0502284318, -0.0112760412, -0.0127729103,
        -0.0361896530, -0.0447209701,  0.0030998290,  0.1003994122])

        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # ------------------------------------------------

        m.use_einsum = True
        test_out = m(test_in)
        print(f"test_out, separable_conv, use_einsum - {test_out[1,1,0,3,:]}")
        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # =======================================================

    def test_benchmark(self):

        device = get_device()

        min_run_time = 5

        forward_time_limit = 50
        backward_time_limit = 100
        all_time_limit = 150
        mem_limit = 19

        B, T, C, H, W = 16, 12, 32, 256, 256
        C_out = 32
        n_head = 32
        test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)

        print(f"{Fore.YELLOW}-----------------------------------------------------------------{Style.RESET_ALL}")
        m = SpatialViTAttention(window_size=None, num_wind=[8, 8],
                                    a_type="conv", 
                                    C_in=C, C_out=C_out, 
                                    H=H, W=W, 
                                    stride_qk=(2,2),
                                    separable_conv=True,
                                    n_head=n_head,
                                    cosine_att=False, 
                                    normalize_Q_K=False, 
                                    att_with_relative_postion_bias=True,
                                    att_with_output_proj=False,
                                    use_einsum=True)

        m.to(device=device)

        with torch.inference_mode():
            y = m(test_in)

        f, b, all1 = benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialViTAttention-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)

        mem = benchmark_memory(m, test_in, desc='SpatialViTAttention-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

        assert f[1].mean*1e3 < forward_time_limit
        assert b[1].mean*1e3 < backward_time_limit
        assert all1[1].mean*1e3 < all_time_limit
        assert mem < mem_limit

        print(f"{Fore.YELLOW}-----------------------------------------------------------------{Style.RESET_ALL}")

        m.use_einsum = False

        f, b, all2 = benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialViTAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)
        mem = benchmark_memory(m, test_in, desc='SpatialViTAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

        assert f[1].mean*1e3 < forward_time_limit
        assert b[1].mean*1e3 < backward_time_limit
        assert all1[1].mean*1e3 < all_time_limit
        assert mem < mem_limit
