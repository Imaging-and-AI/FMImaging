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

class Test_Spatial_Local_Attention(object):

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

        m = SpatialLocalAttention(H=H, W=W, window_size=None, patch_size=None, 
                                    num_wind=[8, 8], num_patch=[4, 4], 
                                    a_type="conv", 
                                    C_in=C, C_out=C_out, 
                                    n_head=n_head,
                                    stride_qk=stride_qk,
                                    separable_conv=separable_conv,
                                    cosine_att=cosine_att, 
                                    normalize_Q_K=normalize_Q_K, 
                                    att_with_relative_postion_bias=att_with_relative_postion_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    use_einsum=False).to(device=device)

        test_out = m(test_in)
        print(f"test_out - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([0.0456418470, 0.1583141983, 0.1616141200, 0.1757398546, 0.1533311009,
        0.1683022380, 0.1676262319, 0.1876847446, 0.1815644056, 0.1691629887,
        0.1695845723, 0.1786854565, 0.1388620734, 0.1236437708, 0.1121998131,
        0.0886064097, 0.1455400586, 0.1570098996, 0.1302569211, 0.1378887594,
        0.1306552142, 0.1141469181, 0.1248630509, 0.1601799577, 0.1167429388,
        0.1715956777, 0.1605068147, 0.1436699629, 0.1740612090, 0.1339018643,
        0.0891680345, 0.0610226095])

        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # ------------------------------------------------

        m.use_einsum = True
        test_out = m(test_in)
        print(f"test_out, use_einsum - {test_out[1,1,0,3,:]}")
        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # =======================================================

        separable_conv = True

        m = SpatialLocalAttention(H=H, W=W, window_size=None, patch_size=None, 
                                    num_wind=[8, 8], num_patch=[4, 4], 
                                    a_type="conv", 
                                    C_in=C, C_out=C_out, 
                                    n_head=n_head,
                                    stride_qk=stride_qk,
                                    separable_conv=separable_conv,
                                    cosine_att=cosine_att, 
                                    normalize_Q_K=normalize_Q_K, 
                                    att_with_relative_postion_bias=att_with_relative_postion_bias,
                                    att_with_output_proj=att_with_output_proj,
                                    use_einsum=False).to(device=device)

        test_out = m(test_in)
        print(f"test_out, separable_conv - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([-0.0151489992, -0.0407282338, -0.0224726144, -0.0172828659,
        -0.0153235001, -0.0523843840, -0.0276088063, -0.0398909003,
        -0.0400013924, -0.0425506011, -0.0145693915, -0.0336207226,
        -0.0400631726, -0.0461868271, -0.0511579663, -0.0551702119,
        -0.0270310510, -0.0315211304, -0.0327133685, -0.0338304192,
        -0.0258924924, -0.0179298874, -0.0215777718, -0.0199000090,
        -0.0357092693, -0.0358278118, -0.0259872936, -0.0428694151,
        -0.0471831448, -0.0187609755, -0.0204386953, -0.0134220300])

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
        mem_limit = 20

        B, T, C, H, W = 16, 12, 32, 256, 256
        C_out = 32
        n_head = 32
        test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)

        window_size=[32, 32]
        patch_size=[8, 8]
        num_wind=[8, 8]
        num_patch=[4, 4]
        att_dropout_p=0.0
        cosine_att=True
        normalize_Q_K=True
        att_with_relative_postion_bias=False
        att_with_output_proj=False

        print(f"{Fore.YELLOW}-----------------------------------------------------------------{Style.RESET_ALL}")

        m = SpatialLocalAttention(C_in=C, C_out=C_out, H=H, W=W,
                            window_size=window_size, patch_size=patch_size, 
                            num_wind=num_wind, num_patch=num_patch, 
                            a_type="conv", n_head=n_head,
                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                            stride_qk=(2, 2), 
                            separable_conv=True,
                            att_dropout_p=att_dropout_p, 
                            cosine_att=cosine_att, 
                            normalize_Q_K=normalize_Q_K, 
                            att_with_relative_postion_bias=att_with_relative_postion_bias,
                            att_with_output_proj=att_with_output_proj,
                            use_einsum=True)

        m.to(device=device)

        with torch.inference_mode():
            y = m(test_in)

        f, b, all1 = benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialLocalAttention-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)

        mem = benchmark_memory(m, test_in, desc='SpatialLocalAttention-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

        assert f[1].mean*1e3 < forward_time_limit
        assert b[1].mean*1e3 < backward_time_limit
        assert all1[1].mean*1e3 < all_time_limit
        assert mem < mem_limit

        print(f"{Fore.YELLOW}-----------------------------------------------------------------{Style.RESET_ALL}")

        m.use_einsum = False

        f, b, all2 = benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialLocalAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)
        mem = benchmark_memory(m, test_in, desc='SpatialLocalAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

        assert f[1].mean*1e3 < forward_time_limit
        assert b[1].mean*1e3 < backward_time_limit
        assert all1[1].mean*1e3 < all_time_limit
        assert mem < mem_limit
