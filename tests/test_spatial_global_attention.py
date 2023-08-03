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

class Test_Spatial_Global_Attention(object):

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

        m = SpatialGlobalAttention(window_size=[16, 16], patch_size=[8, 8], 
                                num_wind=None, num_patch=None, 
                                a_type="conv", 
                                C_in=C, C_out=C_out, 
                                H=H, W=W, 
                                stride_qk=stride_qk,
                                separable_conv=True,
                                cosine_att=cosine_att, 
                                normalize_Q_K=normalize_Q_K, 
                                att_with_relative_postion_bias=att_with_relative_postion_bias,
                                att_with_output_proj=att_with_output_proj,
                                use_einsum=False).to(device=device)

        test_out = m(test_in)
        print(f"test_out - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([-0.0456042849, -0.0316154398,  0.0510408022,  0.0197037328,
         0.0053351177,  0.0121822814,  0.0609576777,  0.0080466531,
         0.0499836020,  0.0503908694,  0.0369840860,  0.0452060811,
        -0.0123959789,  0.0228928961, -0.0133936014, -0.0824305117,
        -0.0247492976, -0.0354663618,  0.0519989729,  0.0182968974,
         0.0107577099,  0.0113464575,  0.0613979697,  0.0114242453,
         0.0443990454,  0.0506563485,  0.0357892327,  0.0426145718,
        -0.0121082012,  0.0219068583, -0.0146635640, -0.0565882660])

        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # ------------------------------------------------

        m.use_einsum = True
        test_out = m(test_in)
        print(f"test_out, use_einsum - {test_out[1,1,0,3,:]}")
        assert torch.allclose(test_out[1,1,0,3,:].cpu(), test_out_GT)

        # =======================================================

        separable_conv = True

        m = SpatialGlobalAttention(window_size=[16, 16], patch_size=[8, 8], 
                                num_wind=None, num_patch=None, 
                                a_type="conv", 
                                C_in=C, C_out=C_out, 
                                H=H, W=W, 
                                stride_qk=stride_qk,
                                separable_conv=True,
                                cosine_att=cosine_att, 
                                normalize_Q_K=normalize_Q_K, 
                                att_with_relative_postion_bias=att_with_relative_postion_bias,
                                att_with_output_proj=att_with_output_proj,
                                use_einsum=False).to(device=device)

        test_out = m(test_in)
        print(f"test_out, separable_conv - {test_out[1,1,0,3,:]}")

        test_out_GT = torch.tensor([-0.0359408520, -0.0295771100, -0.0735319108, -0.0306682158,
        -0.0257704630, -0.0201204643, -0.0011911914, -0.0221619178,
        -0.0724141374,  0.0065838788, -0.0131002665,  0.0175279565,
        -0.0265355743, -0.0556709096, -0.0898940042, -0.0375859663,
        -0.0070649758, -0.0240705721, -0.0733641982, -0.0312120486,
        -0.0163097754, -0.0143062267, -0.0014717449, -0.0225311648,
        -0.0694785640,  0.0032226089, -0.0058338903,  0.0289130770,
        -0.0164748356, -0.0510558933, -0.0871194005, -0.0651520565])

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

        forward_time_limit = 100
        backward_time_limit = 150
        all_time_limit = 250
        mem_limit = 26

        B, T, C, H, W = 16, 12, 32, 256, 256

        test_in = torch.rand(B,T,C,H,W, dtype=torch.float32, device=device)

        C_out = 32
        n_head = 32

        window_size=[16, 16]
        patch_size=[2, 2] 
        num_wind=[8, 8]
        num_patch=[4, 4]

        att_dropout_p=0.1 
        cosine_att=True 
        normalize_Q_K=True 
        att_with_relative_postion_bias=True
        att_with_output_proj=True
        shuffle_in_window=False

        stride_qk = (2, 2)
        separable_conv = True

        print(f"{Fore.YELLOW}-----------------------------------------------------------------{Style.RESET_ALL}")

        m = SpatialGlobalAttention(C_in=C, C_out=C_out, H=H, W=H,
                            window_size=window_size, patch_size=patch_size, 
                            num_wind=num_wind, num_patch=num_patch,  
                            a_type="conv", n_head=n_head,
                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                            stride_qk = stride_qk, separable_conv=separable_conv, 
                            att_dropout_p=att_dropout_p, 
                            cosine_att=cosine_att, 
                            normalize_Q_K=normalize_Q_K, 
                            att_with_relative_postion_bias=att_with_relative_postion_bias,
                            att_with_output_proj=att_with_output_proj,
                            shuffle_in_window=shuffle_in_window,
                            use_einsum=True)

        m.to(device=device)

        with torch.inference_mode():
            y = m(test_in)

        f, b, all1 = benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialGlobalAttention-einsum', verbose=True, amp=True, amp_dtype=torch.bfloat16)

        mem = benchmark_memory(m, test_in, desc='SpatialGlobalAttention-einsum', amp=True, amp_dtype=torch.bfloat16, verbose=True)

        assert f[1].mean*1e3 < forward_time_limit
        assert b[1].mean*1e3 < backward_time_limit
        assert all1[1].mean*1e3 < all_time_limit
        assert mem < mem_limit

        print(f"{Fore.YELLOW}-----------------------------------------------------------------{Style.RESET_ALL}")

        m.use_einsum = False

        f, b, all2 = benchmark_all(m, test_in, grad=None, min_run_time=min_run_time, desc='SpatialGlobalAttention', verbose=True, amp=True, amp_dtype=torch.bfloat16)
        mem = benchmark_memory(m, test_in, desc='SpatialGlobalAttention', amp=True, amp_dtype=torch.bfloat16, verbose=True)

        assert f[1].mean*1e3 < forward_time_limit
        assert b[1].mean*1e3 < backward_time_limit
        assert all1[1].mean*1e3 < all_time_limit
        assert mem < mem_limit
