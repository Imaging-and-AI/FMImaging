# Adapted from https://github.com/HazyResearch/hippo/blob/datasets/benchmark/utils.py
""" Useful functions for writing test code. """

import torch
import torch.utils.benchmark as benchmark
from colorama import Fore, Back, Style

def benchmark_forward(fn, *inputs, min_run_time = 10, desc='', verbose=True, amp=False,
                      amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward pass of an arbitrary function. """
    if verbose:
        print(f"{Fore.RED}{desc} - Forward pass{Style.RESET_ALL}")
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    t = benchmark.Timer(
            stmt='fn_amp(*inputs, **kwinputs)',
            globals={'fn_amp': amp_wrapper, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.blocked_autorange(min_run_time=min_run_time)
    if verbose:
        print(f"{Fore.RED}{m}{Style.RESET_ALL}")
    return t, m


def benchmark_backward(fn, *inputs, grad=None, min_run_time=10, desc='', verbose=True, amp=False,
                       amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the backward pass of an arbitrary function. """
    if verbose:
        print(f"{Fore.GREEN}{desc} - Backward pass{Style.RESET_ALL}")
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError('Grad shape does not match output shape')
    t = benchmark.Timer(
            stmt='y.backward(grad, retain_graph=True)',
            globals={'y': y, 'grad': grad},
            num_threads=torch.get_num_threads(),
            )
    m = t.blocked_autorange(min_run_time=min_run_time)
    if verbose:
        print(f"{Fore.GREEN}{m}{Style.RESET_ALL}")
    return t, m


def benchmark_combined(fn, *inputs, grad=None, min_run_time=10, desc='', verbose=True, amp=False,
                       amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    if verbose:
        print(f"{Fore.YELLOW}{desc} - Forward + Backward pass{Style.RESET_ALL}")
    # y = fn(*inputs, **kwinputs)
    # if grad is None:
    #     grad = torch.randn_like(y)
    # else:
    #     if grad.shape != y.shape:
    #         raise RuntimeError('Grad shape does not match output shape')
    # del y
    def f(grad, *inputs, **kwinputs):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            y = fn(*inputs, **kwinputs)
            if type(y) is tuple:
                y = y[0]
        if grad is None:
            grad = torch.randn_like(y)
        else:
            if grad.shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')
        y.backward(grad, retain_graph=True)
    t = benchmark.Timer(
            stmt='f(grad, *inputs, **kwinputs)',
            globals={'f': f, 'fn': fn, 'inputs': inputs, 'grad': grad, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.blocked_autorange(min_run_time=min_run_time)
    if verbose:
        print(f"{Fore.YELLOW}{m}{Style.RESET_ALL}")
    return t, m


def benchmark_all(fn, *inputs, grad=None, min_run_time=10, desc='', verbose=True, amp=False,
                  amp_dtype=torch.bfloat16, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    return (
        benchmark_forward(fn, *inputs, min_run_time=min_run_time, desc=desc, verbose=verbose,
                          amp=amp, amp_dtype=amp_dtype, **kwinputs),
        benchmark_backward(fn, *inputs, grad=grad, min_run_time=min_run_time, desc=desc, verbose=verbose,
                           amp=amp, amp_dtype=amp_dtype, **kwinputs),
        benchmark_combined(fn, *inputs, grad=grad, min_run_time=min_run_time, desc=desc, verbose=verbose,
                           amp=amp, amp_dtype=amp_dtype, **kwinputs),
    )


def pytorch_profiler(fn, *inputs, trace_filename=None, backward=False, amp=False,
                     amp_dtype=torch.bfloat16, cpu=False, verbose=True, **kwinputs):
    """ Wrap benchmark functions in Pytorch profiler to see CUDA information. """
    if backward:
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            g = torch.randn_like(fn(*inputs, **kwinputs))
    for _ in range(30):   # Warm up
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            if backward:
                for x in inputs:
                    if isinstance(x, torch.Tensor):
                        x.grad = None
            # fn(*inputs, **kwinputs) if not backward else fn(*inputs, **kwinputs).backward(g)
            out = fn(*inputs, **kwinputs)
        # Backward should be done outside autocast
        if backward:
            out.backward(g)
    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            if backward:
                for x in inputs:
                    if isinstance(x, torch.Tensor):
                        x.grad = None
            out = fn(*inputs, **kwinputs)
        if backward: out.backward(g)
    if verbose:
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)


def benchmark_memory(fn, *inputs, desc='', amp=False, amp_dtype=torch.bfloat16, verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
        fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2 ** 20) * 1000)
    if verbose:
        print(f'{desc} max memory: {mem:.4f} GB')
    torch.cuda.empty_cache()
    return mem
