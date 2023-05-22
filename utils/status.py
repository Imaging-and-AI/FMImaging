"""
Utility functions to measure system status
"""
import torch
from collections import OrderedDict
from datetime import datetime
from torchinfo import summary
from colorama import Fore, Style

# -------------------------------------------------------------------------------------------------
    
def start_timer(enable=False):
    
    if enable:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return (start, end)
    else:
        return None

def end_timer(enable=False, t=None, msg=""):
    if enable:
        t[1].record()
        torch.cuda.synchronize()
        print(f"{Fore.LIGHTBLUE_EX}{msg} {t[0].elapsed_time(t[1])} ms ...{Style.RESET_ALL}")
                               
# -------------------------------------------------------------------------------------------------

def get_cuda_info(device):
	return {
		"PyTorch_version": torch.__version__,
		"CUDA_version": torch.version.cuda,
		"cuDNN_version": torch.backends.cudnn.version(),
		"Arch_version": torch._C._cuda_getArchFlags(),
		"device_count": torch.cuda.device_count(),
		"device_name": torch.cuda.get_device_name(device=device),
		"device_id": torch.cuda.current_device(),
		"cuda_capability": torch.cuda.get_device_capability(device=device),
		"device_properties": torch.cuda.get_device_properties(device=device),
		"reserved_memory": torch.cuda.memory_reserved(device=device) / 1024**3,
		"allocated_memory": torch.cuda.memory_allocated(device=device) / 1024**3,
		"max_allocated_memory": torch.cuda.max_memory_allocated(device=device) / 1024**3
	}
# -------------------------------------------------------------------------------------------------
def get_gpu_ram_usage(device='cuda:0'):
    """
    Get info regarding memory usage of a device
    @args:
        - device (torch.device): the device to get info about
    @rets:
        - result_string (str): a string containing the info
    """
    result_string = f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(device=device)/1024/1024/1024:.3}GB\n" + \
                    f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(device=device)/1024/1024/1024:.3f}GB\n" + \
                    f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(device=device)/1024/1024/1024:.3f}GB"

    return result_string
    
# -------------------------------------------------------------------------------------------------
# model info

def get_number_of_params(model):
    """
    Count the total number of parameters
    @args:
        - model (torch model): the model to check parameters of
    @rets:
        - trainable_params (int): the number of trainable params in the model
        - total_params (int): the total number of params in the model
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    return trainable_params, total_params

def model_info(model, config):
    """
    Prints model info and sets total and trainable parameters in the config
    @args:
        - model (torch model): the model to check parameters of
        - config (Namespace): runtime namespace for setup
    @rets:
        - model_summary (ModelStatistics object): the model summary
            see torchinfo/model_statistics.py for more information.
    """
    c = config
    input_size = (c.batch_size, c.time, c.C_in, c.height[0], c.width[0])
    col_names=("num_params", "params_percent", "mult_adds", "input_size", "output_size", "trainable")
    row_settings=["var_names", "depth"]
    dtypes=[torch.float32]

    model_summary = summary(model, verbose=0, mode="train", depth=c.summary_depth,\
                            input_size=input_size, col_names=col_names,\
                            row_settings=row_settings, dtypes=dtypes,\
                            device=config.device)

    c.trainable_params = model_summary.trainable_params
    c.total_params = model_summary.total_params

    torch.cuda.empty_cache()

    return model_summary

# -------------------------------------------------------------------------------------------------
# average metric tracker

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=="__main__":
    pass