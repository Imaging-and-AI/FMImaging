"""
Standard convolution modules
Provides ability to use convolutions instead of attention in 2d and 3d:

Main class:
    ConvolutionModule
"""

import torch
import torch.nn as nn

from attention_modules import *

class ConvolutionModule(nn.Module):
    """
    The standard convolution class.
    Either 2d or 3d depending on the argument.
    """
    def __init__(self, conv_type, C_in, C_out,
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                    separable_conv=False):
        """
        @args:
            - conv_type ("conv2d" or "conv3d"): the type of conv
            - C_in (int): number of input channels
            - C_out (int): number of output channels
            - kernel_size, stride, padding (2 or 3 tuple): the params for conv
                - can extrapolare 2 from 3 and vice versa
            - separable_conv (bool): whether to use separable conv or not
        """
        super().__init__()

        assert conv_type=="conv2d" or conv_type=="conv3d", \
            f"Conv type not implemented: {conv_type}"

        if conv_type=="conv2d":

            if len(kernel_size)==3:
                kernel_size = (kernel_size[0], kernel_size[1])
            if len(stride)==3:
                stride = (stride[0], stride[1])
            if len(padding)==3:
                padding = (padding[0], padding[1])
                
            self.conv = Conv2DExt(in_channels=C_in, out_channels=C_out,\
                                    kernel_size=kernel_size, stride=stride, padding=padding,\
                                    separable_conv=separable_conv)
        elif conv_type=="conv3d":
            
            if len(kernel_size)==2:
                kernel_size = (*kernel_size, kernel_size[0])
            if len(stride)==2:
                stride = (*stride, stride[0])
            if len(padding)==2:
                padding = (*padding, padding[0])

            self.conv = Conv3DExt(in_channels=C_in, out_channels=C_out,\
                                    kernel_size=kernel_size, stride=stride, padding=padding,\
                                    separable_conv=separable_conv)
        else:
            raise NotImplementedError(f"Conv type not implemented: {conv_type}")

    def forward(self, x):
        """
        @args:
            x ([B, T, C_in, H, W]): Input of a batch of time series

        @rets:
            y ([B, T, C_out, H', W']): Output of the batch
        """

        return self.conv(x)

# -------------------------------------------------------------------------------------------------

def tests():
    # tests

    print("Begin Testing")

    import time
    import itertools

    B, T, C, H, W = 2, 16, 3, 16, 16
    C_out = 32

    kernel_size=(3, 3, 3)
    stride=(1, 1, 1)
    padding=(1, 1, 1)

    device = get_device()

    test_in = torch.rand(B,T,C,H,W, device=device)

    conv_types = ["conv2d", "conv3d"]
    separables = [True, False]

    for conv_type, separable in itertools.product(conv_types, separables):

        model = ConvolutionModule(conv_type=conv_type, C_in=C, C_out=C_out, \
                                    kernel_size=kernel_size, stride=stride, padding=padding, \
                                    separable_conv=separable)
        
        model.to(device)

        test_out = model(test_in)

        assert test_out.shape[0] == B
        assert test_out.shape[1] == T
        assert test_out.shape[2] == C_out
        assert test_out.shape[3] == H
        assert test_out.shape[4] == W

    print("Passed all tests")

if __name__=="__main__":
    tests()
