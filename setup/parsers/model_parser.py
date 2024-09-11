
import argparse
import sys
from pathlib import Path

Setup_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Setup_DIR))

from config_utils import *

class model_parser(object):
    """
    Parser that contains args for the model architecture
    @args:
        no args
    @rets:
        no rets; self.parser contains args
    """

    def __init__(self, model_type):
        self.parser = argparse.ArgumentParser("")

        if 'omnivore' in model_type: 
            self.add_omnivore_args()

        #if 'STCNNT' in model_type: 
        self.add_shared_STCNNT_args()
        self.add_hrnet_STCNNT_args()

        # if model_type=='STCNNT_HRNET': 
        #     self.add_hrnet_STCNNT_args()

        if model_type=='STCNNT_UNET': 
            self.add_unet_STCNNT_args()
            
        if model_type=='STCNNT_mUNET': 
            self.add_mixed_unetr_STCNNT_args()
        
    def add_omnivore_args(self):  
        self.parser.add_argument('--omnivore.size', type=str, default='tiny', choices=['tiny','small','base','large','custom'], help="Size of omnivore model")
        self.parser.add_argument('--omnivore.patch_size', nargs='+', type=int, default=[1,1,1], help="Size of swin patches")
        self.parser.add_argument('--omnivore.window_size', nargs='+', type=int, default=[7,7,7], help="Size of swin windows")
        self.parser.add_argument('--omnivore.embed_dim', type=int, default=24, help="Size of embedding dimension")
        self.parser.add_argument('--omnivore.depths', nargs='+', type=int, default=[2,2,6,2], help="Number of transformer blocks per resolution depth")
        self.parser.add_argument('--omnivore.num_heads', nargs='+', type=int, default=[3,6,12,24], help="Number of attention heads per resolution depth")
        
    def add_shared_STCNNT_args(self):
        self.parser.add_argument("--cell_type", type=str, default="sequential", choices=['sequential', 'parallel'], help='cell type, sequential or parallel')
        self.parser.add_argument("--block_dense_connection", type=int, default=1, help='whether to add dense connections between cells in a block')
        self.parser.add_argument("--a_type", type=str, default="conv", choices=['conv', 'lin'], help='type of attention in the spatial attention modules')
        self.parser.add_argument("--mixer_type", type=str, default="conv", choices=['conv', 'lin'], help='conv or lin, type of mixer in the spatial attention modules; only conv is possible for the temporal attention')
        self.parser.add_argument("--window_size", nargs='+', type=int, default=[16, 16], help='size of window for spatial attention. This is the number of pixels in a window. Given image height and weight H and W, number of windows is H/windows_size * W/windows_size')
        self.parser.add_argument("--patch_size", nargs='+', type=int, default=[2, 2], help='size of patch for spatial attention. This is the number of pixels in a patch. An image is first split into windows. Every window is further split into patches.')
        self.parser.add_argument("--window_sizing_method", type=str, default="mixed", choices=['mixed', 'keep_window_size', 'keep_num_window'], help='method to adjust window_size between resolution levels, "keep_window_size", "keep_num_window", "mixed".\
                                    "keep_window_size" means number of pixels in a window is kept after down/upsample the image; \
                                    "keep_num_window" means the number of windows is kept after down/upsample the image; \
                                    "mixed" means interleave both methods.')
        self.parser.add_argument("--n_head", type=int, default=32, help='number of transformer heads')
        self.parser.add_argument("--kernel_size", type=int, default=3, help='size of the square kernel for CNN')
        self.parser.add_argument("--stride", type=int, default=1, help='stride for CNN (equal x and y)')
        self.parser.add_argument("--padding", type=int, default=1, help='padding for CNN (equal x and y)')
        self.parser.add_argument("--stride_s", type=int, default=1, help='stride for spatial attention, q and k (equal x and y)') 
        self.parser.add_argument("--stride_t", type=int, default=1, help='stride for temporal attention, q and k (equal x and y)') 
        self.parser.add_argument("--separable_conv", action="store_true", help='if set, use separable conv')
        self.parser.add_argument("--mixer_kernel_size", type=int, default=5, help='conv kernel size for the mixer')
        self.parser.add_argument("--mixer_stride", type=int, default=1, help='stride for the mixer')
        self.parser.add_argument("--mixer_padding", type=int, default=2, help='padding for the mixer')
        self.parser.add_argument("--normalize_Q_K", action="store_true", help='whether to normalize Q and K before computing attention matrix')
        self.parser.add_argument("--cosine_att", type=int, default=0, help='whether to use cosine attention; if True, normalize_Q_K is ignored')   
        self.parser.add_argument("--att_with_relative_postion_bias", type=int, default=1, help='whether to use relative position bias')   
        self.parser.add_argument("--att_dropout_p", type=float, default=0.0, help='pdrop for the attention coefficient matrix')
        self.parser.add_argument("--dropout_p", type=float, default=0.1, help='pdrop regulization for stochastic residual connections')
        self.parser.add_argument("--att_with_output_proj", type=int, default=1, help='whether to add output projection in attention layer')
        self.parser.add_argument("--scale_ratio_in_mixer", type=float, default=4.0, help='the scaling ratio to increase/decrease dimensions in the mixer of an attention layer')
        self.parser.add_argument("--norm_mode", type=str, default="instance2d", choices=['layer', 'batch2d', 'instance2d', 'batch3d', 'instance3d'], help='normalization mode: "layer", "batch2d", "instance2d", "batch3d", "instance3d"')
        self.parser.add_argument("--shuffle_in_window", type=int, default=0, help='whether to shuffle patches in a window for the global attention')    
        self.parser.add_argument("--is_causal", action="store_true", help='treat timed data as causal and mask future entries')
        self.parser.add_argument("--interp_align_c", action="store_true", help='align corners while interpolating')
        self.parser.add_argument("--use_einsum", action="store_true", help='if set, use einsum implementation.')
        self.parser.add_argument("--temporal_flash_attention", action="store_true", help='if set, temporal attention uses flash attention implementation.')
        self.parser.add_argument('--activation_func', type=str, default="prelu", choices=['elu', 'relu', 'leakyrelu', 'prelu', 'relu6', 'selu', 'celu', 'gelu'], help="nonlinear activation function, elu, relu, leakyrelu, prelu, relu6, selu, celu, gelu ")
        self.parser.add_argument('--upsample_method', type=str, default="linear", choices=['NN', 'linear', 'bspline'], help="upsampling method in backbone, NN, linear or bspline ")

    def add_hrnet_STCNNT_args(self):  
        self.parser.add_argument('--backbone_hrnet.C', type=int, default=32, help="number of channels in main body of hrnet")
        self.parser.add_argument('--backbone_hrnet.num_resolution_levels', type=int, default=2, help="number of resolution levels; image size reduce by x2 for every level")
        self.parser.add_argument('--backbone_hrnet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
            to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")
        self.parser.add_argument('--backbone_hrnet.use_interpolation', type=int, default=1, help="whether to use interpolation in downsample layer; if False, use stride convolution")

    def add_unet_STCNNT_args(self):  
        self.parser.add_argument('--backbone_unet.C', type=int, default=32, help="number of channels in main body of unet")
        self.parser.add_argument('--backbone_unet.num_resolution_levels', type=int, default=2, help="number of resolution levels for unet; image size reduce by x2 for every level")
        self.parser.add_argument('--backbone_unet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
            to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")    
        self.parser.add_argument('--backbone_unet.use_unet_attention', type=int, default=1, help="whether to add unet attention between resolution levels")
        self.parser.add_argument('--backbone_unet.use_interpolation', type=int, default=1, help="whether to use interpolation in downsample layer; if False, use stride convolution")
        self.parser.add_argument('--backbone_unet.with_conv', type=int, default=1, help="whether to add conv in down/upsample layers; if False, only interpolation is performed")

    def add_mixed_unetr_STCNNT_args(self):
        self.parser.add_argument('--backbone_mixed_unetr.C', type=int, default=32, help="number of channels in main body of mixed unetr")
        self.parser.add_argument('--backbone_mixed_unetr.num_resolution_levels', type=int, default=2, help="number of resolution levels for unet; image size reduce by x2 for every level")
        self.parser.add_argument('--backbone_mixed_unetr.block_str', nargs='+', type=str, default=['T1L1G1', 'T1L1G1'], help="block string to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")
        self.parser.add_argument('--backbone_mixed_unetr.use_unet_attention', type=int, default=1, help="whether to add unet attention between resolution levels")
        self.parser.add_argument('--backbone_mixed_unetr.use_interpolation', type=int, default=1, help="whether to use interpolation in downsample layer; if False, use stride convolution")
        self.parser.add_argument('--backbone_mixed_unetr.with_conv', type=int, default=0, help="whether to add conv in down/upsample layers; if False, only interpolation is performed")
        self.parser.add_argument('--backbone_mixed_unetr.min_T', type=int, default=16, help="minimal T/D for downsampling along T or D dimension")
        self.parser.add_argument('--backbone_mixed_unetr.encoder_on_skip_connection', type=int, default=1, help="whether to add encoder on skip connection")
        self.parser.add_argument('--backbone_mixed_unetr.encoder_on_input', type=int, default=1, help="whether to add encoder for input tensor")
        self.parser.add_argument('--backbone_mixed_unetr.transformer_for_upsampling', type=int, default=0, help="whether to use transformer for upsampling branch")
        self.parser.add_argument("--backbone_mixed_unetr.n_heads", nargs='+', type=int, default=[32, 32, 32, 32], help='number of heads in each resolution layer')
        self.parser.add_argument('--backbone_mixed_unetr.use_conv_3d', type=int, default=1, help="whether to use 3D convolution")
        self.parser.add_argument('--backbone_mixed_unetr.use_window_partition', type=int, default=0, help="whether to add window partition on input tensors")

