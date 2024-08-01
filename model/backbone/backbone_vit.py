from __future__ import annotations

# from zoologydev.src.models.mixers.hyena_simple import SimpleHyenaOperator

import itertools
from collections.abc import Sequence
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks import MLPBlock as Mlp
from monai.utils import deprecated_arg
from monai.utils import optional_import
from monai.networks.layers import Conv, DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg

import sys
import os
from pathlib import Path

Model_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Model_DIR))
sys.path.append(os.path.join(str(Model_DIR),'hyena'))
sys.path.append(os.path.join(str(Model_DIR),'hyena','zoologydev'))

from zoologydev.src.models.mixers.hyena_simple import SimpleHyenaOperator

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
rearrange, _ = optional_import("einops", name="rearrange")
SUPPORTED_PATCH_EMBEDDING_TYPES = {"conv", "perceptron"}
SUPPORTED_POS_EMBEDDING_TYPES = {"none", "learnable", "sincos"}

"""
CODE MODIFIED FROM MONAI GITHUB 
"""

#-------------------------------------------------------------------------------------

def custom_ViT(config, input_feature_channels):
    """
    NOTE: THIS ARCHITECTURE IS NOT WRITTEN FOR MORE THAN ONE INPUT TASK
    Wrapper function to set up and return ViT model.
    @args:
        config (Namespace): Namespace object containing configuration parameters.
        input_feature_channels (List[int]): List of ints containing the number of channels in each input tensor.
    @rets:
        model (torch model): pytorch model object 
        output_feature_channels (List[int]): list of ints indicated the number of channels in each output tensor.
    """

    if config.ViT.size=='small':
        # Small params from dino paper https://arxiv.org/pdf/2104.14294.pdf, which mirrors timm implementation
        hidden_size = 384
        mlp_dim = 1536 
        num_layers = 12
        num_heads = 6
        config.ViT.hidden_size = hidden_size
        config.ViT.mlp_dim = mlp_dim
        config.ViT.num_layers = num_layers
        config.ViT.num_heads = num_heads

    elif config.ViT.size=='base':
        # Base params from original ViT paper
        hidden_size = 768
        mlp_dim = 3072
        num_layers = 12
        num_heads = 12
        config.ViT.hidden_size = hidden_size
        config.ViT.mlp_dim = mlp_dim
        config.ViT.num_layers = num_layers
        config.ViT.num_heads = num_heads

    elif config.ViT.size=='custom':
        hidden_size = config.ViT.hidden_size
        mlp_dim = config.ViT.mlp_dim
        num_layers = config.ViT.num_layers
        num_heads = config.ViT.num_heads
        config.ViT.hidden_size = hidden_size
        config.ViT.mlp_dim = mlp_dim
        config.ViT.num_layers = num_layers
        config.ViT.num_heads = num_heads
        
    else:
        raise ValueError(f"Unknown model size {config.ViT.size} specified in config.")
    
    if config.time[0]==1:
        spatial_dims = 2
        if not config.use_patches[0]: input_size = [config.height[0], config.width[0]]
        else: input_size = [config.patch_height[0], config.patch_width[0]]
        if len(config.ViT.patch_size)==3: mod_patch_size = config.ViT.patch_size[1:]
        else: mod_patch_size = config.ViT.patch_size
    else:
        spatial_dims = 3
        if not config.use_patches[0]: input_size = [config.time[0], config.height[0], config.width[0]]
        else: input_size = [config.patch_time[0], config.patch_height[0], config.patch_width[0]]
        mod_patch_size = config.ViT.patch_size

    use_mae = config.task_type[-1]=='ss_image_restoration'
    if use_mae:
        if config.ss_image_restoration.mask_percent==0:
            use_mae = False

    model = ViT_with_hyena(use_hyena=config.ViT.use_hyena,
                in_channels=input_feature_channels[-1],
                img_size=input_size, 
                patch_size=mod_patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout_rate=0.0,
                spatial_dims=spatial_dims,
                classification=config.task_type[0]=='class',
                use_mae=use_mae)
    
    output_feature_channels=[hidden_size]*13

    return model, output_feature_channels

#-------------------------------------------------------------------------------------

class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        use_hyena: bool, 
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.use_hyena = use_hyena

        if not use_hyena:
            self.drop_output = nn.Dropout(dropout_rate)
            self.drop_weights = nn.Dropout(dropout_rate)
            self.head_dim = hidden_size // num_heads
            self.scale = self.head_dim**-0.5
            self.save_attn = save_attn
            self.att_mat = torch.Tensor()

            self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
            self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
            self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        else: 
            self.hyena = SimpleHyenaOperator(d_model=hidden_size,
                                            l_max=66000,
                                            filter_order=64,
                                            num_heads=num_heads,
                                            num_blocks=1,
                                            short_filter_order=5,
                                            bidrectional=True,
                                            dropout=dropout_rate,
                                            filter_dropout=dropout_rate,
                                            activation="id",
                                            # outer_mixing=False,
                                            # filter_cls="hyena-filter",
                                            # return_state=False,

                                            # ablations args
                                            # pre_gating: str="x1", 
                                            # post_gating: str="x2",

                                            # **filter_args,
                                            )

    def forward(self, x):
        if not self.use_hyena:
            output = self.input_rearrange(self.qkv(x))
            q, k, v = output[0], output[1], output[2]
            att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
            if self.save_attn:
                # no gradients and new tensor;
                # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                self.att_mat = att_mat.detach()

            att_mat = self.drop_weights(att_mat)
            x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
            x = self.out_rearrange(x)
            x = self.out_proj(x)
            x = self.drop_output(x)

        else:
            x = self.hyena(x.transpose(1,2)).transpose(1,2)
        
        return x

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        use_hyena: bool,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.use_hyena = use_hyena
        self.attn = SABlock(use_hyena, hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)            

        self.norm2 = nn.LayerNorm(hidden_size)



    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ModPatchEmbeddingBlock(nn.Module):
    """
    Code from Monai
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4,
        >>>                     proj_type="conv", pos_embed_type="sincos")

    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        use_mae: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            proj_type: patch embedding layer type.
            pos_embed_type: position embedding layer type.
            dropout_rate: fraction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            use_mae: whether to use masked auto encoding.
        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.
        """

        super().__init__()


        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate {dropout_rate} should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        self.proj_type = look_up_option(proj_type, SUPPORTED_PATCH_EMBEDDING_TYPES)
        self.pos_embed_type = look_up_option(pos_embed_type, SUPPORTED_POS_EMBEDDING_TYPES)

        self.use_mae = use_mae

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.proj_type == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        if self.proj_type == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.proj_type == "perceptron":
            raise NotImplementedError("Perceptron position embedding is not implemented yet (specifically, the MAE masking has not been implemented for perceptron embeddings).")
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

        if self.pos_embed_type == "none":
            pass
        elif self.pos_embed_type == "learnable":
            trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.pos_embed_type == "sincos":
            grid_size = []
            for in_size, pa_size in zip(img_size, patch_size):
                grid_size.append(in_size // pa_size)
            self.position_embeddings = build_sincos_position_embedding(grid_size, hidden_size, spatial_dims)
        else:
            raise ValueError(f"pos_embed_type {self.pos_embed_type} not supported.")

        self.apply(self._init_weights)

        if self.use_mae:
            if spatial_dims==2: self.mask_token = nn.Parameter(torch.zeros(1, hidden_size, 1, 1))
            else: self.mask_token = nn.Parameter(torch.zeros(1, hidden_size, 1, 1, 1))
            trunc_normal_(self.mask_token, mean=0.0, std=0.02, a=-0.02, b=0.02)
        self.apply(self._init_weights)        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _mask_tokens(self, x_embed, x_original):
        mask = 1.0*(x_original[:,:1] == 0)
        if len(x_original.shape)==4:
            mask = mask[:,:,::self.patch_size[0],::self.patch_size[1]]
            B, _, H, W = mask.shape

            mask_token = self.mask_token.expand(B, -1, H, W)
            x_embed = x_embed * (1 - mask) + mask_token * mask

        else:
            mask = mask[:,:,::self.patch_size[0],::self.patch_size[1],::self.patch_size[2]]
            B, _, D, H, W = mask.shape

            mask_token = self.mask_token.expand(B, -1, D, H, W)
            x_embed = x_embed * (1 - mask) + mask_token * mask

        return x_embed
        
        
    def forward(self, x):
        x_og = x.clone()
        x = self.patch_embeddings(x)
        if self.use_mae:
            x = self._mask_tokens(x, x_og)
            del x_og 
        if self.proj_type == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class ViT_with_hyena(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        use_hyena: bool,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
        use_mae: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            proj_type (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.
            use_mae (bool, optional): use masked auto encoding. Defaults to False.

        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), proj_type='conv', pos_embed_type='sincos')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), proj_type='conv', pos_embed_type='sincos', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), proj_type='conv', pos_embed_type='sincos', classification=True,
            >>>           spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.spatial_dims = spatial_dims
        self.use_mae = use_mae

        if use_hyena: pos_embed_type = "none"
        
        self.patch_embedding = ModPatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            use_mae=use_mae,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(use_hyena, hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification and not use_hyena:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

            # Commenting out classification heads, these are taken care of in post component
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore


    def forward(self, x):
        x = x[-1]
        if self.spatial_dims==2:
            x = x.squeeze(2)
        hidden_states_out = [x]
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        hidden_states_out.append(x) # In ViT, last hidden state is the final output for classification apps
        
        # Commenting out classification heads, these are taken care of in post component
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])

        return hidden_states_out