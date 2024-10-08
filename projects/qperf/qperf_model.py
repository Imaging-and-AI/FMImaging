
"""
MRI models

- STCNNT_MRI: the pre-backbone-post model with a simple pre and post module
- MRI_hrnet: a hrnet backbone + a hrnet post
- MRI_double_net: a hrnet or mixed_unet backbone + a hrnet or mixed_unet post
"""

import os
import sys
import copy
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from colorama import Fore, Back, Style

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

from model.model_base import ModelManager
from model.backbone import identity_model, omnivore_tiny, omnivore_small, omnivore_base, omnivore_large, STCNNT_HRnet_model, STCNNT_Unet_model, STCNNT_Mixed_Unetr_model
from model.backbone import STCNNT_HRnet, STCNNT_Mixed_Unetr, UpSample, set_window_patch_sizes_keep_num_window, set_window_patch_sizes_keep_window_size, STCNNT_Block
from model.imaging_attention import *
from model.transformer import *

# -------------------------------------------------------------------------------------------------
# QPerf model

class QPerfModel(ModelManager):
    """
        QPerf mapping model

        Input: aif and myo Gd curves
        Output: clean myo curves, Fp, Vp, Visf, PS, and delay

        The architecture is quite straight-forward :

        x -> input_proj --> + --> drop_out --> attention layers one after another --> LayerNorm --> output_proj_myo --> logits
                            |                                                                    |--> output_proj_params --> logits
        pos_embedding-------|
    """
    def __init__(self, config, n_layer=8, input_D=2, output_myo_D=1, num_params=5, T=80, is_causal=False, use_pos_embedding=True, n_embd=1024, n_head=32, dropout_p=0.1, att_dropout_p=0.0, residual_dropout_p=0.1):

        self.n_layer = n_layer
        self.input_D = input_D
        self.output_myo_D = output_myo_D
        self.num_params = num_params
        self.T = T
        self.is_causal = is_causal
        self.use_pos_embedding = use_pos_embedding
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.att_dropout_p = att_dropout_p
        self.residual_dropout_p = residual_dropout_p

        super().__init__(config)

        # a good trick to count how many parameters
        logging.info("number of parameters: %d", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def create_pre(self): 
        self.pre = nn.ModuleDict()
        self.pre["input_proj"] = nn.Linear(self.input_D, self.n_embd)
        if self.use_pos_embedding:
            self.pre["pos_emb"] = torch.nn.ParameterList([nn.Parameter(torch.zeros(1, self.T, self.n_embd))])
        self.pre["drop"] = nn.Dropout(self.dropout_p)

    def create_backbone(self, channel_first=True): 
        self.backbone = nn.Sequential(*[Cell(T=self.T, n_embd=self.n_embd, is_causal=self.is_causal, n_head=self.n_head, attn_dropout_p=self.att_dropout_p, residual_dropout_p=self.residual_dropout_p, activation_func=self.config.activation_func) for _ in range(self.n_layer)])
        self.feature_channels = self.n_embd

    def create_post(self): 
        self.post = nn.ModuleDict()
        n_embd = self.n_embd
        self.post["layer_norm"] = nn.LayerNorm(n_embd)
        self.post["output_proj1_myo"] = nn.Linear(n_embd, n_embd*2, bias=True)
        self.post["output_proj2_myo"] = nn.Linear(n_embd*2, self.output_myo_D, bias=True)

        self.post["output_proj1_params"] = nn.Linear(n_embd, n_embd, bias=True)
        self.post["output_proj2_params"] = nn.Linear(self.T*n_embd, self.num_params, bias=True)

        # self.post["output_proj_myo"] = nn.Linear(n_embd, self.output_myo_D, bias=True)

        # self.post["output_proj_params"] = nn.Linear(self.T*n_embd, self.num_params, bias=True)

    def forward(self, x):
        """Forward pass of detector

        Args:
            x ([B, T, C]]): Input membrane waveform with B batches and T time points with C length

            Due to the positional embedding is used, the input T is limited to less or equal to self.T

        Returns:
            logits_myo: [B, T, output_D]
            logits_params: [B, 5]
        """

        B, T, C = x.size()
        assert T <= self.T, "The positional embedding is used, so the maximal series length is %d" % self.T

        # project input from C channels to n_embd channels
        x_proj = self.pre["input_proj"](x)

        if self.use_pos_embedding:
            x = x_proj + self.pre["pos_emb"][0][:, :T, :]
            x = self.pre["drop"](x)
        else:
            x = x_proj + position_encoding(seq_len=T, dim_model=C, device=self.device)
            x = self.pre["drop"](x)

        # go through all layers of attentions
        x = self.backbone(x)

        # project outputs to output_size channel
        x = self.post["layer_norm"](x)
        x = F.gelu(x, approximate='tanh')

        y_myo = self.post["output_proj1_myo"](x)
        y_myo = F.gelu(y_myo, approximate='tanh')
        y_myo = self.post["output_proj2_myo"](y_myo)
        #y_myo = F.relu(y_myo)

        y_params = self.post["output_proj1_params"](x)
        y_params = F.gelu(y_params, approximate='tanh')
        y_params = torch.flatten(y_params, start_dim=1, end_dim=2)
        y_params = self.post["output_proj2_params"](y_params)
        #y_params = F.relu(y_params)

        # y_myo = self.post["output_proj_myo"](x)

        # y_params = torch.flatten(x, start_dim=1, end_dim=2)
        # y_params = self.post["output_proj_params"](y_params)

        return y_myo, y_params

# -------------------------------------------------------------------------------------------------

class QPerfModel_double_net(ModelManager):
    """
        QPerf mapping model

        Input: aif and myo Gd curves
        Output: clean myo curves, Fp, Vp, Visf, PS, and delay

        First part of network recons the Gd curves; the second part estimates parameters
    """
    def __init__(self, config, n_layer=[8, 8], input_D=2, output_myo_D=1, num_params=5, T=80, is_causal=False, use_pos_embedding=True, n_embd=1024, n_head=[32, 32], dropout_p=[0.1, 0.1], att_dropout_p=[0.0, 0.0], residual_dropout_p=[0.1, 0.1]):

        self.n_layer = n_layer
        self.input_D = input_D
        self.output_myo_D = output_myo_D
        self.num_params = num_params
        self.T = T
        self.is_causal = is_causal
        self.use_pos_embedding = use_pos_embedding
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.att_dropout_p = att_dropout_p
        self.residual_dropout_p = residual_dropout_p

        super().__init__(config)

        # a good trick to count how many parameters
        logging.info("number of parameters: %d", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def create_pre(self): 
        self.pre = nn.ModuleDict()
        self.pre["input_proj"] = nn.Linear(self.input_D, self.n_embd)
        if self.use_pos_embedding:
            self.pre["pos_emb"] = torch.nn.ParameterList([nn.Parameter(torch.zeros(1, self.T, self.n_embd))])
        self.pre["drop"] = nn.Dropout(self.dropout_p[0])

    def create_backbone(self, channel_first=True): 
        self.backbone = nn.ModuleDict()

        n_embd = self.n_embd

        self.backbone["1st"] = nn.ModuleList()
        for n in range(self.n_layer[0]):
            t_cell = Cell(T=self.T, 
                n_embd=self.n_embd, 
                is_causal=self.is_causal, 
                n_head=self.n_head[0], 
                attn_dropout_p=self.att_dropout_p[0], 
                residual_dropout_p=self.residual_dropout_p[0], 
                activation_func=self.config.activation_func)

            self.backbone["1st"].append(t_cell)

        self.backbone["1st_layer_norm"] = nn.LayerNorm(n_embd)
        self.backbone["1st_output_proj1_myo"] = nn.Linear(n_embd, n_embd*4, bias=True)
        self.backbone["1st_output_proj2_myo"] = nn.Linear(n_embd*4, self.output_myo_D, bias=True)
        #self.backbone["1st_output_proj_myo"] = nn.Linear(n_embd, self.output_myo_D, bias=True)

        self.feature_channels = self.n_embd

    def create_post(self): 
        self.post = nn.ModuleDict()

        n_embd = self.n_embd

        self.post["input_proj_2nd"] = nn.Linear(self.input_D, self.n_embd)

        self.post["2nd"] = nn.ModuleList()
        for n in range(self.n_layer[1]):
            t_cell = Cell(T=self.T, 
                n_embd=self.n_embd, 
                is_causal=self.is_causal, 
                n_head=self.n_head[1], 
                attn_dropout_p=self.att_dropout_p[1], 
                residual_dropout_p=self.residual_dropout_p[1], 
                activation_func=self.config.activation_func)

            self.post["2nd"].append(t_cell)

        self.post["output_proj1_params"] = nn.Linear(n_embd, n_embd*4, bias=True)
        self.post["output_proj2_params"] = nn.Linear(self.T*n_embd*4, self.num_params, bias=True)
        # self.post["output_proj_params"] = nn.Linear(self.T*n_embd, self.num_params, bias=True)

    def forward_backbone(self, N, backbone, x_input):
        # res_list = []
        # for n in range(N):
        #     if n == 0:
        #         input_1st = x_input
        #     else:
        #         input_1st = torch.clone(res_list[0])
        #         for k in range(1, n):
        #             input_1st += res_list[k]

        #     res = backbone[n](input_1st)
        #     res_list.append(res)

        for n in range(N):
            if n == 0:
                res = backbone[n](x_input)
            else:
                res = backbone[n](res)

        return res

    def forward(self, x):
        """Forward pass of detector

        Args:
            x ([B, T, C]]): Input membrane waveform with B batches and T time points with C length

            Due to the positional embedding is used, the input T is limited to less or equal to self.T

        Returns:
            logits_myo: [B, T, output_D]
            logits_params: [B, 5]
        """

        B, T, C = x.size()
        assert T <= self.T, "The positional embedding is used, so the maximal series length is %d" % self.T

        # project input from C channels to n_embd channels
        x_proj = self.pre["input_proj"](x)

        if self.use_pos_embedding:
            x_input = x_proj + self.pre["pos_emb"][0][:, :T, :]
            x_input = self.pre["drop"](x_input)
        else:
            x_input = x_proj + position_encoding(seq_len=T, dim_model=C, device=self.device)
            x_input = self.pre["drop"](x_input)

        # go through all layers of attentions
        res_1st = self.forward_backbone(self.n_layer[0], self.backbone["1st"], x_input)

        # myo
        y_myo = self.backbone["1st_output_proj1_myo"](res_1st)
        y_myo = F.relu(y_myo)
        y_myo = self.backbone["1st_output_proj2_myo"](y_myo)

        #y_myo = self.backbone["1st_output_proj_myo"](res_1st)

        # y_myo += torch.unsqueeze(x[:,:,1], dim=2)

        # params
        #p_inputs = torch.concatenate((torch.unsqueeze(x[:,:,0], dim=2), y_myo), dim=2)

        #p_proj = self.post["input_proj_2nd"](p_inputs)
        #res_2nd = self.forward_backbone(self.n_layer[1], self.post["2nd"], p_proj)

        res_2nd = self.forward_backbone(self.n_layer[1], self.post["2nd"], res_1st)

        y_params = self.post["output_proj1_params"](res_2nd)
        y_params = F.relu(y_params)
        y_params = torch.flatten(y_params, start_dim=1, end_dim=2)
        y_params = self.post["output_proj2_params"](y_params)

        #y_params = torch.flatten(res_2nd, start_dim=1, end_dim=2)
        #y_params = self.post["output_proj_params"](y_params)

        return y_myo, y_params

# -------------------------------------------------------------------------------------------------

class QPerfBTEXModel(ModelManager):
    """
        QPerf mapping model

        Input: aif and parameters
        Output: clean myo curves

        The architecture is quite straight-forward :

        aif -> input_proj --> + --> drop_out --> attention layers one after another --> LayerNorm --> output_proj_myo --> logits
                            |                  |                                                 
        pos_embedding-------|                  | 
                                               |
        params -> input_proj --> norm --> nl --|
    """
    def __init__(self, config, n_layer=8, 
                            input_D=1, 
                            output_myo_D=1, 
                            num_params=5, 
                            T=80, 
                            is_causal=False, 
                            use_pos_embedding=True, 
                            n_embd=1024, 
                            n_head=32, 
                            dropout_p=0.1, 
                            att_dropout_p=0.0, 
                            residual_dropout_p=0.1):

        self.n_layer = n_layer
        self.input_D = input_D
        self.output_myo_D = output_myo_D
        self.num_params = num_params
        self.T = T
        self.is_causal = is_causal
        self.use_pos_embedding = use_pos_embedding
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.att_dropout_p = att_dropout_p
        self.residual_dropout_p = residual_dropout_p

        super().__init__(config)

        # a good trick to count how many parameters
        logging.info("number of parameters: %d", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def create_pre(self): 
        self.pre = nn.ModuleDict()
        self.pre["input_proj"] = nn.Linear(self.input_D, self.n_embd)
        if self.use_pos_embedding:
            self.pre["pos_emb"] = torch.nn.ParameterList([nn.Parameter(torch.zeros(1, self.T, self.n_embd))])
        self.pre["drop"] = nn.Dropout(self.dropout_p)
        self.pre["nl"] = create_activation_func(name=self.config.activation_func)

        self.pre["input_proj_params"] = nn.Linear(self.num_params, self.T*self.n_embd)

        self.pre["nl_params"] = create_activation_func(name=self.config.activation_func)

    def create_backbone(self, channel_first=True): 
        self.backbone = nn.Sequential(*[Cell(T=self.T, n_embd=self.n_embd, is_causal=self.is_causal, n_head=self.n_head, attn_dropout_p=self.att_dropout_p, residual_dropout_p=self.residual_dropout_p, activation_func=self.config.activation_func) for _ in range(self.n_layer)])
        self.feature_channels = 2*self.n_embd

    def create_post(self): 
        self.post = nn.ModuleDict()
        n_embd = self.n_embd
        self.post["layer_norm"] = nn.LayerNorm(n_embd)
        self.post["output_proj1_myo"] = nn.Linear(n_embd, n_embd*2, bias=True)
        self.post["output_proj2_myo"] = nn.Linear(n_embd*2, self.output_myo_D, bias=True)

        self.post["nl1"] = create_activation_func(name=self.config.activation_func)
        self.post["nl2"] = create_activation_func(name=self.config.activation_func)

    def forward(self, input):
        """Forward pass of detector

        Args:
            x ([B, T, C]]): Input membrane waveform with B batches and T time points with C length
            p ([B, num_params]): parameters

        Returns:
            logits_myo: [B, T, output_D]
        """

        x, p = input

        B = x.shape[0]
        T = x.shape[1]
        assert T <= self.T, "The positional embedding is used, so the maximal series length is %d" % self.T

        # project input from C channels to n_embd channels
        x_proj = self.pre["input_proj"](x)
        x_proj = self.pre["nl"](x_proj)

        if self.use_pos_embedding:
            x_p = x_proj + self.pre["pos_emb"][0][:, :T, :]
            x_p = self.pre["drop"](x_p)
        else:
            x_p = x_proj + position_encoding(seq_len=T, dim_model=C, device=self.device)
            x_p = self.pre["drop"](x_p)

        p_proj = self.pre["input_proj_params"](p)
        p_proj = self.pre["nl_params"](p_proj)
        p_proj = torch.reshape(p_proj, [B, T, self.n_embd])

        #input_backbone = torch.concatenate((x_p, p_proj), dim=2)
        input_backbone = x_p + p_proj

        # go through all layers of attentions
        output_backbone = self.backbone(input_backbone)

        # project outputs to output_size channel
        y_myo = self.post["layer_norm"](output_backbone)
        y_myo = self.post["nl1"](y_myo) 

        y_myo = self.post["output_proj1_myo"](y_myo)
        y_myo = self.post["nl2"](y_myo) 
        y_myo = self.post["output_proj2_myo"](y_myo)

        return y_myo

# -------------------------------------------------------------------------------------------------

class QPerfParamsModel(ModelManager):
    """
        QPerf mapping model for parameters

        Input: aif and myo Gd curves
        Output: Fp, Vp, Visf, PS, and delay

        The architecture is quite straight-forward :

        x -> input_proj --> + --> drop_out --> attention layers one after another --> LayerNorm --> output_proj_params --> p_est --> pre-trained QPerfBTEX model --> myo_est
                            |
        pos_embedding-------|
    """
    def __init__(self, config, n_layer=8, input_D=2, output_myo_D=1, num_params=5, T=80, is_causal=False, use_pos_embedding=True, n_embd=1024, n_head=32, dropout_p=0.1, att_dropout_p=0.0, residual_dropout_p=0.1):

        self.n_layer = n_layer
        self.input_D = input_D
        self.output_myo_D = output_myo_D
        self.num_params = num_params
        self.T = T
        self.is_causal = is_causal
        self.use_pos_embedding = use_pos_embedding
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.att_dropout_p = att_dropout_p
        self.residual_dropout_p = residual_dropout_p

        super().__init__(config)

        # a good trick to count how many parameters
        logging.info("number of parameters: %d", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def create_pre(self): 
        self.pre = nn.ModuleDict()
        self.pre["input_proj"] = nn.Linear(self.input_D, self.n_embd)
        if self.use_pos_embedding:
            self.pre["pos_emb"] = torch.nn.ParameterList([nn.Parameter(torch.zeros(1, self.T, self.n_embd))])
        self.pre["drop"] = nn.Dropout(self.dropout_p)

    def create_backbone(self, channel_first=True): 
        self.backbone = nn.ModuleDict()
        self.backbone["cells"] = nn.Sequential(*[Cell(T=self.T, n_embd=self.n_embd, is_causal=self.is_causal, n_head=self.n_head, attn_dropout_p=self.att_dropout_p, residual_dropout_p=self.residual_dropout_p, activation_func=self.config.activation_func) for _ in range(self.n_layer)])

        n_embd = self.n_embd
        self.backbone["layer_norm"] = nn.LayerNorm(n_embd)
        self.backbone["output_proj1_params"] = nn.Linear(n_embd, n_embd, bias=True)
        self.backbone["output_proj2_params"] = nn.Linear(self.T*n_embd, self.num_params, bias=True)

        self.feature_channels = self.n_embd

    def create_post(self): 
        self.post = QPerfBTEXModel(config=self.config, 
                    n_layer=self.config.n_layer[1], 
                    input_D=1, 
                    output_myo_D=1, 
                    num_params=self.num_params, 
                    T=self.config.qperf_T, 
                    is_causal=False, 
                    use_pos_embedding=self.config.use_pos_embedding, 
                    n_embd=self.config.n_embd, 
                    n_head=self.config.n_head, 
                    dropout_p=self.config.dropout_p, 
                    att_dropout_p=self.config.att_dropout_p, 
                    residual_dropout_p=self.config.residual_dropout_p)

    def forward(self, x):
        """Forward pass of detector

        Args:
            x ([B, T, 2]]): Aif and myo

        Returns:
            logits_myo: [B, T, output_D]
            logits_params: [B, 5]
        """

        B, T, C = x.size()
        assert T <= self.T, "The positional embedding is used, so the maximal series length is %d" % self.T
        assert C==2

        aif = torch.unsqueeze(x[:,:,0], dim=2)

        # project input from C channels to n_embd channels
        x_proj = self.pre["input_proj"](x)

        if self.use_pos_embedding:
            x = x_proj + self.pre["pos_emb"][0][:, :T, :]
            x = self.pre["drop"](x)
        else:
            x = x_proj + position_encoding(seq_len=T, dim_model=C, device=self.device)
            x = self.pre["drop"](x)

        # go through all layers of attentions
        x = self.backbone["cells"](x)

        # project outputs to output_size channel
        x = self.backbone["layer_norm"](x)
        x = F.gelu(x, approximate='tanh')

        y_params = self.backbone["output_proj1_params"](x)
        y_params = F.gelu(y_params, approximate='tanh')
        y_params = torch.flatten(y_params, start_dim=1, end_dim=2)
        p_est = self.backbone["output_proj2_params"](y_params)

        # now send aif and p_est to post model
        myo_est = self.post((aif, p_est))

        return myo_est, p_est

# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    from setup import parse_config_and_setup_run, config_to_yaml
    
    device = get_device()
    x = torch.rand([24, 80, 2], device=device, dtype=torch.float32)

    config = parse_config_and_setup_run()

    # # -------------------------------------------------
    # m = QPerfModel(config, n_layer=16, input_D=2, output_myo_D=1, num_params=5, T=80, is_causal=False, use_pos_embedding=True, n_embd=512, n_head=32, dropout_p=0.1, att_dropout_p=0.0, residual_dropout_p=0.1)
    # m.to(device=device)
    # y_myo, y_params = m(x)

    # # -------------------------------------------------
    # m = QPerfModel_double_net(config, n_layer=[16, 12], input_D=2, output_myo_D=1, num_params=5, T=80, is_causal=False, use_pos_embedding=True, n_embd=512, n_head=[32, 32], dropout_p=[0.1, 0.1], att_dropout_p=[0.0, 0.0], residual_dropout_p=[0.1, 0.1])
    # m.to(device=device)
    # y_myo, y_params = m(x)

    # -------------------------------------------------
    m = QPerfBTEXModel(config, n_layer=16, input_D=1, output_myo_D=1, num_params=5, T=80, is_causal=False, use_pos_embedding=True, n_embd=512, n_head=32, dropout_p=0.1, att_dropout_p=0.0, residual_dropout_p=0.1)
    m.to(device=device)
    x = torch.rand([24, 80, 1], device=device, dtype=torch.float32)
    p = torch.rand([24, 5], device=device, dtype=torch.float32)
    y_myo = m((x, p))
    print(f"y_myo is {y_myo.shape}")
