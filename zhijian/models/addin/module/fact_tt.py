import torch
import torch.nn as nn

from zhijian.models.addin.module.base import AddinBase


def prepare_specific_addin_parser(parser):
    parser.add_argument('--fact-tt-scale', type=float, default=1)
    parser.add_argument('--fact-tt-dim', type=int, default=8)
    return parser

class Sub_FacT_tt(nn.Module):
    def __init__(self, config, model_config, **kwargs):
        super().__init__()

        self.config = config
        self.dim = self.config.fact_tt_dim
        self.model_config = model_config
        self.embed_dims = model_config.hidden_size

        self.q_FacTs = nn.Linear(self.dim, self.dim, bias=False)
        self.k_FacTs = nn.Linear(self.dim, self.dim, bias=False)
        self.v_FacTs = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_FacTs = nn.Linear(self.dim, self.dim, bias=False)
        self.attn_dp = nn.Dropout(0.1)

        self.fc1_FacTs = nn.Linear(self.dim, self.dim * 4, bias=False)
        self.fc2_FacTs = nn.Linear(4 * self.dim, self.dim, bias=False)
        self.mlp_dp = nn.Dropout(0.1)   
        self.B = 0
        self.N = 0
        self.C = 0


class FacT_tt(AddinBase):
    def __init__(self, config, model_config, **kwargs):
        super().__init__()
        self.config = config
        self.dim = self.config.fact_tt_dim
        self.model_config = model_config
        self.embed_dims = model_config.hidden_size

        self.FacTu = nn.Linear(self.embed_dims, self.dim, bias=False)
        self.FacTv = nn.Linear(self.dim, self.embed_dims, bias=False)
        nn.init.zeros_(self.FacTv.weight)
        self.attn_s = self.config.fact_tt_scale
        self.mlp_s = self.config.fact_tt_scale

        self.sub_module = nn.ModuleList()
        for _ in range(self.model_config.transformer_num_layers):
            self.sub_module.append(Sub_FacT_tt(config, model_config, **kwargs))

        for i in range(model_config.transformer_num_layers):
            setattr(self, f'adapt_attn_1_{i}', self.get_adapt_attn_1(i))
            setattr(self, f'adapt_attn_2_{i}', self.get_adapt_attn_2(i))
            setattr(self, f'mlp_attn_1_{i}', self.get_mlp_attn_1(i))
            setattr(self, f'mlp_attn_2_{i}', self.get_mlp_attn_2(i))


    def get_adapt_attn_1(self, idx):
        def core(module, inputs, outputs):
            x = inputs[0]
            q = self.FacTv(self.sub_module[idx].attn_dp(self.sub_module[idx].q_FacTs(self.FacTu(x))))
            k = self.FacTv(self.sub_module[idx].attn_dp(self.sub_module[idx].k_FacTs(self.FacTu(x))))
            v = self.FacTv(self.sub_module[idx].attn_dp(self.sub_module[idx].v_FacTs(self.FacTu(x))))
            outputs += torch.cat([q, k, v], dim = 2) * self.attn_s
            return outputs
        return core

    def get_adapt_attn_2(self, idx):
        def core(module, inputs, outputs):
            x = inputs[0]
            outputs += self.FacTv(self.sub_module[idx].attn_dp(self.sub_module[idx].proj_FacTs(self.FacTu(x)))) * self.attn_s
            return outputs
        return core

    def get_adapt_mlp_1(self, idx):
        def core(module, inputs, outputs):
            x = inputs[0]
            self.sub_module[idx].B, self.sub_module[idx].N, self.sub_module[idx].C = x.shape
            outputs += self.FacTv(self.sub_module[idx].mlp_dp(self.sub_module[idx].fc1_FacTs(self.FacTu(x))).reshape(
            self.sub_module[idx].B, self.sub_module[idx].N, 4, self.dim)).reshape(
            self.sub_module[idx].B, self.sub_module[idx].N, 4 * self.sub_module[idx].C) * self.mlp_s
            return outputs
        return core

    def get_adapt_mlp_2(self, idx):
        def core(module, inputs, outputs):
            x = inputs[0]
            x = x.reshape(self.sub_module[idx].B, self.sub_module[idx].N, 4, self.sub_module[idx].C)
            outputs += self.FacTv(self.sub_module[idx].mlp_dp(self.sub_module[idx].fc2_FacTs(self.FacTu(x).reshape(
            self.sub_module[idx].B, self.sub_module[idx].N, 4 * self.dim)))) * self.mlp_s
            return outputs
        return core
