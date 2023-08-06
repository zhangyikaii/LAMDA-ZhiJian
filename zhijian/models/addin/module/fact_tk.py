import torch
import torch.nn as nn

from zhijian.models.addin.module.base import AddinBase


def prepare_specific_addin_parser(parser):
    parser.add_argument('--fact-tk-scale', type=float, default=1)
    parser.add_argument('--fact-tk-dim', type=int, default=8)
    return parser

class FacT_tk(AddinBase):
    def __init__(self, config, model_config, **kwargs):
        super().__init__()
        self.config = config
        self.s = self.config.fact_tk_scale
        self.dim = self.config.fact_tk_dim
        self.model_config = model_config
        self.embed_dims = model_config.hidden_size

        self.FacTu = nn.Linear(self.embed_dims, self.dim, bias=False)
        self.FacTv = nn.Linear(self.dim, self.embed_dims, bias=False)
        nn.init.zeros_(self.FacTv.weight)
        self.FacTp = nn.Parameter(torch.zeros([self.dim, 12 * self.model_config.transformer_num_layers], dtype=torch.float))
        nn.init.xavier_uniform_(self.FacTp)
        self.FacTc = nn.Parameter(torch.zeros([self.dim, self.dim, self.dim], dtype=torch.float))
        nn.init.xavier_uniform_(self.FacTc)   
        self.dp = nn.Dropout(0.1)

        for i in range(model_config.transformer_num_layers):
            setattr(self, f'adapt_attn_1_{i}', self.get_adapt_attn_1(i))
            setattr(self, f'adapt_attn_2_{i}', self.get_adapt_attn_2(i))
            setattr(self, f'mlp_attn_1_{i}', self.get_mlp_attn_1(i))
            setattr(self, f'mlp_attn_2_{i}', self.get_mlp_attn_2(i))


    def get_adapt_attn_1(self, idx):
        def core(module, inputs, outputs):
            x = inputs[0]
            FacTc = self.FacTc @ self.FacTp[:, 12 * idx : 12 * idx + 4]
            q_FacTc, k_FacTc, v_FacTc = FacTc[:, :, 0], FacTc[:, :, 1], FacTc[:, :, 2]
        
            q = self.FacTv(self.dp(self.FacTu(x) @ q_FacTc))
            k = self.FacTv(self.dp(self.FacTu(x) @ k_FacTc))
            v = self.FacTv(self.dp(self.FacTu(x) @ v_FacTc))
            outputs += torch.cat([q, k, v], dim=2) * self.s
            return outputs
        return core

    def get_adapt_attn_2(self, idx):
        def core(module, inputs, outputs):
            x = inputs[0]
            FacTc = self.FacTc @ self.FacTp[:, 12 * idx : 12 * idx + 4]
            proj_FacTc = FacTc[:, :, 3]
            outputs += self.FacTv(self.dp(self.FacTu(x) @ proj_FacTc)) * self.s
        return core

    def get_adapt_mlp_1(self, idx):
        def core(module, inputs, outputs):
            x = inputs[0]
            self.B, self.N, self.C = x.shape
            FacTc = self.FacTc @ self.FacTp[:, 12 * idx + 4 : 12 * (idx + 1)]
            fc1_FacTc = FacTc[:, :, :4].reshape(self.dim, self.dim * 4)
            outputs += self.FacTv(self.dp(self.FacTu(x) @ fc1_FacTc).reshape(self.B, self.N, 4, self.dim)).reshape(self.B, self.N, 4 * self.C) * self.s
            return outputs
        return core    

    def get_adapt_mlp_2(self, idx):
        def core(module, inputs, outputs):
            x = inputs[0]
            x = x.reshape(self.B, self.N, 4, self.C)
            FacTc = self.FacTc @ self.FacTp[:, 12 * idx + 4 : 12 * (idx + 1)]
            fc2_FacTc = FacTc[:, :, 4:].reshape(self.dim, self.dim * 4)
            outputs += self.FacTv(self.dp(self.FacTu(x).reshape(self.B, self.N, 4 * self.dim) @ fc2_FacTc.t())) * self.s
            return outputs
        return core
