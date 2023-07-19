import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import math
from functools import reduce
from operator import mul
from timm.models.layers import Mlp

from zhijian.models.addin.module.base import AddinBase

def prepare_specific_addin_parser(parser):
    parser.add_argument('--vqt-dropout', type=float, default=0)
    parser.add_argument('--vqt-num-query-tokens', type=int, default=3)
    parser.add_argument('--vqt-pool-feats', action='store_true', default=False)
    parser.add_argument('--vqt-weighted-sum-feats', action='store_true', default=False)
    parser.add_argument('--vqt-norm-feats', action='store_true', default=False)

    return parser

class VQT(AddinBase):
    def __init__(self, config, model_config, **kwargs):
        super().__init__()
        self.num_query_tokens = config.vqt_num_query_tokens
        self.prompt_dropout = nn.Dropout(config.vqt_dropout)

        self.norm_feats = config.vqt_norm_feats
        self.pool_feats = config.vqt_pool_feats
        self.weighted_sum_feats = config.vqt_weighted_sum_feats
        self.query_prompt = []
        self.query_outputs = []
        self.head = Mlp(
                model_config.hidden_size + model_config.hidden_size*self.num_query_tokens*model_config.transformer_num_layers,
                out_features=model_config.head_output
            )
        if self.num_query_tokens > 0:
            patch_size = _pair(model_config.patches_size)
            self.patch_nums = patch_size[0]*patch_size[1] + 1 
            self.query_prompt_embeddings = nn.Parameter(torch.zeros(
                model_config.transformer_num_layers, self.num_query_tokens, model_config.hidden_size))
            prompt_dim = model_config.hidden_size
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            nn.init.uniform_(self.query_prompt_embeddings.data, -val, val)
        else:
            self.register_parameter('query_prompt_embeddings', None)
        
        self.num_heads = model_config.num_heads
        self.head_dim = model_config.head_dim

        for i in range(model_config.transformer_num_layers):
            setattr(self, f'cat_query_prompt_1_{i}', self.cat_query_prompt_1(i))
            setattr(self, f'cat_query_prompt_2_{i}', self.cat_query_prompt_2(i))
            setattr(self, f'get_query_prompt_{i}', self.get_query_prompt(i))

    def cat_query_prompt_1(self, cur_depth):
        def core(module, inputs):
            hidden_states = inputs[0]
            B = hidden_states.shape[0]
            if self.query_prompt_embeddings is not None:
                q_states = self.prompt_dropout(
                        self.query_prompt_embeddings[cur_depth].expand(B, -1, -1))
                hidden_states = torch.cat([q_states, hidden_states], dim=1)
            return hidden_states
        return core

    def divide_query_prompt(self, module, inputs):
        x = inputs[0]
        x = x[:, self.num_query_tokens:, :]
        return x

    def get_query_output(self, module, inputs, outputs):
        x = outputs
        self.query_outputs.append(x[:, :self.num_query_tokens, :])
        x = x[:, self.num_query_tokens:, :]
        return x

    def get_query_prompt(self, cur_depth):
        def core(module, inputs, outputs):
            x = inputs[0]
            B,N,C = inputs[0].shape
            q_states = x[:, :self.num_query_tokens, :]
            prompt_value = module.forward(q_states).reshape(B, self.num_query_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            query_prompt, _, _ = prompt_value.unbind(0)

            qkv = outputs.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            _, k, v = qkv.unbind(0)
            scale = self.head_dim ** -0.5
            query_prompt = query_prompt * scale
            attn = query_prompt @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v

            self.query_prompt.append(x.transpose(1, 2).reshape(B, self.num_query_tokens, C))

            return outputs
        
        return core

    def cat_query_prompt_2(self, cur_depth):
        def core(module, inputs):
            hidden_states = inputs[0]
            x = self.query_prompt[cur_depth]
            hidden_states = torch.cat([x.cuda(), hidden_states], dim=1)

            return hidden_states
        return core

    def make_logits(self, module, inputs, outputs):
        x = inputs[0]
        B = x.shape[0]
        included_features = [x]
        for l_idx, q in enumerate(self.query_outputs):
            if self.pool_feats:
                included_features.append(q.mean(dim=1))
            elif self.weighted_sum_feats:
                included_features.append(
                        torch.sum(self.combine_params[l_idx].expand(B, -1).unsqueeze(-1)*q, dim=1))
            else:
                included_features.append(q.view(B, -1))
        self.query_outputs = []
        self.query_prompt = []
        if self.norm_feats:
            included_features = [F.normalize(x) for x in included_features]

        feats = torch.cat(included_features, dim=1)
        outputs = self.head(feats)
        return outputs