import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from functools import reduce

import math

from operator import mul


from zhijian.models.addin.module.base import AddinBase

def prepare_specific_addin_parser(parser):
    parser.add_argument('--vpt-initiation', type=str, default='random')
    parser.add_argument('--vpt-project', type=int, default=-1)
    parser.add_argument('--vpt-deep', type=bool, default=False)
    parser.add_argument('--vpt-dropout', type=float, default=0.5)
    parser.add_argument('--vpt-num-tokens', type=int, default=3)

    return parser

class VPT(AddinBase):
    def __init__(self, config, model_config, **kwargs):
        super().__init__()
        from torch.nn import Dropout
        patch_size = _pair(model_config.patches_size)

        self.num_tokens = config.vpt_num_tokens
        self.prompt_dropout = Dropout(config.vpt_dropout)

        if config.vpt_project > -1:
            prompt_dim = config.vpt_project
            self.prompt_proj = nn.Linear(
                prompt_dim, model_config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = model_config.hidden_size
            self.prompt_proj = nn.Identity()

        if config.vpt_initiation == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_tokens, prompt_dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if config.vpt_deep:
                total_d_layer = model_config.transformer_num_layers-1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.num_tokens, prompt_dim))
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")


        for i in range(1, model_config.transformer_num_layers):
            setattr(self, f'cat_prompt_deep_{i}', self.get_cat_prompt_deep(i - 1))


    def cat_prompt_shallow(self, module, inputs):
        x = inputs[0]
        B = x.shape[0]
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)

        return x

    def get_cat_prompt_deep(self, cur_depth):
        def core(module, inputs):
            B = inputs[0].shape[0]

            hidden_states = inputs[0]
            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                self.deep_prompt_embeddings[cur_depth]).expand(B, -1, -1)
                )

            hidden_states = torch.cat((
                hidden_states[:, :1, :],
                deep_prompt_emb,
                hidden_states[:, (1+self.num_tokens):, :]
            ), dim=1)

            return hidden_states
        return core

    def get_log_str(self):
        return f'{self.num_tokens}'