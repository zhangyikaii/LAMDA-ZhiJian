import torch
import torch.nn as nn
import math
from zhijian.models.addin.module.base import AddinBase

def prepare_specific_addin_parser(parser):
    parser.add_argument('--adaptformer-reduction-dim', type=int, default=64)
    parser.add_argument('--adaptformer-layernorm-option', type=str, default='none')
    parser.add_argument('--adaptformer-scalar', type=float, default=0.1)
    parser.add_argument('--adaptformer-dropout', type=float, default=0.1)
    parser.add_argument('--adaptformer-add-residual', type=bool, default=False)
    parser.add_argument('--adaptformer-ffn-option', type=str, default='parallel')
    return parser

class AdaptFormer(AddinBase):
    def __init__(self, config, model_config):
        super().__init__()
        self.adaptformer_config = config

        self.embed_dims = model_config.hidden_size
        self.reduction_dims = config.adaptformer_reduction_dim
        self.adapter_layernorm_option = config.adaptformer_layernorm_option
        self.adapter_layer_norm_before = None

        if self.adapter_layernorm_option == "in" or self.adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.embed_dims)

        if config.adaptformer_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(config.adaptformer_scalar)

        self.down_proj = nn.Linear(self.embed_dims, self.reduction_dims)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(self.reduction_dims, self.embed_dims)

        self.add_residual = config.adaptformer_add_residual

        self.dropout = nn.Dropout(config.adaptformer_dropout)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if self.add_residual:
            output = up + residual
        else:
            output = up
        
        if self.adaptformer_config.adaptformer_ffn_option == 'parallel':
            self.inputs_cache = output + residual  
        elif self.adaptformer_config.adaptformer_ffn_option == 'sequential':
            output += self.inputs_cache
        else:
            raise ValueError(self.adaptformer_config.adaptformer_ffn_option)

        return output

    def adapt(self, module, inputs):
        return self.forward(inputs[0])