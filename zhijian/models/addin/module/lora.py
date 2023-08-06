import torch.nn as nn
import math

from zhijian.models.addin.module.base import AddinBase


def prepare_specific_addin_parser(parser):
    parser.add_argument('--lora-dropout', type=float, default=0.5)
    parser.add_argument('--lora-dim', type=int, default=8)
    return parser

class LoRA(AddinBase):
    def __init__(self,config, model_config):
        super().__init__()
        self.lora_config = config
        self.embed_dims = model_config.hidden_size

        self.lora_dim = self.lora_config.lora_dim

        self.lora_drop = nn.Dropout(self.lora_config.lora_dropout)

        self.lora_a = nn.Linear(self.embed_dims, self.lora_dim, bias = False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        self.lora_b = nn.Linear(self.lora_dim, self.embed_dims*3, bias = False)
        nn.init.zeros_(self.lora_b.weight)


    def adapt(self, module, inputs, outputs):
        x = inputs[0]
        B, N, C = x.shape

        qkv_delta = self.lora_a(x)
        qkv_delta = self.lora_b(qkv_delta)
        outputs = outputs + qkv_delta

        return outputs
    
