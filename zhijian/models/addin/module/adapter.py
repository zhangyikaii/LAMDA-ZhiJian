import torch
import torch.nn as nn

from zhijian.models.addin.module.base import AddinBase


def prepare_specific_addin_parser(parser):
    parser.add_argument('--adapter-dropout', type=float, default=0)
    parser.add_argument('--adapter-reduction-dim', type=int, default=8)
    return parser

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Adapter(AddinBase):
    def __init__(self, config, model_config):
        super(Adapter, self).__init__()

        self.config = config
        self.embed_dims = model_config.hidden_size

        self.reduction_dim = self.config.adapter_reduction_dim
        self.adapter_dropout = nn.Dropout(self.config.adapter_dropout)

        if self.reduction_dim > 0:
            self.adapter_ln1 = nn.Linear(self.embed_dims, self.reduction_dim)
            nn.init.xavier_uniform_(self.adapter_ln1.weight)
            nn.init.normal_(self.adapter_ln1.bias, std=1e-6)
            self.adapter_activate = QuickGELU()
            self.adapter_ln2 = nn.Linear(self.reduction_dim, self.embed_dims)
            nn.init.xavier_uniform_(self.adapter_ln2.weight)
            nn.init.normal_(self.adapter_ln2.bias, std=1e-6)

    def forward(self, x):
        identity = x 
        out = self.adapter_ln1(identity)
        out = self.adapter_activate(out)
        out = self.adapter_dropout(out)
        out = self.adapter_ln2(out)

        return identity + out

    def adapt(self, module, inputs, outputs):

        return self.forward(outputs)
