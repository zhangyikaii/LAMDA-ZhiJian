import torch
import torch.nn as nn

from timm.models.layers import DropPath

from zhijian.models.addin.module.base import AddinBase

def prepare_specific_addin_parser(parser):
    parser.add_argument('--convpass-dim', type=int, default=8)
    parser.add_argument('--convpass-scale', type=float, default=0.01)
    parser.add_argument('--convpass-drop-path', type=float, default=0)
    parser.add_argument('--convpass-xavier-init', type=bool, default=True)
    return parser

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Convpass(AddinBase):
    def __init__(self, config, model_config, **kwargs):
        super().__init__()
        from torch.nn import Dropout
        self.config = config
        self.model_config = model_config
        self.dim = config.convpass_dim
        self.xavier_init = config.convpass_xavier_init
        self.embed_dims = model_config.hidden_size
        self.s = config.convpass_scale

        self.adapter_conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
        if self.xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)
        self.adapter_down = nn.Linear(self.embed_dims, self.dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(self.dim, self.embed_dims)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = QuickGELU()
        self.dropout = Dropout(0.1)
        self.drop_path = DropPath(self.config.convpass_drop_path) if self.config.convpass_drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        output = self.drop_path(x_up) * self.s

        return output

    def adapt(self, module, inputs, outputs):
        outputs = outputs + self.forward(self.inputs_cache)

        return outputs

