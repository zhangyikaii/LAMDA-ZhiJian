import torch
import torch.nn as nn

from zhijian.models.addin.module.base import AddinBase


def prepare_specific_addin_parser(parser):
    return parser

def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    device = torch.device('cuda')
    if x.shape[-1] == scale.shape[0]:
        return x * scale.to(device) + shift.to(device)
    elif x.shape[1] == scale.shape[0]:
        return x * scale.to(device).view(1, -1, 1, 1) + shift.to(device).view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

class SSF(AddinBase):
    def __init__(self, config, model_config, **kwargs):
        super().__init__()
        dim = kwargs['dim'] if 'dim' in kwargs.keys() else model_config.hidden_size

        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))


    def init_ssf_scale_shift(self, dim):
        scale = nn.Parameter(torch.ones(dim))
        shift = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(scale, mean=1, std=.02)
        nn.init.normal_(shift, std=.02)

        return scale, shift

    def adapt_input(self, module, inputs):
        x = inputs[0]
        if self.scale.shape[0] != x.shape[-1]:
            self.scale, self.shift = self.init_ssf_scale_shift(x.shape[-1])
        return ssf_ada(x, self.scale, self.shift)

    def adapt_output(self, module, inputs, outputs):
        x = outputs
        if self.scale.shape[0] != x.shape[-1]:
            self.scale, self.shift = self.init_ssf_scale_shift(x.shape[-1])

        return ssf_ada(x, self.scale, self.shift)

    def adapt_output_post(self, module, inputs, outputs):
        x = outputs
        if self.scale.shape[0] != x.shape[-1]:
            self.scale, self.shift = self.init_ssf_scale_shift(x.shape[-1])
        if isinstance(module, nn.Identity):
            return outputs
        else:
            return ssf_ada(x, self.scale, self.shift)
