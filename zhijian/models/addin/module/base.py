import torch.nn as nn

def prepare_specific_addin_parser(parser):
    return parser

class AddinBase(nn.Module):
    def __init__(self, config=None, model_config=None, **kwargs):
        super(AddinBase, self).__init__()
        self.inputs_cache = None

    def get_pre(self, module, inputs):
        self.inputs_cache = inputs[0]
        return inputs[0]

    def get_post(self, module, inputs, output):
        self.inputs_cache = output
        return output

    def add_pre(self, module, inputs):
        return inputs[0] + self.inputs_cache

    def add_post(self, module, inputs, output):
        return output + self.inputs_cache
    
    def get_log_str(self):
        return ''
