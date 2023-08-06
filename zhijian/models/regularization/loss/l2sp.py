import torch.nn as nn


class L2SP(nn.Module):
    def __init__(self, reuse_keys, alpha=0.1, beta=0.01):
        super().__init__()
        self.reuse_keys = ['.'.join([f'{j}' for j in i]) for i in reuse_keys]
        def _is_reuse_keys(cur_key):
            for i in self.reuse_keys:
                if i in cur_key:
                    return True
            return False
        self._is_reuse_keys = _is_reuse_keys
        self.alpha = alpha
        self.beta = beta

    def forward(self, source_weights, target_weights):
        loss = 0
        for name in target_weights.keys():
            if 'head' in name or target_weights[name].shape != source_weights[name].shape:
                loss += 0.5 * self.beta * target_weights[name].norm(2)**2
            elif self._is_reuse_keys(name):
                loss += 0.5 * self.alpha * (target_weights[name] - source_weights[name]).norm(2)**2
        return loss
