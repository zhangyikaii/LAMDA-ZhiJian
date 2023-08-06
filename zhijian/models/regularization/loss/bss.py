import torch
import torch.nn as nn


class BSS(nn.Module):
    def __init__(self, alpha=0.001):
        super().__init__()
        self.alpha = alpha

    def forward(self, target_features):
        target_features = target_features.view(target_features.shape[0], -1)
        u, s, v = torch.svd(target_features.t())
        loss = 0
        ll = s.size(0)
        for i in range(1):
            loss += torch.pow(s[ll-1-i], 2)
        return self.alpha * loss
