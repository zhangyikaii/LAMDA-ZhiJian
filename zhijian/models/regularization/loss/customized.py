import torch
from zhijian.models.regularization.loss.l2sp import L2SP


class Customized(L2SP):
    def __init__(self, reuse_keys, alpha=0.1, beta=0.01):
        super().__init__(reuse_keys, alpha, beta)

    def forward(self, target_weights, source_features, target_features):
        loss = 0
        return loss

