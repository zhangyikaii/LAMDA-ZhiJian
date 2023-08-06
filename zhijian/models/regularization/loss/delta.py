import torch
from zhijian.models.regularization.loss.l2sp import L2SP


class DELTA(L2SP):
    def __init__(self, reuse_keys, alpha=0.1, beta=0.01):
        super().__init__(reuse_keys, alpha, beta)

    def reg_classifier(self, target_weights):
        l2_cls = 0
        for name in target_weights.keys():
            if self._is_reuse_keys(name):
                l2_cls += 0.5 * torch.norm(target_weights[name]) ** 2
        return l2_cls

    def reg_fea_map(self, source_features, target_features):
        loss = 0
        for fm_src, fm_tgt in zip(source_features, target_features):
            loss += 0.5 * (torch.norm(fm_tgt - fm_src) ** 2)
        return loss

    def forward(self, target_weights, source_features, target_features):
        return self.alpha * self.reg_fea_map(source_features, target_features) + self.beta * self.reg_classifier(target_weights)
