import torch.nn as nn
import torch.nn.functional as F

class Customized(nn.Module):
    def __init__(self, **kwargs):
        self.T = kwargs['temperature']

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss
