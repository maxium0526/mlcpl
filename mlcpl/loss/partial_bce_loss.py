import torch
from torch import nn

class PartialBCE(nn.Module):
    def __init__(self, alpha=-4.45, beta=5.45, gamma=1.00):
        super(PartialBCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y):
        partial_ratio = 1 - torch.sum(torch.isnan(y), dim=1) / y.shape[1]
        y_ignore = torch.where(torch.isnan(y), nn.functional.sigmoid(x), y)
        bce_loss = torch.mean(nn.functional.binary_cross_entropy_with_logits(x, y_ignore, reduction='none'), dim=1)
        g = self.alpha * (torch.pow(partial_ratio, self.gamma)) + self.beta
        partial_bce_loss = bce_loss * g
        return torch.mean(partial_bce_loss)