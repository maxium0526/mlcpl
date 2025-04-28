import torch
from torch import nn
from ..label_strategies import *

class LargeLossRejection(nn.Module):
    def __init__(self, delta_rel=0.1, reduction='none'):
        super(LargeLossRejection, self).__init__()
        self.delta_rel = delta_rel
        self.reduction = reduction

    def forward(self, logits, targets, epoch):
        losses = nn.functional.binary_cross_entropy_with_logits(logits, unknown_to_negative(targets), reduction='none')

        with torch.no_grad():
            unknown_label_losses = losses * torch.isnan(targets)
            percent = epoch * self.delta_rel
            percent = 1 if percent > 1 else percent

            k = round(torch.count_nonzero(unknown_label_losses).cpu().numpy() * percent)
            k = 1 if k == 0 else k
            
            loss_threshold = torch.topk(unknown_label_losses.flatten(), k).values.min()

            lambdas = torch.where(unknown_label_losses > loss_threshold, 0, 1)

        final_loss = losses * lambdas

        if self.reduction == 'mean':
            return torch.mean(final_loss)
        if self.reduction == 'sum':
            return torch.sum(final_loss)

        return final_loss