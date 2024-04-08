import torch
from torch import nn
from mlcpl.label_strategy import *
from .loss import PartialNegativeBCELoss

class LargeLossRejection(nn.Module):
    def __init__(self, loss_fn=PartialNegativeBCELoss(reduction=None), delta_rel=0.1):
        super(LargeLossRejection, self).__init__()
        self.delta_rel = delta_rel
        self.loss_fn = loss_fn

    def forward(self, logits, targets, epoch):
        losses = self.loss_fn(logits, targets)

        unknown_label_losses = losses * torch.isnan(targets)
        percent = epoch * self.delta_rel / 100
        percent = 1 if percent > 1 else percent

        k = round(torch.count_nonzero(unknown_label_losses).cpu().detach().numpy() * percent)
        k = 1 if k == 0 else k
        
        loss_threshold = torch.topk(unknown_label_losses.flatten(), k).values.min()

        lambdas = torch.where(unknown_label_losses > loss_threshold, 0, 1)

        final_loss = torch.sum(losses * lambdas)

        return final_loss