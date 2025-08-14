import torch
from torch import nn
from . import PartialNegativeBCEWithLogitLoss

class LargeLossRejection(nn.Module):
    """The Large Loss-Reject in https://arxiv.org/abs/2206.03740.

    Args:
        loss_fn (Callable, optional): The base loss function. Defaults to PartialNegativeBCEWithLogitLoss(reduction=None).

        delta_rel (float, optional): The delta_rel hyperparameter. Defaults to 0.1.

        reduction (str, optional): reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.
    """
    def __init__(self, loss_fn=PartialNegativeBCEWithLogitLoss(reduction=None), delta_rel=0.1, reduction='mean'):
        
        super(LargeLossRejection, self).__init__()
        self.delta_rel = delta_rel
        self.loss_fn = loss_fn
        self.reduction = reduction

    def forward(self, logits, targets, epoch):
        losses = self.loss_fn(logits, targets)

        unknown_label_losses = losses * torch.isnan(targets)
        percent = epoch * self.delta_rel / 100
        percent = 1 if percent > 1 else percent

        k = round(torch.count_nonzero(unknown_label_losses).cpu().detach().numpy() * percent)
        k = 1 if k == 0 else k
        
        loss_threshold = torch.topk(unknown_label_losses.flatten(), k).values.min()

        lambdas = torch.where(unknown_label_losses > loss_threshold, 0, 1)

        final_loss = losses * lambdas

        if self.reduction == 'sum':
            return torch.sum(final_loss)
        elif self.reduction == 'mean':
            return torch.mean(final_loss)

        return final_loss
    
class LargeLossCorrectionTemporary(nn.Module):
    """The Large Loss-Correct (temporary) in https://arxiv.org/abs/2206.03740.

    Args:
        loss_fn (Callable, optional): The base loss function. Defaults to PartialNegativeBCEWithLogitLoss(reduction=None).

        delta_rel (float, optional): The delta_rel hyperparameter. Defaults to 0.1.
        
        reduction (str, optional): reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.
    """
    def __init__(self, loss_fn=PartialNegativeBCEWithLogitLoss(reduction=None), delta_rel=0.1, reduction='mean'):
        
        super(LargeLossCorrectionTemporary, self).__init__()
        self.delta_rel = delta_rel
        self.loss_fn = loss_fn
        self.reduction = reduction

    def forward(self, logits, targets, epoch):
        with torch.no_grad():
            losses = self.loss_fn(logits, targets)

            unknown_label_losses = losses * torch.isnan(targets)
            percent = epoch * self.delta_rel / 100
            percent = 1 if percent > 1 else percent

            k = round(torch.count_nonzero(unknown_label_losses).cpu().detach().numpy() * percent)
            k = 1 if k == 0 else k
            
            loss_threshold = torch.topk(unknown_label_losses.flatten(), k).values.min()

            lambdas = torch.where(unknown_label_losses > loss_threshold, 0, 1)

            change_label = torch.logical_and(torch.logical_not(lambdas), torch.isnan(targets))

            new_targets = torch.where(change_label, 1, targets)

        final_loss = self.loss_fn(logits, new_targets)

        if self.reduction == 'sum':
            return torch.sum(final_loss)
        elif self.reduction == 'mean':
            return torch.mean(final_loss)

        return final_loss