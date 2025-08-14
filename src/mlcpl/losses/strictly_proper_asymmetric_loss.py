import torch
from torch import nn as nn
from torch.nn.functional import sigmoid as sigmoid

class StrictlyProperAsymmetricWithLogitLoss(nn.Module):
    def __init__(self, zeta_p=1, k_p=1, b_p=0, zeta_n=5, k_n=3, b_n=1, reduction='mean'):
        super(StrictlyProperAsymmetricWithLogitLoss, self).__init__()
        self.zeta_p = zeta_p
        self.k_p = k_p
        self.b_p = b_p
        self.zeta_n = zeta_n
        self.k_n = k_n
        self.b_n = b_n

        self.reduction = reduction

    def forward(self, preds, target):
        # inverse link
        preds = 1 / (1+((self.k_p * self.zeta_n * sigmoid(-self.k_p * (preds - self.b_p)))/(self.k_n * self.zeta_p * sigmoid(self.k_n * (preds - self.b_n)))))

        # loss
        loss_p = - torch.log(sigmoid(self.k_p * (preds - self.b_p))) / self.zeta_p
        loss_n = - torch.log(sigmoid(- self.k_n * (preds - self.b_n))) / self.zeta_n
        loss = target * loss_p + (1-target) * loss_n

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss

class PartialStrictlyProperAsymmetricWithLogitLoss(nn.Module):
    """ The strictly proper asymmetric loss that ignores unknown labels (0 gradient) (https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html).

    Args:
        zeta_p (int, optional): A hyperparameter. Defaults to 1.

        k_p (int, optional): A hyperparameter. Defaults to 1.

        b_p (int, optional): A hyperparameter. Defaults to 0.

        zeta_n (int, optional): A hyperparameter. Defaults to 5.

        k_n (int, optional): A hyperparameter. Defaults to 3.

        b_n (int, optional): A hyperparameter. Defaults to 1.
        
        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.
    """
    def __init__(self, zeta_p=1, k_p=1, b_p=0, zeta_n=5, k_n=3, b_n=1, reduction='mean'):
        super(PartialStrictlyProperAsymmetricWithLogitLoss, self).__init__()
        self.loss_fn = StrictlyProperAsymmetricWithLogitLoss(zeta_p=zeta_p, k_p=k_p, b_p=b_p, zeta_n=zeta_n, k_n=k_n, b_n=b_n, reduction=reduction)

    def forward(self, preds, target):
        preds, target = preds.flatten(), target.flatten()
        
        labeled_map = ~torch.isnan(target)
        preds, target = preds[labeled_map], target[labeled_map]

        return self.loss_fn(preds, target)