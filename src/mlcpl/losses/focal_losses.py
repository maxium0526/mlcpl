import torch
from torch import Tensor
from torch import nn as nn
from typing import Literal, Callable

def PartialNegativeBCEWithLogitLoss(
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
):
    """ Get the binary crossentropy loss function that treats unknown labels as negative.

    Args:
        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos),
        lossfn_neg = FocalLossTerm(alpha_neg),

        partial_loss_mode = 'negative',
        normalize = normalize,
        reduction = reduction,
    )

def PartialBCEWithLogitLoss(
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
):
    """ Get the binary crossentropy loss function that ignores unknown labels (0 gradient).

    Args:
        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos),
        lossfn_neg = FocalLossTerm(alpha_neg),

        partial_loss_mode = 'ignore',
        normalize = normalize,
        reduction = reduction,
    )

def PartialSelectiveBCEWithLogitLoss(
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
    label_priors: Tensor = None,
    likelihood_topk: int = 5,
    prior_threshold: float = 0.05,
):
    """ Get the class-aware selective loss function with binary crossentropy (https://arxiv.org/abs/2110.10955).

    Args:
        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

        label_priors (Tensor, optional): The label priors of the class-aware selective loss. Default to ``None``.

        likelihood_topk (int, optional): The Top-K likelihood of the class-aware selective loss. Default to ``5``.

        prior_threshold (float, optional): The threshold that determine the mode used for each category. Default to ``0.05``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos),
        lossfn_neg = FocalLossTerm(alpha_neg),

        partial_loss_mode = 'selective',
        normalize = normalize,
        reduction = reduction,
        
        label_priors = label_priors,
        likelihood_topk = likelihood_topk,
        prior_threshold = prior_threshold,
    )

def PartialNegativeFocalWithLogitLoss(
    gamma: float = 1,
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    discard_focal_grad: bool = True,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
):
    """ Get the focal loss function that treats unknown labels as negative.

    Args:
        gamma (float, optional): The focal term. Defaults to 1.

        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        discard_focal_grad: Whether discarding the gradient of the focal term. Defaults to True.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'negative',
        normalize = normalize,
        reduction = reduction,
    )

def PartialFocalWithLogitLoss(
    gamma: float = 1,
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    discard_focal_grad: bool = True,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
):
    """ Get the focal loss function that ignores unknown labels (0 gradient).

    Args:
        gamma (float, optional): The focal term. Defaults to 1.

        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        discard_focal_grad: Whether discarding the gradient of the focal term. Defaults to True.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'ignore',
        normalize = normalize,
        reduction = reduction,
    )

def PartialSelectiveFocalWithLogitLoss(
    gamma: float = 1,
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    discard_focal_grad: bool = True,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
    label_priors: Tensor = None,
    likelihood_topk: int = 5,
    prior_threshold: float = 0.05,
):
    """ Get the class-aware selective loss function with focal (https://arxiv.org/abs/2110.10955).

    Args:
        gamma (float, optional): The focal term. Defaults to 1.

        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        discard_focal_grad: Whether discarding the gradient of the focal term. Defaults to True.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

        label_priors (Tensor, optional): The label priors of the class-aware selective loss. Default to ``None``.

        likelihood_topk (int, optional): The Top-K likelihood of the class-aware selective loss. Default to ``5``.

        prior_threshold (float, optional): The threshold that determine the mode used for each category. Default to ``0.05``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'selective',
        normalize = normalize,
        reduction = reduction,
        
        label_priors = label_priors,
        likelihood_topk = likelihood_topk,
        prior_threshold = prior_threshold,
    )

def PartialNegativeAsymmetricWithLogitLoss(
    clip: float = 0,
    gamma_pos: float = 0,
    gamma_neg: float = 1,
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    discard_focal_grad: bool = True,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
):
    """ Get the asymmetric loss function that treats unknown labels as negative (https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html).

    Args:
        clip (float, optional): The threshold that discards easy negative labels.

        gamma_pos (float, optional): The focal term of the positive loss term. Defaults to 1.

        gamma_neg (float, optional): The focal term of the negative loss term. Defaults to 1.

        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        discard_focal_grad: Whether discarding the gradient of the focal term. Defaults to True.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma_pos, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma_neg, clip, discard_focal_grad=discard_focal_grad),
        
        partial_loss_mode = 'negative',
        normalize = normalize,
        reduction = reduction,
    )

def PartialAsymmetricWithLogitLoss(
    clip: float = 0,
    gamma_pos: float = 0,
    gamma_neg: float = 1,
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    discard_focal_grad: bool = True,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
):
    """ Get the asymmetric loss function that ignores unknown labels (0 gradient) (https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html).

    Args:
        clip (float, optional): The threshold that discards easy negative labels.

        gamma_pos (float, optional): The focal term of the positive loss term. Defaults to 1.

        gamma_neg (float, optional): The focal term of the negative loss term. Defaults to 1.

        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        discard_focal_grad: Whether discarding the gradient of the focal term. Defaults to True.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma_pos, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma_neg, clip, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'ignore',
        normalize = normalize,
        reduction = reduction,
    )

def PartialSelectiveAsymmetricWithLogitLoss(
    clip: float = 0,
    gamma_pos: float = 0,
    gamma_neg: float = 1,
    alpha_pos: float = 1,
    alpha_neg: float = 1,
    normalize: bool = False,
    discard_focal_grad: bool = True,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
    label_priors: Tensor = None,
    likelihood_topk: int = 5,
    prior_threshold: float = 0.05,
):
    """ Get the class-aware selective loss function with asymmetric (https://arxiv.org/abs/2110.10955).

    Args:
        clip (float, optional): The threshold that discards easy negative labels.

        gamma_pos (float, optional): The focal term of the positive loss term. Defaults to 1.

        gamma_neg (float, optional): The focal term of the negative loss term. Defaults to 1.

        alpha_pos (float, optional): The weight of the positive loss term. Defaults to 1.

        alpha_neg (float, optional): The weight of the negative loss term. Defaults to 1.
        
        normalize (bool, optional): Whether applying normalization to the losses (https://arxiv.org/pdf/1902.09720.pdfs). Defaults to False.

        discard_focal_grad: Whether discarding the gradient of the focal term. Defaults to True.

        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of elements in the output, ``'sum'``: the output will be summed. Defaults to ``'mean'``.

        label_priors (Tensor, optional): The label priors of the class-aware selective loss. Default to ``None``.

        likelihood_topk (int, optional): The Top-K likelihood of the class-aware selective loss. Default to ``5``.

        prior_threshold (float, optional): The threshold that determine the mode used for each category. Default to ``0.05``.

    """
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma_pos, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma_neg, clip, discard_focal_grad=discard_focal_grad),
        
        partial_loss_mode = 'selective',
        normalize = normalize,
        reduction = reduction,
        
        label_priors = label_priors,
        likelihood_topk = likelihood_topk,
        prior_threshold = prior_threshold,
    )

# Ignore the label
class NoneLossTerm(nn.Module):
    def __init__(self) -> None:
        super(NoneLossTerm, self).__init__()
    
    def forward(self, p):
        return 0 * p
    
# class BCELossTerm(nn.Module):
#     def __init__(self, alpha=1) -> None:
#         super(BCELossTerm, self).__init__()
#         self.alpha = alpha
    
#     def forward(self, z):
#         return self.alpha * torch.binary_cross_entropy_with_logits(z, torch.ones_like(z), None, None, 0)
    
class FocalLossTerm(nn.Module):
    def __init__(self,
                 alpha: float = 1, 
                 gamma: float = 0, 
                 shift: float = 0, 
                 discard_focal_grad: bool = True
                 ) -> None:
        super(FocalLossTerm, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.shift = torch.tensor(shift) # negative term of asymmetric loss
        self.discard_focal_grad = discard_focal_grad
    
    def forward(self, p):

        p = torch.clamp(p + self.shift, max=1)

        if self.gamma == 0:
            return - self.alpha * torch.log(p)

        p_focal = p.detach() if self.discard_focal_grad else p

        return - self.alpha * torch.pow(1 - p_focal, self.gamma) * torch.log(p)

class PartialLoss(nn.Module):
    def __init__(
            self,
            lossfn_pos: Callable = NoneLossTerm(),
            lossfn_neg: Callable = NoneLossTerm(),

            partial_loss_mode: Literal['ignore', 'negative', 'selective'] = 'ignore',
            normalize: bool = False,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',

            label_priors: Tensor = None,
            likelihood_topk: int = 5,
            prior_threshold: float = 0.05,
            ):
        super(PartialLoss, self).__init__()

        self.lossfn_pos = lossfn_pos
        self.lossfn_neg = lossfn_neg

        self.label_priors = label_priors
        self.partial_loss_mode = partial_loss_mode
        self.likelihood_topk = likelihood_topk
        self.prior_threshold = prior_threshold
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, logits, targets):

        preds = torch.sigmoid(logits)

        if self.partial_loss_mode == 'ignore':
            pseudo_target = targets
        elif self.partial_loss_mode == 'negative':
            pseudo_target = torch.where(torch.isnan(targets), 0, targets)
        elif self.partial_loss_mode == 'selective':
            selective_target = torch.zeros_like(targets)
            if self.label_priors is not None and self.prior_threshold:
                idx_ignore = torch.where(self.label_priors > self.prior_threshold)[0]
                selective_target[:, idx_ignore] = torch.nan
            # ignore top-k
            with torch.no_grad():
                num_top_k = self.likelihood_topk * targets.shape[0]
                targets_flatten = targets.flatten()
                cond_flatten = torch.where(torch.isnan(targets_flatten))[0]
                selective_target_flatten = selective_target.flatten()
                xs_neg_flatten = (1-preds).flatten()
                ind_class_sort = torch.argsort(xs_neg_flatten[cond_flatten])
                selective_target_flatten[cond_flatten[ind_class_sort[:num_top_k]]] = torch.nan
                selective_target = selective_target_flatten.view(*selective_target.shape)
                pseudo_target = torch.where(torch.isnan(targets), selective_target, targets)

        # Positive, Negative and Unknown labels # as long as weights for soft labels
        weights_pos = torch.where(torch.isnan(pseudo_target), 0, pseudo_target)
        weights_neg = torch.where(torch.isnan(pseudo_target), 0, 1-pseudo_target)

        # Loss calculation
        loss_pos = self.lossfn_pos(preds)
        loss_neg = self.lossfn_neg(1-preds)

        total_loss = loss_pos * weights_pos + loss_neg * weights_neg

        if self.normalize: # https://arxiv.org/pdf/1902.09720.pdfs
            alpha_norm, beta_norm = 1, 1
            num_known_labels = 1 + torch.sum(~torch.isnan(pseudo_target), axis=1)    # Add 1 to avoid dividing by zero
            g_norm = alpha_norm * (1 / num_known_labels) + beta_norm
            total_loss *= g_norm.repeat([pseudo_target.shape[1], 1]).T

        if self.reduction == 'mean':
            return total_loss.nansum() / torch.sum(~torch.isnan(pseudo_target)).detach()
        if self.reduction == 'sum':
            return total_loss.nansum()
        return total_loss
