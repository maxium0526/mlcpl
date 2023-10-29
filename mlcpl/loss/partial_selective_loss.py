import torch
from torch import nn as nn, Tensor

# from https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/src/loss_functions/partial_asymmetric_loss.py 
class PartialSelectiveLoss(nn.Module):
    def __init__(
            self,
            clip = 0,
            gamma_pos = 0,
            gamma_neg = 1,
            gamma_unann = 2,
            alpha_pos = 1,
            alpha_neg = 1,
            alpha_unann = 1,
            prior_path = None,
            partial_loss_mode = 'negative',
            likelihood_topk = 5,
            prior_threshold = 0.05,
            reduction = 'mean',
            ):
        super(PartialSelectiveLoss, self).__init__()

        self.clip = clip
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_unann = gamma_unann
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.alpha_unann = alpha_unann

        self.lossfn_pos = FocalLossTerm(self.alpha_pos, self.gamma_pos)
        self.lossfn_neg = FocalLossTerm(self.alpha_neg, self.gamma_neg)
        self.lossfn_unann = FocalLossTerm(self.alpha_unann, self.gamma_unann)

        self.prior_path = prior_path
        self.partial_loss_mode = partial_loss_mode
        self.likelihood_topk = likelihood_topk
        self.prior_threshold = prior_threshold

        self.reduction = reduction

        self.targets_weights = None

        if self.prior_path is not None:
            self.prior_classes = torch.load(self.prior_path)
            print("Prior file was loaded successfully. ")
            
    def forward(self, logits, targets):
        targets = torch.where(torch.isnan(targets), -1, targets) # adopt to my code
        
        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (targets == -1).float()

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        prior_classes = None
        if hasattr(self, "prior_classes"):
            # prior_classes = torch.tensor(list(self.prior_classes.values())).cuda()
            prior_classes = self.prior_classes

        targets_weights = self.targets_weights
        targets_weights, xs_neg = edit_targets_parital_labels(self.partial_loss_mode, self.likelihood_topk, self.prior_threshold, targets, targets_weights, xs_neg,
                                                              prior_classes=prior_classes)

        # Loss calculation
        loss_pos = targets_pos * self.lossfn_pos(torch.clamp(xs_pos, min=1e-8))
        loss_neg = targets_neg * self.lossfn_neg(torch.clamp(xs_neg, min=1e-8))
        loss_unann = targets_unann * self.lossfn_unann(torch.clamp(xs_neg, min=1e-8))

        total_loss = loss_pos + loss_neg + loss_unann

        # partial labels weights
        total_loss *= targets_weights

        if self.reduction == 'mean':
            return total_loss.mean()
        if self.reduction == 'sum':
            return total_loss.sum()
        return total_loss
    
class FocalLossTerm():
    def __init__(self, alpha=1, gamma=1) -> None:
        super(FocalLossTerm, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, p):
        return - self.alpha * torch.pow(1 - p, self.gamma) * torch.log(p)


def edit_targets_parital_labels(partial_loss_mode, likelihood_topk, prior_threshold, targets, targets_weights, xs_neg, prior_classes=None):
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if partial_loss_mode is None:
        targets_weights = 1.0

    elif partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets_weights = 1.0

    elif partial_loss_mode == 'ignore':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        targets_weights[targets == -1] = 0

    elif partial_loss_mode == 'ignore_normalize_classes':
        # remove all unsure targets and normalize by Durand et al. https://arxiv.org/pdf/1902.09720.pdfs
        alpha_norm, beta_norm = 1, 1
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        n_annotated = 1 + torch.sum(targets != -1, axis=1)    # Add 1 to avoid dividing by zero

        g_norm = alpha_norm * (1 / n_annotated) + beta_norm
        n_classes = targets_weights.shape[1]
        targets_weights *= g_norm.repeat([n_classes, 1]).T
        targets_weights[targets == -1] = 0

    elif partial_loss_mode == 'selective':
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        else:
            targets_weights[:] = 1.0
        num_top_k = likelihood_topk * targets_weights.shape[0]

        xs_neg_prob = xs_neg
        if prior_classes is not None:
            if prior_threshold:
                idx_ignore = torch.where(prior_classes > prior_threshold)[0]
                targets_weights[:, idx_ignore] = 0
                targets_weights += (targets != -1).float()
                targets_weights = targets_weights.bool()

        targets_weights = negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)

    return targets_weights, xs_neg


# @torch.jit.script
def negative_backprop_fun_jit(targets: Tensor, xs_neg_prob: Tensor, targets_weights: Tensor, num_top_k: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_prob_flatten = xs_neg_prob.flatten()
        ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_k]]] = 0
        return targets_weights_flatten.reshape(targets.shape)