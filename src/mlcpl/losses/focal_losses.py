import torch
from torch import nn as nn

def PartialNegativeBCELoss(
    alpha_pos = 1,
    alpha_neg = 1,
    normalize = False,
    reduction = 'mean',
):
    return PartialLoss(
        lossfn_pos = BCELossTerm(alpha_pos),
        lossfn_neg = BCELossTerm(alpha_neg),
        lossfn_unann = BCELossTerm(alpha_neg),

        partial_loss_mode = 'negative',
        normalize = normalize,
        reduction = reduction,
    )

def PartialBCELoss(
    alpha_pos = 1,
    alpha_neg = 1,
    normalize = False,
    reduction = 'mean',
):
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos),
        lossfn_neg = FocalLossTerm(alpha_neg),

        partial_loss_mode = 'ignore',
        normalize = normalize,
        reduction = reduction,
    )

def PartialSelectiveBCELoss(
    alpha_pos = 1,
    alpha_neg = 1,
    alpha_unann = 1,
    normalize = False,
    reduction = 'mean',
    class_priors = None,
    likelihood_topk = 5,
    prior_threshold = 0.05,
):
    return PartialLoss(
        lossfn_pos = BCELossTerm(alpha_pos),
        lossfn_neg = BCELossTerm(alpha_neg),
        lossfn_unann = BCELossTerm(alpha_unann),

        partial_loss_mode = 'selective',
        normalize = normalize,
        reduction = reduction,
        
        class_priors = class_priors,
        likelihood_topk = likelihood_topk,
        prior_threshold = prior_threshold,
    )

def PartialNegativeFocalLoss(
    gamma = 1,
    alpha_pos = 1,
    alpha_neg = 1,
    normalize = False,
    discard_focal_grad = True,
    reduction = 'mean',
):
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma, discard_focal_grad=discard_focal_grad),
        lossfn_unann = FocalLossTerm(alpha_neg, gamma, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'negative',
        normalize = normalize,
        reduction = reduction,
    )

def PartialFocalLoss(
    gamma = 1,
    alpha_pos = 1,
    alpha_neg = 1,
    normalize = False,
    discard_focal_grad = True,
    reduction = 'mean',
):
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'ignore',
        normalize = normalize,
        reduction = reduction,
    )

def PartialSelectiveFocalLoss(
    gamma = 1,
    alpha_pos = 1,
    alpha_neg = 1,
    alpha_unann = 1,
    normalize = False,
    discard_focal_grad = True,
    reduction = 'mean',
    class_priors = None,
    likelihood_topk = 5,
    prior_threshold = 0.05,
):
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma, discard_focal_grad=discard_focal_grad),
        lossfn_unann = FocalLossTerm(alpha_unann, gamma, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'selective',
        normalize = normalize,
        reduction = reduction,
        
        class_priors = class_priors,
        likelihood_topk = likelihood_topk,
        prior_threshold = prior_threshold,
    )

def PartialNegativeAsymmetricLoss(
    clip = 0,
    gamma_pos = 0,
    gamma_neg = 1,
    alpha_pos = 1,
    alpha_neg = 1,
    normalize = False,
    discard_focal_grad = True,
    reduction = 'mean',
):
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma_pos, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma_neg, clip, discard_focal_grad=discard_focal_grad),
        lossfn_unann = FocalLossTerm(alpha_neg, gamma_neg, clip, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'negative',
        normalize = normalize,
        reduction = reduction,
    )

def PartialAsymmetricLoss(
    clip = 0,
    gamma_pos = 0,
    gamma_neg = 1,
    alpha_pos = 1,
    alpha_neg = 1,
    normalize = False,
    discard_focal_grad = True,
    reduction = 'mean',
):
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma_pos, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma_neg, clip, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'ignore',
        normalize = normalize,
        reduction = reduction,
    )

def PartialSelectiveAsymmetricLoss(
    clip = 0,
    gamma_pos = 0,
    gamma_neg = 1,
    gamma_unann = 2,
    alpha_pos = 1,
    alpha_neg = 1,
    alpha_unann = 1,
    normalize = False,
    discard_focal_grad = True,
    reduction = 'mean',
    class_priors = None,
    likelihood_topk = 5,
    prior_threshold = 0.05,
):
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma_pos, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma_neg, clip, discard_focal_grad=discard_focal_grad),
        lossfn_unann = FocalLossTerm(alpha_unann, gamma_unann, clip, discard_focal_grad=discard_focal_grad),

        partial_loss_mode = 'selective',
        normalize = normalize,
        reduction = reduction,
        
        class_priors = class_priors,
        likelihood_topk = likelihood_topk,
        prior_threshold = prior_threshold,
    )

# Ignore the label
class NoneLossTerm(nn.Module):
    def __init__(self) -> None:
        super(NoneLossTerm, self).__init__()
    
    def forward(self, p):
        return 0 * p
    
class BCELossTerm(nn.Module):
    def __init__(self, alpha=1) -> None:
        super(BCELossTerm, self).__init__()
        self.alpha = alpha
    
    def forward(self, z):
        return self.alpha * torch.binary_cross_entropy_with_logits(z, torch.ones_like(z), None, None, 0)
    
class FocalLossTerm(nn.Module):
    def __init__(self, alpha=1, gamma=1, shift=0, discard_focal_grad=True) -> None:
        super(FocalLossTerm, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.shift = shift # negative term of asymmetric loss
        self.discard_focal_grad = discard_focal_grad
    
    def forward(self, z):
        z = z if self.shift == 0 else torch.where(z > torch.log(-1/self.shift), torch.inf, z)

        if self.gamma == 1:
            return self.alpha * torch.binary_cross_entropy_with_logits(z, torch.ones_like(z), None, None, 0)

        p_focal = torch.sigmoid(z.detach() if self.discard_focal_grad else z)

        return self.alpha * torch.pow(1 - p_focal, self.gamma) * torch.binary_cross_entropy_with_logits(z, torch.ones_like(z), None, None, 0)

# from https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/src/loss_functions/partial_asymmetric_loss.py 
class PartialLoss(nn.Module):
    def __init__(
            self,
            lossfn_pos = NoneLossTerm(),
            lossfn_neg = NoneLossTerm(),
            lossfn_unann = NoneLossTerm(),

            partial_loss_mode = 'ignore',
            normalize = False,
            reduction = 'mean',

            # arguments for selective mode
            class_priors = None,
            likelihood_topk = 5,
            prior_threshold = 0.05,

            fully_labeled_warning = True # print the warning if the labels seems fully-labeled
            ):
        super(PartialLoss, self).__init__()

        self.lossfn_pos = lossfn_pos
        self.lossfn_neg = lossfn_neg
        self.lossfn_unann = lossfn_unann

        self.class_priors = class_priors
        self.partial_loss_mode = partial_loss_mode
        self.likelihood_topk = likelihood_topk
        self.prior_threshold = prior_threshold
        self.normalize = normalize
        self.reduction = reduction

        self.fully_labeled_warning = fully_labeled_warning
        self.fully_labeled_warning_trigger = True if self.fully_labeled_warning else False

    def forward(self, logits, targets):

        if self.partial_loss_mode == 'ignore':
            pseudo_target = targets
        elif self.partial_loss_mode == 'negative':
            pseudo_target = torch.where(torch.isnan(targets), 0, targets)
        elif self.partial_loss_mode == 'selective':
            pseudo_target = targets

        # Positive, Negative and Unknown labels # as long as weights for soft labels
        weights_pos = torch.where(torch.isnan(pseudo_target), 0, pseudo_target)
        weights_neg = torch.where(torch.isnan(pseudo_target), 0, 1-pseudo_target)

        # if self.fully_labeled_warning_trigger and targets_unann.nonzero().sum() == 0:
        #     print('The input batch data seems to be fully-labeled. This warning is only logged once.')
        #     self.fully_labeled_warning_trigger = False

        # Activation
        # xs_pos = torch.sigmoid(logits)
        # xs_neg = 1.0 - xs_pos

        # targets_weights = self.__compute_targets_weights(xs_neg, targets)

        # Loss calculation
        loss_pos = weights_pos * self.lossfn_pos(logits)
        loss_neg = weights_neg * self.lossfn_neg(-logits)
        # loss_unann = targets_unann * self.lossfn_unann(-logits)

        total_loss = loss_pos + loss_neg

        # partial labels weights
        # total_loss *= targets_weights

        if self.reduction == 'mean':
            return total_loss.mean()
        if self.reduction == 'sum':
            return total_loss.sum()
        return total_loss
    
    def __compute_targets_weights(self, xs_neg, targets):

        targets_weights = 1.0

        if self.partial_loss_mode == 'negative':
            # set all unsure targets as negative
            targets_weights = 1.0

        elif self.partial_loss_mode == 'ignore':
            # remove all unsure targets (targets_weights=0)
            targets_weights = torch.ones(targets.shape, device=targets.device)
            targets_weights[targets == -1] = 0

        elif self.partial_loss_mode == 'selective':
            targets_weights = torch.ones(targets.shape, device=targets.device)

            if self.class_priors is not None:
                if self.prior_threshold:
                    idx_ignore = torch.where(self.class_priors > self.prior_threshold)[0]
                    targets_weights[:, idx_ignore] = 0
                    targets_weights += (targets != -1).float()
                    targets_weights = targets_weights.bool()

            # ignore top-k
            with torch.no_grad():
                num_top_k = self.likelihood_topk * targets_weights.shape[0]

                targets_flatten = targets.flatten()
                cond_flatten = torch.where(targets_flatten == -1)[0]
                targets_weights_flatten = targets_weights.flatten()
                xs_neg_flatten = xs_neg.flatten()
                ind_class_sort = torch.argsort(xs_neg_flatten[cond_flatten])
                targets_weights_flatten[
                    cond_flatten[ind_class_sort[:num_top_k]]] = 0
                targets_weights

        if self.normalize: # https://arxiv.org/pdf/1902.09720.pdfs
            alpha_norm, beta_norm = 1, 1
            n_annotated = 1 + torch.sum(targets_weights == 1, axis=1)    # Add 1 to avoid dividing by zero
            g_norm = alpha_norm * (1 / n_annotated) + beta_norm
            n_classes = targets_weights.shape[1]
            targets_weights *= g_norm.repeat([n_classes, 1]).T

        return targets_weights


