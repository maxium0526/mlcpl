"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
# from __future__ import print_function

import torch
import torch.nn as nn

class NewTestLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super(NewTestLoss, self).__init__()
        self.loss_fn = SupConLoss()
        self.alpha = alpha

    def forward(self, anchor, contrast, anchor_target, contrast_target):
        main_loss = self.loss_fn(anchor, contrast)
        category_losses = []
        num_categories = anchor_target.shape[1]

        for c in range(num_categories):
            cateogry_loss = self.loss_fn(anchor, contrast, anchor_target[:, c], contrast_target[:, c])
            main_loss += self.alpha * cateogry_loss
            category_losses.append(cateogry_loss)
        
        return main_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor, contrast=None, anchor_target=None, contrast_target=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = torch.device('cuda') if anchor.is_cuda else torch.device('cpu')
        
        anchor = anchor
        anchor_size = anchor.shape[0]

        if contrast is None:
            contrast = anchor
        contrast_size = contrast.shape[0]

        if anchor_target is not None and contrast_target is None:
            contrast_target = anchor_target

        if anchor_target is not None:
            anchor_target = anchor_target.contiguous().view(-1, 1)
            contrast_target = contrast_target.contiguous().view(-1, 1)
            mask = torch.eq(anchor_target, contrast_target.T).float().to(device)

        else:
            mask = torch.zeros((anchor.shape[0], contrast.shape[0])).to(device)
            mask.fill_diagonal_(1)

        anchor_feature = torch.cat(torch.unbind(anchor, dim=1), dim=0)
        anchor_count = anchor.shape[1]
        contrast_feature = torch.cat(torch.unbind(contrast, dim=1), dim=0)
        contrast_count = contrast.shape[1]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.ones_like(logits)
        for i in range(min(anchor_count, contrast_count)):

            logits_mask[anchor_size*i:anchor_size*(i+1), contrast_size*i:contrast_size*(i+1)] = torch.abs(1 - mask)

        # tile masks
        mask = mask.repeat(anchor_count, contrast_count)

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        return loss.mean()


# maxium0526@github
class MLSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(MLSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, anchor, contrast, anchor_target, contrast_target):
        """Compute loss for model.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            target: one-hot ground truth of shape [bsz, C].
        Returns:
            A loss scalar.
        """
        
        device = torch.device('cuda') if anchor.is_cuda else torch.device('cpu')

        anchor_size = anchor.shape[0]
        contrast_size = contrast.shape[0]

        anchor_feature = torch.cat(torch.unbind(anchor, dim=1), dim=0)
        anchor_count = anchor.shape[1]
        contrast_feature = torch.cat(torch.unbind(contrast, dim=1), dim=0)
        contrast_count = contrast.shape[1]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        batch_size = anchor.shape[0]
        num_categories = anchor_target.shape[1]
        
        losses = [] # [C, N]
        for c in range(num_categories):
            anchor_labels = anchor_target[:, c].contiguous().view(-1, 1)
            contrast_labels = contrast_target[:, c].contiguous().view(-1, 1)

            mask = torch.matmul(anchor_labels, contrast_labels.T).float().to(device)
            mask.fill_diagonal_(1)

            # mask-out self-contrast cases
            logits_mask = torch.ones_like(logits)
            for i in range(min(anchor_count, contrast_count)):
                sub = torch.ones((anchor_size, contrast_size))
                sub.fill_diagonal_(0)
                logits_mask[anchor_size*i:anchor_size*(i+1), contrast_size*i:contrast_size*(i+1)] = sub

            # tile mask
            mask = mask.repeat(anchor_count, contrast_count)

            mask = mask * logits_mask

            # here mask indicates the samples in numerator, logits mask indicate the samples in denominator

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            # modified to handle edge cases when there is no positive pair
            # for an anchor point. 
            # Edge case e.g.:- 
            # features of shape: [4,1,...]
            # labels:            [0,1,1,2]
            # loss before mean:  [nan, ..., ..., nan] 
            mask_pos_pairs = mask.sum(1)
            mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

            losses.append(mean_log_prob_pos.view(-1, 1))

        losses = torch.concat(losses, dim=1) * anchor_target.repeat(2, 1)

        num_pos = anchor_target.repeat(2, 1).sum()
        num_pos = torch.where(num_pos<1e-6, 1, num_pos) # avoid there is no positive label in a sample

        loss = losses.sum() / num_pos
            # loss
        loss = - (self.temperature / self.base_temperature) * loss
        # loss = loss.view(anchor_count, batch_size).mean()

        return loss

class PartialMLSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', unknown_mode='ignore',
                 base_temperature=0.07):
        super(PartialMLSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.unknown_mode = unknown_mode

    def forward(self, features, targets):
        """Compute loss for model.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            target: one-hot ground truth of shape [bsz, C].
        Returns:
            A loss scalar.
        """
        if targets is None:
            raise ValueError('`targets` is None. Use SupConLoss for unsupervised learning.')

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        batch_size = features.shape[0]
        num_categories = targets.shape[1]
        
        losses = [] # [C, N]
        for c in range(num_categories):

            labels = targets[:, c]
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.matmul(labels, labels.T).float().to(device)
            mask.fill_diagonal_(1)

            # tile mask
            mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                0
            )

            unknown_mask = torch.isnan(mask)

            mask = mask * logits_mask
            mask = torch.nan_to_num(mask, nan=0)

            if self.unknown_mode == 'ignore':
                logits_mask = logits_mask * ~unknown_mask
            elif self.unknown_mode == 'negative':
                pass
            else:
                raise ValueError('Unknown mode: {}'.format(self.unknown_mode))
            
            # here mask indicates the samples in numerator, logits mask indicate the samples in denominator

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask

            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            # modified to handle edge cases when there is no positive pair
            # for an anchor point. 
            # Edge case e.g.:- 
            # features of shape: [4,1,...]
            # labels:            [0,1,1,2]
            # loss before mean:  [nan, ..., ..., nan] 
            mask_pos_pairs = mask.sum(1)
            mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

            losses.append(mean_log_prob_pos.view(-1, 1))

        losses = torch.concat(losses, dim=1) * targets.repeat(contrast_count, 1).nan_to_num(nan=0)

        num_pos = targets.repeat(2, 1).nan_to_num(nan=0).sum()
        num_pos = torch.where(num_pos<1e-6, 1, num_pos) # avoid there is no positive label in a sample

        loss = losses.sum() / num_pos
            # loss
        loss = - (self.temperature / self.base_temperature) * loss

        return loss