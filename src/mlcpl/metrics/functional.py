import torchmetrics
import torchmetrics.functional.classification
from .core import *
from typing import Literal

def binary_drop_unknown(preds, target):
    known_index = ~torch.isnan(target)
    return preds[known_index], target[known_index]

def partial_binary_wrapper(binary_metric, preds, target, **kwargs):

    preds, target = binary_drop_unknown(preds, target)
    
    if target.numel() == 0 :
        score = torch.tensor(torch.nan)
    elif (target==1.0).sum() == 0:
        score = torch.tensor(torch.nan)
    elif (target==0.0).sum() == 0:
        score = torch.tensor(torch.nan)
    else:
        score = binary_metric(preds, target.to(torch.int32), **kwargs)

    return score

def partial_multilabel_wrapper(binary_metric, preds, target, average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro', **kwargs):
    if average == 'micro':
        preds, target = preds.flatten(), target.flatten()
        return partial_binary_wrapper(binary_metric, preds, target)

    num_categories = preds.shape[1]
    scores = torch.zeros(num_categories, dtype=torch.float32)

    for i in range(num_categories):
        category_preds, category_target = preds[:, i], target[:, i]
        scores[i] = partial_binary_wrapper(binary_metric, category_preds, category_target, **kwargs)

    if average == 'macro':
        return torch.mean(scores[~torch.isnan(scores)])
    elif average == 'weighted':
        positive_counts = torch.sum(target == 1, dim=0)
        weights = positive_counts[~torch.isnan(scores)] / torch.sum(positive_counts[~torch.isnan(scores)])
        return (scores[~torch.isnan(scores)] * weights).sum()
    else:
        return scores

def partial_multilabel_average_precision(
        preds, target, 
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro', thresholds=None, ignore_index=None, validate_args=True
        ):
    binary_metric = torchmetrics.functional.classification.binary_average_precision

    return partial_multilabel_wrapper(binary_metric, preds, target, average, thresholds=None, ignore_index=None, validate_args=True)