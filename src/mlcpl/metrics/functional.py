import torchmetrics
import torchmetrics.functional.classification
from .core import *
from typing import Literal

def binary_drop_unknown(preds, target):
    known_index = ~torch.isnan(target)
    return preds[known_index], target[known_index]

def partial_binary_wrapper(binary_metric, preds, target, check_labels, **kwargs):

    preds, target = binary_drop_unknown(preds, target)
    
    if target.numel() == 0 :
        return torch.tensor(torch.nan)
    
    if check_labels == 'p+n':
        if (target==1.0).sum() == 0 or (target==0.0).sum() == 0:
            return torch.tensor(torch.nan)
    elif check_labels == 'p/n':
        if (target==1.0).sum() == 0 and (target==0.0).sum() == 0:
            return torch.tensor(torch.nan)
    elif check_labels == 'p':
        if (target==1.0).sum() == 0:
            return torch.tensor(torch.nan)       
    elif check_labels == 'n':
        if (target==0.0).sum() == 0:
            return torch.tensor(torch.nan)

    return binary_metric(preds, target, **kwargs)


def partial_multilabel_wrapper(binary_metric, preds, target, check_labels, average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro', **kwargs):
    if average == 'micro':
        preds, target = preds.flatten(), target.flatten()
        return partial_binary_wrapper(binary_metric, preds, target, check_labels,  **kwargs)

    num_categories = preds.shape[1]
    scores = torch.zeros(num_categories, dtype=torch.float32)

    for i in range(num_categories):
        category_preds, category_target = preds[:, i], target[:, i]
        scores[i] = partial_binary_wrapper(binary_metric, category_preds, category_target, check_labels, **kwargs)

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
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        thresholds=None,
        ignore_index=None,
        validate_args=True,
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_average_precision
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        thresholds=thresholds, 
        ignore_index=ignore_index, 
        validate_args=validate_args)

def partial_multilabel_auroc(
        preds, target, 
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        thresholds=None,
        ignore_index=None,
        validate_args=True
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_auroc
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target,
        check_labels=check_labels,
        average=average,
        thresholds=thresholds, 
        ignore_index=ignore_index, 
        validate_args=validate_args)

def partial_multilabel_fbeta_score(
        preds, target,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ignore_index=None,
        validate_args=True
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_fbeta_score
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        beta=beta,
        threshold=threshold, 
        ignore_index=ignore_index, 
        validate_args=validate_args)

def partial_multilabel_f1_score(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ignore_index=None,
        validate_args=True
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_f1_score
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=ignore_index, 
        validate_args=validate_args)

def partial_multilabel_precision(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ignore_index=None,
        validate_args=True,
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_precision
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=ignore_index, 
        validate_args=validate_args)

def partial_multilabel_recall(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ignore_index=None,
        validate_args=True,
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_recall
    check_labels = 'p'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=ignore_index, 
        validate_args=validate_args)