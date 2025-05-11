import torchmetrics
import torchmetrics.classification
import torchmetrics.functional.classification
from .core import *
from .calibration_error import _binning_bucketize
from typing import Literal

def binary_drop_unknown(preds, target):
    known_index = ~torch.isnan(target)
    return preds[known_index], target[known_index]

def partial_binary_wrapper(binary_metric, preds, target, check_labels, **kwargs):

    preds, target = binary_drop_unknown(preds, target)

    target = target.to(torch.int64)
    
    if target.numel() == 0 :
        return torch.tensor(torch.nan)
    
    if check_labels == 'p+n':
        if (target==1).sum() == 0 or (target==0).sum() == 0:
            return torch.tensor(torch.nan)
    elif check_labels == 'p/n':
        if (target==1).sum() == 0 and (target==0).sum() == 0:
            return torch.tensor(torch.nan)
    elif check_labels == 'p':
        if (target==1).sum() == 0:
            return torch.tensor(torch.nan)       
    elif check_labels == 'n':
        if (target==0).sum() == 0:
            return torch.tensor(torch.nan)

    return binary_metric(preds, target, **kwargs)

def partial_multilabel_wrapper(binary_metric, preds, target, check_labels, num_returns=1, return_list=False, average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro', **kwargs):
    if average == 'micro':
        preds, target = preds.flatten(), target.flatten()
        return partial_binary_wrapper(binary_metric, preds, target, check_labels,  **kwargs)

    num_categories = preds.shape[1]
    # scores = torch.zeros(num_categories, dtype=torch.float32)
    outputs = []

    for i in range(num_categories):
        category_preds, category_target = preds[:, i], target[:, i]
        outputs.append(partial_binary_wrapper(binary_metric, category_preds, category_target, check_labels, **kwargs))

    if num_returns > 1:
        for i in range(len(outputs)):
            outputs[i] = outputs[i] if isinstance(outputs[i], tuple) or not torch.isnan(outputs[i]) else (outputs[i], ) * num_returns
        outputs = list(map(list, zip(*outputs)))
        if return_list is True:
            return outputs
        else:
            return [torch.tensor(output) for output in outputs]

    scores = torch.tensor(outputs, dtype=torch.float32)

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
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_average_precision
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        thresholds=thresholds, 
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_auroc(
        preds, target, 
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        thresholds=None,
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_auroc
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target,
        check_labels=check_labels,
        average=average,
        thresholds=thresholds, 
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_fbeta_score(
        preds, target,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
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
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_f1_score(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_f1_score
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_precision(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_precision
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_recall(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_recall
    check_labels = 'p'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_sensitivity(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    return partial_multilabel_recall(
        preds, target,
        threshold=threshold,
        average=average,
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_specificity(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_specificity
    check_labels = 'n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_precision_at_fixed_recall(
        preds, target,
        min_recall: float,
        thresholds = None,
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_precision_at_fixed_recall
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        num_returns=2,
        average='none',
        min_recall=min_recall,
        thresholds=thresholds, 
        ignore_index=None,
        validate_args=False)

def partial_multilabel_recall_at_fixed_precision(
        preds, target,
        min_precision: float,
        thresholds = None,
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_recall_at_fixed_precision
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        num_returns=2,
        average='none',
        min_precision=min_precision,
        thresholds=thresholds, 
        ignore_index=None,
        validate_args=False)

def partial_multilabel_sensitivity_at_specificity(
        preds, target,
        min_specificity: float,
        thresholds = None,
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_sensitivity_at_specificity
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        num_returns=2,
        average='none',
        min_specificity=min_specificity,
        thresholds=thresholds, 
        ignore_index=None,
        validate_args=False)

def partial_multilabel_specificity_at_sensitivity(
        preds, target,
        min_sensitivity: float,
        thresholds = None,
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_specificity_at_sensitivity
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        num_returns=2,
        average='none',
        min_sensitivity=min_sensitivity,
        thresholds=thresholds, 
        ignore_index=None,
        validate_args=False)

def partial_multilabel_roc(
        preds, target,
        thresholds = None, # not tested yet
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_roc
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        num_returns=3,
        return_list=True,
        average='none',
        thresholds=thresholds, 
        ignore_index=None,
        validate_args=False)

def partial_multilabel_precision_recall_curve(
        preds, target,
        thresholds = None, # not tested yet
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_precision_recall_curve
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        num_returns=3,
        return_list=True,
        average='none',
        thresholds=thresholds, 
        ignore_index=None,
        validate_args=False)

def partial_multilabel_accuracy(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_accuracy
    check_labels = 'p/n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=None, 
        validate_args=False)

def binary_calibration_error(preds, target, n_bins=15, norm='l1'):

    preds = torch.nn.functional.sigmoid(preds)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    confidences, accuracies = preds, target

    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(confidences, accuracies, bin_boundaries)

    if norm == "l1":
        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    if norm == "max":
        return torch.max(torch.abs(acc_bin - conf_bin))
    if norm == "l2":
        ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
    if norm == 'ECE':
        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    if norm == 'ACE':
        return torch.mean(torch.abs(acc_bin - conf_bin))
    if norm == 'MCE':
        return torch.max(torch.abs(acc_bin - conf_bin))

    return None

def partial_multilabel_calibration_error(
        preds, target,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        norm: Literal['l1', 'l2', 'max', 'ECE', 'ACE', 'MCE'] = 'l1'
        ):
    
    binary_metric = binary_calibration_error
    check_labels = 'p/n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        norm=norm)

def partial_multilabel_cohen_kappa(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        weights: Literal['linear', 'quadratic', 'none'] = 'none',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_cohen_kappa
    check_labels = 'p+n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold,
        weights=weights,
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_confusion_matrix(
        preds, target,
        threshold: float = 0.5,
        normalize: Literal['none', 'true', 'pred', 'all'] = 'none',
        ):
    
    num_categories = preds.shape[1]
    
    binary_metric = torchmetrics.functional.classification.binary_confusion_matrix
    
    outputs = torch.zeros((num_categories, 2, 2), dtype=torch.float32)
    
    for i in range(num_categories):
        category_preds, category_target = binary_drop_unknown(preds[:, i], target[:, i])
        
        if (category_target==1.0).sum() == 0 and (category_target==0.0).sum() == 0:
            outputs[i, :, :] = torch.nan
        else:
            outputs[i, :, :] = binary_metric(category_preds, category_target, threshold=threshold, normalize=normalize)

    return outputs

# def partial_multilabel_coverage_error(
#         preds, target,
#         ):
#     preds = torch.sigmoid(preds)

#     min_preds = preds.min()

#     coverages = torch.zeros((preds.shape[0]))

#     for i, (pred, label) in enumerate(zip(preds, target)):

#         pred, label = binary_drop_unknown(pred, label)
#         if (label==1).sum() == 0:
#             continue
#         pred_min = torch.min(pred[label==1])

#         coverages[i] = (pred >= pred_min).sum()
    
#     label_counts = torch.sum(~torch.isnan(target), dim=1)

#     print(label_counts)

#     return (coverages * label_counts).sum() / label_counts.sum()

def partial_multilabel_dice(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'samples', 'none'] = 'micro',
        ):
    
    binary_metric = torchmetrics.functional.classification.dice
    check_labels = 'p+n'

    if average != 'samples':
        return partial_multilabel_wrapper(
            binary_metric, 
            preds, target, 
            check_labels=check_labels,
            average=average,
            threshold=threshold,
            ignore_index=None)
    
    elif average == 'samples':
        return partial_multilabel_wrapper(
            binary_metric, 
            preds.T, target.T, 
            check_labels=check_labels,
            average='macro',
            threshold=threshold,
            ignore_index=None)
    
def partial_multilabel_exact_match(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    preds = torch.sigmoid(preds)

    scores = torch.zeros(preds.shape[0], dtype=torch.float32)

    for i, (pred, label) in enumerate(zip(preds, target)):

        pred, label = binary_drop_unknown(pred, label)
        if (label==1).sum() + (label==0).sum() == 0:
            scores[i] = torch.nan
            continue
        
        pred_label = pred > threshold
        if all(pred_label == label):
            scores[i] = 1
        else:
            scores[i] = 0

    return torch.nanmean(scores)

def partial_multilabel_hamming_distance(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_hamming_distance
    check_labels = 'p/n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_hinge_loss(
        preds, target,
        squared: bool = False,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_hinge_loss
    check_labels = 'p/n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        squared=squared,
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_jaccard_index(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    
    binary_metric = torchmetrics.functional.classification.binary_jaccard_index
    check_labels = 'p/n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        threshold=threshold, 
        ignore_index=None, 
        validate_args=False)

def partial_multilabel_ranking_average_precision(
        preds, target,
        ):
    return torchmetrics.functional.classification.multilabel_ranking_average_precision(
        preds, target,
        num_labels=preds.shape[1],
        validate_args=False
    )

def partial_multilabel_ranking_loss(
        preds, target,
        ):
    
    scores = torch.zeros(preds.shape[0])
    for i, (pred, label) in enumerate(zip(preds, target)):
        pred, label = binary_drop_unknown(pred, label)
        pred, label = pred.unsqueeze(0), label.unsqueeze(0)
        scores[i] = torchmetrics.functional.classification.multilabel_ranking_loss(
            pred, label,
            num_labels=pred.shape[1],
            validate_args=False
        )
        
    return scores.nanmean()

def partial_multilabel_matthews_corrcoef(
        preds, target,
        threshold: float = 0.5,
        ):
    
    check_labels = 'p/n'

    confusion_metrics = partial_multilabel_confusion_matrix(preds, target,threshold=threshold)

    # reduce to binary task
    confusion_matrix = confusion_metrics.nansum(dim=0)
    
    tn, fp, fn, tp = confusion_matrix.reshape(-1)

    nominator = tp*tn - fp*fn
    denominator = torch.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

    denominator = 1 if denominator == 0 else denominator

    return nominator / denominator