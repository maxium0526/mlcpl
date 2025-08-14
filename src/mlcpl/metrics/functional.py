import torch
import torchmetrics
import torchmetrics.classification
import torchmetrics.functional.classification
from .core import *
from .calibration_error import _binning_bucketize
from typing import Literal

def partial_multilabel_average_precision(
        preds, target, 
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        thresholds=None,
        ):
    """Compute the average precision (AP) score.

    Args:
        preds:
            Tensor with predictions

        target:
            Tensor with true labels

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction

        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.
    """
    
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
    """Compute Area Under the Receiver Operating Characteristic Curve.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction

        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

    """
    
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
    """Compute F_Beta score.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight

        threshold: Threshold for transforming probability to binary (0,1) predictions

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
    """
    
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
    """Compute F-1 score.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        threshold: Threshold for transforming probability to binary (0,1) predictions

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

    """
    
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
    """Compute precision.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        threshold: Threshold for transforming probability to binary (0,1) predictions

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

    """
    
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

    """Compute recall.


    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
        
        threshold: Threshold for transforming probability to binary (0,1) predictions

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
    """
    
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
    """Compute sensitivity.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
        
        threshold: Threshold for transforming probability to binary (0,1) predictions
        
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
    """
    
    return partial_multilabel_recall(
        preds, target,
        threshold=threshold,
        average=average)

def partial_multilabel_specificity(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    """Compute specificity.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
        
        threshold: Threshold for transforming probability to binary (0,1) predictions
        
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
    """
    
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
    """Compute the highest possible precision value given the minimum recall thresholds.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        min_recall: float value specifying minimum recall threshold.

        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~torch.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

    """
    
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
    """Compute the highest possible recall value given the minimum precision thresholds.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        min_precision: float value specifying minimum precision threshold.

        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~torch.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

    """
    
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
    """Compute the highest possible sensitivity value given the minimum specificity thresholds.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        min_specificity: float value specifying minimum specificity threshold.

        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~torch.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

    """
    
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
    """Compute the highest possible specificity value given the minimum sensitivity thresholds.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        min_sensitivity: float value specifying minimum sensitivity threshold.

        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~torch.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

    """
    
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
    """Compute the Receiver Operating Characteristic Curves.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

    """
    
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
    """Compute the Precision Recall Curves.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

    """
    
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
    """Compute accuracy.


    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
        
        threshold: Threshold for transforming probability to binary (0,1) predictions
        
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
    """
    
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

    preds = torch.sigmoid(preds)

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
        n_bins=15,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        norm: Literal['l1', 'l2', 'max', 'ECE', 'ACE', 'MCE'] = 'l1'
        ):
    """Compute calibration error.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        n_bins: Number of bins to use when computing the metric.

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        norm: Norm used to compare empirical and expected probability bins.

    """
    
    binary_metric = binary_calibration_error
    check_labels = 'p/n'

    return partial_multilabel_wrapper(
        binary_metric, 
        preds, target, 
        check_labels=check_labels,
        average=average,
        n_bins=n_bins,
        norm=norm)

def partial_multilabel_expected_calibration_error(
        preds, target,
        n_bins=15,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    """Compute expected calibration error.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        n_bins: Number of bins to use when computing the metric.

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

    """
    
    return partial_multilabel_calibration_error(
        preds, target, average=average, norm='ECE', n_bins=n_bins,
    )

def partial_multilabel_average_calibration_error(
        preds, target,
        n_bins=15,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    """Compute average calibration error.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        n_bins: Number of bins to use when computing the metric.

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

    """
    
    return partial_multilabel_calibration_error(
        preds, target, average=average, norm='ACE', n_bins=n_bins,
    )

def partial_multilabel_maximum_calibration_error(
        preds, target,
        n_bins=15,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        ):
    """Compute maximum calibration error.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        n_bins: Number of bins to use when computing the metric.

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

    """
    
    return partial_multilabel_calibration_error(
        preds, target, average=average, norm='MCE', n_bins=n_bins,
    )

def partial_multilabel_cohen_kappa(
        preds, target,
        threshold: float = 0.5,
        average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
        weights: Literal['linear', 'quadratic', 'none'] = 'none',
        ):
    """Calculate Cohen's kappa score.


    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        threshold: Threshold for transforming probability to binary (0,1) predictions

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting

    """
    
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
    """Compute the `confusion matrix`.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        threshold: Threshold for transforming probability to binary (0,1) predictions

        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

    Returns:
        A ``[num_labels, 2, 2]`` tensor

    """
    
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
    """Compute Dice.

    Args:
        preds: Predictions from model (probabilities, logits or labels)

        target: Ground truth values
        
        zero_division: The value to use for the score if denominator equals zero

        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).


    """
    
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
        ):
    """Compute Exact match (also known as subset accuracy).


    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        threshold: Threshold for transforming probability to binary (0,1) predictions

    """
    
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
    
    """Compute hamming distance.


    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
        
        threshold: Threshold for transforming probability to binary (0,1) predictions
        
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
    """
    
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
    """Compute the mean Hinge loss typically used for Support Vector Machines.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels

        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.

        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

    """
    
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
    """Compute jaccard index score.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
        
        threshold: Threshold for transforming probability to binary (0,1) predictions
        
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
    """
    
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
    """Compute ranking average precision.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
        
    """
    return torchmetrics.functional.classification.multilabel_ranking_average_precision(
        preds, target,
        num_labels=preds.shape[1],
        validate_args=False
    )

def partial_multilabel_ranking_loss(
        preds, target,
        ):
    """Compute multilabel ranking loss.


    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
    """
    
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
    
    """Compute mattews correlation coefficient.

    Args:
        preds: Tensor with predictions

        target: Tensor with true labels
        
        threshold: Threshold for transforming probability to binary (0,1) predictions
        
    """
    
    check_labels = 'p/n'

    confusion_metrics = partial_multilabel_confusion_matrix(preds, target,threshold=threshold)

    # reduce to binary task
    confusion_matrix = confusion_metrics.nansum(dim=0)
    
    tn, fp, fn, tp = confusion_matrix.reshape(-1)

    nominator = tp*tn - fp*fn
    denominator = torch.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

    denominator = 1 if denominator == 0 else denominator

    return nominator / denominator