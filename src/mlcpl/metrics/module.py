from .functional import *

class PartialMultilabelAveragePrecision:
    def __init__(
            self,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            thresholds=None,
            ):
        self.average = average
        self.thresholds = thresholds
    
    def __call__(self, preds, target):
        return partial_multilabel_average_precision(
            preds,
            target,
            average=self.average,
            thresholds=self.thresholds
        )

class PartialMultilabelAUROC:
    def __init__(
            self,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            thresholds=None,
            ):
        self.average = average
        self.thresholds = thresholds
    
    def __call__(self, preds, target):
        return partial_multilabel_auroc(
            preds,
            target,
            average=self.average,
            thresholds=self.thresholds
        )

class PartialMultilabelFBetaScore:
    def __init__(
            self,
            beta: float = 1.0,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.beta = beta
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_fbeta_score(
            preds,
            target,
            beta=self.beta,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelF1Score:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_f1_score(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelPrecision:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_precision(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelRecall:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_recall(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelSensitivity:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_sensitivity(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelSpecificity:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_specificity(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )
    
class PartialMultilabelPrecisionAtFixedRecall:
    def __init__(
            self,
            min_recall: float,
            thresholds = None,
            ):
        self.min_recall = min_recall
        self.thresholds = thresholds
    
    def __call__(self, preds, target):
        return partial_multilabel_precision_at_fixed_recall(
            preds,
            target,
            min_recall=self.min_recall,
            thresholds=self.thresholds
        )

class PartialMultilabelRecallAtFixedPrecision:
    def __init__(
            self,
            min_precision: float,
            thresholds = None,
            ):
        self.min_precision = min_precision
        self.thresholds = thresholds
    
    def __call__(self, preds, target):
        return partial_multilabel_recall_at_fixed_precision(
            preds,
            target,
            min_precision=self.min_precision,
            thresholds=self.thresholds
        )

class PartialMultilabelSensitivityAtSpecificity:
    def __init__(
            self,
            min_specificity: float,
            thresholds = None,
            ):
        self.min_specificity = min_specificity
        self.thresholds = thresholds
    
    def __call__(self, preds, target):
        return partial_multilabel_sensitivity_at_specificity(
            preds,
            target,
            min_specificity=self.min_specificity,
            thresholds=self.thresholds
        )

class PartialMultilabelSpecificityAtSensitivity:
    def __init__(
            self,
            min_sensitivity: float,
            thresholds = None,
            ):
        self.min_sensitivity = min_sensitivity
        self.thresholds = thresholds
    
    def __call__(self, preds, target):
        return partial_multilabel_specificity_at_sensitivity(
            preds,
            target,
            min_sensitivity=self.min_sensitivity,
            thresholds=self.thresholds
        )

class PartialMultilabelROC:
    def __init__(
            self,
            thresholds = None,
            ):
        self.thresholds = thresholds
    
    def __call__(self, preds, target):
        return partial_multilabel_roc(
            preds,
            target,
            thresholds=self.thresholds
        )

class PartialMultilabelPrecisionRecallCurve:
    def __init__(
            self,
            thresholds = None,
            ):
        self.thresholds = thresholds
    
    def __call__(self, preds, target):
        return partial_multilabel_precision_recall_curve(
            preds,
            target,
            thresholds=self.thresholds
        )

class PartialMultilabelAccuracy:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_accuracy(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelCalibrationError:
    def __init__(
            self,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            norm: Literal['l1', 'l2', 'max', 'ECE', 'ACE', 'MCE'] = 'l1'
            ):
        self.average = average
        self.norm = norm
    
    def __call__(self, preds, target):
        return partial_multilabel_calibration_error(
            preds,
            target,
            average=self.average,
            norm=self.norm
        )

class PartialMultilabelExpectedCalibrationError:
    def __init__(
            self,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_expected_calibration_error(
            preds,
            target,
            average=self.average
        )

class PartialMultilabelAverageCalibrationError:
    def __init__(
            self,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_average_calibration_error(
            preds,
            target,
            average=self.average
        )

class PartialMultilabelMaximumCalibrationError:
    def __init__(
            self,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_maximum_calibration_error(
            preds,
            target,
            average=self.average
        )

class PartialMultilabelCohenKappa:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            weights: Literal['linear', 'quadratic', 'none'] = 'none',
            ):
        self.threshold = threshold
        self.average = average
        self.weights = weights
    
    def __call__(self, preds, target):
        return partial_multilabel_cohen_kappa(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
            weights=self.weights
        )

class PartialMultilabelConfusionMatrix:
    def __init__(
            self,
            threshold: float = 0.5,
            normalize: Literal['none', 'true', 'pred', 'all'] = 'none',
            ):
        self.threshold = threshold
        self.normalize = normalize
    
    def __call__(self, preds, target):
        return partial_multilabel_confusion_matrix(
            preds,
            target,
            threshold=self.threshold,
            normalize=self.normalize
        )

class PartialMultilabelDice:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'samples', 'none'] = 'micro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_dice(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelExactMatch:
    def __init__(
            self,
            threshold: float = 0.5,
            ):
        self.threshold = threshold
    
    def __call__(self, preds, target):
        return partial_multilabel_exact_match(
            preds,
            target,
            threshold=self.threshold
        )

class PartialMultilabelHammingDistance:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_hamming_distance(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelHingeLoss:
    def __init__(
            self,
            squared: bool = False,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.squared = squared
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_hinge_loss(
            preds,
            target,
            squared=self.squared,
            average=self.average,
        )

class PartialMultilabelJaccardIndex:
    def __init__(
            self,
            threshold: float = 0.5,
            average: Literal['macro', 'micro', 'weighted', 'none'] = 'macro',
            ):
        self.threshold = threshold
        self.average = average
    
    def __call__(self, preds, target):
        return partial_multilabel_jaccard_index(
            preds,
            target,
            threshold=self.threshold,
            average=self.average,
        )

class PartialMultilabelRankingAveragePrecision:
    def __init__(
            self,
            ):
        pass
    
    def __call__(self, preds, target):
        return partial_multilabel_ranking_average_precision(
            preds,
            target,
        )

class PartialMultilabelRankingLoss:
    def __init__(
            self,
            ):
        pass
    
    def __call__(self, preds, target):
        return partial_multilabel_ranking_loss(
            preds,
            target,
        )

class PartialMultilabelMatthewsCorrCoef:
    def __init__(
            self,
            threshold: float = 0.5,
            ):
        self.threshold = threshold
    
    def __call__(self, preds, target):
        return partial_multilabel_matthews_corrcoef(
            preds,
            target,
            threshold=self.threshold,
        )