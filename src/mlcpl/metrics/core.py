import torch
import numpy as np

class PartialMultilabelMetric():
    def __init__(self, binary_metric, mask=None, reduction='mean'):
        self.binary_metric = binary_metric
        self.mask = mask
        self.reduction = reduction
    def __call__(self, preds, target, threshold=None):
        with torch.no_grad():
            num_categories = preds.shape[1]
            scores = torch.zeros(num_categories, dtype=torch.float32)

            for i in range(num_categories):
                category_preds, category_target = preds[:, i], target[:, i]
                scores[i] = PartialBinaryMetric(self.binary_metric)(category_preds, category_target, threshold=threshold)
            
            if self.reduction is None:
                return scores
            if self.reduction=='mean':
                if self.mask is None:
                    mean_score = torch.mean(scores[~torch.isnan(scores)])
                else:
                    selected_scores = scores[self.mask]
                    mean_score = torch.mean(selected_scores[~torch.isnan(selected_scores)])

                return mean_score
            
class PartialPerSampleMultilabelMetric():
    def __init__(self, binary_metric, mask=None, reduction='mean'):
        self.metric = PartialMultilabelMetric(binary_metric, reduction=reduction)
        self.mask = mask

    def __call__(self, preds, target, threshold=None):
        with torch.no_grad():
            
            if self.mask:
                preds = preds[:, self.mask]
                target = target[:, self.mask]

        return self.metric(preds.transpose(0, 1), target.transpose(0, 1), threshold=threshold)

class PartialOverallMultilabelMetric():
    def __init__(self, binary_metric, mask=None):
        self.binary_metric = binary_metric
        self.mask = mask
    def __call__(self, preds, target, threshold=None):
        with torch.no_grad():

            if self.mask:
                preds = preds[:, self.mask]
                target = target[:, self.mask]
            
            preds = preds.flatten()
            target = target.flatten()

            return PartialBinaryMetric(self.binary_metric)(preds, target, threshold=threshold)
 

class PartialBinaryMetric():
    def __init__(self, binary_metric):
        self.binary_metric = binary_metric

    def __call__(self, preds, target, threshold=None):
        with torch.no_grad():
            if target.dtype == torch.int8:
                labeled_map = (target != -1)
            else:
                labeled_map = ~torch.isnan(target)

            preds, target = preds[labeled_map], target[labeled_map]

            if target.numel() == 0 :
                score = torch.tensor(torch.nan)
            elif (target==1.0).sum() == 0:
                score = torch.tensor(torch.nan)
            elif (target==0.0).sum() == 0:
                score = torch.tensor(torch.nan)
            else:
                if threshold:
                    score = self.binary_metric(preds, target.to(torch.int32), threshold=threshold)
                else:
                    score = self.binary_metric(preds, target.to(torch.int32))

            return score
        
class Open_Images_V3_Group_Metric():
    def __init__(self, binary_metric, categories_num_labels):
        self.binary_metric = binary_metric

        _, sorted_indices = torch.sort(torch.tensor(categories_num_labels))
        self.groups = []
        for i in range(0, 5000, 1000):
            self.groups.append(sorted_indices[i: i+1000])
    
    def __call__(self, preds, target):
        
        group_metric = PartialMultilabelMetric(self.binary_metric)
        group_scores = []
        for group in self.groups:
            score = group_metric(preds[:, group], target[:, group])
            group_scores.append(score)

        return torch.tensor(group_scores)

class SearchThreshold():
    def __init__(self, metric, iterations=2, interval=11, mode='max'):
        self.metric = metric
        self.iterations = iterations
        self.interval = interval
        self.mode = mode

    def __call__(self, preds, target):
        self.records = []
        low, high = 0, 1

        for i in range(self.iterations):
            best_threshold, best_score = None, None

            thresholds, step = np.linspace(low, high, num=self.interval, retstep=True)
            for threshold in thresholds:
                score = self.metric(preds, target, threshold=threshold)
                if best_score is None or (self.mode=='max' and score>best_score) or (self.mode=='min' and score<best_score):
                    best_score = score
                    best_threshold = threshold
                    low = np.max([threshold - step, 0])
                    high = np.min([threshold + step, 1])

                self.records.append((threshold, score))

        self.records.sort(key=lambda x: x[0])

        return max(self.records, key=lambda x: x[1])[1]
            
class PartialMultilabelPairwiseCosineSimilarity():
    def __init__(self, reduction='mean', batch_size=512):
        self.reduction = reduction
        self.batch_size = batch_size
    
    def __call__(self, preds, targets):
        num_categories = targets.shape[1]

        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(preds, targets), batch_size=self.batch_size)

        with torch.no_grad():

            positive_category_similarities = torch.zeros((len(dataloader), num_categories))
            negative_category_similarities = torch.zeros((len(dataloader), num_categories))
            for batch, (pred, target) in enumerate(dataloader):
                similarity_matrix = torch.matmul(pred, pred.T)

                for c in range(num_categories):

                    labels = target[:, c].view(-1, 1)
                    mask = torch.matmul(labels, labels.T)

                    positive_mask = mask == 1
                    negative_mask = mask == 0

                    positive_similarity = (similarity_matrix * positive_mask).sum() / positive_mask.sum()
                    negative_similarity = (similarity_matrix * negative_mask).sum() / negative_mask.sum()

                    positive_category_similarities[batch, c] = positive_similarity
                    negative_category_similarities[batch, c] = negative_similarity

            if self.reduction == 'mean':
                return positive_category_similarities.nanmean(), negative_category_similarities.nanmean()
            
            return positive_category_similarities.nanmean(dim=0), negative_category_similarities.nanmean(dim=0)

class PartialMLCalibrationError():
    def __init__(self, n_bins=15, mode='l1'):
        self.n_bins = n_bins
        self.mode = mode

        self.bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)

    def __call__(self, preds, target):
        preds = torch.nn.functional.sigmoid(preds.flatten())
        target = target.flatten()

        if target.dtype == torch.int8:
            labeled_map = (target != -1)
        else:
            labeled_map = ~torch.isnan(target)
        preds, target = preds[labeled_map], target[labeled_map]

        confidences, accuracies = preds, target

        with torch.no_grad():
            acc_bin, conf_bin, prop_bin = self._binning_bucketize(confidences, accuracies, self.bin_boundaries)
            
        if self.mode == "l1":
            return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        if self.mode == "max":
            return torch.max(torch.abs(acc_bin - conf_bin))
        if self.mode == "l2":
            return torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
            return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
        
        if self.mode == 'ECE':
            return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        if self.mode == 'ACE':
            return torch.mean(torch.abs(acc_bin - conf_bin))
        if self.mode == 'MCE':
            return torch.max(torch.abs(acc_bin - conf_bin))

        return None

    def _binning_bucketize(self, confidences, accuracies, bin_boundaries):
        """Compute calibration bins using ``torch.bucketize``. Use for ``pytorch >=1.6``.

        Args:
            confidences: The confidence (i.e. predicted prob) of the top1 prediction.
            accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
            bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

        Returns:
            tuple with binned accuracy, binned confidence and binned probabilities

        """
        accuracies = accuracies.to(dtype=confidences.dtype)
        acc_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
        conf_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
        count_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)

        indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1

        count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

        conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
        conf_bin = torch.nan_to_num(conf_bin / count_bin)

        acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
        acc_bin = torch.nan_to_num(acc_bin / count_bin)

        prop_bin = count_bin / count_bin.sum()
        return acc_bin, conf_bin, prop_bin

class CalibrationError():
    def __init__(self, n_bins=15, mode='l1'):
        self.n_bins = n_bins
        self.mode = mode

        self.bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)

    def __call__(self, preds, target):
        preds = torch.nn.functional.sigmoid(preds)
        target = target.flatten()

        confidences, accuracies = preds, target

        with torch.no_grad():
            acc_bin, conf_bin, prop_bin = self._binning_bucketize(confidences, accuracies, self.bin_boundaries)

        if self.mode == "l1":
            return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        if self.mode == "max":
            return torch.max(torch.abs(acc_bin - conf_bin))
        if self.mode == "l2":
            return torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
            return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
        
        if self.mode == 'ECE':
            return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        if self.mode == 'ACE':
            return torch.mean(torch.abs(acc_bin - conf_bin))
        if self.mode == 'MCE':
            return torch.max(torch.abs(acc_bin - conf_bin))

        return None

    def _binning_bucketize(self, confidences, accuracies, bin_boundaries):
        """Compute calibration bins using ``torch.bucketize``. Use for ``pytorch >=1.6``.

        Args:
            confidences: The confidence (i.e. predicted prob) of the top1 prediction.
            accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
            bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

        Returns:
            tuple with binned accuracy, binned confidence and binned probabilities

        """
        accuracies = accuracies.to(dtype=confidences.dtype)
        acc_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
        conf_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
        count_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)

        indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1

        count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

        conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
        conf_bin = torch.nan_to_num(conf_bin / count_bin)

        acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
        acc_bin = torch.nan_to_num(acc_bin / count_bin)

        prop_bin = count_bin / count_bin.sum()
        return acc_bin, conf_bin, prop_bin

# metric = PartialMLCalibrationError()
# preds = torch.tensor([[0.01, 0.25, 0.25, 0.55, 0.75, 0.75]])
# target = torch.tensor([[0, 0, 0, 1, 1, 1]])
# metric(preds, target)
