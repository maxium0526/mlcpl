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
            
            
