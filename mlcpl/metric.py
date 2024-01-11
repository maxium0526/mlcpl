import torch

class PartialMultilabelMetric():
    def __init__(self, binary_metric, mask=None, reduction='mean'):
        self.binary_metric = binary_metric
        self.mask = mask
        self.reduction = reduction
    def __call__(self, preds, target):
        with torch.no_grad():
            num_categories = preds.shape[1]
            scores = torch.zeros(num_categories, dtype=torch.float32)

            for i in range(num_categories):
                category_preds, category_target = preds[:, i], target[:, i]
                scores[i] = PartialBinaryMetric(self.binary_metric)(category_preds, category_target)
            
            if self.reduction is None:
                return scores
            if self.reduction=='mean':
                if self.mask is None:
                    mean_score = torch.mean(scores[~torch.isnan(scores)])
                else:
                    selected_scores = scores[self.mask]
                    mean_score = torch.mean(selected_scores[~torch.isnan(selected_scores)])

                return mean_score

class PartialBinaryMetric():
    def __init__(self, binary_metric):
        self.binary_metric = binary_metric

    def __call__(self, preds, target):
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