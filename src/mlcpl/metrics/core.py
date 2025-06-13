import torch
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