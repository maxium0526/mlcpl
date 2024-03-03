import torch
from ..dataset import *

def mixup(tensor_1, tensor_2, lam=0.5):
    return lam * tensor_1 + (1 - lam) * tensor_2

def mix_images(images):
    images = torch.stack(images)
    new_image = torch.mean(images, 0)

    return new_image

def logic_mix_targets(targets, strict_negative=False, unknown_as=None):
        targets = torch.stack(targets)

        if unknown_as is not None:
            targets = torch.where(torch.isnan(targets), unknown_as, targets)

        #compute must positive
        t = torch.where(torch.isnan(targets), 0, targets)
        t = torch.sum(t, dim=0)
        must_positive = torch.sign(t)

        #compute must negative
        if strict_negative:
            t = torch.where(targets == 0, 0, 1)
            t = torch.sum(t, dim=0)
            must_negative = torch.where(t==0, 1, 0)
        else:
            t = torch.where(targets == 0, 1, 0)
            t = torch.sum(t, dim=0)
            must_negative = torch.where(torch.logical_and(t > 0, must_positive != 1), 1, 0)

        if torch.max(must_negative+must_positive) > 1:
            print('Found a label is both positive and negative!')

        new_target = must_positive
        new_target = torch.where(must_negative == 1, -1, new_target)
        new_target = torch.where(new_target == 0, torch.nan, new_target)
        new_target = torch.where(new_target == -1, 0, new_target)

        return new_target

class LogicMixTargets():
    def __init__(self, strict_negative=False, unknown_as=None) -> None:
        self.strict_negative = strict_negative
        self.unknown_as = unknown_as

    def __call__(self, targets):
        return logic_mix_targets(targets, strict_negative=self.strict_negative, unknown_as=self.unknown_as)

def estimate_target_mix_strategy(strategy, partial_dataset, full_dataset, seed=526):
    if len(partial_dataset) != len(full_dataset):
        raise Exception('The datasets have different sizes.')
    
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    
    num_samples = len(partial_dataset)
    num_categories = partial_dataset.num_categories

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    records = []
    for i in range(num_samples):
        print(f'{i}/{num_samples}; tp={total_tp}, tn={total_tn}, fp={total_fp}, fn={total_fn}', end='\r')
        random_index = rng.integers(0, num_samples)

        partial_records = [partial_dataset.records[a] for a in [i, random_index]]
        partial_targets = [labels_to_one_hot(p, n, u, num_categories) for _, _, p, n, u in partial_records]

        full_records = [full_dataset.records[a] for a in [i, random_index]]
        full_targets = [labels_to_one_hot(p, n, u, num_categories) for _, _, p, n, u in full_records]

        partial_new_target = strategy(partial_targets)
        full_new_target = logic_mix_targets(full_targets, strict_negative=True)

        tp, tn, fp, fn = 0, 0, 0, 0
        for partial_target, full_target in zip(partial_new_target, full_new_target):
            if torch.isnan(partial_target):
                continue

            if partial_target == 1:
                if full_target == 1:
                    tp += 1
                elif full_target == 0:
                    fp += 1
                else:
                    raise Exception(f'Full target = {full_target}. Neither 0 nor 1.')
            elif partial_target == 0:
                if full_target == 1:
                    fn += 1
                elif full_target == 0:
                    tn += 1
                else:
                    raise Exception(f'Full target = {full_target}. Neither 0 nor 1.')
            else:
                raise Exception(f'Full target = {partial_target}. Neither 0 nor 1 nor nan.')
            
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        record = (tp, tn, fp, fn)
        records.append(record)
            
    print()

    pd.DataFrame(records, columns=['TP', 'TN', 'FP', 'FN']).to_csv('false_labels.csv')