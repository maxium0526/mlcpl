import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from PIL import Image
import json
from ..helper import dotdict
import copy

def read_jpg(img_path):
    return Image.open(img_path).convert('RGB')

class MLCPLDataset(Dataset):
    def __init__(self, dataset_path, records, num_categories, transform, categories=None, read_func=read_jpg):
        self.dataset_path = dataset_path
        self.records = records
        self.categories = categories
        self.num_categories = num_categories
        self.transform = transform
        self.read_func = read_func

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        id, path, pos_category_nos, neg_category_nos = self.records[idx]
        img_path = os.path.join(self.dataset_path, path)
        img = self.read_func(img_path)
        img = self.transform(img)
        target = self.__to_one_hot(pos_category_nos, neg_category_nos)
        return img, target
    
    def __to_one_hot(self, pos_category_nos, neg_category_nos):
        return labels_to_one_hot(pos_category_nos, neg_category_nos, self.num_categories)
    
    def get_statistics(self):
        return get_statistics(self.records, self.num_categories)
    
    def drop_labels_random(self, target_partial_ratio, seed=526):
        self.records = drop_labels(self.records, target_partial_ratio, seed=seed)
        return self
    
    def drop_labels_fix_per_category(self, max_num_labels_per_category, seed=526):
        # https://openaccess.thecvf.com/content/CVPR2022/papers/Ben-Baruch_Multi-Label_Classification_With_Partial_Annotations_Using_Class-Aware_Selective_Loss_CVPR_2022_paper.pdf
        rng = np.random.Generator(np.random.PCG64(seed=seed))

        per_category_positive_samples = []
        per_category_negative_samples = []
        for c in range(self.num_categories):
            per_category_positive_samples.append([])
            per_category_negative_samples.append([])

        for i, path, positives, negatives, uncertains in self.records:
            for positive in positives:
                per_category_positive_samples[positive].append(i)
            for negative in negatives:
                per_category_negative_samples[negative].append(i)

        for c in range(self.num_categories):
            num_positive = len(per_category_positive_samples[c])
            num_negative = len(per_category_negative_samples[c])
            N_s = np.min([num_positive, num_negative, max_num_labels_per_category//2])
            
            rng.shuffle(per_category_positive_samples[c])
            rng.shuffle(per_category_negative_samples[c])

            per_category_positive_samples[c] = per_category_positive_samples[c][:N_s]
            per_category_negative_samples[c] = per_category_negative_samples[c][:N_s]

        new_records = []
        for i, path, _, _, _ in self.records:
            positives = [c for c in range(self.num_categories) if i in per_category_positive_samples[c]]
            negatives = [c for c in range(self.num_categories) if i in per_category_negative_samples[c]]
            new_records.append((i, path, positives, negatives, []))
        
        self.records = new_records
        return self
    
    def drop_labels_natural(self, N, alpha=1, beta=5, seed=526):
        rng = np.random.Generator(np.random.PCG64(seed=seed))

        new_records = []
        for id, path, positives, negatives, uncertains in self.records:
            num_positives = len(positives)
            num_negatives = len(negatives)

            rng.shuffle(positives)
            rng.shuffle(negatives)

            new_num_positives = np.min([round(rng.beta(alpha, beta) * N), num_positives])
            new_num_negatives = np.min([round(rng.beta(alpha, beta) * N), num_negatives])

            new_records.append((id, path, positives[:new_num_positives], negatives[:new_num_negatives], uncertains))
        
        self.records = new_records

        return self

def labels_to_one_hot(positives, negatives, num_categories):
    one_hot = torch.full((num_categories, ), torch.nan, dtype=torch.float32)
    one_hot[np.array(positives)] = 1.0
    one_hot[np.array(negatives)] = 0.0
    return one_hot

def one_hot_to_labels(one_hot):
    positives = (one_hot ==1).nonzero().flatten().tolist()
    negatives = (one_hot ==0).nonzero().flatten().tolist()
    return positives, negatives
    
def get_statistics(records, num_categories):
    num_categories = num_categories
    num_samples = len(records)
    num_labels = num_samples * num_categories

    all_positive_labels = []
    all_negative_labels = []

    for i, (_, _, positive_labels, negative_labels) in enumerate(records):
        print(f'Counting Labels: {i+1}/{len(records)}.', end='\r')
        
        all_positive_labels += positive_labels
        all_negative_labels += negative_labels
    
    print()

    all_positive_labels = pd.Series(all_positive_labels)
    all_negative_labels = pd.Series(all_negative_labels)

    num_positive_labels = len(all_positive_labels)
    num_negative_labels = len(all_negative_labels)

    num_known_labels = num_positive_labels + num_negative_labels
    num_unknown_labels = num_labels - num_known_labels

    label_ratio = num_known_labels / num_labels

    categories_has_pos = set(all_positive_labels)
    categories_has_neg = set(all_negative_labels)

    category_distributions = pd.DataFrame([(0)]*num_categories, columns=['dummy'])
    category_distributions['Num Positive'] = all_positive_labels.value_counts()
    category_distributions['Num Negative'] = all_negative_labels.value_counts()
    category_distributions = category_distributions.drop('dummy', axis=1)
    category_distributions = category_distributions.fillna(0)
    category_distributions['Total'] = category_distributions.sum(axis=1)

    sample_distributions = records_to_df(records)
    sample_distributions['Num Positive'] = sample_distributions.apply(lambda row: len(row['Positive']), axis=1)
    sample_distributions['Num Negative'] = sample_distributions.apply(lambda row: len(row['Negative']), axis=1)
    sample_distributions['Total'] = sample_distributions.apply(lambda row: row['Num Positive']+row['Num Negative'], axis=1)
    sample_distributions = sample_distributions.drop(columns=['Path', 'Positive', 'Negative'])

    statistics = dotdict({
        'num_categories': num_categories,
        'num_samples': num_samples,
        'num_labels': num_labels,
        'num_positive_labels': num_positive_labels,
        'num_negative_labels': num_negative_labels,
        'num_known_labels': num_known_labels,
        'num_unknown_labels': num_unknown_labels,
        'label_ratio': label_ratio,
        'num_trainable_categories': len(categories_has_pos.union(categories_has_neg)),
        'num_evaluatable_categories': len(categories_has_pos.intersection(categories_has_neg)),
        'category_distributions': category_distributions,
        'sample_distributions': sample_distributions,
    })

    return statistics

def fill_nan_to_negative(old_records, num_categories):
    new_records = []
    for (i, path, pos_category_nos, neg_category_nos) in old_records:
        new_neg_category_nos = [x for x in range(num_categories) if x not in pos_category_nos]
        new_records.append((i, path, pos_category_nos, new_neg_category_nos))
    return new_records

def drop_labels(old_records, target_partial_ratio, seed=526):
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    new_records = []
    for (i, path, pos_category_nos, neg_category_nos) in old_records:
        new_pos_category_nos = [no for no in pos_category_nos if rng.random() < target_partial_ratio]
        new_neg_category_nos = [no for no in neg_category_nos if rng.random() < target_partial_ratio]
        new_records.append((i, path, new_pos_category_nos, new_neg_category_nos))
    return new_records

def records_to_df(records):
    df = pd.DataFrame(records, columns=['Id', 'Path', 'Positive', 'Negative'])
    return df

def df_to_records(df):
    records = []
    for i, row in df.iterrows():
        id = row['Id']
        path = row['Path']
        pos_category_nos = json.loads(row['Positive'])
        neg_category_nos = json.loads(row['Negative'])
        records.append((id, path, pos_category_nos, neg_category_nos))
    
    return records

def divide(dataset, divide_ratio=0.5, shuffle=True, seed=526):
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    records = copy.deepcopy(dataset.records)
    rng.shuffle(records)
    split_at = round(len(records)*divide_ratio)
    records_1 = records[:split_at]
    records_2 = records[split_at:]

    dataset_1 = copy.deepcopy(dataset)
    dataset_1.records = records_1

    dataset_2 = copy.deepcopy(dataset)
    dataset_2.records = records_2

    return dataset_1, dataset_2
