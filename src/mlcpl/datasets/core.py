import torch
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from PIL import Image
import json
from ..helper import dotdict
import copy
from typing import List, Callable, Tuple, Dict

def read_jpg(img_path):
    return Image.open(img_path).convert('RGB')

class MLCPLDataset(Dataset):
    """ A subclass of the torch Dataset for partially labeled multi-label datasets.
    """

    def __init__(self,
                 name: str,
                 dataset_path: str,
                 records: List[Tuple],
                 num_categories: int,
                 transform: Callable = transforms.ToTensor(),
                 categories: List[str] = None,
                 read_func: Callable = read_jpg,
                 ):
        """Construct a MLCPLDataset object

        Args:
            name (str): Dataset name

            dataset_path (str): The absolute/relative path of the dataset folder.

            records (List[tuple]): In consists of information of samples. Each tuple store a sample's (id, img_path, list of positive categories, list of negative categories)
            
            num_categories (int): The total number of categories.
            
            transform (Callable): The transform function applied to images.
            
            categories (List[str], optional): Categories's name. Defaults to None.
            
            read_func (Callable, optional): The function to read an image into a PILLOW Image instance. Defaults to read_jpg.
        """
        self.name = name
        self.dataset_path = dataset_path
        self.records = records
        self.categories = categories
        self.num_categories = num_categories
        self.transform = transform
        self.read_func = read_func

    def __len__(self):
        return len(self.records)
    
    def summary(self):
        """
        print general infromation of the dataset
        """
        statistics = self.get_statistics()

        print(statistics)

        info = f'{"-"*40}\n' \
            f'# Name: {self.name}\n' \
            f'# Dataset Path: {self.dataset_path}\n' \
            f'# # of Samples: {len(self)}\n' \
            f'# # of Categories: {self.num_categories}\n' \
            f'# Categories: {self.categories}\n' \
            f'# Label Proportion: {statistics["label_ratio"]*100:.2f}%\n' \
            f'# Positive-Negative Balance: {statistics["num_positive_labels"] / statistics["num_known_labels"]*100:.2f} : {statistics["num_negative_labels"] / statistics["num_known_labels"]*100:.2f}%\n' \
            f'{"-"*40}\n'
        
        print(info)


    def __getitem__(self, idx):
        """loads and returns a sample from the dataset at the given index idx. See https://pytorch.org/tutorials/beginner/basics/data_tutorial.html.
        """

        id, path, pos_categories, neg_categories = self.records[idx]
        img_path = os.path.join(self.dataset_path, path)
        img = self.read_func(img_path)
        img = self.transform(img)
        target = self.__to_one_hot(pos_categories, neg_categories)
        return img, target
    
    def __to_one_hot(self, pos_categories: List[int], neg_categories: List[int]) -> Tensor:
        """Produce an one-hot vector based on given categories.

        Args:
            pos_categories (List[int]): Positive categories
            
            neg_categories (List[int]): Negative categories

        Returns:
            Tensor: An one-hot tensor with shape (N).
        """
        return labels_to_one_hot(pos_categories, neg_categories, self.num_categories)
    
    def get_statistics(self):
        """Return some statistics given the dataset's informations

        Args:
            records (List[Tuple]): In consists of information of samples. Each tuple store a sample's (id, img_path, list of positive categories, list of negative categories)
            
            num_categories (int): The total number of categories

        Returns:
            Dict: A dict consisting of statistics
        """
        return get_statistics(self.records, self.num_categories)
    
    def drop_labels_uniform(self, target_label_proportion: float, seed: int = 526):
        self.records = drop_labels_uniform(self.records, target_label_proportion, seed=seed)
        return self
    
    def drop_labels_single_positive(self, seed: int = 526) -> List[Tuple]:
        """ Only one positive label is retained and all other labels are dropped for each sample.

        Args:
            seed (int, optional): The random seed. Defaults to 526.

        Returns:
            Self: self
        """

        rng = np.random.Generator(np.random.PCG64(seed=seed))

        new_records = []
        for (i, path, pos_categories, neg_categories) in self.records:
            new_pos_categories = rng.choice(pos_categories, 1).tolist() if len(pos_categories)> 0 else []
            new_records.append((i, path, new_pos_categories, []))
        self.records = new_records
        return self
    
    def drop_labels_fix_per_category(self, max_num_labels_per_category: int, seed: int = 526):
        """Drop labels with the FPC method. See https://openaccess.thecvf.com/content/CVPR2022/papers/Ben-Baruch_Multi-Label_Classification_With_Partial_Annotations_Using_Class-Aware_Selective_Loss_CVPR_2022_paper.pdf

        Args:
            max_num_labels_per_category (int): The maximum number of labels for each category
            
            seed (int, optional): The random seed. Defaults to 526.

        Returns:
            Self: self
        """

        rng = np.random.Generator(np.random.PCG64(seed=seed))

        per_category_positive_samples = []
        per_category_negative_samples = []
        for c in range(self.num_categories):
            per_category_positive_samples.append([])
            per_category_negative_samples.append([])

        for i, path, positives, negatives in self.records:
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
        for i, path, _, _ in self.records:
            positives = [c for c in range(self.num_categories) if i in per_category_positive_samples[c]]
            negatives = [c for c in range(self.num_categories) if i in per_category_negative_samples[c]]
            new_records.append((i, path, positives, negatives))
        
        self.records = new_records
        return self
    
    def __drop_labels_natural(self, N, alpha=1, beta=5, seed=526): # deprecated
        rng = np.random.Generator(np.random.PCG64(seed=seed))

        new_records = []
        for id, path, positives, negatives, uncertains in self.records:
            num_positives = len(positives)
            num_negatives = len(negatives)

            rng.shuffle(positives)
            rng.shuffle(negatives)

            new_num_positives = np.min([round(rng.beta(alpha, beta) * N), num_positives])
            new_num_negatives = np.min([round(rng.beta(alpha, beta) * N), num_negatives])

            new_records.append((id, path, positives[:new_num_positives], negatives[:new_num_negatives]))
        
        self.records = new_records

        return self

def labels_to_one_hot(positives: List[int], negatives: List[int], num_categories: int) -> Tensor:
    """Produce an one-hot vector based on given categories.

    Args:
        positives (List[int]): Positive categories
        negatives (List[int]): Negative categories
        num_categories (int): The number of categories (N).

    Returns:
        Tensor: An one-hot tensor with shape (N).
    """

    one_hot = torch.full((num_categories, ), torch.nan, dtype=torch.float32)
    one_hot[np.array(positives)] = 1.0
    one_hot[np.array(negatives)] = 0.0
    return one_hot

def one_hot_to_labels(one_hot: Tensor) -> Tuple[List[int], List[int]]:
    """Inverse function of labels_to_one_hot

    Args:
        one_hot (Tensor): An one-hot tensor with shape (N)

    Returns:
        Tuple[List[int], List[int]]: Two lists containing positive categories and negative categories, respectively.
    """

    positives = (one_hot ==1).nonzero().flatten().tolist()
    negatives = (one_hot ==0).nonzero().flatten().tolist()
    return positives, negatives
    
def get_statistics(records: List[Tuple], num_categories: int) -> Dict:
    """Return some statistics given the dataset's informations

    Args:
        records (List[Tuple]): In consists of information of samples. Each tuple store a sample's (id, img_path, list of positive categories, list of negative categories)
        num_categories (int): The total number of categories

    Returns:
        Dict: A dict consisting of statistics
    """

    num_categories = num_categories
    num_samples = len(records)
    num_labels = num_samples * num_categories

    all_positive_labels = []
    all_negative_labels = []

    for i, (_, _, positive_labels, negative_labels) in enumerate(records):
        all_positive_labels += positive_labels
        all_negative_labels += negative_labels

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

def fill_nan_to_negative(old_records: List[Tuple], num_categories: int) -> List[Tuple]:
    """ Put categories that are implicitly negative (i.e., neither positive nor negaitve) into negative

    Args:
        old_records (List[Tuple]): Original records
        num_categories (int): The total number of categories

    Returns:
        List[Tuple]: New records
    """
    new_records = []
    for (i, path, pos_categories, neg_categories) in old_records:
        new_neg_categories = [x for x in range(num_categories) if x not in pos_categories]
        new_records.append((i, path, pos_categories, new_neg_categories))
    return new_records

def drop_labels_uniform(old_records: List[Tuple], target_label_proportion: float, seed: int = 526) -> List[Tuple]:
    """Randomly drop labels that each label has the same probability to be dropped.

    Args:
        old_records (List[Tuple]): Original records.
        target_label_proportion (float): Desired label proportion of the new records.
        seed (int, optional): The random seed. Defaults to 526.

    Returns:
        List[Tuple]: New records.
    """

    rng = np.random.Generator(np.random.PCG64(seed=seed))

    new_records = []
    for (i, path, pos_categories, neg_categories) in old_records:
        new_pos_categories = [no for no in pos_categories if rng.random() < target_label_proportion]
        new_neg_categories = [no for no in neg_categories if rng.random() < target_label_proportion]
        new_records.append((i, path, new_pos_categories, new_neg_categories))
    return new_records

def records_to_df(records: List[Tuple]) -> pd.DataFrame:
    """Convert records to a dataframe

    Args:
        records (List[Tuple]): Records

    Returns:
        pd.DataFrame: A dataframe
    """

    df = pd.DataFrame(records, columns=['Id', 'Path', 'Positive', 'Negative'])
    return df

def df_to_records(df: pd.DataFrame) -> List[Tuple]:
    """Convert a dataframe to records

    Args:
        df (pd.DataFrame): A dataframe

    Returns:
        List[Tuple]: records
    """
    records = []
    for i, row in df.iterrows():
        id = row['Id']
        path = row['Path']
        pos_categories = json.loads(row['Positive'])
        neg_categories = json.loads(row['Negative'])
        records.append((id, path, pos_categories, neg_categories))
    
    return records

def split_dataset(dataset: MLCPLDataset, split_ratio: float =0.5, shuffle: bool = True, seed: int = 526) -> Tuple[MLCPLDataset, MLCPLDataset]:
    """Split a MLCPLDataset into two MLCPLDatasets

    Args:
        dataset (MLCPLDataset): The dataset to be divied
        divide_ratio (float, optional): Ratio of split. Defaults to 0.5.
        shuffle (bool, optional): Whether shuffling the samples before split. Defaults to True.
        seed (int, optional): The random seed. Defaults to 526.

    Returns:
        Tuple[MLCPLDataset, MLCPLDataset]: The two datasets
    """

    rng = np.random.Generator(np.random.PCG64(seed=seed))
    records = copy.deepcopy(dataset.records)
    rng.shuffle(records)
    split_at = round(len(records)*split_ratio)
    records_1 = records[:split_at]
    records_2 = records[split_at:]

    dataset_1 = copy.deepcopy(dataset)
    dataset_1.records = records_1

    dataset_2 = copy.deepcopy(dataset)
    dataset_2.records = records_2

    return dataset_1, dataset_2
