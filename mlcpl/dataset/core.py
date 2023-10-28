import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from PIL import Image
import json
import pydicom as dicom

def read_jpg(img_path):
    return Image.open(img_path).convert('RGB')

def read_dicom(img_path):
    numpy_img = dicom.dcmread(img_path).pixel_array
    pil_img = Image.fromarray(numpy_img).convert('RGB')
    return pil_img

class MLCPLDataset(Dataset):
    def __init__(self, dataset_path, records, num_categories, transform, read_func=read_jpg):
        self.dataset_path = dataset_path
        self.records = records
        self.num_categories = num_categories
        self.transform = transform
        self.read_func = read_func

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        id, path, pos_category_nos, neg_category_nos, unc_category_nos = self.records[idx]
        img_path = os.path.join(self.dataset_path, path)
        img = self.read_func(img_path)
        img = self.transform(img)
        target = to_one_hot(self.num_categories, pos_category_nos, neg_category_nos, unc_category_nos)
        return img, target

    def test(self):
        return self.__getitem__(0)

    def get_samples(self, indices):
        samples = []
        if indices is None:
            sub_df = self.df
        else:
            sub_df = self.df.iloc[indices]
        for i, row in sub_df.iterrows():
            print(f'Loading {i+1}/{sub_df.shape[0]}', end='\r')
            samples.append(self.__getitem__(i))
        return samples

    def statistics(self):
        num_categories = self.num_categories
        num_samples = len(self.df)
        num_labels = num_samples * num_categories
        
        num_positive_labels = 0
        num_negative_labels = 0
        num_uncertain_labels = 0
        for i, row in self.df.iterrows():
            num_positive_labels += len(json.loads(row['Positive'].replace(';', ',')))
            num_negative_labels += len(json.loads(row['Negative'].replace(';', ',')))
            num_uncertain_labels += len(json.loads(row['Uncertain'].replace(';', ',')))
        
        num_known_labels = num_positive_labels + num_negative_labels + num_uncertain_labels
        num_unknown_labels = num_labels - num_known_labels

        label_ratio = num_known_labels / num_labels

        return {
            'num_categories': num_categories,
            'num_samples': num_samples,
            'num_labels': num_labels,
            'num_positive_labels': num_positive_labels,
            'num_negative_labels': num_negative_labels,
            'num_uncertain_labels': num_uncertain_labels,
            'num_known_labels': num_known_labels,
            'num_unknown_labels': num_unknown_labels,
            'label_ratio': label_ratio,
        }

def fill_nan_to_negative(old_records, num_categories):
    new_records = []
    for (i, path, pos_category_nos, neg_category_nos, unc_category_nos) in old_records:
        new_neg_category_nos = [x for x in range(num_categories) if x not in pos_category_nos+unc_category_nos]
        new_records.append((i, path, pos_category_nos, new_neg_category_nos, unc_category_nos))
    return new_records

def drop_labels(old_records, target_partial_ratio, seed=526):
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    new_records = []
    for (i, path, pos_category_nos, neg_category_nos, unc_category_nos) in old_records:
        new_pos_category_nos = [no for no in pos_category_nos if rng.random() < target_partial_ratio]
        new_neg_category_nos = [no for no in neg_category_nos if rng.random() < target_partial_ratio]
        new_unc_category_nos = [no for no in unc_category_nos if rng.random() < target_partial_ratio]
        new_records.append((i, path, new_pos_category_nos, new_neg_category_nos, new_unc_category_nos))
    return new_records

def to_one_hot(num_categories, pos_category_nos, neg_category_nos, unc_category_nos):
    one_hot = torch.full((num_categories, ), torch.nan, dtype=torch.float32)
    one_hot[np.array(pos_category_nos)] = 1.0
    one_hot[np.array(neg_category_nos)] = 0.0
    one_hot[np.array(unc_category_nos)] = -1.0
    return one_hot

def records_to_df(records):
    df = pd.DataFrame(records, columns=['Id', 'Path', 'Positive', 'Negative', 'Uncertain'])
    return df

def df_to_records(df):
    records = []
    for i, row in df.iterrows():
        id = row['Id']
        path = row['Path']
        pos_category_nos = json.loads(row['Positive'])
        neg_category_nos = json.loads(row['Negative'])
        unc_category_nos = json.loads(row['Uncertain'])
        records.append((id, path, pos_category_nos, neg_category_nos, unc_category_nos))
    
    return records
