import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from PIL import Image
import xmltodict
import json
import glob
from tqdm import tqdm
import time
import pydicom as dicom
import glob

def read_jpg(img_path):
    return Image.open(img_path).convert('RGB')

def read_dicom(img_path):
    numpy_img = dicom.dcmread(img_path).pixel_array
    pil_img = Image.fromarray(numpy_img).convert('RGB')
    return pil_img

class MLCPLDataset(Dataset):
    def __init__(self, dataset_path, df, num_categories, transform, read_func=read_jpg):
        self.dataset_path = dataset_path
        self.df = df
        self.num_categories = num_categories
        self.transform = transform
        self.read_func = read_func

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.dataset_path, row['Path'])
        img = self.read_func(img_path)
        img = self.transform(img)
        pos_category_nos = json.loads(row['Positive'].replace(';', ','))
        neg_category_nos = json.loads(row['Negative'].replace(';', ','))
        unc_category_nos = json.loads(row['Uncertain'].replace(';', ','))
        target = to_one_hot(self.num_categories, np.array(pos_category_nos), np.array(neg_category_nos), np.array(unc_category_nos))
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
    one_hot[pos_category_nos] = 1.0
    one_hot[neg_category_nos] = 0.0
    one_hot[unc_category_nos] = -1.0
    return one_hot

def records_to_df(records):
    records = [(i, path, json.dumps(pos_category_nos).replace(',', ';'), json.dumps(neg_category_nos).replace(',', ';'), json.dumps(unc_category_nos).replace(',', ';')) for (i, path, pos_category_nos, neg_category_nos, unc_category_nos) in records]
    df = pd.DataFrame(records, columns=['Id', 'Path', 'Positive', 'Negative', 'Uncertain'])
    return df

def MSCOCO(dataset_path, year='2014', split='train', partial_ratio=1.0, transform=transforms.ToTensor()):
    from pycocotools.coco import COCO

    if split == 'train':
        subset = 'train'
    if split == 'valid':
        subset = 'val'

    coco = COCO(os.path.join(dataset_path, 'annotations', f'instances_{subset}{year}.json'))
    all_category_ids = coco.getCatIds()
    num_categories = len(all_category_ids)

    records = []
    image_ids = coco.getImgIds()
    for i, img_id in enumerate(image_ids):
        print(f'Loading row: {i+1} / {len(image_ids)}', end='\r')
        img_filename = coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(subset+year, img_filename)
        pos_category_ids = [coco.loadAnns(annotation_id)[0]['category_id'] for annotation_id in coco.getAnnIds(imgIds=img_id)]
        pos_category_ids = list(set(pos_category_ids))
        pos_category_nos = [all_category_ids.index(category_id) for category_id in pos_category_ids]
        pos_category_nos.sort()
        records.append((img_id, path, pos_category_nos, [], []))
    
    records = fill_nan_to_negative(records, num_categories)
    records = drop_labels(records, partial_ratio)

    return MLCPLDataset(dataset_path, records_to_df(records), num_categories, transform)

def Pascal_VOC_2007(dataset_path, split='train', partial_ratio=1.0, transform=transforms.ToTensor()):

    if split == 'train':
        subset = 'trainval'
    elif split == 'valid':
        subset = 'test'

    all_category_ids = set({})
    paths = glob.glob(os.path.join(dataset_path, 'ImageSets', 'Main', '*.txt'))
    for i, path in enumerate(paths):
        print(f'Loading row: {i+1} / {len(paths)}', end='\r')
        basename = os.path.basename(path)
        if '_' in basename:
            all_category_ids.add(basename.split('_')[0])
    all_category_ids = list(all_category_ids)
    all_category_ids.sort()
    num_categories = len(all_category_ids)

    img_nos = pd.read_csv(os.path.join(dataset_path, 'ImageSets', 'Main', subset+'.txt'), sep=' ', header=None, names=['Id'], dtype=str)
    records = []
    for i, row in img_nos.iterrows():
        print(f'Loading row: {i+1} / {img_nos.shape[0]}', end='\r')
        img_no = row['Id']
        path = os.path.join('JPEGImages', img_no+'.jpg')

        xml_path = os.path.join(dataset_path, 'Annotations', f'{img_no}.xml')
        with open(xml_path, 'r') as f:
            data = f.read()
        xml = xmltodict.parse(data)
        detections = xml['annotation']['object']
        if isinstance(detections, list):
            pos_category_ids = list(set([detection['name'] for detection in detections]))
            pos_category_ids.sort()
        else:
            pos_category_ids = [detections['name']]
        
        pos_category_nos = [all_category_ids.index(i) for i in pos_category_ids]
        records.append((img_no, path, pos_category_nos, [], []))

    records = fill_nan_to_negative(records, num_categories)
    records = drop_labels(records, partial_ratio)

    return MLCPLDataset(dataset_path, records_to_df(records), num_categories, transform)

def LVIS(dataset_path, split='train', transform=transforms.ToTensor()):
    from lvis import LVIS

    if split == 'train':
        subset = 'train'
    elif split == 'valid':
        subset = 'val'

    print(f'Loading split {subset}')
    lvis = LVIS(os.path.join(dataset_path, 'annotations', f'lvis_v1_{subset}.json'))       
    all_category_ids = lvis.get_cat_ids()
    num_categories = len(all_category_ids)

    records = []
    imgs = lvis.load_imgs(lvis.get_img_ids())
    for i, img in enumerate(imgs):
        print(f'Loading row: {i+1} / {len(imgs)}', end='\r')
        img_id = img['id']
        path = os.path.join(*img['coco_url'].split('/')[-2:])
        annotation_ids = lvis.get_ann_ids(img_ids=[img_id])
        pos_category_ids = [annotation['category_id'] for annotation in lvis.load_anns(annotation_ids)] + img['not_exhaustive_category_ids']
        pos_category_ids = list(set(pos_category_ids))
        pos_category_ids.sort()
        pos_category_nos = [all_category_ids.index(pos_category_id) for pos_category_id in pos_category_ids]
        neg_category_nos = [all_category_ids.index(neg_category_id) for neg_category_id in img['neg_category_ids']]
        records.append((img_id, path, pos_category_nos, neg_category_nos, []))

    return MLCPLDataset(dataset_path, records_to_df(records), num_categories, transform)

def Open_Images(dataset_path, split=None, transform=transforms.ToTensor(), use_cache=True, cache_dir='output/dataset'):
    from pathlib import Path
    num_categories = 9605

    if use_cache and os.path.exists(os.path.join(cache_dir, 'train.csv')) and os.path.exists(os.path.join(cache_dir, 'valid.csv')):
        train_dataset = MLCPLDataset(dataset_path, pd.read_csv(os.path.join(cache_dir, 'train.csv')), num_categories, transform)
        valid_dataset = MLCPLDataset(dataset_path, pd.read_csv(os.path.join(cache_dir, 'valid.csv')), num_categories, transform)
    else:
        raw_data = pd.read_csv(os.path.join(dataset_path, 'data.csv'))

        categories = set({})
        for i, raw in raw_data.iterrows():
            print(f'Finding categories: {len(categories)}; {i+1} / {raw_data.shape[0]}', end='\r')
            if len(categories) == num_categories:
                break
            pos_category_ids = json.loads(raw['label'].replace("'", '"'))
            neg_category_ids = json.loads(raw['label_neg'].replace("'", '"'))
            categories.update(pos_category_ids)
            categories.update(neg_category_ids)
        categories = list(categories)
        categories.sort()
        num_categories = len(categories)
        print()

        category_map = {}
        for i, category in enumerate(categories):
            category_map[category] = i

        train_records, valid_records = [], []
        for i, raw in raw_data.iterrows():
            print(f'Loading row: {i+1} / {raw_data.shape[0]}', end='\r')
            pos_category_ids = json.loads(raw['label'].replace("'", '"'))
            neg_category_ids = json.loads(raw['label_neg'].replace("'", '"'))
            pos_category_nos = list(map(lambda x: category_map[x], pos_category_ids))
            neg_category_nos = list(map(lambda x: category_map[x], neg_category_ids))

            record = (i, raw['filepath'], pos_category_nos, neg_category_nos, [])
            if raw['split_name'] == 'train':
                train_records.append(record)
            else:
                valid_records.append(record)

        train_dataset = MLCPLDataset(dataset_path, records_to_df(train_records), num_categories, transform)
        valid_dataset = MLCPLDataset(dataset_path, records_to_df(valid_records), num_categories, transform)

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        train_dataset.df.to_csv(os.path.join(cache_dir, 'train.csv'))
        valid_dataset.df.to_csv(os.path.join(cache_dir, 'valid.csv'))

    if split == 'train':
        return train_dataset
    elif split == 'valid':
        return valid_dataset
    else:
        return train_dataset, valid_dataset

def CheXpert(dataset_path, competition_categories=False):
    results = []
    for split in ['train', 'valid']:
        df = pd.read_csv(os.path.join(dataset_path, split+'.csv'))

        if competition_categories is True:
            categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        else:
            categories = df.columns.tolist()[5:]
            categories.sort()
            
        records = []
        for i, row in df.iterrows():
            print(f'Loading row: {i+1} / {df.shape[0]}', end='\r')
            path = os.path.join(*(row['Path'].split('/')[1:]))
            pos_category_nos = [no for no, category in enumerate(categories) if row[category]==1]
            neg_category_nos = [no for no, category in enumerate(categories) if row[category]==0]
            unc_category_nos = [no for no, category in enumerate(categories) if row[category]==-1]
            records.append((i, path, pos_category_nos, neg_category_nos, unc_category_nos))
        results.append(records)
    results.append(categories)

    return results

def RSNA_2023_2D(dataset_path, train_valid_ratio=0.8, seed=526, transform=transforms.ToTensor()):
    df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    df = df.sample(frac=1, random_state=seed)
    df_train, df_valid = np.split(df, [round(len(df) * train_valid_ratio)])

    categories = df.columns[1:-1].tolist() # remove patient_id and any_injury
    num_categories = len(categories)

    outputs = []

    for df_sub in [df_train, df_valid]:
        records = []
        counter = 0
        for i, row in df_sub.iterrows():
            patient_id = row['patient_id']
            patient_dir = os.path.join(dataset_path, 'train_images', str(patient_id))
            img_dirs = glob.glob(patient_dir+'/**/*.dcm')
            positive_nos = [no for no, category in enumerate(categories) if row[category] == 1]
            negative_nos = [no for no, category in enumerate(categories) if row[category] == 0]
            for img_dir in img_dirs:
                img_dir = img_dir.replace(dataset_path, '')
                img_dir = img_dir[1:] if img_dir[0] == '/' else img_dir
                records.append((patient_id, img_dir, positive_nos, negative_nos, []))
                counter += 1

        dataset = MLCPLDataset(dataset_path, records_to_df(records), num_categories, transform, read_func=read_dicom)
        outputs.append(dataset)

    return outputs

def RSNA_2023_3D(dataset_path, train_valid_ratio=0.8, seed=526, transform=transforms.ToTensor()):
    df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    df = df.sample(frac=1, random_state=seed)
    df_train, df_valid = np.split(df, [round(len(df) * train_valid_ratio)])

    categories = df.columns[1:-1].tolist() # remove patient_id and any_injury
    num_categories = len(categories)

    outputs = []

    for df_sub in [df_train, df_valid]:
        records = []
        counter = 0
        for i, row in df_sub.iterrows():
            patient_id = row['patient_id']
            patient_dir = os.path.join(dataset_path, 'train_images', str(patient_id))
            img_dirs = glob.glob(patient_dir+'/**/*.dcm')
            positive_nos = [no for no, category in enumerate(categories) if row[category] == 1]
            negative_nos = [no for no, category in enumerate(categories) if row[category] == 0]
            for img_dir in img_dirs:
                img_dir = img_dir.replace(dataset_path, '')
                img_dir = img_dir[1:] if img_dir[0] == '/' else img_dir
                records.append((patient_id, img_dir, positive_nos, negative_nos, []))
                counter += 1

        dataset = MLCPLDataset(dataset_path, records_to_df(records), num_categories, transform, read_func=read_dicom)
        outputs.append(dataset)

    return outputs