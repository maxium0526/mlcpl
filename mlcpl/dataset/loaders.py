from torchvision import transforms
import numpy as np
import os
import pandas as pd
import xmltodict
import json
import glob
from .core import *

def MSCOCO(dataset_path, year='2014', split='train', partial_ratio=1.0, transform=transforms.ToTensor()):
    from pathlib import Path
    from pycocotools.coco import COCO

    num_categories = 80

    if split == 'train':
        subset = 'train'
    if split == 'valid':
        subset = 'val'

    coco = COCO(os.path.join(dataset_path, 'annotations', f'instances_{subset}{year}.json'))
    all_category_ids = coco.getCatIds()

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

    return MLCPLDataset(dataset_path, records, num_categories, transform)

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

    return MLCPLDataset(dataset_path, records, num_categories, transform)

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

    return MLCPLDataset(dataset_path, records, num_categories, transform)

def Open_Images(dataset_path, split=None, transform=transforms.ToTensor(), use_cache=True, cache_dir='output/dataset'):
    from pathlib import Path
    num_categories = 9605

    if use_cache and os.path.exists(os.path.join(cache_dir, 'train.csv')) and os.path.exists(os.path.join(cache_dir, 'valid.csv')):
        train_dataset = MLCPLDataset(dataset_path, df_to_records(pd.read_csv(os.path.join(cache_dir, 'train.csv'))), num_categories, transform)
        valid_dataset = MLCPLDataset(dataset_path, df_to_records(pd.read_csv(os.path.join(cache_dir, 'valid.csv'))), num_categories, transform)
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

        train_dataset = MLCPLDataset(dataset_path, train_records, num_categories, transform)
        valid_dataset = MLCPLDataset(dataset_path, valid_records, num_categories, transform)

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        records_to_df(train_records).to_csv(os.path.join(cache_dir, 'train.csv'))
        records_to_df(valid_records).to_csv(os.path.join(cache_dir, 'valid.csv'))

    if split == 'train':
        return train_dataset
    elif split == 'valid':
        return valid_dataset
    else:
        return train_dataset, valid_dataset
    
def Open_Images_V3(dataset_path, split='train', transform=transforms.ToTensor(), use_cache=True, cache_dir='output/dataset', check_images=True):
    from pathlib import Path

    if split == 'train':
        subset = 'train'
    elif split == 'valid':
        subset = 'validation'
    elif split == 'test':
        subset = 'test'

    categories = pd.read_csv(os.path.join(dataset_path, 'classes-trainable.txt'), header=None)[0].tolist()
    num_categories = len(categories)

    if use_cache and os.path.exists(os.path.join(cache_dir, split+'.csv')) and os.path.exists(os.path.join(cache_dir, 'valid.csv')):
        return MLCPLDataset(dataset_path, df_to_records(pd.read_csv(os.path.join(cache_dir, split+'.csv'))), num_categories, transform)

    df = pd.read_csv(os.path.join(dataset_path, subset, 'annotations-human.csv'))
    df = df.drop('Source', axis=1)
    df = df[df['LabelName'].isin(categories)] # drop the annotations not belong to trainable categories
    df['LabelName'] = df['LabelName'].apply(lambda x: categories.index(x))
    df_pos = df[df['Confidence'] == 1].drop('Confidence', axis=1)

    df_neg = df[df['Confidence'] == 0].drop('Confidence', axis=1)
    df_pos = df_pos.groupby('ImageID').agg(list).rename(columns={'LabelName': 'Positive'})
    df_neg = df_neg.groupby('ImageID').agg(list).rename(columns={'LabelName': 'Negative'})
    df = pd.merge(df_pos, df_neg, on='ImageID', how='outer')
    df = df.reset_index()
    df = df.rename(columns={'ImageID': 'Id'})

    df['Uncertain'] = np.nan
    df['Positive'] = df['Positive'].fillna("").apply(list).apply(lambda x: json.dumps(x))
    df['Negative'] = df['Negative'].fillna("").apply(list).apply(lambda x: json.dumps(x))
    df['Uncertain'] = df['Uncertain'].fillna("").apply(list).apply(lambda x: json.dumps(x))

    paths = [f'{subset}/{img_id}.jpg' for img_id in df['Id'].tolist()]
    df.insert(loc=1, column='Path', value=paths)

    if check_images:
        # check if the images exists:
        non_exist_indices = []
        num_exist, num_non_exist = 0, 0
        for i, row in df.iterrows():
            if os.path.isfile(os.path.join(dataset_path, row['Path'])):
                num_exist += 1
            else:
                num_non_exist += 1
                non_exist_indices.append(i)
            print(f'Checked {i+1}/{len(df)} images. Exist: {num_exist}. Not found: {num_non_exist}', end='\r')
        print()

        df = df.drop(non_exist_indices)
    
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(cache_dir, split+'.csv'))

    return MLCPLDataset(dataset_path, df_to_records(df), num_categories, transform)

def CheXpert(dataset_path, split='train', competition_categories=False, transform=transforms.ToTensor()):

    if split == 'train':
        subset = 'train'
    elif split == 'valid':
        subset = 'valid'
    elif split == 'test':
        subset = 'test'

    df = pd.read_csv(os.path.join(dataset_path, subset+'.csv'))

    if competition_categories is True:
        categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    else:
        categories = [
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Pleural Effusion',
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Lung Opacity',
            'Lung Lesion',
            'Pneumonia',
            'Pneumothorax',
            'Pleural Other',
            'Fracture',
            'Support Devices',
            ]
        categories.sort()

    num_categories = len(categories)
        
    records = []
    for i, row in df.iterrows():
        print(f'Loading row: {i+1} / {df.shape[0]}', end='\r')
        if subset in ['train', 'valid']:
            path = os.path.join(*(row['Path'].split('/')[1:]))
        else: # test
            path = os.path.join(*(row['Path'].split('/')))
        pos_category_nos = [no for no, category in enumerate(categories) if row[category]==1]
        neg_category_nos = [no for no, category in enumerate(categories) if row[category]==0]
        unc_category_nos = [no for no, category in enumerate(categories) if row[category]==-1]
        records.append((i, path, pos_category_nos, neg_category_nos, unc_category_nos))

    return MLCPLDataset(dataset_path, records, num_categories, transform=transform, categories=categories)

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

        dataset = MLCPLDataset(dataset_path, records, num_categories, transform, read_func=read_dicom)
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

        dataset = MLCPLDataset(dataset_path, records, num_categories, transform, read_func=read_dicom)
        outputs.append(dataset)

    return outputs