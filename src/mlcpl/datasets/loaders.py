from torchvision import transforms
import numpy as np
import os
import pandas as pd
import xmltodict
import json
import glob
from .core import *

def MSCOCO(dataset_path, year='2014', split='train', transform=transforms.ToTensor()):
    """Load the MS-COCO dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        year:
            The year of split. Defaults to '2014'.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

    Returns:
        A ``MLCPLDataset`` object.

    """
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
        # print(f'Loading MSCOCO {split}: {i+1} / {len(image_ids)}', end='\r')
        img_filename = coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(subset+year, img_filename)
        pos_category_ids = [coco.loadAnns(annotation_id)[0]['category_id'] for annotation_id in coco.getAnnIds(imgIds=img_id)]
        pos_category_ids = list(set(pos_category_ids))
        pos_category_nos = [all_category_ids.index(category_id) for category_id in pos_category_ids]
        pos_category_nos.sort()
        records.append((img_id, path, pos_category_nos, []))
    print()
    
    records = fill_nan_to_negative(records, num_categories)

    return MLCPLDataset(f'MS-COCO ({split})', dataset_path, records, num_categories, transform)

def Pascal_VOC_2007(dataset_path, split='train', transform=transforms.ToTensor()):

    """Load the Pascal VOC 2007 dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

    Returns:
        A ``MLCPLDataset`` object.

    """

    if split == 'train':
        subset = 'trainval'
    elif split == 'valid':
        subset = 'test'

    all_category_ids = set({})
    paths = glob.glob(os.path.join(dataset_path, 'ImageSets', 'Main', '*.txt'))
    for i, path in enumerate(paths):
        print(f'Finding categories of Pascal VOC 2007: {i+1} / {len(paths)}', end='\r')
        basename = os.path.basename(path)
        if '_' in basename:
            all_category_ids.add(basename.split('_')[0])
    all_category_ids = list(all_category_ids)
    all_category_ids.sort()
    num_categories = len(all_category_ids)

    img_nos = pd.read_csv(os.path.join(dataset_path, 'ImageSets', 'Main', subset+'.txt'), sep=' ', header=None, names=['Id'], dtype=str)
    records = []
    for i, row in img_nos.iterrows():
        print(f'Loading Pascal VOC 2007 {split}: {i+1} / {img_nos.shape[0]}', end='\r')
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
        records.append((img_no, path, pos_category_nos, []))

    records = fill_nan_to_negative(records, num_categories)

    return MLCPLDataset(f'Pascal VOC 2007 ({split})', dataset_path, records, num_categories, transform)

def LVIS(dataset_path, split='train', transform=transforms.ToTensor()):
    """Load the LVIS dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

    Returns:
        A ``MLCPLDataset`` object.

    """
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
        print(f'Loading LVIS {split}: {i+1} / {len(imgs)}', end='\r')
        img_id = img['id']
        path = os.path.join(*img['coco_url'].split('/')[-2:])
        annotation_ids = lvis.get_ann_ids(img_ids=[img_id])
        pos_category_ids = [annotation['category_id'] for annotation in lvis.load_anns(annotation_ids)] + img['not_exhaustive_category_ids']
        pos_category_ids = list(set(pos_category_ids))
        pos_category_ids.sort()
        pos_category_nos = [all_category_ids.index(pos_category_id) for pos_category_id in pos_category_ids]
        neg_category_nos = [all_category_ids.index(neg_category_id) for neg_category_id in img['neg_category_ids']]
        records.append((img_id, path, pos_category_nos, neg_category_nos))
    print()

    return MLCPLDataset(f'LVIS ({split})', dataset_path, records, num_categories, transform)

def Open_Images_V6(dataset_path, split=None, transform=transforms.ToTensor(), use_cache=True, cache_dir='output/dataset'):
    """Load the Open Images v6 dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

        use_cache:
            Whether saving the loaded metadata to cache. Defaults to True.
        
        cache_dir:
            The path to the cache. Defaults to 'output/dataset'.

    Returns:
        A ``MLCPLDataset`` object.

    """
    from pathlib import Path
    num_categories = 9605

    if use_cache and os.path.exists(os.path.join(cache_dir, 'train.csv')) and os.path.exists(os.path.join(cache_dir, 'valid.csv')):
        train_dataset = MLCPLDataset(f'Open Images v6 (train)', dataset_path, df_to_records(pd.read_csv(os.path.join(cache_dir, 'train.csv'))), num_categories, transform)
        valid_dataset = MLCPLDataset(f'Open Images v6 (valid)', df_to_records(pd.read_csv(os.path.join(cache_dir, 'valid.csv'))), num_categories, transform)
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

            record = (i, raw['filepath'], pos_category_nos, neg_category_nos)
            if raw['split_name'] == 'train':
                train_records.append(record)
            else:
                valid_records.append(record)

        train_dataset = MLCPLDataset(f'Open Images v6 (train)', dataset_path, train_records, num_categories, transform)
        valid_dataset = MLCPLDataset(f'Open Images v6 (valid)', dataset_path, valid_records, num_categories, transform)

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
    """Load the Open Images v3 dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

        use_cache:
            Whether saving the loaded metadata to cache. Defaults to True.
        
        cache_dir:
            The path to the cache. Defaults to 'output/dataset'.

        check_images:
            Whether perform a check to detect if each image file in the metadata exists. Defaults to True.

    Returns:
        A ``MLCPLDataset`` object.

    """
    
    from pathlib import Path

    if split == 'train':
        subset = 'train'
    elif split == 'valid':
        subset = 'validation'
    elif split == 'test':
        subset = 'test'

    categories = pd.read_csv(os.path.join(dataset_path, 'classes-trainable.txt'), header=None)[0].tolist()
    num_categories = len(categories)

    if use_cache and os.path.exists(os.path.join(cache_dir, split+'.csv')):
        print('Loading Open Images V3 from cache...')
        return MLCPLDataset(f'Open Images v3 ({split})', dataset_path, df_to_records(pd.read_csv(os.path.join(cache_dir, split+'.csv'))), num_categories, transform)

    print('Loading Open Images V3...')

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

    df['Positive'] = df['Positive'].fillna("").apply(list).apply(lambda x: json.dumps(x))
    df['Negative'] = df['Negative'].fillna("").apply(list).apply(lambda x: json.dumps(x))

    paths = [f'{subset}/{img_id}.jpg' for img_id in df['Id'].tolist()]
    df.insert(loc=1, column='Path', value=paths)

    print('Checking if images exist...')

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

    return MLCPLDataset(f'Open Images v3 ({split})', dataset_path, df_to_records(df), num_categories, transform)

def CheXpert(dataset_path, split='train', competition_categories=False, transform=transforms.ToTensor()):
    """Load the CheXpert dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        competition_categories:
            If True, the returned dataset only consists of 5 categories of the CheXpert competition: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion. Defaults to False.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

    Returns:
        A ``MLCPLDataset`` object.

    """

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
        # records.append((i, path, pos_category_nos, neg_category_nos, unc_category_nos))
        records.append((i, path, pos_category_nos, neg_category_nos))

    return MLCPLDataset(f'CheXpert ({split})', dataset_path, records, num_categories, transform=transform, categories=categories)

def VAW(dataset_path, vg_dataset_path, split='train', use_cache=True, cache_dir='output/dataset', transform=transforms.ToTensor()):
    """Load the VAW dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        vg_dataset_path: 
            Path to the Visual Gerome dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

        use_cache:
            Whether saving the loaded metadata to cache. Defaults to True.
        
        cache_dir:
            The path to the cache. Defaults to 'output/dataset'.

    Returns:
        A ``MLCPLDataset`` object.

    """
    
    from pathlib import Path
    from PIL import Image

    vg_folder_1 = 'VG_100K'
    vg_folder_2 = 'VG_100K_2'
    
    if split == 'train':
        subsets = ['train_part1', 'train_part2']
    elif split == 'valid':
        subsets = ['val']
    elif split == 'test':
        subsets = ['test']

    with open(os.path.join(dataset_path, 'data', 'attribute_index.json'), 'r') as f:
        category_dict = json.load(f)
        categories = list(category_dict.keys())
        num_categories = len(categories)

    if use_cache and os.path.exists(os.path.join(cache_dir, split+'.csv')):
        print(f'Loading Open VAW {split} from cache...')
        return MLCPLDataset(f'VAW ({split})', cache_dir, df_to_records(pd.read_csv(os.path.join(cache_dir, split+'.csv'))), num_categories, transform)

    folder_1 = os.listdir(os.path.join(vg_dataset_path, vg_folder_1))
    folder_1 = set([int(os.path.splitext(name)[0]) for name in folder_1])

    folder_2 = os.listdir(os.path.join(vg_dataset_path, vg_folder_2))
    folder_2 = set([int(os.path.splitext(name)[0]) for name in folder_2])

    Path(os.path.join(cache_dir, split)).mkdir(parents=True, exist_ok=True)
    
    records = []
    for subset in subsets:
        with open(os.path.join(dataset_path, 'data', subset+'.json'), 'r') as f:
            samples = json.load(f)

        for i, sample in enumerate(samples):
            print(f'Loading VAW {split}({subset}): {i+1} / {len(samples)}', end='\r')

            image_id = int(sample['image_id'])
            instance_id = int(sample['instance_id'])
            x, y, w, h = sample['instance_bbox']
            positive_attributes = sample['positive_attributes']
            negative_attributes = sample['negative_attributes']

            if image_id in folder_1:
                folder = vg_folder_1
            elif image_id in folder_2:
                folder = vg_folder_2
            else:
                print(f'Image {image_id} not found.')

            img_path = os.path.join(vg_dataset_path, folder, f'{image_id}.jpg')
            img = Image.open(img_path)
            instance_img = img.crop((x, y, x+w, y+h))

            instance_img_path = os.path.join(split, f'{instance_id}.jpg')
            instance_img.save(os.path.join(cache_dir, instance_img_path))

            positive_category_nos = [categories.index(attribute) for attribute in positive_attributes]
            negative_category_nos = [categories.index(attribute) for attribute in negative_attributes]
            
            records.append((instance_id, instance_img_path, positive_category_nos, negative_category_nos))
        print()

    records_to_df(records).to_csv(os.path.join(cache_dir, f'{split}.csv'))

    return MLCPLDataset(f'VAW ({split})', cache_dir, records, num_categories, transform=transform, categories=categories)

def NUS_WIDE(dataset_path, split='train', transform=transforms.ToTensor()):
    """Load the NUS-WIDE dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

    Returns:
        A ``MLCPLDataset`` object.

    """

    if split == 'train':
        subset = 'Train'
    elif split == 'valid':
        subset = 'Test'

    categories = pd.read_csv(os.path.join(dataset_path, 'Concepts81.txt'), header=None)[0].to_list()
    num_categories = len(categories)

    df = pd.read_csv(os.path.join(dataset_path, 'ImageList', f'{subset}Imagelist.txt'), header=None)
    df = df.rename(columns={0: 'Path'})
    df['Path'] = df['Path'].apply(lambda x: os.path.join('images', x.split('\\')[1]))

    for i, category in enumerate(categories):
        labels = pd.read_csv(os.path.join(dataset_path, 'GroundTruth', 'TrainTestLabels', f'Labels_{category}_{subset}.txt'), header=None)[0]
        df[i] = labels

    records = []
    for i, row in df.iterrows():
        print(f'Loading NUS-WIDE {split}: {i+1} / {len(df)}', end='\r')

        positives = []
        negatives = []
        for c in range(num_categories):
            if row[c] == 1:
                positives.append(c)
            else:
                negatives.append(c)
        
        records.append((i, row['Path'], positives, negatives))
    
    print()

    return MLCPLDataset(f'NUS-WIDE ({split})', dataset_path, records, num_categories, transform=transform, categories=categories)

def VISPR(dataset_path, split='train', transform=transforms.ToTensor()):
    """Load the VISPR dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

    Returns:
        A ``MLCPLDataset`` object.

    """

    if split == 'train':
        subset = 'train2017'
    elif split == 'valid':
        subset = 'val2017'
    elif split == 'test':
        subset = 'test2017'

    categories = set()

    paths = glob.glob(os.path.join(dataset_path, subset, '*.json'))
    paths.sort()

    records_temp = []
    for i, path in enumerate(paths):
        print(f'Loading VISPR {split}: {i+1} / {len(paths)}', end='\r')
        
        with open(os.path.join(path), 'r') as f:
            data = json.load(f)

        id = data['id']
        img_path = data['image_path']
        labels = data['labels']
        records_temp.append((id, img_path.replace('images/', ''), labels))
        categories.update(labels)

    print()

    categories = list(categories)
    categories.sort()
    num_categories = len(categories)

    records = []
    for id, path, labels in records_temp:
        positives = [categories.index(label) for label in labels]
        records.append((id, path, positives, [],))
    
    records = fill_nan_to_negative(records, num_categories=num_categories)

    return MLCPLDataset(f'VISPR ({split})', dataset_path, records, num_categories, transform=transform, categories=categories)

def Vireo_Food_172(dataset_path, split='train', transform=transforms.ToTensor()):
    """Load the Vireo Food 172 dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

    Returns:
        A ``MLCPLDataset`` object.

    """

    if split == 'train':
        subset = 'TR'
    elif split == 'valid':
        subset = 'VAL'
    elif split == 'test':
        subset = 'TE'

    images = pd.read_csv(os.path.join(dataset_path, 'SplitAndIngreLabel', f'{subset}.txt'), header=None, sep=' ')[0]

    df = pd.read_csv(os.path.join(dataset_path, 'SplitAndIngreLabel', 'IngreLabel.txt'), header=None, sep=' ', index_col=0).rename(lambda x: x-1, axis=1)

    df = df.loc[images]

    records = []
    for i, (id, row) in enumerate(df.iterrows()):
        print(f'Loading Vireo_Food_172 {split}: {i+1} / {len(df)}', end='\r')

        positives = []
        negatives = []
        for c in df.columns:
            if row[c] == 1:
                positives.append(c)
            else:
                negatives.append(c)
        records.append((i, f'ready_chinese_food{id}', positives, negatives))

    print()

    categories = pd.read_csv(os.path.join(dataset_path, 'SplitAndIngreLabel', 'IngredientList.txt'), header=None, sep=',')[0].to_list()
    
    return MLCPLDataset(f'Vireo Food 172 ({split})', dataset_path, records, len(categories), transform=transform, categories=categories)

def VG_200(dataset_path, metadata_path=None, split='train', transform=transforms.ToTensor()):
    """Load the VG-200 dataset.

    Args:
        dataset_path: 
            Path to the dataset folder.

        metadata_path:
            Path to the folder of the metadata file.

        split: 
            The sub-split of the dataset. Defaults to 'train'.

        transform: 
            Transformation applied to images. Defaults to transforms.ToTensor().

    Returns:
        A ``MLCPLDataset`` object.

    """

    if split == 'train':
        subset = 'train'
    elif split == 'valid':
        subset = 'test'

    metadata_path = dataset_path if metadata_path is None else metadata_path

    num_categories = 200

    vg_folder_1 = 'VG_100K'
    vg_folder_2 = 'VG_100K_2'

    folder_1 = os.listdir(os.path.join(dataset_path, vg_folder_1))
    folder_2 = os.listdir(os.path.join(dataset_path, vg_folder_2))

    records = []

    image_ids = pd.read_csv(os.path.join(metadata_path, f'{subset}_list_500.txt'), header=None)[0].tolist()

    with open(os.path.join(metadata_path, 'vg_category_200_labels_index.json'), 'r') as f:
        labels = json.load(f)

    for i, image_id in enumerate(image_ids):
        print(f'Loading VG_200 ({split}): {i+1} / {len(image_ids)}', end='\r')

        positives = labels[image_id]

        if image_id in folder_1:
            folder = vg_folder_1
        elif image_id in folder_2:
            folder = vg_folder_2
        else:
            raise Exception(f'Image {image_id} not found.')

        img_path = os.path.join(dataset_path, folder, f'{image_id}')
        records.append((image_id, img_path, positives, []))

    print()

    records = fill_nan_to_negative(records, num_categories=num_categories)

    return MLCPLDataset(f'Visual Genome-200 ({split})', dataset_path, records, num_categories, transform=transform)