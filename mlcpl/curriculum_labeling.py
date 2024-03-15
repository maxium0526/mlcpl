from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class CurriculumLabeling(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_categories = self.dataset.num_categories
        self.selections = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.bool)
        self.labels = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.int8)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        selection = torch.logical_and(self.selections[idx], torch.isnan(target))

        target_cl = torch.where(selection, self.labels[idx], target)

        return img, target_cl
    
    def getitem(self, idx):
        return self.__getitem__(idx)
    
    def update(self, model, batch_size=32, num_workers=20, selection_strategy='score', selection_threshold=0.5, verbose=False):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)

        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                if not verbose:
                    print(f'Updating Labels with {selection_strategy} strategy: {batch+1}/{len(dataloader)}', end='\r')

                x, y = x.to('cuda'), y.to('cuda')
                logit = model(x)
                
                label = torch.sign(logit)
                label = torch.where(label==-1, 0, label)
                self.labels[batch*batch_size: (batch+1)*batch_size] = label
                
                if selection_strategy == 'score':
                    selection = torch.where(torch.abs(logit)>selection_threshold, 1, 0)

                if selection_strategy == 'positive_score':
                    selection = torch.where(logit>selection_threshold, 1, 0)
                
                self.selections[batch*batch_size: (batch+1)*batch_size] = torch.logical_and(selection, torch.isnan(y))
        
        if not verbose:
            print()

    def get_pseudo_label_proportion(self):
        num_pseudo_labels = torch.count_nonzero(self.selections)
        return num_pseudo_labels / (len(self.dataset) * self.dataset.num_categories)
