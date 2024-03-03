import torch
import numpy as np
from .core import *

class MixUpPME(torch.utils.data.Dataset):
    def __init__(self, dataset, alpha=0.75, transform=None):
        self.dataset = dataset
        self.alpha = alpha
        self.transform = transform
        self.num_categories = self.dataset.num_categories

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        img1, target1 = self.dataset[idx]
        img2, target2 = self.dataset[np.random.randint(0, len(self.dataset))]

        mask = ~torch.isnan(target1)

        target1 = torch.where(torch.isnan(target1), 0.5, target1)
        target2 = torch.where(torch.isnan(target2), 0.5, target2)

        lam = np.random.uniform(self.alpha, 1)

        img = mixup(img1, img2, lam=lam)
        target = mixup(target1, target2, lam=lam)
        target[~mask] = torch.nan

        if self.transform:
            img = self.transform(img)

        return img, target