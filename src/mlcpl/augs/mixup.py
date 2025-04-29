import torch
import numpy as np
from .core import *

class MixUp(torch.utils.data.Dataset):
    def __init__(self, dataset, alpha=0.2, transform=None, unknown_as=None):
        self.dataset = dataset
        self.alpha = alpha
        self.transform = transform
        self.unknown_as = unknown_as
        self.num_categories = self.dataset.num_categories

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        img1, target1 = self.dataset[idx]
        img2, target2 = self.dataset[np.random.randint(0, len(self.dataset))]

        if self.unknown_as is not None:
            target1 = torch.where(torch.isnan(target1), self.unknown_as, target1)
            target2 = torch.where(torch.isnan(target2), self.unknown_as, target2)

        if torch.isnan(target1).any() or torch.isnan(target2).any():
            raise Exception('Target contains nan.')

        lam = np.random.beta(self.alpha, self.alpha)

        img = mixup(img1, img2, lam=lam)
        target = mixup(target1, target2, lam=lam)

        if self.transform:
            img = self.transform(img)

        return img, target