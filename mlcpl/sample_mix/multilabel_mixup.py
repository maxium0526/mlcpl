import torch
import numpy as np
from .core import *

class MultilabelMixUp(torch.utils.data.Dataset):
    def __init__(self, dataset, alpha=0.2, transform=None, unknown_as=0):
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

        lam = np.random.beta(self.alpha, self.alpha)

        img = mixup(img1, img2, lam=lam)
        target = logic_mix_targets([target1, target2], strict_negative=True, unknown_as=self.unknown_as)

        if self.transform:
            img = self.transform(img)

        return img, target