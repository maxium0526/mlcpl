import torch
import numpy as np
from .core import *
from ..dataset import *

class LogicMix(torch.utils.data.Dataset):
    def __init__(self, dataset, probability=1, mix_num_samples=2, strict_negative=False, transform=None):
        self.dataset = dataset
        self.probability = probability
        self.mix_num_samples = mix_num_samples
        self.strict_negative = strict_negative
        self.transform = transform
        self.num_categories = self.dataset.num_categories

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if np.random.rand() > self.probability:
            return self.dataset[idx]

        if type(self.mix_num_samples) is int:
            num_samples = self.mix_num_samples
        elif type(self.mix_num_samples) is tuple:
            num_samples = np.random.randint(*self.mix_num_samples)

        indices = np.random.randint(len(self.dataset), size=(num_samples))
        indices[0] = idx

        samples = [self.dataset[i] for i in indices]

        image = mix_images([image for image, target in samples])
        target = logic_mix_targets([target for image, target in samples], strict_negative=self.strict_negative)

        if self.transform:
            image = self.transform(image)

        return image, target
    
    def estimate_statistics(self):
        records = []
        for i, (image, target) in enumerate(torch.utils.data.DataLoader(self, batch_size=None, num_workers=20)):
            record = (i, None, *one_hot_to_labels(target))
            records.append(record)

        return get_statistics(records, self.num_categories)