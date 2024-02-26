from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torchvision import transforms

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
        target = mix_targets([target for image, target in samples], strict_negative=self.strict_negative)

        if self.transform:
            image = self.transform(image)

        return image, target
    
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

        if self.unknown_as:
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
    
class MultilabelMixUp(torch.utils.data.Dataset):
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

        if self.unknown_as:
            target1 = torch.where(torch.isnan(target1), self.unknown_as, target1)
            target2 = torch.where(torch.isnan(target2), self.unknown_as, target2)

        if torch.isnan(target1).any() or torch.isnan(target2).any():
            raise Exception('Target contains nan.')

        lam = np.random.beta(self.alpha, self.alpha)

        img = mixup(img1, img2, lam=lam)
        target = mix_targets([target1, target2], strict_negative=True)

        if self.transform:
            img = self.transform(img)

        return img, target

def mixup(tensor_1, tensor_2, lam=0.5):
    return lam * tensor_1 + (1 - lam) * tensor_2

def mix_images(images):
    images = torch.stack(images)
    new_image = torch.mean(images, 0)

    return new_image

def mix_targets(targets, strict_negative=False):
        targets = torch.stack(targets)

        #compute must positive
        t = torch.where(torch.isnan(targets), 0, targets)
        t = torch.sum(t, dim=0)
        must_positive = torch.sign(t)

        #compute must negative
        if strict_negative:
            t = torch.where(targets == 0, 0, 1)
            t = torch.sum(t, dim=0)
            must_negative = torch.where(t==0, 1, 0)
        else:
            t = torch.where(targets == 0, 1, 0)
            t = torch.sum(t, dim=0)
            must_negative = torch.where(torch.logical_and(t > 0, must_positive != 1), 1, 0)

        if torch.max(must_negative+must_positive) > 1:
            print('Found a label is both positive and negative!')

        new_target = must_positive
        new_target = torch.where(must_negative == 1, -1, new_target)
        new_target = torch.where(new_target == 0, torch.nan, new_target)
        new_target = torch.where(new_target == -1, 0, new_target)

        return new_target

if __name__=='__main__':
    n = torch.nan
    a = torch.tensor([n, n, 1, n, 0, n])
    b = torch.tensor([0, 1, 0, n, 0, n])
    c = torch.tensor([n, 0, 0, n, 0, 1])
    print(mix_targets([a, b, c], strict_negative=False))
    print(mix_targets([a, b, c], strict_negative=True))