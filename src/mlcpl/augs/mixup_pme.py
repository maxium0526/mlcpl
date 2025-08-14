import torch
import numpy as np

def interpolate(tensor_1, tensor_2, lam=0.5):
    return lam * tensor_1 + (1 - lam) * tensor_2

class MixUpPME(torch.utils.data.Dataset):
    """Implementation of MixUpPME.

    """
    def __init__(self, dataset, alpha=0.75, transform=None):
        """Initialization

        Args:
            dataset:
                The ``'MLCPLDataset'`` to be applied.

            alpha:
                A hyperparameter. Defaults to 0.75.

            transform: 
                Optional transformation applied to augment images. Defaults to None.
        """
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

        img = interpolate(img1, img2, lam=lam)
        target = interpolate(target1, target2, lam=lam)
        target[~mask] = torch.nan

        if self.transform:
            img = self.transform(img)

        return img, target

def mixup_pme_with_batch(images, target, alpha=0.75):
    batch_size = images.shape[0]
    images_1, images_2 = images[:batch_size//2], images[batch_size//2:]
    target_1, target_2 = target[:batch_size//2], target[batch_size//2:]

    mask = ~torch.isnan(target_1)

    target_1 = torch.where(torch.isnan(target_1), 0.5, target_1)
    target_2 = torch.where(torch.isnan(target_2), 0.5, target_2)

    lam = np.random.uniform(alpha, 1)

    new_images = interpolate(images_1, images_2, lam=lam)
    new_target = interpolate(target_1, target_2, lam=lam)
    new_target[~mask] = torch.nan

    return new_images, new_target