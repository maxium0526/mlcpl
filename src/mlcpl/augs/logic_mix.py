import torch
import numpy as np

def logic_mix_images(images):
    if isinstance(images, torch.Tensor):
        pass
    else:
        images = torch.stack(images)

    new_image = torch.mean(images, 0)

    return new_image

def logic_mix_target(target, unknown_as=None):
    if isinstance(target, torch.Tensor):
        pass
    else:
        target = torch.stack(target)

    if unknown_as is not None:
        target = torch.where(torch.isnan(target), unknown_as, target)

    #compute must positive
    t = torch.where(torch.isnan(target), 0, target)
    t = torch.sum(t, dim=0)
    must_positive = torch.sign(t)

    #compute must negative
    t = torch.where(target == 0, 0, 1)
    t = torch.sum(t, dim=0)
    must_negative = torch.where(t==0, 1, 0)

    new_target = must_positive
    new_target = torch.where(must_negative == 1, -1, new_target)
    new_target = torch.where(new_target == 0, torch.nan, new_target)
    new_target = torch.where(new_target == -1, 0, new_target)

    return new_target

class LogicMix(torch.utils.data.Dataset):
    """Implementation of LogicMix.

    """
    def __init__(self, dataset, probability=1, k_min=2, k_max=3, transform=None):
        """Initialization

        Args:
            dataset:
                The ``'MLCPLDataset'`` to be applied.

            probability:
                A hyperparameter. Defaults to 1.

            k_min: 
                A hyperparameter. Defaults to 2.

            k_max: 
                A hyperparameter. Defaults to 3.

            transform: 
                Optional transformation applied to augment images. Defaults to None.
        """
        self.dataset = dataset
        self.probability = probability
        self.k_min = k_min
        self.k_max = k_max
        self.transform = transform
        self.num_categories = self.dataset.num_categories

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if np.random.rand() > self.probability:
            return self.dataset[idx]

        k = np.random.randint(self.k_min, self.k_max + 1)

        indices = np.random.randint(len(self.dataset), size=(k))
        indices[0] = idx

        samples = [self.dataset[i] for i in indices]

        image = logic_mix_images([image for image, target in samples])
        target = logic_mix_target([target for image, target in samples])

        if self.transform:
            image = self.transform(image)

        return image, target
    
def logic_mix_with_batch(images, target, probability=0.5, k_min=2, k_max=3):

    batch_size = target.shape[0]

    new_images, new_target, is_augmented = torch.zeros_like(images), torch.zeros_like(target), torch.zeros(batch_size, dtype=torch.bool)

    for i in range(batch_size):
        if torch.rand(1) > probability:
            new_images[i] = images[i]
            new_target[i] = target[i]
            is_augmented[i] = False
            continue

        k = torch.randint(k_min, k_max + 1, size=(1, ))
        
        indices = torch.randint(0, batch_size, size=(k, ))
        indices[0] = i

        new_images[i] = logic_mix_images(images[indices])
        new_target[i] = logic_mix_target(target[indices])
        is_augmented[i] = True

    return new_images, new_target, is_augmented