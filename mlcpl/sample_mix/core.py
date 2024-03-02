import torch

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