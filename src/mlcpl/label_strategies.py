import torch

def unknown_to_unknown(y):
    return y

def unknown_to_negative(y):
    return torch.nan_to_num(y, 0)

def unknown_to_positive(y):
    return torch.nan_to_num(y, 1)
