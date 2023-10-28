import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def forward(self, x, y):
        return sigmoid_focal_loss(x, y, alpha=0.25, gamma=2, reduction='mean')