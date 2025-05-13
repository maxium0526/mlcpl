import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from mlcpl.label_strategies import *
from .losses import PartialNegativeBCEWithLogitLoss

class LargeLossRejection(nn.Module):
    def __init__(self, loss_fn=PartialNegativeBCEWithLogitLoss(reduction=None), delta_rel=0.1, reduction='mean'):
        super(LargeLossRejection, self).__init__()
        self.delta_rel = delta_rel
        self.loss_fn = loss_fn
        self.reduction = reduction

    def forward(self, logits, targets, epoch):
        losses = self.loss_fn(logits, targets)

        unknown_label_losses = losses * torch.isnan(targets)
        percent = epoch * self.delta_rel / 100
        percent = 1 if percent > 1 else percent

        k = round(torch.count_nonzero(unknown_label_losses).cpu().detach().numpy() * percent)
        k = 1 if k == 0 else k
        
        loss_threshold = torch.topk(unknown_label_losses.flatten(), k).values.min()

        lambdas = torch.where(unknown_label_losses > loss_threshold, 0, 1)

        final_loss = losses * lambdas

        if self.reduction == 'sum':
            return torch.sum(final_loss)
        elif self.reduction == 'mean':
            final_loss = torch.mean(final_loss)

        return final_loss

class LargeLossCorrectionTemporary(nn.Module):
    def __init__(self, loss_fn=PartialNegativeBCEWithLogitLoss(reduction=None), delta_rel=0.1):
        super(LargeLossCorrectionTemporary, self).__init__()
        self.delta_rel = delta_rel
        self.loss_fn = loss_fn

    def forward(self, logits, targets, epoch):
        with torch.no_grad():
            losses = self.loss_fn(logits, targets)

            unknown_label_losses = losses * torch.isnan(targets)
            percent = epoch * self.delta_rel / 100
            percent = 1 if percent > 1 else percent

            k = round(torch.count_nonzero(unknown_label_losses).cpu().detach().numpy() * percent)
            k = 1 if k == 0 else k
            
            loss_threshold = torch.topk(unknown_label_losses.flatten(), k).values.min()

            lambdas = torch.where(unknown_label_losses > loss_threshold, 0, 1)

            change_label = torch.logical_and(torch.logical_not(lambdas), torch.isnan(targets))

            new_targets = torch.where(change_label, 1, targets)

        final_loss = torch.sum(self.loss_fn(logits, targets))

        return final_loss

class LargeLossCorrectionPermenent(Dataset):
    def __init__(self, dataset, loss_fn, delta_rel=0.1):
        self.dataset = dataset
        self.num_categories = self.dataset.num_categories
        self.selections = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.bool)
        self.labels = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.int8)
        self.loss_fn = loss_fn
        self.delta_rel = delta_rel


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        selection = torch.logical_and(self.selections[idx], torch.isnan(target))

        target_cl = torch.where(selection, self.labels[idx], target)

        return img, target_cl
    
    def getitem(self, idx):
        return self.__getitem__(idx)
    
    def update(self, model, epoch, batch_size=32, num_workers=20, verbose=False):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)

        model.eval()

        with torch.no_grad():
            for batch, (x, targets) in enumerate(dataloader):
                if not verbose:
                    print(f'Updating Labels: {batch+1}/{len(dataloader)}', end='\r')

                x, targets = x.to('cuda'), targets.to('cuda')
                logit = model(x)
                losses = self.loss_fn(logit, targets)

                unknown_label_losses = losses * torch.isnan(targets)
                percent = (epoch+1) * self.delta_rel / 100
                percent = 1 if percent > 1 else percent

                k = round(torch.count_nonzero(unknown_label_losses).cpu().detach().numpy() * percent)
                k = 1 if k == 0 else k
        
                loss_threshold = torch.topk(unknown_label_losses.flatten(), k).values.min()

                changes = torch.where(unknown_label_losses > loss_threshold, 1, 0).cpu()

                self.labels[batch*batch_size: (batch+1)*batch_size] = torch.logical_or(self.labels[batch*batch_size: (batch+1)*batch_size], changes)

                self.selections[batch*batch_size: (batch+1)*batch_size] = torch.isnan(targets)
        
        if not verbose:
            print()

    def get_pseudo_label_proportion(self):
        num_pseudo_labels = torch.count_nonzero(self.selections)
        return num_pseudo_labels / (len(self.dataset) * self.dataset.num_categories)

    def get_positive_pseudo_label_proportion(self):
        num_pseudo_labels = torch.sum(self.labels)
        return num_pseudo_labels / (len(self.dataset) * self.dataset.num_categories)
