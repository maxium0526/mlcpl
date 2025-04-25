from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import math

class CurriculumLabeling(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_categories = self.dataset.num_categories
        self.selections = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.bool)
        self.labels = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.int8)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        selection = torch.logical_and(self.selections[idx], torch.isnan(target))

        target_cl = torch.where(selection, self.labels[idx], target)

        return img, target_cl
    
    def getitem(self, idx):
        return self.__getitem__(idx)
    
    def update(self, model, batch_size=32, num_workers=20, selection_strategy='score', selection_threshold=0.5, verbose=False):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)

        model.eval()

        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                if not verbose:
                    print(f'Updating Labels with {selection_strategy} strategy: {batch+1}/{len(dataloader)}', end='\r')

                x, y = x.to('cuda'), y.to('cuda')
                logit = model(x)
                
                label = torch.sign(logit)
                label = torch.where(label==-1, 0, label)
                self.labels[batch*batch_size: (batch+1)*batch_size] = label
                
                if selection_strategy == 'score':
                    if isinstance(selection_threshold, (int, float)):
                        negative_threshold = -selection_threshold
                        positive_threshold = selection_threshold
                    elif isinstance(selection_threshold, tuple):
                        negative_threshold, positive_threshold = selection_threshold

                    negative_selection = torch.where(logit<negative_threshold, 1, 0)
                    positive_selection = torch.where(logit>positive_threshold, 1, 0)

                    selection = torch.logical_or(negative_selection, positive_selection)

                if selection_strategy == 'positive_score':
                    selection = torch.where(logit>selection_threshold, 1, 0)

                if selection_strategy == 'top%/category':
                    k = round(logit.shape[0] * selection_threshold)
                    selection = torch.zeros_like(label)
                    for c in range(logit.shape[1]):
                        threshold = torch.topk(logit[:, c], k)[0].min()
                        selection[:, c] = logit[:, c] > threshold

                if selection_strategy == 'top%':
                    k = round(torch.numel(logit) * selection_threshold)
                    selection = torch.zeros_like(label)
                    threshold = torch.topk(logit.flatten(), k)[0].min()
                    selection = logit > threshold

                self.selections[batch*batch_size: (batch+1)*batch_size] = torch.logical_and(selection, torch.isnan(y))
        
        if not verbose:
            print()

    def get_pseudo_label_proportion(self):
        num_pseudo_labels = torch.count_nonzero(self.selections)
        return num_pseudo_labels / (len(self.dataset) * self.dataset.num_categories)

class BayesianUncertaintyCurriculumLabeling(Dataset):
    def __init__(self, dataset, uncertainty_samples = 20):
        self.dataset = dataset
        self.num_categories = self.dataset.num_categories
        self.selections = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.bool)
        self.labels = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.int8)
        self.uncertainty_samples = uncertainty_samples

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        selection = torch.logical_and(self.selections[idx], torch.isnan(target))

        target_cl = torch.where(selection, self.labels[idx], target)

        return img, target_cl
    
    def getitem(self, idx):
        return self.__getitem__(idx)
    
    def update(self, model, batch_size=32, num_workers=20, selection_threshold=0.5, verbose=False):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)

        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print(m)
                m.train()

        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                if not verbose:
                    print(f'Updating Labels: {batch+1}/{len(dataloader)}', end='\r')

                x, y = x.to('cuda'), y.to('cuda')
                
                logits = [model(x) for n in range(self.uncertainty_samples)]
                logits = torch.stack(logits)
                
                logit_var, logit_mean = torch.var_mean(logits, dim=0)
                
                label = torch.sign(logit_mean)
                label = torch.where(label==-1, 0, label)
                self.labels[batch*batch_size: (batch+1)*batch_size] = label

                selection = logit_var < selection_threshold

                self.selections[batch*batch_size: (batch+1)*batch_size] = torch.logical_and(selection, torch.isnan(y))
        
        if not verbose:
            print()

    def get_pseudo_label_proportion(self):
        num_pseudo_labels = torch.count_nonzero(self.selections)
        return num_pseudo_labels / (len(self.dataset) * self.dataset.num_categories)


class GNN(torch.nn.Module):
    def __init__(self, hidden_dim=1024, msg_dim=1024, T=3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.T = T

        self.msg_update_fn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, msg_dim),
            torch.nn.ReLU(),
        )

        self.hidden_update_fn = torch.nn.GRUCell(msg_dim, hidden_dim)

        self.s = torch.nn.Conv1d(hidden_dim, out_channels=1, kernel_size=1, stride=1, padding=0)


    def forward(self, x): # x: [B, Z]
        batch_size, num_categories = x.shape

        # hidden:[B, V, hidden_dim]
        hidden_0 = torch.unsqueeze(x, -1)
        pad_left = (self.hidden_dim - 1) //2
        pad_right = self.hidden_dim - 1 - pad_left
        hidden_0 = torch.nn.functional.pad(hidden_0, (pad_left, pad_right))
        
        hidden = hidden_0

        for t in range(self.T):

            temp = self.msg_update_fn(hidden) #[B, V, msg_dim]
            temp = temp.unsqueeze(1) #[B, 1, V, msg_dim]
            temp = temp.expand(batch_size, num_categories, num_categories, self.msg_dim) # [B, V(boardcast), V, msg_dim]
            temp = temp.permute(0, 3, 1, 2) # [B, msg_dim, V(boardcast), V]

            weights = torch.ones((num_categories, num_categories)) - torch.eye(num_categories)
            weights = weights.to('cuda')

            msg = temp * weights # [B, msg_dim, V(boardcast), V]

            msg = msg.sum(dim=-1) # [B, msg_dim, V(boardcast)]
            msg = msg / (num_categories - 1)
            msg = msg.permute(0, 2, 1) # [B, V(boardcast), msg_dim]

            #reshape to forward the GRUCell
            hidden = hidden.reshape(-1, self.hidden_dim) # [B*V, hidden_dim]
            msg = msg.reshape(-1, self.msg_dim) # [B*V, msg_dim]

            hidden = self.hidden_update_fn(msg, hidden)

            hidden = hidden.reshape(batch_size, num_categories, self.hidden_dim) # [B, V, msg_dim]

        hidden_T = hidden

        x_gnn = self.s(hidden_T.permute(0, 2, 1)) # [B, V, 1]
        x_gnn = x_gnn.squeeze() # [B, V]

        return x_gnn

class CalibrationAwareThresholdingFreePseudoLabeling(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_categories = self.dataset.num_categories
        self.selections = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.bool)
        self.labels = torch.zeros((len(self.dataset), self.dataset.num_categories), dtype=torch.int8)

        self.calibrator = GNNCalibrator()
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        selection = torch.logical_and(self.selections[idx], torch.isnan(target))

        target_cl = torch.where(selection, self.labels[idx], (target + self.labels[idx]) / 2)

        return img, target_cl
    
    def getitem(self, idx):
        return self.__getitem__(idx)
    
    def update(self, model, batch_size=256, num_workers=20, verbose=False, logger=None):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)

        preds = torch.zeros((len(self.dataset), self.dataset.num_categories))
        target = torch.zeros((len(self.dataset), self.dataset.num_categories))

        model.eval()

        for batch, (x, y) in enumerate(dataloader):
            if not verbose:
                print(f'Updating Labels: {batch+1}/{len(dataloader)}', end='\r')

            with torch.no_grad():
                x, y = x.to('cuda'), y.to('cuda')
                preds[batch*batch_size: (batch+1)*batch_size] = model(x)
                target[batch*batch_size: (batch+1)*batch_size] = y
        
        if not verbose:
            print()
            
        losses = self.calibrator.fit(preds.to('cuda'), target.to('cuda'), epochs=10, lr=0.001, batch_size=256)
        if logger is not None:
            logger.add('train_calibrator_loss', losses.mean().numpy())

        prob = self.calibrator(preds.to('cuda')).to('cpu')

        with torch.no_grad():
            self.labels = prob.detach()
            self.selections = torch.isnan(target)

    def get_pseudo_label_proportion(self):
        num_pseudo_labels = torch.count_nonzero(self.selections)
        return num_pseudo_labels / (len(self.dataset) * self.dataset.num_categories)