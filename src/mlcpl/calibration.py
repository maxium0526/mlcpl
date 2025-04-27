import torch
from torch import nn
import math
from .loss import PartialBCELoss

class HistogramBinning():
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.boundaries = None
        self.calibrated_probabilities = None

    def fit(self, preds, target):
        # preds \in [num_samples, num_categories]

        num_samples, num_categories = preds.shape

        self.boundaries = torch.zeros((num_categories, self.n_bins+1))
        self.calibrated_probabilities = torch.zeros((num_categories, self.n_bins))

        for c in range(num_categories):
            predictions = preds[:, c]
            labels = target[:, c]
            sorted_predictions, sorted_indices = torch.sort(predictions)
            sorted_labels = labels[sorted_indices]

            interval = (num_samples / self.n_bins)

            for b_idx in range(self.n_bins):
                interval = (num_samples / self.n_bins)
                self.boundaries[c, b_idx] = (sorted_predictions[math.floor(interval * b_idx)] + sorted_predictions[math.ceil(interval * b_idx)]) / 2
            self.boundaries[c, 0] = 0
            self.boundaries[c, -1] = 1

            for n in range(self.n_bins):
                binned_sorted_labels = sorted_labels[math.floor(interval * n): math.floor(interval * (n+1))]
                self.calibrated_probabilities[c, n] = torch.sum(binned_sorted_labels) / torch.numel(binned_sorted_labels)

    def __call__(self, preds):
        output = torch.zeros_like(preds)
        _, num_categories = preds.shape

        for c in range(num_categories):
            predictions = preds[:, c]

            for n in range(self.n_bins):
                output[:, c] = torch.where(predictions > self.boundaries[c, n], self.calibrated_probabilities[c, n], output[:, c])
        
        return output

class PlattScaling():
    def __init__(self, num_categories, device='cuda'):
        self.a = nn.Parameter(torch.ones(num_categories, device=device))
        self.b = nn.Parameter(torch.zeros(num_categories, device=device))
    
    def fit(self, preds, target, loss_fn=PartialBCELoss(reduction='mean'), optimizer=torch.optim.Adam, lr=0.001, batch_size=256, epochs=10000):
        num_samples, num_categories = preds.shape

        dataloader = torch.utils.data.DataLoader(list(zip(preds.detach(), target.detach())), batch_size=batch_size)

        optimizer = optimizer([self.a, self.b], lr=lr)

        for epoch in range(epochs):
            for i, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = loss_fn(self.a * x + self.b, y)
                loss.backward()
                optimizer.step()
            print(f'Fitting calibrator: {epoch}/{epochs}, Loss: {loss:.4f}', end='\r')
        print()

    def __call__(self, preds):
        return nn.functional.sigmoid(self.a * preds + self.b)

class GNNCalibrator():
    def __init__(self):
        self.gnn = GNN(hidden_dim=8, msg_dim=8, T=25)
    
    def fit(self, preds, target, loss_fn=PartialBCELoss(reduction='mean'), optimizer=torch.optim.Adam, lr=0.001, batch_size=256, epochs=10000):
        num_samples, num_categories = preds.shape

        dataloader = torch.utils.data.DataLoader(list(zip(preds.detach(), target.detach())), batch_size=batch_size)

        self.gnn.to(preds.device)

        optimizer = optimizer(self.gnn.parameters(), lr=lr)

        losses = torch.zeros(len(dataloader))

        self.gnn.train()
        for epoch in range(epochs):
            for batch, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = loss_fn(self.gnn(x), y)
                loss.backward()
                optimizer.step()
                losses[batch] = loss.detach().cpu()
            print(f'Fitting calibrator: {epoch}/{epochs}, Loss: {loss:.4f}', end='\r')
        print()

        return losses

    def __call__(self, preds, batch_size=256):
        num_samples, num_categories = preds.shape

        dataloader = torch.utils.data.DataLoader(preds.detach(), batch_size=batch_size)

        probs = torch.zeros((num_samples, num_categories))

        self.gnn.to(preds.device)
        self.gnn.eval()

        with torch.no_grad():
            for batch, x in enumerate(dataloader):
                x = x.to(preds.device)
                probs[batch*batch_size: (batch+1)*batch_size, :] = nn.functional.sigmoid(self.gnn(x).detach().cpu())

        return probs

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
        # self.hidden_update_fn = lambda msg_dim, hidden_dim: msg_dim * hidden_dim

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