import torch
import numpy as np
from ..label_strategies import *
from ..helper import *
from torch.utils.data import Dataset
from copy import deepcopy

class FTDataset(Dataset):
    def __init__(self, Z, Y):
        self.Z = Z
        self.Y = Y
        self.num_categories = self.Y.shape[1]

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        z, y = self.Z[idx], self.Y[idx]
        return z, y
        
    def test(self):
        return self.__getitem__(0)

def CFT(
    heads,
    training_data=None,
    validation_data=None,
    optimizer=None,
    batch_size=None,
    epochs=1,
    early_stopping=None,
    validation_metric=None,
    device='cuda',
    ):

    num_categories = len(heads)

    z_train, y_train = training_data
    z_valid, y_valid = validation_data

    finetuned_heads, category_records = [], []

    for i in range(num_categories):
        
        print(f'Fine-tuning category {i}/{num_categories}.')

        finetuned_head, records = finetune_head(
            heads[i],
            training_data=(z_train, y_train[:, i:i+1]),
            validation_data=(z_valid, y_valid[:, i:i+1]),
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping=early_stopping,
            validation_metric=validation_metric,
            device=device,
        )

        finetuned_heads.append(finetuned_head)
        category_records.append(records)

    return finetuned_heads, category_records

def finetune_head(
    head,
    training_data=None,
    validation_data=None,
    optimizer=None,
    batch_size=None,
    epochs = 1,
    early_stopping=None,
    validation_metric=None,
    device='cuda',
    ):

    head = deepcopy(head).to(device)

    z_train, y_train = training_data
    z_valid, y_valid = validation_data
    z_train, y_train = z_train.to(device), y_train.to(device)
    z_valid, y_valid = z_valid.to(device), y_valid.to(device)

    if y_train.dtype == torch.int8:
        y_train = y_train.to(torch.float32)
        y_train[y_train==-1] = torch.nan
    train_label_map = ~torch.isnan(y_train).view(-1)
    z_train, y_train = z_train[train_label_map, :], y_train[train_label_map, :]

    if y_valid.dtype == torch.int8:
        y_valid = y_valid.to(torch.float32)
        y_valid[y_valid==-1] = torch.nan
    valid_label_map = ~torch.isnan(y_valid).view(-1)
    z_valid, y_valid = z_valid[valid_label_map, :], y_valid[valid_label_map, :]

    if (y_train==0).sum() == 0 :
        print(f'Negative training samples not found. Skip.')
        return head, []
    if (y_train==1).sum() == 0:
        print(f'Positive training samples not found. Skip.')
        return head, []
    if (y_valid==0).sum() == 0 :
        print(f'Negative validation samples not found. Skip.')
        return head, []
    if (y_valid==1).sum() == 0:
        print(f'Positive validation samples not found. Skip.')
        return head, []

    training_dataset = FTDataset(z_train, y_train)
    validation_dataset = FTDataset(z_valid, y_valid)

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size = len(training_dataset) if batch_size is None else batch_size,
        shuffle=True,
        )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size = len(training_dataset) if batch_size is None else batch_size,
        )

    optimizer.set_head(head)

    original_validation_score = validation_metric(head(z_valid), y_valid)
    print(f'Original Vaild Score: {original_validation_score:.4f}')

    best_validation_score = original_validation_score
    best_state_dict = deepcopy(head.state_dict())
    best_at = -1

    records = []

    for epoch in range(epochs):
        record = {'Epoch': epoch}

        if batch_size is None:
            train_log = optimizer.step(z_train, y_train)
            train_log = {name: value.cpu().detach().numpy() for name, value in train_log.items()}
            
            with torch.no_grad():
                validation_score = validation_metric(head(z_valid), y_valid)
            
            record.update(train_log)
            record['Valid Score'] = validation_score.cpu().detach().numpy()

        else:
            train_logs = []
            for batch, (z, y) in enumerate(training_dataloader):
                train_log = optimizer.step(z, y)
                if train_log is not None:
                    train_logs.append(train_log)

            with torch.no_grad():
                validation_score = validation_metric(head(z_valid), y_valid)
            
            for name in train_logs[0].keys():
                record[name] = torch.mean(torch.tensor([log[name] for log in train_logs])).cpu().detach().numpy()
            record['Valid Score'] = validation_score.cpu().detach().numpy()

        if validation_score > best_validation_score:
            best_validation_score = validation_score
            best_at = epoch
            best_state_dict = deepcopy(head.state_dict())

        records.append(record)
        print_record(record)

        if early_stopping is not None and epoch - best_at >= early_stopping:
            print()
            print(f'Early stopping. Best Valid Score: {best_validation_score:.4f} (+{best_validation_score-original_validation_score:.4f})')
            print()
            break

        if epoch == epochs - 1:
            print()
            print(f'Done. Best Valid Score: {best_validation_score:.4f} (+{best_validation_score-original_validation_score:.4f})')
            print()
            break
    
    del z_train, z_valid, y_train, y_valid, training_dataloader, training_dataset, validation_dataloader, validation_dataset
    torch.cuda.empty_cache()

    head.load_state_dict(best_state_dict)

    return head, records

def greedy(parameters, data, validation_metric):
    z, y = data

    num_categories = y.shape[1]
    best_category_scores = np.zeros(num_categories, dtype=np.float32)
    best_category_scores[:] = np.nan
    best_weight = torch.zeros((num_categories, z.shape[-1]), dtype=torch.float32)
    best_bias = torch.zeros(num_categories, dtype=torch.float32)

    records = []
    for name, parameter in parameters:
        record = {'Name': name, 'Mean': np.nan}
        weight, bias = parameter

        if len(z.shape) == 3:             # (batch_size, num_categories, feature_dim)     # For SSGRLs
            pred = torch.sum(z * weight, dim=2) + bias
        else:                             # (batch_size, feature_dim)                     # For vanilla CNNs
            pred = torch.nn.functional.linear(z, weight, bias)
            
        category_scores = torch.zeros((num_categories))
        for i in range(num_categories):
            score = validation_metric(pred[:, i:i+1], y[:, i:i+1])
            category_scores[i] = score

            record['Category_'+str(i)] = score

            if not torch.isnan(score):
                if score > best_category_scores[i] or np.isnan(best_category_scores[i]):
                    best_category_scores[i] = score
                    best_weight[i:i+1] = weight[i:i+1]
                    best_bias[i] = bias[i]

        record['Mean'] = np.mean([score.numpy() for score in category_scores if not torch.isnan(score)])
        records.append(record)
    
    return best_weight, best_bias, records