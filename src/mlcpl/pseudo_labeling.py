from torch.utils.data import Dataset, DataLoader
import torch

class CurriculumLabeling(Dataset):
    """An implementation of curriculum labeling.

    """

    def __init__(self, dataset):
        """

        Args:
            dataset: The ``'MLCPLDataset`` that is applied to.

        """

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
        """Update pseudo-labels.

        Args:
            model:
                the model that produces predictions.
            
            batch_size: size of mini-batches.

            num_workers: the number of data loader workers.

            selection_strategy: The method to select predictions to generate pseudo-labels.

            selection_threshold: The thresholds for generating pseudo-labels.

            verbose: Whether showing the process of generation.
        """
        
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
