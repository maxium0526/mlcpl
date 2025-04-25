import torch

class MultiViewTransform():
    def __init__(self, transform, num_views=2):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        output = []
        for i in range(self.num_views):
            output.append(self.transform(x))
        
        return output

# class DictionaryQueue():
#     def __init__(self, n=256):
#         self.n = n
#         self.batches = [None] * self.n
#         self.iter = 0
    
#     def add(self, batch):
#         self.batches[self.iter] = batch
#         self.iter += 1
#         self.iter = self.iter % self.n

#     def get(self):
#         batches = [batch for batch in self.batches if batch is not None]  
#         if len(batches) > 0:
#             X = torch.cat([batch[0] for batch in batches])
#             Y = torch.cat([batch[1] for batch in batches])

#             return X, Y

#         return None, None

class DictionaryQueue():
    def __init__(self, n=256):
        self.n = n
        self.batches = [None] * self.n
        self.iter = 0
    
    def add(self, batch):
        self.batches[self.iter] = batch
        self.iter += 1
        self.iter = self.iter % self.n

    def get(self):
        batches = [batch for batch in self.batches if batch is not None]
        if len(batches) > 0:
            if isinstance(batches[0], torch.Tensor):
                return torch.cat(batches, dim=0)
            elif isinstance(batches[0], tuple):
                tuple_length = len(batches[0])
                output = []
                for i in range(tuple_length):
                    elements = torch.cat([batch[i] for batch in batches], dim=0)
                    output.append(elements)
                return tuple(output)
        return None

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)