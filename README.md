# mlcpl: A Python Package for Deep Multi-label Image Classification with Partial-labels on PyTorch

(This is the Introduction part of the package. It will be filled after the paper is published.

---

## Requirements

This mlcpl package requires Python having a minimum version of 3.8.20. Additionally, it also requires the following packages:
- `Cython>=0.29.33`
- `lvis>=0.5.3`
- `pandas>=2.0`
- `protobuf>=3.20.1`
- `pycocotools>=2.0.7`
- `tensorboard>=2.14.0`
- `torch>=1.13.1`
- `torchmetrics>=1.5.2`
- `torchvision>=0.14.1`
- `xmltodict>=0.13.0`

These requirements should be automatically installed when installing the mlcpl package.

## Installation

The mlcpl package can be easily installed via the Python package index (PyPI). For example:

```
# with pip
pip install mlcpl

# or with uv
uv add mlcpl
```

Once the package is installed, it should be able to be used by calling:

```
import mlcpl
```

## Usage

### Loss Functions

This package provides three popular loss functions compatible with unknown labels, including:

- Binary Crossentropy Loss
- Focal Loss
- Asymmetric Loss

The usages of the loss functions are very similar to those in PyTorch. The unknown labels are ignored in the loss calculation. Soft labels are also accepted.

```
preds = torch.tensor([
    [-3.0, -2.6, 1.9, 0.7, 2.2, -0.7],
    [1.0, 0.5, -0.6, -2.8, 7.1, -0.3]
    ])
target = torch.tensor([
    [0., torch.nan, 1., torch.nan, torch.nan, 1.],
    [0., 0., torch.nan, torch.nan, 1., torch.nan]
    ])

bce = mlcpl.losses.PartialBCEWithLogitLoss( alpha_neg=1, alpha_pos=1)
print(bce(preds, target))

focal = mlcpl.losses.PartialFocalWithLogitLoss(alpha_neg=1, alpha_pos=1, gamma=4)
print(focal(preds, target))

asy = mlcpl.losses.PartialAsymmetricWithLogitLoss(alpha_neg=1, alpha_pos=1, gamma_neg=4, gamma_pos=0, clip=0.05)
print(asy(preds, target))
```

### Metrics

This package provides 27 performance metrics that support unknown labels, covering most popular MLC metrics.

```
# per-category mean average precision (mAP)
map_ = mlcpl.metrics.partial_multilabel_average_precision(preds, target, average='macro')
print(map_)

# overall mAP
omap = mlcpl.metrics.partial_multilabel_average_precision(preds, target, average='micro')
print(omap)
%\out{tensor(0.8962)}%

# AP of categories
aps = mlcpl.metrics.partial_multilabel_average_precision(preds, target, average='none')
print(aps)

# per-category mean AUROC
mauroc = mlcpl.metrics.partial_multilabel_auroc(preds, target, average='macro')
print(mauroc)

# overall F1 with threshold=0.3
of1 = mlcpl.metrics.partial_multilabel_f1_score(preds, target, threshold=0.3, average='micro')
print(of1)

```

### Dataset Loading

This package also provides a convenient way for loading popular datasets.

```
train_dataset = mlcpl.datasets.CheXpert(
    dataset_path='/datasets/CheXpert-v1.0-small',
    split='train',
    competition_categories=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset)

for i, (images, target) in enumerate(train_dataloader):
    print(images.shape)
    print(target)

    #... some code for training
```

### Examples

Several Jupyter notebook examples are provided in the `notebook_example` folder for reference.

## Documentation
A comprehensive documentation for this package can be found here:
[https://maxium0526.github.io/mlcpl/build/index.html]()

## Citation

If you find our software is useful, consider citing us:
(bib info, to be filled)
