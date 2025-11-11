# mlcpl: A Python Package for Deep Multi-label Image Classification with Partial-labels on PyTorch

(This is the Introduction part of the package. It will be filled after the paper is published.

---

## Requirements

This mlcpl package requires Python having a minimum version of 3.8.20. Additionally, it also requires the following packages:
- `"Cython==0.29.33"`,
- `"lvis==0.5.3"`,
- `"pandas==1.5.2"`,
- `"protobuf==3.20.1"`,
- `"pycocotools>=2.0.7"`,
- `"tensorboard>=2.14.0"`,
- `"torch>=1.13.1"`,
- `"torchmetrics>=1.5.2"`,
- `"torchvision>=0.14.1"`,
- `"xmltodict==0.13.0"`

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
