import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from copy import deepcopy
from PIL import ImageDraw
import random
from pathlib import Path

def print_record(log, end='\r'):
    str = ''
    for key, value in log.items():
        str += f'{key}: {value:.4f}; '
    print(str, end=end)

class CSVLogger():
    def __init__(self, path):
        self.sheets = {}
        self.path = path
    
    def add(self, tag, scalar):
        if tag not in self.sheets.keys():
            self.sheets[tag] = []
        sheet = self.sheets[tag]
        sheet.append(scalar)

    def flush(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)
        for tag, sheet in self.sheets.items():
            if type(sheet[0]) == dict:
                pd.DataFrame.from_records(sheet).to_csv(os.path.join(self.path, f'{tag}.csv'))
            else:
                pd.DataFrame(sheet, columns=[tag]).to_csv(os.path.join(self.path, f'{tag}.csv'))

class MultiLogger():
    def __init__(self, path, csvlog = True, tblog = True) -> None:
        self.csvlog = CSVLogger(os.path.join(path, 'csv_log')) if csvlog is True else None
        self.tblog = SummaryWriter(log_dir=path) if tblog is True else None
        
        self.tag_counts = {}

    def get_tag_count(self, tag):
        if tag not in self.tag_counts:
            self.tag_counts[tag] = 0
            return 0
        self.tag_counts[tag] += 1
        return self.tag_counts[tag]

    def add(self, tag, value):
        tag_count = self.get_tag_count(tag)

        if self.tblog:
            if type(value) == dict:
                self.tblog.add_scalars(tag, value, tag_count)
            elif torch.is_tensor or isinstance(value, np.ndarray):
                if len(value.shape) == 0:
                    # scalar
                    self.tblog.add_scalar(tag, value, tag_count)
                else:
                    #vector
                    self.tblog.add_scalars(tag, {str(i): v for i, v in enumerate(value)}, tag_count)
            else:
                self.tblog.add_scalar(tag, value, tag_count)

        if self.csvlog:
            if type(value) == dict:
                self.csvlog.add(tag, value)
            elif torch.is_tensor or isinstance(value, np.ndarray):
                if len(value.shape) == 0:
                    # scalar
                    self.csvlog.add(tag, value)
                else:
                    #vector
                    self.csvlog.add(tag, {str(i): v for i, v in enumerate(value)})
            else:
                self.csvlog.add(tag, value)

    def adds(self, tag, values):
        for value in values:
            self.add(tag, value)

    def flush(self):
        if self.tblog:
            self.tblog.flush()
        if self.csvlog:
            self.csvlog.flush()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# from https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/src/helper_functions/helper_functions.py
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

# from https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/src/helper_functions/helper_functions.py
class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x
