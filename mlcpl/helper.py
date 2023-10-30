import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

def print_record(log, end='\r'):
    str = ''
    for key, value in log.items():
        str += f'{key}: {value:.4f}; '
    print(str, end=end)

class ExcelLogger():
    def __init__(self, path):
        self.sheets = {}
        self.path = path
    
    def add(self, tag, scalar):
        if tag not in self.sheets.keys():
            self.sheets[tag] = []
        sheet = self.sheets[tag]
        sheet.append(scalar)

    def flush(self):
        with pd.ExcelWriter(self.path) as writer:
            for tag, sheet in self.sheets.items():
                if type(sheet[0]) == dict:
                    pd.DataFrame.from_records(sheet).to_excel(writer, sheet_name=tag)
                else:
                    pd.DataFrame(sheet, columns=[tag]).to_excel(writer, sheet_name=tag)

class MultiLogger():
    def __init__(self, path, excellog = True, tblog = True) -> None:
        self.excellog = ExcelLogger(os.path.join(path, 'excel_log.xlsx')) if excellog is True else None
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
                
        if self.excellog:
            if type(value) == dict:
                self.excellog.add(tag, value)
            elif torch.is_tensor or isinstance(value, np.ndarray):
                if len(value.shape) == 0:
                    # scalar
                    self.excellog.add(tag, value)
                else:
                    #vector
                    self.excellog.add(tag, {str(i): v for i, v in enumerate(value)})
            else:
                self.excellog.add(tag, value)

    def flush(self):
        if self.tblog:
            self.tblog.flush()
        if self.excellog:
            self.excellog.flush()



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__