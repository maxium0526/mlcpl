import pandas as pd

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

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__