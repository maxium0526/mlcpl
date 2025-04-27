from .core import MLCPLDataset, labels_to_one_hot
from .loaders import (
    MSCOCO,
    Pascal_VOC_2007,
    LVIS,
    Open_Images,
    Open_Images_V3,
    CheXpert,
    VAW,
    NUS_WIDE,
    VISPR,
    Vireo_Food_172,
    VG_200,
)

del core, loaders