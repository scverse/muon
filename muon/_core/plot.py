import sys
from typing import Union, Optional, List, Iterable
import logging

import numpy as np
from matplotlib.axes import Axes
import scanpy as sc
from anndata import AnnData
from .mudata import MuData

def mofa(mdata: MuData, **kwargs) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot in MOFA factors coordinates

    See sc.pl.embedding for details.
    """
    return sc.pl.embedding(mdata, 'mofa', **kwargs)