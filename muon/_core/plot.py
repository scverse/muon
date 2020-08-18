from typing import Union, List

from matplotlib.axes import Axes
import scanpy as sc
from .mudata import MuData

def mofa(mdata: MuData, **kwargs) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot in MOFA factors coordinates

    See sc.pl.embedding for details.
    """
    return sc.pl.embedding(mdata, 'mofa', **kwargs)