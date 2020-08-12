from typing import Union
from pathlib import Path

import scanpy as sc
from .mudata import MuData

def read_10x_h5(filename: Union[str, Path],
				*args, **kwargs) -> MuData:
	adata = sc.read_10x_h5(filename, gex_only=False, *args, **kwargs)
	return MuData(adata)