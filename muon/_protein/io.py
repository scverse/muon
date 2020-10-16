from typing import Union
from pathlib import Path

import scanpy as sc
from anndata import AnnData


def read_10x_h5(filename: Union[str, Path], prot_only: bool = True, *args, **kwargs) -> AnnData:
    adata = sc.read_10x_h5(filename, gex_only=False, *args, **kwargs)
    if prot_only:
        adata = adata[:, list(map(lambda x: x == "Antibody Capture", adata.var["feature_types"]))]
    return adata


def read_10x_mtx(filename: Union[str, Path], prot_only: bool = True, *args, **kwargs) -> AnnData:
    adata = sc.read_10x_mtx(filename, gex_only=False, *args, **kwargs)
    if prot_only:
        adata = adata[:, list(map(lambda x: x == "Antibody Capture", adata.var["feature_types"]))]
    return adata
