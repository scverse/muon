from os import PathLike

import scanpy as sc  # type: ignore[import-untyped]
from anndata import AnnData


def read_10x_h5(filename: PathLike, prot_only: bool = True, *args, **kwargs) -> AnnData:
    """Read a 10x Genomics ``.h5`` file, keeping only the protein (Antibody Capture) features by default."""
    adata = sc.read_10x_h5(filename, *args, gex_only=False, **kwargs)
    if prot_only:
        adata = adata[:, [x == "Antibody Capture" for x in adata.var["feature_types"]]].copy()
    return adata


def read_10x_mtx(filename: PathLike, prot_only: bool = True, *args, **kwargs) -> AnnData:
    """Read a 10x Genomics ``mtx`` directory, keeping only the protein (Antibody Capture) features by default."""
    adata = sc.read_10x_mtx(filename, *args, gex_only=False, **kwargs)
    if prot_only:
        adata = adata[:, [x == "Antibody Capture" for x in adata.var["feature_types"]]].copy()
    return adata
