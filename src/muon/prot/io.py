from os import PathLike

import scanpy as sc
from anndata import AnnData


def read_10x_h5(filename: PathLike, prot_only: bool = True, *args, **kwargs) -> AnnData:
    """Read a 10x Genomics ``.h5`` file, keeping only the protein (Antibody Capture) features by default.

    Args:
        filename: Path to the 10x Genomics HDF5 file (``.h5``).
        prot_only: Only keep features of type ``Antibody Capture``, discarding e.g. gene expression features.
        args: Positional arguments passed to :func:`scanpy.read_10x_h5`.
        kwargs: Keyword arguments passed to :func:`scanpy.read_10x_h5`.
    """
    adata = sc.read_10x_h5(filename, *args, gex_only=False, **kwargs)
    if prot_only:
        adata = adata[:, [x == "Antibody Capture" for x in adata.var["feature_types"]]].copy()
    return adata


def read_10x_mtx(filename: PathLike, prot_only: bool = True, *args, **kwargs) -> AnnData:
    """Read a 10x Genomics ``mtx`` directory, keeping only the protein (Antibody Capture) features by default.

    Args:
        filename: Path to the directory with the ``mtx`` matrix and its features and barcodes files.
        prot_only: Only keep features of type ``Antibody Capture``, discarding e.g. gene expression features.
        args: Positional arguments passed to :func:`scanpy.read_10x_mtx`.
        kwargs: Keyword arguments passed to :func:`scanpy.read_10x_mtx`.
    """
    adata = sc.read_10x_mtx(filename, *args, gex_only=False, **kwargs)
    if prot_only:
        adata = adata[:, [x == "Antibody Capture" for x in adata.var["feature_types"]]].copy()
    return adata
