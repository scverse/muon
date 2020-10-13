from os import PathLike

import scanpy as sc
from .. import MuData


def read_10x_h5(filename: PathLike, *args, **kwargs) -> MuData:
    adata = sc.read_10x_h5(filename, gex_only=False, *args, **kwargs)
    protein = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()
    rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    return MuData({"rna": rna, "cite": protein})


def read_10x_mtx(filename: PathLike, *args, **kwargs) -> MuData:
    adata = sc.read_10x_mtx(filename, gex_only=False, *args, **kwargs)
    protein = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()
    rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    return MuData({"rna": rna, "cite": protein})
