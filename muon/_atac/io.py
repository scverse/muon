from os import PathLike
from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def read_10x_h5(filename: PathLike, atac_only: bool = True, *args, **kwargs) -> AnnData:
    adata = sc.read_10x_h5(filename, gex_only=False, *args, **kwargs)
    if atac_only:
        adata = adata[:, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))]
    return adata


def read_10x_mtx(filename: PathLike, atac_only: bool = True, *args, **kwargs) -> AnnData:
    adata = sc.read_10x_mtx(filename, gex_only=False, *args, **kwargs)
    if atac_only:
        adata = adata[:, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))]
    return adata


def read_snap(filename: PathLike, matrix: str, bin_size: Optional[int] = None):
    """
    Read a matrix from a .snap file.

    Parameters
    ----------
    filename : str
            Path to .snap file.
    matrix : str
            Count matrix to be read, which can be
            - cell-by-peak ('peaks', 'PM'),
            - cell-by-gene ('genes', 'GM'),
            - cell-by-bin matrix ('bins', 'AM').
            In the latter case `bin_size` has to be provided.
    bin_size : Optional[int]
            Bin size, only relevant and necessary when cells x bins matrix (AM) is read.
    """

    try:
        from snaptools import snap
    except ImportError:
        raise ImportError(
            "SnapTools library is not available. Install SnapTools from PyPI (`pip install snaptools`) or from GitHub (`pip install git+https://github.com/r3fang/SnapTools`)"
        )

    from scipy.sparse import csr_matrix
    import h5py

    # Allow both PM and pm
    matrix = matrix.lower()
    assert matrix in ["pm", "gm", "am", "peaks", "genes", "bins"]
    if bin_size is not None:
        if matrix not in ["bm", "bins"]:
            warn("Argument bin_size is only relevant for bins matrix (BM) and will be ignored")

    f = h5py.File(filename, "r")

    if matrix == "pm" or matrix == "peaks":
        if "PM" in f:
            chrom = np.array(f["PM"]["peakChrom"]).astype(str)
            start = np.array(f["PM"]["peakStart"])
            end = np.array(f["PM"]["peakEnd"])
            idx = np.array(f["PM"]["idx"]) - 1
            idy = np.array(f["PM"]["idy"]) - 1
            count = np.array(f["PM"]["count"])

            features = (
                np.char.array(chrom)
                + ":"
                + np.char.array(start).astype("str")
                + "-"
                + np.char.array(end).astype("str")
            )
            var = pd.DataFrame({"Chromosome": chrom, "Start": start, "End": end}, index=features)
        else:
            raise AttributeError("PM is not available in the snap file")

    elif matrix == "gm" or matrix == "genes":
        if "GM" in f:
            name = np.array(f["GM"]["name"]).astype(str)
            idx = np.array(f["GM"]["idx"]) - 1
            idy = np.array(f["GM"]["idy"]) - 1
            count = np.array(f["GM"]["count"])

            var = pd.DataFrame(index=name)
        else:
            raise AttributeError("GM is not available in the snap file")

    elif matrix == "bm" or matrix == "bins":
        if "AM" in f:
            bin_sizes = list(f["AM"]["binSizeList"])
            if bin_size is None or int(bin_size) not in bin_sizes:
                raise ValueError(
                    f"Argument bin_size has to be defined. Available bin sizes: {', '.join([str(i) for i in bin_sizes])}."
                )

            am = f["AM"][str(bin_size)]
            chrom = np.array(am["binChrom"]).astype(str)
            start = np.array(am["binStart"])
            idx = np.array(am["idx"]) - 1
            idy = np.array(am["idy"]) - 1
            count = np.array(am["count"])

            features = (
                np.char.array(chrom)
                + ":"
                + np.char.array(start - 1).astype("str")
                + "-"
                + np.char.array(start + bin_size - 1).astype("str")
            )
            var = pd.DataFrame({"Chromosome": chrom, "Start": start - 1}, index=features)

        else:
            raise AttributeError("AM is not available in the snap file")

    f.close()

    bcs = snap.getBarcodesFromSnap(filename)
    obs = pd.DataFrame([bcs[i].__dict__ for i in bcs.keys()], index=bcs.keys())

    x = csr_matrix((count, (idx, idy)), shape=(obs.shape[0], var.shape[0]))

    return AnnData(X=x, obs=obs, var=var)
