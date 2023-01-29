from typing import Union
from os import PathLike
import os
from warnings import warn

import numpy as np
import h5py
import anndata as ad
from anndata import AnnData
from pathlib import Path
import scanpy as sc

from mudata import MuData
from mudata._core.file_backing import MuDataFileManager, AnnDataFileManager

from .._atac.tools import initialise_default_files

#
# Reading data
#


def read_10x_h5(filename: PathLike, extended: bool = True, *args, **kwargs) -> MuData:
    """
    Read data from 10X Genomics-formatted HDF5 file

    This function uses scanpy.read_10x_h5() internally
    and patches its behaviour to:
    - attempt to read `interval` field for features;
    - attempt to locate peak annotation file and add peak annotation;
    - attempt to locate fragments file.

    Parameters
    ----------
    filename : str
            Path to 10X HDF5 file (.h5)
    extended : bool, optional (default: True)
            Perform extended functionality automatically such as
            locating peak annotation and fragments files.
    """

    adata = sc.read_10x_h5(filename, gex_only=False, *args, **kwargs)

    # Patches sc.read_10x_h5 behaviour to:
    # - attempt to read `interval` field for features from the HDF5 file
    # - attempt to add peak annotation
    # - attempt to locate fragments file

    if extended:

        # 1) Read interval field from the HDF5 file
        h5file = h5py.File(filename, "r")

        if "interval" in h5file["matrix"]["features"]:
            intervals = np.array(h5file["matrix"]["features"]["interval"]).astype(str)

            h5file.close()

            adata.var["interval"] = intervals

            print(f"Added `interval` annotation for features from {filename}")

        else:
            # Make sure the file is closed
            h5file.close()

    mdata = MuData(adata)

    if extended:
        if "atac" in mdata.mod:
            initialise_default_files(mdata, filename)

    return mdata


def read_10x_mtx(path: PathLike, extended: bool = True, *args, **kwargs) -> MuData:
    """
    Read data from 10X Genomics-formatted files
    (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz)

    This function uses scanpy.read_10x_mtx() internally
    and patches its behaviour to:
    - attempt to read `interval` field for features;
    - (for ATAC-seq) attempt to locate peak annotation file and add peak annotation;
    - (for ATAC-seq) attempt to locate fragments file.

    Parameters
    ----------
    path : str
            Path to 10X folder (filtered_feature_bc_matrix or raw_feature_bc_matrix)
            or to the matrix file inside it
    extended : bool, optional (default: True)
            Perform extended functionality automatically such as
            locating peak annotation and fragments files.
    """

    adata = sc.read_10x_mtx(path, gex_only=False, *args, **kwargs)

    mdata = MuData(adata)

    # Patches sc.read_10x_h5 behaviour to:
    # - attempt to add peak annotation
    # - attempt to locate fragments file
    if extended:
        if "atac" in mdata.mod:
            initialise_default_files(mdata, path)

    return mdata
