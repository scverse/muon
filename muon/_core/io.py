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

from .mudata import MuData
from .file_backing import MuDataFileManager, AnnDataFileManager

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


#
# Saving multimodal data objects
#


def _write_h5mu(file: h5py.File, mdata: MuData, write_data=True, **kwargs):
    from anndata._io.utils import write_attribute
    from .. import __version__, __mudataversion__, __anndataversion__

    write_attribute(
        file,
        "obs",
        mdata.strings_to_categoricals(mdata._shrink_attr("obs", inplace=False)),
        dataset_kwargs=kwargs,
    )
    write_attribute(
        file,
        "var",
        mdata.strings_to_categoricals(mdata._shrink_attr("var", inplace=False)),
        dataset_kwargs=kwargs,
    )
    write_attribute(file, "obsm", mdata.obsm, dataset_kwargs=kwargs)
    write_attribute(file, "varm", mdata.varm, dataset_kwargs=kwargs)
    write_attribute(file, "obsp", mdata.obsp, dataset_kwargs=kwargs)
    write_attribute(file, "varp", mdata.varp, dataset_kwargs=kwargs)
    write_attribute(file, "uns", mdata.uns, dataset_kwargs=kwargs)

    write_attribute(file, "obsmap", mdata.obsmap, dataset_kwargs=kwargs)
    write_attribute(file, "varmap", mdata.varmap, dataset_kwargs=kwargs)

    mod = file.require_group("mod")
    for k, v in mdata.mod.items():
        group = mod.require_group(k)

        adata = mdata.mod[k]

        adata.strings_to_categoricals()
        if adata.raw is not None:
            adata.strings_to_categoricals(adata.raw.var)

        if write_data:
            write_attribute(group, "X", adata.X, dataset_kwargs=kwargs)
        if adata.raw is not None:
            write_h5ad_raw(group, "raw", adata.raw)

        write_attribute(group, "obs", adata.obs, dataset_kwargs=kwargs)
        write_attribute(group, "var", adata.var, dataset_kwargs=kwargs)
        write_attribute(group, "obsm", adata.obsm, dataset_kwargs=kwargs)
        write_attribute(group, "varm", adata.varm, dataset_kwargs=kwargs)
        write_attribute(group, "obsp", adata.obsp, dataset_kwargs=kwargs)
        write_attribute(group, "varp", adata.varp, dataset_kwargs=kwargs)
        write_attribute(group, "layers", adata.layers, dataset_kwargs=kwargs)
        write_attribute(group, "uns", adata.uns, dataset_kwargs=kwargs)

        attrs = group.attrs
        attrs["encoding-type"] = "AnnData"
        attrs["encoding-version"] = __anndataversion__
        attrs["encoder"] = "muon"
        attrs["encoder-version"] = __version__

    attrs = file.attrs
    attrs["encoding-type"] = "MuData"
    attrs["encoding-version"] = __mudataversion__
    attrs["encoder"] = "muon"
    attrs["encoder-version"] = __version__

    # Restore top-level annotation
    if not mdata.is_view or not mdata.isbacked:
        mdata.update()


def write_h5mu(filename: PathLike, mdata: MuData, **kwargs):
    """
    Write MuData object to the HDF5 file

    Matrices - sparse or dense - are currently stored as they are.
    """
    from .. import __version__, __mudataversion__, __anndataversion__

    with h5py.File(filename, "w", userblock_size=512) as f:
        _write_h5mu(f, mdata, **kwargs)
    with open(filename, "br+") as f:
        nbytes = f.write(
            f"MuData (format-version={__mudataversion__};creator=muon;creator-version={__version__})".encode(
                "utf-8"
            )
        )
        f.write(
            b"\0" * (512 - nbytes)
        )  # this is only needed because the H5file was written in append mode


def write_h5ad(filename: PathLike, mod: str, data: Union[MuData, AnnData]):
    """
    Write AnnData object to the HDF5 file with a MuData container

    Currently is based on anndata._io.h5ad.write_h5ad internally.
    Matrices - sparse or dense - are currently stored as they are.

    Ideally this is merged later to anndata._io.h5ad.write_h5ad.
    """
    from anndata._io.utils import write_attribute
    from anndata._io.h5ad import write_h5ad
    from .. import __version__, __anndataversion__

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod[mod]
    else:
        raise TypeError(f"Expected AnnData or MuData object with {mod} modality")

    with h5py.File(filename, "r+") as f:
        # Check that 'mod' is present
        if not "mod" in f:
            raise ValueError("The .h5mu object has to contain .mod slot")
        fm = f["mod"]

        # Remove the modality if it exists
        if mod in fm:
            del fm[mod]

        fmd = fm.create_group(mod)

        adata.strings_to_categoricals()
        if adata.raw is not None:
            adata.strings_to_categoricals(adata.raw.var)

        filepath = Path(filename)

        if not (adata.isbacked and Path(adata.filename) == Path(filepath)):
            write_attribute(fmd, f"X", adata.X)

        # NOTE: Calling write_attribute() does not allow writing .raw into .h5mu modalities
        if adata.raw is not None:
            write_h5ad_raw(f, f"mod/{mod}/raw", adata.raw)

        write_attribute(fmd, "obs", adata.obs)
        write_attribute(fmd, "var", adata.var)
        write_attribute(fmd, "obsm", adata.obsm)
        write_attribute(fmd, "varm", adata.varm)
        write_attribute(fmd, "obsp", adata.obsp)
        write_attribute(fmd, "varp", adata.varp)
        write_attribute(fmd, "layers", adata.layers)
        write_attribute(fmd, "uns", adata.uns)

        attrs = fmd.attrs
        attrs["encoding-type"] = "AnnData"
        attrs["encoding-version"] = __anndataversion__
        attrs["encoder"] = "muon"
        attrs["encoder-version"] = __version__


write_anndata = write_h5ad


def write_h5ad_raw(f, key, raw, **kwargs):
    """
    Replicates write_raw() in anndata/_io/h5ad.py but allow
    to write raw slots to modalities inside .h5mu files
    """
    from anndata._io.utils import write_attribute, EncodingVersions

    group = f.create_group(key)
    group.attrs["encoding-type"] = "raw"
    group.attrs["encoding-version"] = EncodingVersions.raw.value
    group.attrs["shape"] = raw.shape
    write_attribute(f, f"{key}/X", raw.X, dataset_kwargs=kwargs)
    write_attribute(f, f"{key}/var", raw.var, dataset_kwargs=kwargs)
    write_attribute(f, f"{key}/varm", raw.varm, dataset_kwargs=kwargs)


def write(filename: PathLike, data: Union[MuData, AnnData]):
    """
    Write MuData or AnnData to an HDF5 file

    This function is designed to enhance I/O ease of use.
    It recognises the following formats of filename:
      - for MuData
            - FILE.h5mu
      - for AnnData
              - FILE.h5mu/MODALITY
              - FILE.h5mu/mod/MODALITY
              - FILE.h5ad
    """

    import re

    if filename.endswith(".h5mu") or isinstance(data, MuData):
        assert filename.endswith(".h5mu") and isinstance(
            data, MuData
        ), "Can only save MuData object to .h5mu file"

        write_h5mu(filename, data)

    else:
        assert isinstance(data, AnnData), "Only MuData and AnnData objects are accepted"

        m = re.search("^(.+)\.(h5mu)[/]?([A-Za-z]*)[/]?([/A-Za-z]*)$", filename)
        if m is not None:
            m = m.groups()
        else:
            raise ValueError("Expected non-empty .h5ad or .h5mu file name")

        filepath = ".".join([m[0], m[1]])

        if m[1] == "h5mu":
            if m[3] == "":
                # .h5mu/<modality>
                return write_h5ad(filepath, m[2], data)
            elif m[2] == "mod":
                # .h5mu/mod/<modality>
                return write_h5ad(filepath, m[3], data)
            else:
                raise ValueError(
                    "If a single modality to be written from a .h5mu file, \
                    provide it after the filename separated by slash symbol:\
                    .h5mu/rna or .h5mu/mod/rna"
                )
        elif m[1] == "h5ad":
            return data.write(filepath)
        else:
            raise ValueError()


#
# Reading from multimodal data objects
#


def read_h5mu(filename: PathLike, backed: Union[str, bool, None] = None):
    """
    Read MuData object from HDF5 file
    """
    assert backed in [
        None,
        True,
        False,
        "r",
        "r+",
    ], "Argument `backed` should be boolean, or r/r+, or None"

    from anndata._io.utils import read_attribute
    from anndata._io.h5ad import read_dataframe

    if backed is True or not backed:
        mode = "r"
    else:
        mode = backed
    manager = MuDataFileManager(filename, mode) if backed else MuDataFileManager()
    with open(filename, "rb") as f:
        ish5mu = f.read(6) == b"MuData"
    if not ish5mu:
        if h5py.is_hdf5(filename):
            warn(
                "The HDF5 file was not created by muon, we can't guarantee that everything will work correctly"
            )
        else:
            raise ValueError("The file is not an HDF5 file")

    with h5py.File(filename, mode) as f:
        d = {}
        for k in f.keys():
            if k in ["obs", "var"]:
                d[k] = read_dataframe(f[k])
            if k == "mod":
                mods = {}
                gmods = f[k]
                for m in gmods.keys():
                    ad = _read_h5mu_mod_backed(gmods[m]) if backed else read_h5ad(filename, m)
                    ad.file = AnnDataFileManager(ad, m, manager)
                    mods[m] = ad
                d[k] = mods
            else:
                d[k] = read_attribute(f[k])

    mu = MuData._init_from_dict_(**d)
    mu.file = manager
    return mu


def _read_h5mu_mod_backed(g: "h5py.Group") -> dict:
    from anndata._io.utils import read_attribute
    from anndata._io.h5ad import read_dataframe, _read_raw
    from anndata import Raw

    d = {}

    for k in g.keys():
        if k in ("obs", "var"):
            d[k] = read_dataframe(g[k])
        elif k == "X":
            X = g["X"]
            if isinstance(X, h5py.Group):
                dtype = X["data"].dtype
            elif hasattr(X, "dtype"):
                dtype = X.dtype
            else:
                raise ValueError()
            d["dtype"] = dtype
        elif k != "raw":
            d[k] = read_attribute(g[k])
    ad = AnnData(**d)

    raw = _read_raw(g, attrs={"var", "varm"})
    if raw:
        ad._raw = Raw(ad, **raw)
    return ad


def read_h5ad(
    filename: PathLike,
    mod: str,
    backed: Union[str, bool, None] = None,
) -> AnnData:
    """
    Read AnnData object from inside a .h5mu file
    or from a standalone .h5ad file

    Currently replicates and modifies anndata._io.h5ad.read_h5ad.
    Matrices are loaded as they are in the file (sparse or dense).

    Ideally this is merged later to anndata._io.h5ad.read_h5ad.
    """
    assert backed in [
        None,
        True,
        False,
        "r",
        "r+",
    ], "Argument `backed` should be boolean, or r/r+, or None"

    from anndata._io.utils import read_attribute
    from anndata._io.h5ad import read_dataframe, _read_raw

    d = {}

    hdf5_mode = "r"
    if backed not in {None, False}:
        hdf5_mode = backed
        if hdf5_mode is True:
            hdf5_mode = "r+"
        assert hdf5_mode in {"r", "r+"}
        backed = True

        manager = MuDataFileManager(filename, hdf5_mode)

    with h5py.File(filename, hdf5_mode) as f_root:
        f = f_root["mod"][mod]
        if backed:
            ad = _read_h5mu_mod_backed(f, manager)
            ad.file = AnnDataFileManager(ad, mod, manager)
            return ad

        for k in f.keys():
            if k in ["obs", "var"]:
                d[k] = read_dataframe(f[k])
            elif k != "raw":
                d[k] = read_attribute(f[k])

        d["raw"] = _read_raw(f)

        X_dset = f.get("X", None)
        if X_dset is None:
            pass
        elif isinstance(X_dset, h5py.Group):
            d["dtype"] = X_dset["data"].dtype
        elif hasattr(X_dset, "dtype"):
            d["dtype"] = f["X"].dtype
        else:
            raise ValueError()

    return AnnData(**d)


read_anndata = read_h5ad


def read(filename: PathLike, **kwargs) -> Union[MuData, AnnData]:
    """
    Read MuData object from HDF5 file
    or AnnData object (a single modality) inside it

    This function is designed to enhance I/O ease of use.
    It recognises the following formats:
      - FILE.h5mu
      - FILE.h5mu/MODALITY
      - FILE.h5mu/mod/MODALITY
      - FILE.h5ad
    """
    import re

    m = re.search("^(.+)\.(h5mu)[/]?([A-Za-z]*)[/]?([/A-Za-z]*)$", filename)
    if m is not None:
        m = m.groups()
    else:
        if filename.endswith(".h5ad"):
            m = [filename[:-5], "h5ad", "", ""]
        else:
            raise ValueError("Expected non-empty .h5ad or .h5mu file name")

    filepath = ".".join([m[0], m[1]])

    if m[1] == "h5mu":
        if all(i == 0 for i in map(len, m[2:])):
            # Ends with .h5mu
            return read_h5mu(filepath, **kwargs)
        elif m[3] == "":
            # .h5mu/<modality>
            return read_h5ad(filepath, m[2], **kwargs)
        elif m[2] == "mod":
            # .h5mu/mod/<modality>
            return read_h5ad(filepath, m[3], **kwargs)
        else:
            raise ValueError(
                "If a single modality to be read from a .h5mu file, \
                provide it after the filename separated by slash symbol:\
                .h5mu/rna or .h5mu/mod/rna"
            )
    elif m[1] == "h5ad":
        return ad.read_h5ad(filepath, **kwargs)
    else:
        raise ValueError("The file format is not recognised, expected to be an .h5mu or .h5ad file")
