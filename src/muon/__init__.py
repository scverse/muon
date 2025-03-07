"""Multimodal omics analysis framework"""

try:  # See https://github.com/maresb/hatch-vcs-footgun-example
    from setuptools_scm import get_version

    __version__ = get_version(root="../..", relative_to=__file__)
except (ImportError, LookupError):
    try:
        from ._version import __version__
    except ModuleNotFoundError:
        raise RuntimeError("muon is not correctly installed. Please install it, e.g. with pip.")


from mudata import MuData
from mudata._core.io import (
    read,
    read_anndata,
    read_h5ad,
    read_h5mu,
    read_zarr,
    write_anndata,
    write_h5ad,
    write_h5mu,
    write_zarr,
)

from . import _atac as atac
from . import _prot as prot
from . import _rna as rna
from ._core import plot as pl
from ._core import preproc as pp
from ._core import tools as tl
from ._core import utils
from ._core.config import set_options
from ._core.io import read_10x_h5, read_10x_mtx

__all__ = [
    "__version__",
    "MuData",
    "pp",
    "tl",
    "pl",
    "utils",
    # mudata I/O
    "read_h5mu",
    "read_h5ad",
    "read_anndata",
    "read_zarr",
    "write_h5mu",
    "write_h5ad",
    "write_anndata",
    "write_zarr",
    "read",
    # muon I/O
    "read_10x_h5",
    "read_10x_mtx",
    "set_options",
    "atac",
    "prot",
    "rna",
]
