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
from mudata._core.io import *

from ._core import preproc as pp
from ._core import tools as tl
from ._core import plot as pl
from ._core import utils
from ._core.io import *
from ._core.config import set_options

from . import atac
from . import prot

__all__ = [
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
]