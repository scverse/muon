"""Multimodal omics analysis framework."""

from mudata import MuData
from mudata._core.io import *  # noqa: F403

from . import atac, pl, pp, prot, tl, utils
from ._hv import register_if_scanpy_v2
from ._version import __version__
from .config import set_options
from .io import *  # noqa: F403

# HoloViews can only read a MuData once an interface is registered (see `_hv.py`).
register_if_scanpy_v2()
