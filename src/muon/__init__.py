"""Multimodal omics analysis framework"""

from importlib.metadata import version

from mudata import MuData  # type: ignore[import-untyped]
from mudata._core.io import *  # type: ignore[import-untyped]  # noqa: F403

from . import atac, pl, pp, prot, tl, utils
from .config import set_options
from .io import *  # noqa: F403

__version__ = version("muon")
