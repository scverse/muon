"""Multimodal omics analysis framework"""

from mudata import MuData
from mudata._core.io import *  # noqa: F403

from . import atac, pl, pp, prot, tl, utils
from ._version import __version__
from .config import set_options
from .io import *  # noqa: F403
