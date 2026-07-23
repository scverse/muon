"""Multimodal omics analysis framework"""

from importlib.metadata import version

from mudata import MuData  # type: ignore[import-untyped]
from mudata._core.io import *  # type: ignore[import-untyped]  # noqa: F403

from . import atac, prot
from ._core import plot as pl
from ._core import preproc as pp
from ._core import tools as tl
from ._core import utils
from ._core.config import set_options
from ._core.io import *  # noqa: F403

__version__ = version("muon")
