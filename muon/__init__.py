"""Multimodal omics analysis framework"""

from ._core.mudata import MuData
from ._core import preproc as pp
from ._core import tools as tl
from ._core import plot as pl
from ._core import utils
from ._core.io import *
from ._core.config import set_options

from . import atac
from . import prot

__version__ = "0.1.0"
__mudataversion__ = "0.1.0"
__anndataversion__ = "0.1.0"
