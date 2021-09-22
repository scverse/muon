"""Multimodal omics analysis framework"""

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

__version__ = "0.1.1"
