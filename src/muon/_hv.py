"""Let HoloViews read a :class:`~mudata.MuData`.

``hv-anndata`` registers an interface that lets HoloViews elements be created from an
:class:`~anndata.AnnData`. A :class:`~mudata.MuData` is not an ``AnnData``, so without an interface
HoloViews falls back to treating it as a dictionary and never finishes.

:class:`MuDataInterface` is a subclass of ``hv_anndata.AnnDataInterface`` that only changes what is specific
to ``AnnData``: the ``types`` and ``datatype`` attributes, and the two ``isinstance`` checks. Everything else
(reading values, selecting, grouping) is inherited, because ``MuData`` understands the same ``anndata.acc``
references (``mdata[A.obs["x"]]``).

The interface is registered when scanpy's preset is ``ScanpyV2Preview``, which is the preset in which scanpy's
plotting functions return HoloViews elements. ``holoviews`` and ``hv-anndata`` are imported lazily, so nothing
here is imported unless that preset is active.
"""

from __future__ import annotations

import warnings
from typing import Any

_registered = False


def _scanpy_preset_is_v2() -> bool:
    """Whether the installed scanpy has presets and the active one is ``ScanpyV2Preview``."""
    import scanpy as sc

    preset_enum = getattr(sc, "Preset", None)  # scanpy versions without presets have no `Preset`
    v2 = getattr(preset_enum, "ScanpyV2Preview", None)
    return v2 is not None and getattr(sc.settings, "preset", None) is v2


def register_mudata_interface() -> None:
    """Register the HoloViews interface for :class:`~mudata.MuData` (does nothing if already registered)."""
    global _registered
    if _registered:
        return

    import holoviews as hv
    from hv_anndata import AdDim, AnnDataInterface
    from mudata import MuData

    class MuDataInterface(AnnDataInterface):
        """HoloViews interface for :class:`~mudata.MuData`."""

        types = (MuData,)
        datatype = "mudata"

        @classmethod
        def init(cls, eltype: Any, data: Any, kdims: Any, vdims: Any) -> Any:
            # Same as `AnnDataInterface.init`, which only differs in requiring an `AnnData`.
            if not isinstance(data, MuData):
                msg = f"{cls.__name__} only supports MuData."
                raise TypeError(msg)
            key_dimensions = [AdDim.from_dimension(d) for d in kdims or []]
            value_dimensions = [AdDim.from_dimension(d) for d in vdims or []]
            vdim = value_dimensions[0] if value_dimensions else None
            ndims = len(vdim.dims) if vdim else 1
            if not cls.gridded and ndims > 1:
                msg = f"{cls.__name__} cannot handle gridded data."
                raise ValueError(msg)
            if cls.gridded and ndims == 1:
                msg = f"{cls.__name__} cannot handle tabular data."
                raise ValueError(msg)
            return data, {"kdims": key_dimensions, "vdims": value_dimensions}, {}

        @classmethod
        def unpack_scalar(cls, dataset: Any, data: Any) -> Any:
            # Selecting returns a MuData, which must not be unpacked like a one-value DataFrame.
            if isinstance(data, MuData):
                return data
            return super().unpack_scalar(dataset, data)

    if MuDataInterface.datatype not in hv.core.data.datatypes:
        hv.core.data.datatypes.append(MuDataInterface.datatype)
    hv.core.Interface.register(MuDataInterface)
    _registered = True


def register_if_scanpy_v2() -> bool:
    """Register the interface if scanpy's preset is ``ScanpyV2Preview``; return whether it is registered."""
    if not _scanpy_preset_is_v2():
        return False
    try:
        register_mudata_interface()
    except ImportError as e:
        msg = f"scanpy's preset is ScanpyV2Preview, but HoloViews support for MuData could not be enabled: {e}"
        warnings.warn(msg, stacklevel=2)
        return False
    return True
