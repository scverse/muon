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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import holoviews as hv
    from anndata import AnnData

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


def _v2_refs(ad: AnnData, A: Any, keys: str | Sequence[str] | None) -> Any:
    """Turn ``.obs`` column names into ``hv-anndata`` references (``None`` stays ``None``)."""
    if keys is None:
        return None
    names = [keys] if isinstance(keys, str) else list(keys)
    missing = [k for k in names if k not in ad.obs.columns]
    if missing:
        msg = (
            f"{missing} not found in .obs. With scanpy's ScanpyV2Preview preset, plain AnnData input can only be "
            "coloured by .obs columns; for a MuData, features from any modality are supported."
        )
        raise NotImplementedError(msg)
    return A.obs[names[0]] if len(names) == 1 else A.obs[names]


def _v2_options(plot: Any, options: dict[str, Any]) -> Any:
    """Add a hover tool (Bokeh) and apply extra keyword arguments as options of every scatter plot.

    The options are applied to the plots and not to a surrounding ``Layout``, which HoloViews does not support.
    Keyword arguments override the defaults, e.g. ``tools=["pan"]`` removes the hover tool.
    """
    import holoviews as hv

    if "bokeh" in getattr(hv.Store, "renderers", {}):
        plot = plot.opts(hv.opts.Scatter(tools=["hover"], backend="bokeh"))
    if options:
        plot = plot.opts(hv.opts.Scatter(**options))
    return plot


def v2_scatter(
    ad: AnnData, x: str | None, y: str | None, color: str | Sequence[str] | None = None, **kwargs: Any
) -> hv.Scatter | hv.Layout:
    """Counterpart of ``scanpy.pl.scatter(ad, x=x, y=y, color=color)`` under the ``ScanpyV2Preview`` preset.

    ``x``, ``y`` and ``color`` are columns of ``ad.obs``. Extra keyword arguments are HoloViews options, not
    matplotlib ones.
    """
    import scanpy as sc

    if x is None or y is None:
        msg = "`x` and `y` are required with scanpy's ScanpyV2Preview preset."
        raise ValueError(msg)
    A = sc.pl.hv_init(None)  # registers the AnnData interface; does not load a plotting backend
    kdims = [_v2_refs(ad, A, x), _v2_refs(ad, A, y)]
    return _v2_options(sc.pl.scatter(ad, kdims, color=_v2_refs(ad, A, color)), kwargs)


def v2_embedding(
    ad: AnnData, basis: str, color: str | Sequence[str] | None = None, **kwargs: Any
) -> hv.Scatter | hv.Layout:
    """Counterpart of ``scanpy.pl.embedding(ad, basis=basis, color=color)`` under the ``ScanpyV2Preview`` preset.

    The first two components of ``ad.obsm[basis]`` are plotted. ``color`` are columns of ``ad.obs``.
    Extra keyword arguments are HoloViews options, not matplotlib ones.
    """
    import scanpy as sc

    A = sc.pl.hv_init(None)
    if basis not in ad.obsm:
        available = ", ".join(ad.obsm) or "none"
        msg = f"Basis {basis!r} is not present in .obsm (available: {available})."
        raise KeyError(msg)
    return _v2_options(sc.pl.scatter(ad, A.obsm[basis][:, [0, 1]], color=_v2_refs(ad, A, color)), kwargs)


def reject_v2_options(**options: Any) -> None:
    """Raise if a matplotlib/scanpy-v1 only option was set while the ``ScanpyV2Preview`` preset is active."""
    set_options = [k for k, v in options.items() if v not in (None, False)]
    if set_options:
        msg = f"{set_options} are not supported with scanpy's ScanpyV2Preview preset for AnnData input."
        raise NotImplementedError(msg)
