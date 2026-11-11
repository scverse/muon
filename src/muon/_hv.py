"""Plotting under scanpy's ``ScanpyV2Preview`` preset, where scanpy's plotting functions return HoloViews elements.

Muon registers no HoloViews interface itself. ``sc.pl.hv_init`` registers the one from ``hv-anndata``, which reads a
:class:`~mudata.MuData` as well as an :class:`~anndata.AnnData` since version 0.1.3 (``MuData`` understands the same
``anndata.acc`` references, e.g. ``mdata[A.obs["x"]]``). ``scanpy`` and ``holoviews`` are imported lazily, so nothing
here is imported unless the preset is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import holoviews as hv
    from anndata import AnnData
    from anndata.acc import AdAcc, AdRef


def _scanpy_preset_is_v2() -> bool:
    """Whether the installed scanpy has presets and the active one is ``ScanpyV2Preview``."""
    import scanpy as sc

    preset_enum = getattr(sc, "Preset", None)  # scanpy versions without presets have no `Preset`
    v2 = getattr(preset_enum, "ScanpyV2Preview", None)
    return v2 is not None and getattr(sc.settings, "preset", None) is v2


def _v2_refs(ad: AnnData, A: AdAcc, keys: str | Sequence[str] | None) -> AdRef | list[AdRef] | None:
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


def _v2_options(plot: hv.Scatter | hv.Layout, options: dict[str, object]) -> hv.Scatter | hv.Layout:
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
    ad: AnnData, x: str | None, y: str | None, color: str | Sequence[str] | None = None, **kwargs: object
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
    ad: AnnData, basis: str, color: str | Sequence[str] | None = None, **kwargs: object
) -> hv.Scatter | hv.Layout:
    """Counterpart of ``scanpy.pl.embedding(ad, basis=basis, color=color)`` under the ``ScanpyV2Preview`` preset.

    The first two components of ``ad.obsm[basis]`` are plotted. ``color`` are columns of ``ad.obs``.
    Extra keyword arguments are HoloViews options, not matplotlib ones.
    """
    import scanpy as sc

    A = sc.pl.hv_init(None)
    if basis not in ad.obsm:
        available = ", ".join(ad.obsm.keys()) or "none"
        msg = f"Basis {basis!r} is not present in .obsm (available: {available})."
        raise KeyError(msg)
    return _v2_options(sc.pl.scatter(ad, A.obsm[basis][:, [0, 1]], color=_v2_refs(ad, A, color)), kwargs)


def reject_v2_options(**options: object) -> None:
    """Raise if a matplotlib/scanpy-v1 only option was set while the ``ScanpyV2Preview`` preset is active."""
    set_options = [k for k, v in options.items() if v not in (None, False)]
    if set_options:
        msg = f"{set_options} are not supported with scanpy's ScanpyV2Preview preset for AnnData input."
        raise NotImplementedError(msg)
