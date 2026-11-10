"""HoloViews interface for :class:`~mudata.MuData` (needs ``hv-anndata``, so only import it lazily)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hv_anndata import AdDim, AnnDataInterface
from mudata import MuData

if TYPE_CHECKING:
    import holoviews as hv
    import pandas as pd
    from holoviews.core.data import Dataset
    from holoviews.core.dimension import Dimension
    from hv_anndata import Dims

__all__ = ["MuDataInterface"]


class MuDataInterface(AnnDataInterface):
    """HoloViews interface for :class:`~mudata.MuData`.

    A ``MuData`` is not an ``AnnData``, so the overrides below deliberately widen or change the ``AnnData``
    types of the parent class (hence the ``type: ignore`` comments).
    """

    types = (MuData,)  # type: ignore[assignment]
    datatype = "mudata"

    @classmethod
    def init(  # type: ignore[override]
        cls, eltype: hv.Element, data: MuData, kdims: list[Dimension] | None, vdims: list[Dimension] | None
    ) -> tuple[MuData, Dims, dict[str, object]]:
        """Initialize the interface; same as ``AnnDataInterface.init``, but requires a ``MuData``."""
        if not isinstance(data, MuData):
            msg = f"{cls.__name__} only supports MuData."
            raise TypeError(msg)
        key_dimensions: list[AdDim] = [AdDim.from_dimension(d) for d in kdims or []]
        value_dimensions: list[AdDim] = [AdDim.from_dimension(d) for d in vdims or []]
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
    def unpack_scalar(cls, dataset: Dataset, data: MuData | pd.DataFrame) -> object:  # type: ignore[override]
        """Like ``AnnDataInterface.unpack_scalar``, but a selected ``MuData`` is returned as is."""
        if isinstance(data, MuData):
            return data
        return super().unpack_scalar(dataset, data)
