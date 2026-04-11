"""Test that random_state=0 correctly sets the RNG seed in leiden/louvain clustering."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import scanpy as sc
from anndata import AnnData
from mudata import MuData

import muon as mu


def _make_mudata():
    """Create a minimal MuData with precomputed neighbors."""
    np.random.seed(42)
    a = AnnData(np.random.rand(50, 10).astype(np.float32))
    b = AnnData(np.random.rand(50, 10).astype(np.float32))
    sc.pp.neighbors(a)
    sc.pp.neighbors(b)
    return MuData({"a": a, "b": b})


def _make_mock_leidenalg():
    """Create a mock leidenalg module with the interfaces the code needs."""
    mock_module = ModuleType("leidenalg")

    mock_optimiser_instance = MagicMock()
    mock_optimiser_instance.optimise_partition_multiplex.return_value = 0.0

    mock_partition = MagicMock()
    mock_partition.membership = [0] * 50

    mock_optimiser_cls = MagicMock(return_value=mock_optimiser_instance)
    mock_partition_cls = MagicMock(return_value=mock_partition)

    mock_module.Optimiser = mock_optimiser_cls
    mock_module.RBConfigurationVertexPartition = mock_partition_cls

    # Also mock the VertexPartition submodule so module-level imports work.
    vp = ModuleType("leidenalg.VertexPartition")
    vp.MutableVertexPartition = MagicMock()
    mock_module.VertexPartition = vp

    return mock_module, mock_optimiser_instance


def test_leiden_random_state_zero_sets_seed():
    """Regression test for https://github.com/scverse/muon/issues/154.

    random_state=0 must call optimiser.set_rng_seed(0), not skip it.
    """
    mdata = _make_mudata()
    mock_module, mock_opt = _make_mock_leidenalg()

    saved = sys.modules.get("leidenalg")
    saved_vp = sys.modules.get("leidenalg.VertexPartition")
    try:
        sys.modules["leidenalg"] = mock_module
        sys.modules["leidenalg.VertexPartition"] = mock_module.VertexPartition
        mu.tl.leiden(mdata, random_state=0)
        mock_opt.set_rng_seed.assert_called_once_with(0)
    finally:
        if saved is None:
            sys.modules.pop("leidenalg", None)
        else:
            sys.modules["leidenalg"] = saved
        if saved_vp is None:
            sys.modules.pop("leidenalg.VertexPartition", None)
        else:
            sys.modules["leidenalg.VertexPartition"] = saved_vp


def test_leiden_random_state_none_skips_seed():
    """When random_state is None, set_rng_seed should not be called."""
    mdata = _make_mudata()
    mock_module, mock_opt = _make_mock_leidenalg()

    saved = sys.modules.get("leidenalg")
    saved_vp = sys.modules.get("leidenalg.VertexPartition")
    try:
        sys.modules["leidenalg"] = mock_module
        sys.modules["leidenalg.VertexPartition"] = mock_module.VertexPartition
        mu.tl.leiden(mdata, random_state=None)
        mock_opt.set_rng_seed.assert_not_called()
    finally:
        if saved is None:
            sys.modules.pop("leidenalg", None)
        else:
            sys.modules["leidenalg"] = saved
        if saved_vp is None:
            sys.modules.pop("leidenalg.VertexPartition", None)
        else:
            sys.modules["leidenalg.VertexPartition"] = saved_vp
