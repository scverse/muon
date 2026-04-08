"""Test that random_state=0 correctly sets the RNG seed in leiden/louvain clustering."""

from unittest.mock import patch, MagicMock

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


def test_leiden_random_state_zero_sets_seed():
    """Regression test for https://github.com/scverse/muon/issues/154.

    random_state=0 must call optimiser.set_rng_seed(0), not skip it.
    """
    mdata = _make_mudata()

    with patch("leidenalg.Optimiser") as MockOptimiser:
        mock_opt = MagicMock()
        mock_opt.optimise_partition_multiplex.return_value = 0.0
        MockOptimiser.return_value = mock_opt

        mu.tl.leiden(mdata, random_state=0)

        mock_opt.set_rng_seed.assert_called_once_with(0)


def test_leiden_random_state_none_skips_seed():
    """When random_state is None, set_rng_seed should not be called."""
    mdata = _make_mudata()

    with patch("leidenalg.Optimiser") as MockOptimiser:
        mock_opt = MagicMock()
        mock_opt.optimise_partition_multiplex.return_value = 0.0
        MockOptimiser.return_value = mock_opt

        mu.tl.leiden(mdata, random_state=None)

        mock_opt.set_rng_seed.assert_not_called()
