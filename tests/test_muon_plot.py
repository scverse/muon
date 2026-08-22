import warnings

import pytest

import numpy as np
from scipy import sparse
import pandas as pd
import anndata
from anndata import AnnData
import muon as mu
from muon import MuData
import matplotlib

matplotlib.use("Agg")


@pytest.fixture()
def mdata():
    mdata = MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )
    mdata.var_names_make_unique()
    yield mdata


class TestScatter:
    def test_pl_scatter(self, mdata):
        mdata = mdata.copy()
        np.random.seed(42)
        mdata.obs["condition"] = np.random.choice(["a", "b"], mdata.n_obs)
        mu.pl.scatter(mdata, x="mod1:0", y="mod2:0", color="condition")


def test_embedding_layer_does_not_mutate():
    # Regression test for https://github.com/scverse/muon/issues/183
    rng = np.random.default_rng(0)
    n_obs, n_var = 20, 5
    X = rng.random((n_obs, n_var))
    ad = AnnData(X=X.copy())
    ad.var_names = [f"g{i}" for i in range(n_var)]
    ad.layers["counts"] = X.copy() * 10
    ad.obsm["X_umap"] = rng.random((n_obs, 2))
    mdata = MuData({"mod": ad})

    X_before = mdata["mod"].X.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", anndata.ImplicitModificationWarning)
        mu.pl.embedding(mdata, basis="mod:umap", color="g0", layer="counts", show=False)

    assert np.array_equal(X_before, mdata["mod"].X)
