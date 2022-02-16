import unittest
import pytest

import os

import numpy as np
import pandas as pd
from anndata import AnnData
import muon as mu
from muon import MuData

from muon._core.utils import _get_values


@pytest.fixture(
    params=[
        {"same_obs": True, "same_obs": False},
    ]
)
def mdata(request):
    mdata = MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )

    n, d = mdata.shape
    n1, d1 = mdata["mod1"].shape
    n2, d2 = mdata["mod2"].shape

    # var

    mdata["mod1"].var_names = [f"var1_{i+1}" for i in range(d1)]
    mdata["mod2"].var_names = [f"var2_{i+1}" for i in range(d2)]

    # obs

    mdata.obs["global_trait"] = "trait_g"

    mdata["mod1"].obs["mod1_trait"] = "trait1"
    mdata["mod2"].obs["mod2_trait"] = "trait2"

    mdata["mod1"].obs["common_trait"] = "trait_c1"
    mdata["mod2"].obs["common_trait"] = "trait_c2"

    # obsm

    mdata.obsm["global_emb"] = np.random.normal(size=(n, 2))

    mdata["mod1"].obsm["mod1_emb"] = np.random.normal(size=(n1, 20))
    mdata["mod2"].obsm["mod2_emb"] = np.random.normal(size=(n2, 20))

    mdata.update()

    if not request.param["same_obs"]:
        mdata.mod["mod1"] = mdata["mod1"][
            np.random.choice(np.arange(n1), size=n1 // 2, replace=False)
        ].copy()
        mdata.update()

    yield mdata


class TestTraitParsing:

    # Observations

    def test_global_obs(self, mdata):
        assert len(_get_values(mdata, "global_trait")) == mdata.n_obs

    def test_common_obs(self, mdata):
        with pytest.raises(ValueError):
            assert len(_get_values(mdata, "common_trait")) == mdata.n_obs
        assert len(_get_values(mdata, "mod1:common_trait")) == mdata.n_obs
        assert len(_get_values(mdata, "mod2:common_trait")) == mdata.n_obs

    def test_mod_obs(self, mdata):
        mod1_trait = _get_values(mdata, "mod1:mod1_trait")
        assert len(mod1_trait) == mdata.n_obs

        mod2_trait = _get_values(mdata, "mod2:mod2_trait")
        assert len(mod2_trait) == mdata.n_obs

        if mdata["mod1"].n_obs == mdata["mod2"].n_obs:
            assert all(mod1_trait == "trait1")
            assert all(mod2_trait == "trait2")
        else:
            print(mod1_trait)
            assert all(mod1_trait[~pd.isnull(mod1_trait)] == "trait1")
            assert all(mod2_trait[~pd.isnull(mod2_trait)] == "trait2")

    # Variables

    def test_var_name(self, mdata):
        var1_0 = mdata["mod1"].var_names[0]
        var2_0 = mdata["mod2"].var_names[0]
        assert len(_get_values(mdata, var1_0)) == mdata.n_obs
        assert len(_get_values(mdata, var2_0)) == mdata.n_obs


if __name__ == "__main__":
    unittest.main()
