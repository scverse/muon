import pytest

import numpy as np
from scipy import sparse
import pandas as pd
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
