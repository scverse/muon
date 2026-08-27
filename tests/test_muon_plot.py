import matplotlib
import numpy as np
import pytest
from anndata import AnnData
from mudata import MuData

import muon as mu

matplotlib.use("Agg")


@pytest.fixture
def mdata() -> MuData:
    mdata = MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )
    mdata.var_names_make_unique()
    return mdata


def test_pl_scatter(mdata: MuData, rng: np.random.Generator) -> None:
    mdata.obs["condition"] = rng.choice(["a", "b"], mdata.n_obs)
    mu.pl.scatter(mdata, x="mod1:0", y="mod2:0", color="condition")
