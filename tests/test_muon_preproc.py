import unittest
import pytest

import os

import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData
import muon as mu
from muon import MuData


@pytest.fixture()
def mdata():
    yield MuData(
        {
            "mod1": AnnData(np.arange(0, 100, 0.1).reshape(-1, 10)),
            "mod2": AnnData(np.arange(101, 2101, 1).reshape(-1, 20)),
        }
    )


@pytest.mark.usefixtures("filepath_h5mu")
class TestInPlaceFiltering:
    def test_filter_obs_adata(self, mdata, filepath_h5mu):
        ad = mdata["mod1"].copy()
        sub = np.random.binomial(1, 0.5, ad.n_obs).astype(bool)
        mu.pp.filter_obs(ad, sub)
        assert ad.n_obs == sub.sum()

    def test_filter_obs_adata_backed(self, mdata, filepath_h5mu):
        mdata.write(filepath_h5mu)
        mdata_ = mu.read_h5mu(filepath_h5mu, backed="r")
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]

        # When backed, in-place filtering should throw a warning
        with pytest.warns(UserWarning):
            sub = np.random.binomial(1, 0.5, mdata_.mod["mod1"].n_obs).astype(bool)
            print("Sub:\t", len(sub))
            print("Size:\t", mdata_.mod["mod1"].n_obs)
            mu.pp.filter_obs(mdata_.mod["mod1"], sub)

    def test_filter_obs_adata_view(self, mdata, filepath_h5mu):
        pov = np.random.binomial(1, 0.4, mdata.mod["mod1"].n_obs).astype(bool)
        view = mdata.mod["mod1"][pov, :]
        # When backed, in-place filtering should throw an error
        with pytest.raises(ValueError):
            sub = np.random.binomial(1, 0.5, view.n_obs).astype(bool)
            mu.pp.filter_obs(view, sub)


if __name__ == "__main__":
    unittest.main()
