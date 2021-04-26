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

    # Observations

    def test_filter_obs_adata(self, mdata, filepath_h5mu):
        ad = mdata["mod1"].copy()
        sub = np.random.binomial(1, 0.5, ad.n_obs).astype(bool)
        mu.pp.filter_obs(ad, sub)
        assert ad.n_obs == sub.sum()

    def test_filter_obs_mdata(self, mdata, filepath_h5mu):
        md = mdata.copy()
        sub = np.random.binomial(1, 0.5, md.n_obs).astype(bool)
        mu.pp.filter_obs(md, sub)
        assert md.n_obs == sub.sum()
        assert md["mod1"].n_obs == mdata.obsm["mod1"][sub].sum()
        assert md["mod2"].n_obs == mdata.obsm["mod2"][sub].sum()

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

        mdata_.file.close()

    def test_filter_obs_adata_view(self, mdata, filepath_h5mu):
        pov = np.random.binomial(1, 0.4, mdata.mod["mod1"].n_obs).astype(bool)
        view = mdata.mod["mod1"][pov, :]
        # When backed, in-place filtering should throw an error
        with pytest.raises(ValueError):
            sub = np.random.binomial(1, 0.5, view.n_obs).astype(bool)
            mu.pp.filter_obs(view, sub)

    # Variables

    def test_filter_var_adata(self, mdata, filepath_h5mu):
        ad = mdata["mod1"].copy()
        sub = np.random.binomial(1, 0.5, ad.n_vars).astype(bool)
        mu.pp.filter_var(ad, sub)
        assert ad.n_vars == sub.sum()

    def test_filter_var_mdata(self, mdata, filepath_h5mu):
        md = mdata.copy()
        sub = np.random.binomial(1, 0.5, md.n_vars).astype(bool)
        sub_mod1 = mdata.varm["mod1"][sub].sum()
        sub_mod2 = mdata.varm["mod2"][sub].sum()
        print(md)
        mu.pp.filter_var(md, sub)
        assert md.n_vars == sub.sum()
        print(md)
        print(sub.sum())
        print(sub_mod1)
        print(sub_mod2)
        assert md["mod1"].n_vars == sub_mod1
        assert md["mod2"].n_vars == sub_mod2

    def test_filter_var_adata_backed(self, mdata, filepath_h5mu):
        mdata.write(filepath_h5mu)
        mdata_ = mu.read_h5mu(filepath_h5mu, backed="r")
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]

        # When backed, in-place filtering should throw a warning
        with pytest.warns(UserWarning):
            sub = np.random.binomial(1, 0.5, mdata_.mod["mod1"].n_vars).astype(bool)
            print("Sub:\t", len(sub))
            print("Size:\t", mdata_.mod["mod1"].n_vars)
            mu.pp.filter_var(mdata_.mod["mod1"], sub)

        mdata_.file.close()

    def test_filter_var_adata_view(self, mdata, filepath_h5mu):
        pov = np.random.binomial(1, 0.4, mdata.mod["mod1"].n_obs).astype(bool)
        view = mdata.mod["mod1"][pov, :]
        # When backed, in-place filtering should throw an error
        with pytest.raises(ValueError):
            sub = np.random.binomial(1, 0.5, view.n_vars).astype(bool)
            mu.pp.filter_var(view, sub)


if __name__ == "__main__":
    unittest.main()
