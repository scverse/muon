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
class TestMuData:
    def test_write_read_h5mu_basic(self, mdata, filepath_h5mu):
        mdata.write(filepath_h5mu)
        mdata_ = mu.read(filepath_h5mu)
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]
        assert mdata.mod["mod1"].X[51, 9] == pytest.approx(51.9)
        assert mdata.mod["mod2"].X[42, 18] == pytest.approx(959)


@pytest.mark.usefixtures("filepath_h5mu")
class TestMuDataMod:
    def test_h5mu_mod_backed(self, mdata, filepath_h5mu):
        mdata.write(
            filepath_h5mu,
        )
        mdata_ = mu.read_h5mu(filepath_h5mu, backed="r")
        assert list(mdata_.mod.keys()) == ["mod1", "mod2"]

        # When backed, the matrix is read-only
        with pytest.raises(OSError):
            mdata_.mod["mod1"].X[10, 5] = 0


if __name__ == "__main__":
    unittest.main()
