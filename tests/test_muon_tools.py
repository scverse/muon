import unittest

import numpy as np
from anndata import AnnData
import muon as mu
from muon import atac as ac
from muon import MuData


class TestMOFASimple(unittest.TestCase):
    def setUp(self):
        # Create a dataset using 5 factors
        np.random.seed(1000)
        z = np.random.normal(size=(100, 5))
        w1 = np.random.normal(size=(1000, 5))
        w2 = np.random.normal(size=(500, 5))
        e1 = np.random.normal(size=(100, 1000))
        e2 = np.random.normal(size=(100, 500))
        y1 = np.dot(z, w1.T) + e1
        y2 = np.dot(z, w2.T) + e2
        self.mdata = MuData({"y1": AnnData(y1), "y2": AnnData(y2)})

    def test_mofa_nfactors(self):
        n_factors = 10
        mu.tl.mofa(
            self.mdata,
            n_factors=n_factors,
            quiet=True,
            verbose=False,
            outfile="/tmp/test_mofa_muon_tools.hdf5",
        )
        y = np.concatenate([self.mdata.mod["y1"].X, self.mdata.mod["y2"].X], axis=1)
        yhat = np.dot(self.mdata.obsm["X_mofa"], self.mdata.varm["LFs"].T)

        r2 = []
        for i in range(n_factors):
            yhat = np.dot(self.mdata.obsm["X_mofa"][:, [i]], self.mdata.varm["LFs"][:, [i]].T)
            r2.append(1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2))

        # Only first 5 factors should have high R2
        self.assertTrue(all([i > 0.1 for i in r2[:5]]))
        self.assertFalse(any([i > 0.1 for i in r2[5:]]))


if __name__ == "__main__":
    unittest.main()
