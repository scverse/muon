import pytest
import unittest

import numpy as np
import pandas as pd
from anndata import AnnData
import muon as mu
from muon import MuData


class TestMOFASimple(unittest.TestCase):
    def setUp(self):
        # Create a dataset using 5 factors
        np.random.seed(1000)
        z = np.random.normal(size=(100, 5))
        w1 = np.random.normal(size=(90, 5))
        w2 = np.random.normal(size=(50, 5))
        e1 = np.random.normal(size=(100, 90))
        e2 = np.random.normal(size=(100, 50))
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


@pytest.mark.usefixtures("filepath_hdf5")
class TestMOFA2D:
    def test_multi_group(self, filepath_hdf5):
        pytest.importorskip("mofapy2")

        views_names = ["view1", "view2"]
        groups_names = ["groupA", "groupB"]

        # Set dimensions
        n_g1, n_g2 = 10, 20
        d_m1, d_m2 = 30, 40
        k = 5
        n = n_g1 + n_g2

        # Generate data
        np.random.seed(42)
        z1 = np.random.normal(size=(n_g1, k))
        z2 = np.random.normal(size=(n_g2, k))
        z = np.concatenate([z1, z2], axis=0)

        w1 = np.random.normal(size=(d_m1, k))
        w2 = np.random.normal(size=(d_m2, k))

        e11 = np.random.normal(size=(n_g1, d_m1))
        e12 = np.random.normal(size=(n_g2, d_m1))
        e21 = np.random.normal(size=(n_g1, d_m2))
        e22 = np.random.normal(size=(n_g2, d_m2))
        e1 = np.concatenate([e11, e12], axis=0)
        e2 = np.concatenate([e21, e22], axis=0)

        y1 = np.dot(z, w1.T) + e1
        y2 = np.dot(z, w2.T) + e2

        # Make sample names
        samples_names = [
            f"sample{i}_group{g}"
            for g, g_size in {"A": n_g1, "B": n_g2}.items()
            for i in range(g_size)
        ]
        np.random.shuffle(samples_names)
        samples_groups = [s.split("_")[1] for s in samples_names]

        ad1 = AnnData(X=y1, obs=pd.DataFrame(index=samples_names))
        ad2 = AnnData(X=y2, obs=pd.DataFrame(index=samples_names))

        mdata = MuData({views_names[0]: ad1, views_names[1]: ad2})
        obs = pd.DataFrame({"sample": samples_names, "group": samples_groups}, index=samples_names)
        mdata.obs = mdata.obs.join(obs)

        mu.tl.mofa(mdata, groups_label="group", outfile=filepath_hdf5)

        mdata.obs["true_group"] = [s.split("_")[1] for s in mdata.obs["sample"]]

        assert all(mdata.obs.group.values == mdata.obs.true_group.values)

        for sample, value in (("sample9_groupA", 1.719391), ("sample17_groupB", -2.057848)):
            si = np.where(mdata.obs.index == sample)[0]
            assert mdata.obsm["X_mofa"][si, 0] == pytest.approx(value)


if __name__ == "__main__":
    unittest.main()
