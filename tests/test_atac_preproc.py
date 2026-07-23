import unittest

import numpy as np
from anndata import AnnData
from scipy.sparse import rand  # type: ignore[import-untyped]

from muon import atac as ac


class TestTFIDF(unittest.TestCase):
    def setUp(self):
        np.random.seed(2020)
        x = np.abs(np.random.normal(size=(4, 5)))
        self.adata = AnnData(x)

    def test_tfidf(self):
        adata = self.adata.copy()
        ac.pp.tfidf(adata, log_tf=True, log_idf=True)
        self.assertEqual(str(f"{adata.X[0, 0]:.3f}"), "4.659")
        self.assertEqual(str(f"{adata.X[3, 0]:.3f}"), "4.770")

    def test_tfidf_view(self):
        view = self.adata[:, :]
        ac.pp.tfidf(view, log_tf=True, log_idf=True)
        self.assertEqual(str(f"{view.X[0, 0]:.3f}"), "4.659")

    def test_tfidf_copy(self):
        adata = self.adata.copy()
        orig_value = adata.X[0, 0]
        copy = ac.pp.tfidf(adata, log_tf=True, log_idf=True, copy=True)
        self.assertEqual(adata.X[0, 0], orig_value)
        self.assertEqual(str(f"{copy.X[0, 0]:.3f}"), "4.659")

    def test_tfidf_inplace(self):
        orig_value = self.adata.X[0, 0]
        res = ac.pp.tfidf(self.adata, log_tf=True, log_idf=True, inplace=False)
        self.assertEqual(self.adata.X[0, 0], orig_value)
        self.assertEqual(str(f"{res[0, 0]:.3f}"), "4.659")

    def test_tfidf_to_layer(self):
        adata = self.adata.copy()
        orig_value = adata.X[0, 0]
        ac.pp.tfidf(adata, log_tf=True, log_idf=True, to_layer="new")
        self.assertEqual(adata.X[0, 0], orig_value)
        self.assertEqual(str("{:.3f}".format(adata.layers["new"][0, 0])), "4.659")

    def test_tfidf_from_layer(self):
        adata = self.adata.copy()
        adata.layers["counts"] = adata.X.copy() + 1
        adata.X = None
        ac.pp.tfidf(adata, from_layer="counts")
        self.assertEqual(str(f"{adata.X[0, 0]:.3f}"), "2.856")


class TestTFIDFSparse(unittest.TestCase):
    def setUp(self):
        np.random.seed(2020)
        x = rand(100, 10, density=0.2, format="csr")
        self.adata = AnnData(x)

    def test_tfidf(self):
        ac.pp.tfidf(self.adata, log_tf=True, log_idf=True)
        self.assertEqual(str(f"{self.adata.X[10, 9]:.3f}"), "18.749")
        self.assertEqual(str(f"{self.adata.X[50, 5]:.3f}"), "0.000")


if __name__ == "__main__":
    unittest.main()
