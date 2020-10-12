import unittest

import numpy as np
from anndata import AnnData
from muon import atac as ac

from scipy.sparse import rand


class TestTFIDF(unittest.TestCase):
    def setUp(self):
        np.random.seed(2020)
        x = np.abs(np.random.normal(size=(4, 5)))
        self.adata = AnnData(x)

    def test_tfidf(self):
        ac.pp.tfidf(self.adata, log_tf=True, log_idf=True)
        self.assertEqual(str("%.3f" % self.adata.X[0, 0]), "4.659")
        self.assertEqual(str("%.3f" % self.adata.X[3, 0]), "4.770")


class TestTFIDFSparse(unittest.TestCase):
    def setUp(self):
        np.random.seed(2020)
        x = rand(100, 10, density=0.2, format="csr")
        self.adata = AnnData(x)

    def test_tfidf(self):
        ac.pp.tfidf(self.adata, log_tf=True, log_idf=True)
        self.assertEqual(str("%.3f" % self.adata.X[10, 9]), "18.748")
        self.assertEqual(str("%.3f" % self.adata.X[50, 5]), "0.000")


if __name__ == "__main__":
    unittest.main()
