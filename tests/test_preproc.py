import os
import unittest

import numpy as np
from anndata import AnnData
import muon as mu


class TestTFIDF(unittest.TestCase):
    def setUp(self):
        np.random.seed(2020)
        x = np.random.normal(size=(4,5))
        self.adata = AnnData(x)

    def test_tfidf(self):
        mu.pp.tfidf(self.adata, log_tf=True, log_idf=True)
        self.assertEqual(str("%.3f" % self.adata.X[0,0]), '-11.726')
        self.assertEqual(str("%.3f" % self.adata.X[3,0]), '-12.452')

if __name__ == "__main__":
    unittest.main()
