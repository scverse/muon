import unittest
import io
import numpy as np
from anndata import AnnData
from muon import atac as ac
import muon as mu
import pandas as pd

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


class TestFragments(unittest.TestCase):
    "Tests functions in fragments.py"

    def setUp(self):
        mdata = mu.read("tests/data/atac/test_rna_atac.h5mu")
        atac = mdata.mod["atac"]
        test_regions = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr777", "chr20"],
                "Start": [1000, 1, 23642777],
                "End": [10000, 9, 23643653],
            }
        )
        self.mdata = mdata
        self.atac = atac
        self.test_regions = test_regions

    def test_fetch_regions_to_df(self):
        df = ac.fr.fetch_regions_to_df(self.atac.uns["files"]["fragments"], self.test_regions)

        # Fragments should be sorted by start position
        np.testing.assert_array_equal(df.Start.sort_values().values, df.Start.values)
        assert df.iloc[4, 0] == "chr20"
        assert df.iloc[4, 1] == 23642556
        assert df.iloc[4, 2] == 23642844
        assert df.iloc[4, 3] == "GGGCGAATCCTCGATC-1"

    def test_region_pileup(self):
        adata = ac.fr.region_pileup(
            fragments=self.atac.uns["files"]["fragments"],
            cells=self.atac.obs.index.values,
            chromosome="chr20",
            start=23642777,
            end=23643653,
        )
        assert adata.X.sum() == 1177
        assert adata.X.sum(axis=0)[111] == 2

    def test_tss_pileup(self):

        # genes = pd.read_csv(
        #     io.StringIO(
        #         """Chromosome,Start,End,gene_id,gene_name
        #         chr7,92833916,92836594,ENSG00000105810,CDK6
        #         chr9,107489765,107489769,ENSG00000136826,KLF4
        #         chr10,32935557,32958230,ENSG00000150093,ITGB1"""
        #     )
        # )
        genes = mu.utils.get_gene_annotation_from_rna(self.mdata)

        adata = ac.fr._tss_pileup(
            adata=self.atac, features=genes, extend_upstream=1000, extend_downstream=1000
        )

        assert adata.X.sum() == 1239536
        assert adata.X.sum(axis=0)[111] == 304

    def test_tss_pileup(self):
        adata = ac.fr.count_fragments_features(self.mdata)
        assert adata.X.sum() == 23852.0
        assert adata.X.sum(axis=0)[0, 42] == 384.0


if __name__ == "__main__":
    unittest.main()
