import unittest
from io import StringIO

import numpy as np
import pandas as pd
from anndata import AnnData
import muon


class TestAddPeakAnnotation(unittest.TestCase):
    """Regression tests for add_peak_annotation with empty distance values (#181)."""

    def test_empty_distance_values(self):
        """Intergenic peaks with empty distance should not raise."""
        tsv = StringIO(
            "chrom\tstart\tend\tgene\tdistance\tpeak_type\n"
            "chr1\t100\t200\t\t\tintergenic\n"
            "chr1\t300\t400\tGeneA\t-173268\tdistal\n"
        )
        pa = pd.read_csv(tsv, sep="\t")
        peaks = ["chr1:100-200", "chr1:300-400"]
        adata = AnnData(np.zeros((2, 2)))
        adata.var_names = peaks

        result = muon.atac.tl.add_peak_annotation(adata, pa, return_annotation=True)

        assert result.distance.dtype == pd.Int64Dtype()
        assert result.distance.iloc[0] is pd.NA
        assert result.distance.iloc[1] == -173268
        assert (result.peak == peaks).all()

    def test_semicolon_separated_distances(self):
        """Multi-gene peaks with semicolon-separated distances should work."""
        tsv = StringIO(
            "chrom\tstart\tend\tgene\tdistance\tpeak_type\n"
            "chr1\t100\t200\tGeneA;GeneB\t-100;200\tpromoter;distal\n"
        )
        pa = pd.read_csv(tsv, sep="\t")
        adata = AnnData(np.zeros((1, 1)))
        adata.var_names = ["chr1:100-200"]

        result = muon.atac.tl.add_peak_annotation(adata, pa, return_annotation=True)

        assert result.distance.dtype == np.int64
        assert result.distance.iloc[0] == -100
        assert result.distance.iloc[1] == 200
        assert (result.peak.iloc[0] == result.peak.iloc[1] == adata.var_names).all()


if __name__ == "__main__":
    unittest.main()
