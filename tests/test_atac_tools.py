import unittest
from io import StringIO

import numpy as np
import pandas as pd
from anndata import AnnData
from muon._atac.tools import add_peak_annotation


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

        result = add_peak_annotation(adata, pa, return_annotation=True)

        self.assertEqual(result.distance.dtype, np.int64)
        # Intergenic peak distance should be 0
        self.assertIn(0, result.distance.values)
        # Distal peak distance should be preserved
        self.assertIn(-173268, result.distance.values)

    def test_semicolon_separated_distances(self):
        """Multi-gene peaks with semicolon-separated distances should work."""
        tsv = StringIO(
            "chrom\tstart\tend\tgene\tdistance\tpeak_type\n"
            "chr1\t100\t200\tGeneA;GeneB\t-100;200\tpromoter;distal\n"
        )
        pa = pd.read_csv(tsv, sep="\t")
        adata = AnnData(np.zeros((1, 1)))
        adata.var_names = ["chr1:100-200"]

        result = add_peak_annotation(adata, pa, return_annotation=True)

        self.assertEqual(result.distance.dtype, np.int64)
        self.assertIn(-100, result.distance.values)
        self.assertIn(200, result.distance.values)


if __name__ == "__main__":
    unittest.main()
