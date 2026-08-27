from io import StringIO

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import muon.atac as ac


@pytest.mark.parametrize("empty_distance", [True, False])
def test_add_peak_annotation(empty_distance: bool) -> None:
    """Intergenic peaks with empty distance should not raise."""
    tsv = StringIO(
        "chrom\tstart\tend\tgene\tdistance\tpeak_type\n"
        "chr1\t100\t200\tGeneA\t400\tintergenic\n"
        "chr1\t300\t400\tGeneB\t-173268\tdistal\n"
        "chr1\t500\t600\tGeneC;GeneD\t-100;200\tdistal;proximal\n"
    )  # fmt: skip
    pa = pd.read_csv(tsv, sep="\t")
    if empty_distance:
        pa.iloc[0, 3:5] = pd.NA

    peaks = ["chr1:100-200", "chr1:300-400", "chr1:500-600", "chr1:500-600"]
    adata = AnnData(np.zeros((3, 3)))
    adata.var_names = peaks[:3]

    result = ac.tl.add_peak_annotation(adata, pa, return_annotation=True)
    assert result is not None

    assert result.distance.dtype == pd.Int64Dtype() if empty_distance else np.int64
    assert result.distance.iloc[0] is pd.NA if empty_distance else 400
    assert result.distance.iloc[1] == -173268
    assert result.distance.iloc[2] == -100
    assert result.distance.iloc[3] == 200
    assert (result.peak == peaks).all()
