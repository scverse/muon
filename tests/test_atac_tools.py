from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest
from anndata import AnnData
from scipy.sparse import csr_array

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


def test_count_fragments_features(tmp_path: Path) -> None:
    fragments = (
        "1\t63082\t63282\tcell1\t2\n"
        "1\t65082\t65282\tcell1\t2\n"
        "1\t67082\t67282\tcell1\t1\n"
        "1\t69082\t69282\tcell1\t1\n"
        "1\t71082\t71282\tcell1\t1\n"
        "1\t83678\t83878\tcell1\t10\n"
        "1\t85678\t85878\tcell1\t10\n"
        "1\t87678\t87878\tcell1\t10\n"
        "1\t89678\t89878\tcell1\t100\n"
        "1\t91678\t91878\tcell1\t100\n"
        "2\t131043\t131243\tcell2\t200\n"
        "2\t133043\t133243\tcell2\t200\n"
        "2\t135043\t135243\tcell2\t20\n"
        "2\t137043\t137243\tcell2\t20\n"
        "2\t139043\t139243\tcell2\t20\n"
        "2\t215701\t215901\tcell2\t2\n"
        "2\t217701\t217901\tcell2\t2\n"
        "2\t219701\t219901\tcell2\t2\n"
        "2\t221701\t221901\tcell2\t4\n"
        "2\t223701\t223901\tcell2\t4\n"
    )

    fragments_path = str(tmp_path / "fragments.txt")
    with open(fragments_path, mode="w") as f:
        f.write(fragments)
    fragments_path = pysam.tabix_index(fragments_path, preset="bed")

    tsv = StringIO("chromosome\tstart\tend\tstrand\n1\t71582\t83178\t+\n2\t139543\t215201\t-\n")
    annotation = pd.read_csv(tsv, sep="\t")

    adata = AnnData(obs=pd.DataFrame(index=["cell1", "cell2"]))
    ac.tl.locate_fragments(adata, fragments_path)

    result: csr_array = ac.tl.count_fragments_features(
        adata, annotation, extend_upstream=5000, extend_downstream=0, count_reads=True
    ).X
    assert np.all(result.toarray() == np.asarray([[3, 0], [0, 6]]))

    result = ac.tl.count_fragments_features(
        adata, annotation, extend_upstream=0, extend_downstream=5000, count_reads=True
    ).X
    assert np.all(result.toarray() == np.asarray([[30, 0], [0, 60]]))
