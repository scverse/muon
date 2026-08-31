import io
import os
import pkgutil
from collections.abc import Iterable
from contextlib import suppress
from datetime import datetime
from glob import glob
from itertools import islice
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scanpy import logging
from scipy.sparse import lil_array
from scipy.sparse.linalg import svds
from tqdm import tqdm

from . import utils

#
# Computational methods for transforming and analysing count data
#


def lsi(data: AnnData | MuData, scale_embeddings: bool = True, n_comps: int = 50, layer: str | None = None) -> None:
    """Run Latent Semantic Indexing.

    This function performs only the SVD dimensionality reduction. The canonical usage of LSI for ATAC-seq applies
    :func:`TF-IDF <muon.atac.pp.tfidf>` beforehand.

    Args:
        data: AnnData object or MuData object with an 'atac' modality.
        scale_embeddings: Scale embeddings to zero mean and unit variance.
        n_comps: Number of components to calculate with SVD.
        layer: Layer to use as input. Will use `.X` if set to `None`.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    # In an unlikely scnenario when there are less 50 features, set n_comps to that value
    n_comps = min(n_comps, adata.n_vars)

    logging.info("Performing SVD")
    cell_embeddings, svalues, peaks_loadings = svds(adata.X if layer is None else adata.layers[layer], k=n_comps)

    # Re-order components in the descending order
    cell_embeddings = cell_embeddings[:, ::-1]
    svalues = svalues[::-1]
    peaks_loadings = peaks_loadings[::-1, :]

    if scale_embeddings:
        cell_embeddings = (cell_embeddings - cell_embeddings.mean(axis=0)) / cell_embeddings.std(axis=0)

    stdev = svalues / np.sqrt(adata.n_obs - 1)

    adata.obsm["X_lsi"] = cell_embeddings
    adata.uns["lsi"] = {"stdev": stdev}
    adata.varm["LSI"] = peaks_loadings.T

    return None


#
# Peak annotation
#
# Peak annotation can include peak type (e.g. promoter, distal, intergenic),
# genes that the peak can be linked to (by proximity),
# as well as distances to these genes.
#


def get_gene_annotation_from_rna(data: AnnData | MuData) -> pd.DataFrame:
    """Get a data frame with start and end positions from the ``interval`` column of the 'rna' modality ``.var``.

    Args:
        data: AnnData or MuData object with an 'rna' modality.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "rna" in data.mod:
        adata = data.mod["rna"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'rna' modality")

    if "interval" in adata.var.columns:
        features = pd.DataFrame([s.replace(":", "-", 1).split("-") for s in adata.var["interval"]])
        features.columns = ["Chromosome", "Start", "End"]
        features["gene_id"] = adata.var["gene_ids"].values
        features["gene_name"] = adata.var.index.values
        features.index = adata.var.index
        # Remove genes with no coordinates indicated
        features = features.loc[~features.Start.isnull()]
        features.Start = features.Start.astype(int)
        features.End = features.End.astype(int)
    else:
        raise ValueError(".var object does not have a column named interval")
    return features


def add_peak_annotation(
    data: AnnData | MuData, annotation: str | Path | pd.DataFrame, sep: str = "\t", return_annotation: bool = False
) -> pd.DataFrame | None:
    """Parse peak annotation file and add it to the .uns["atac"]["peak_annotation"].

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        annotation: A path to the peak annotation file (e.g. peak_annotation.tsv) or DataFrame with it.
            Annotation has to contain columns: peak, gene, distance, peak_type.
        sep: Separator for the peak annotation file. Only used if the file name is provided.
            Tab by default.
        return_annotation: Whether to return adata.uns['atac']['peak_annotation']. False by default.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if isinstance(annotation, pd.DataFrame):
        pa = annotation
    else:
        pa = pd.read_csv(annotation, sep=sep)

    pa = pa.convert_dtypes()

    # If peak name is not in the annotation table, reconstruct it:
    # peak = chrom:start-end
    if "peak" not in pa.columns:
        if "chrom" in pa.columns and "start" in pa.columns and "end" in pa.columns:
            pa["peak"] = pa["chrom"].astype(str) + ":" + pa["start"].astype(str) + "-" + pa["end"].astype(str)
        else:
            raise AttributeError(
                "Peak annotation does not in contain neighter peak column nor chrom, start, and end columns."
            )
    else:
        # chrX_NNNNN_NNNNN -> chrX:NNNNN-NNNNN
        pa["peak"] = pa["peak"].str.replace("_", ":", 1).str.replace("_", "-", 1)

    # Split genes, distances, and peaks into individual records

    if pd.api.types.is_string_dtype(pa.distance):
        pa = pa.set_index("peak")
        pa_g = pa.gene.str.split(";").explode()
        pa_d = pa.distance.str.split(";").explode().astype(pd.Int64Dtype())
        pa_p = pa.peak_type.str.split(";").explode()

        # Make a long dataframe indexed by gene
        pa = pd.concat((pa_g, pa_d, pa_p), axis=1).reset_index()
    else:
        pa = pa[["peak", "gene", "distance", "peak_type"]]

    with suppress(ValueError):  # missing values
        pa["distance"] = pa["distance"].astype(int)

    # TODO: nullable strings work with anndata >= 0.13
    for col in ("peak", "gene", "peak_type"):
        pa[col] = pa[col].fillna("").astype(object)

    pa = pa.set_index("gene")

    if "atac" not in adata.uns:
        adata.uns["atac"] = {}
    adata.uns["atac"]["peak_annotation"] = pa

    if return_annotation:
        return pa

    return None


def add_peak_annotation_gene_names(
    data: AnnData | MuData,
    gene_names: pd.DataFrame | None = None,
    join_on: str | None = None,
    return_annotation: bool = False,
) -> pd.DataFrame | None:
    """Add gene names to peak annotation table in .uns["atac"]["peak_annotation"].

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        gene_names: A DataFrame indexed on the gene name
        join_on: Name of the column in the gene_names DataFrame corresponding to the peak annotation index.
            It is automatically set to "gene_ids" or "gene_name" if none is provided.
        return_annotation: Whether to return adata.uns['atac']['peak_annotation']. False by default.
    """
    if isinstance(data, AnnData):
        adata = data

        if gene_names is None:
            raise ValueError("`gene_names` has to be provided as a pd.DataFrame when passing an AnnData object.")
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]

        if gene_names is None:
            if "rna" in data.mod:
                gene_names = data.mod["rna"].var
            else:
                raise ValueError("There is no .mod['rna'] modality. Provide `gene_names` as a pd.DataFrame.")
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if "atac" not in adata.uns or "peak_annotation" not in adata.uns["atac"]:
        raise KeyError("There is no peak annotation yet. Run muon.atac.tl.add_peak_annotation first.")

    ann = adata.uns["atac"]["peak_annotation"]

    if join_on is None:
        join_on = "gene_ids"

    # Extract a table with gene IDs and gene names only
    gene_id_name = gene_names.loc[:, [join_on]].rename_axis("gene_name").reset_index(drop=False).set_index(join_on)

    # Add gene names to the peak annotation table, then reset the index on gene IDs.
    # If join_on == "gene_name", rename "gene" -> "gene_name" to stay unambiguous.

    # Check whether the annotation index is not gene IDs
    if len(np.intersect1d(ann.index.values, gene_id_name.index.values)) == 0:
        # Check whether the annotation index is gene name already
        if len(np.intersect1d(ann.index.values, gene_names.index.values)) != 0:
            join_on = "gene_name"
            ann.index.names = ["gene_name"]
            adata.uns["atac"]["peak_annotation"] = ann

        if return_annotation:
            return ann
        return None

    ann = ann.join(gene_id_name).rename_axis(join_on).reset_index(drop=False)

    # Use empty strings for intergenic peaks when there is no gene
    ann.loc[ann.gene_name.isnull(), "gene_name"] = ""

    # Finally, set the index to gene name
    ann = ann.set_index("gene_name")
    adata.uns["atac"]["peak_annotation"] = ann

    if return_annotation:
        return ann

    return None


# Gene names for peaks
def add_genes_peaks_groups(data: AnnData | MuData, add_peak_type: bool = False, add_distance: bool = False) -> None:
    """Add gene names to peaks ranked by clustering group.

    To add gene names to ranked peaks, peaks have to be ranked first.
    For that, run `sc.tl.rank_genes_groups`.

    Gene names are picked as indices of the peak annotation table.
    To create annotation table, first run `muon.atac.tl.add_peak_annotation`.
    To add gene names instead of gene IDs, consider
    running `muon.atac.tl.add_peak_annotation_gene_names` then.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        add_peak_type: Whether to add peak type to the ranked peaks per group.
        add_distance: Whether to add distance to the ranked peaks per group.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if "rank_genes_groups" not in adata.uns:
        raise KeyError("There is no .uns['rank_genes_groups'] yet. Run sc.tl.rank_genes_groups first.")

    if "atac" not in adata.uns or "peak_annotation" not in adata.uns["atac"]:
        raise KeyError("There is no peak annotation yet. Run muon.atac.pp.add_peak_annotation first.")

    annotation = adata.uns["atac"]["peak_annotation"]
    if "peak" not in annotation.columns:
        raise KeyError("Peak annotation has to contain 'peak' column.")

    # Add gene names
    index_name = annotation.index.name
    columns = [index_name]
    if add_peak_type:
        if "peak_type" not in annotation.columns:
            raise KeyError("Peak annotation has to contain 'peak_type' column.")
        columns.append("peak_type")
        adata.uns["rank_genes_groups"]["peak_type"] = {}
    if add_distance:
        if "distance" not in annotation.columns:
            raise KeyError("Peak annotation has to contain 'distance' column.")
        columns.append("distance")
        adata.uns["rank_genes_groups"]["distance"] = {}
        annotation.distance = annotation.distance.astype(str)  # in order to join as strings
    peaks_genes = annotation.reset_index(drop=False).loc[:, ["peak", *columns]].set_index("peak")

    adata.uns["rank_genes_groups"]["genes"] = {}
    for i in adata.uns["rank_genes_groups"]["names"].dtype.names:
        ann_ordered = (
            pd.DataFrame(adata.uns["rank_genes_groups"]["names"][i])
            .rename({0: "peak"}, axis=1)
            .join(peaks_genes, on="peak", how="inner", sort=False)
            .groupby("peak", sort=False)
            .agg(", ".join)
        )
        genes = ann_ordered[index_name].values
        adata.uns["rank_genes_groups"]["genes"][i] = genes
        if add_peak_type:
            peak_types = ann_ordered.peak_type.values
            adata.uns["rank_genes_groups"]["peak_type"][i] = peak_types
        if add_distance:
            peak_distances = ann_ordered.distance.values
            adata.uns["rank_genes_groups"]["distance"][i] = peak_distances

    # Convert to rec.array to match 'names', 'scores', and 'pvals'
    adata.uns["rank_genes_groups"]["genes"] = pd.DataFrame(adata.uns["rank_genes_groups"]["genes"]).to_records(
        index=False
    )


def rank_peaks_groups(
    data: AnnData | MuData, groupby: str, add_peak_type: bool = False, add_distance: bool = False, **kwargs
) -> None:
    """Rank peaks in clusters groups.

    Shorthand for running sc.tl.rank_genes_groups
    followed by muon.atac.tl.add_genes_peaks_groups.

    See sc.tl.rank_genes_groups for details.

    Args:
        data: AnnData object with peak counts or MuData object with 'atac' modality.
        groupby: The key of the observations grouping to consider.
        add_peak_type: Whether to add peak type to the ranked peaks per group
        add_distance: Whether to add distance to the ranked peaks per group
        **kwargs: Keyword arguments passed to :func:`scanpy.tl.rank_genes_groups`.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData):
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    sc.tl.rank_genes_groups(adata, groupby, **kwargs)

    add_genes_peaks_groups(adata, add_peak_type=add_peak_type, add_distance=add_distance)


#
# Sequences and motifs
#


def _parse_motif_ids(filename: str | None = None):
    if filename is None:
        # Use a file from the embedded JASPAR database
        filename = io.BytesIO(pkgutil.get_data(__name__, "_ref/jaspar/motif_to_gene.txt"))  # type: ignore[assignment, arg-type]
    motifs = pd.read_csv(filename, sep="\t", header=None)
    motifs.columns = ["motif_id", "tf_gene_name"]
    motifs = motifs.set_index("motif_id")

    return motifs


def _parse_motif_matrices(files: list[str] | None = None, background: int | Iterable = 4, pseudocount: float = 0.0001):
    try:
        import MOODS.parsers
        import MOODS.tools
    except ImportError:
        raise ImportError(
            "MOODS is not available. Install MOODS from PyPI (`pip install MOODS-python`) \
            or from GitHub (`pip install git+https://github.com/jhkorhonen/MOODS`)"
        ) from None

    if files is None:
        # Use pfm files from the embedded JASPAR database
        files = glob(os.path.join(os.path.dirname(__file__), "_ref/jaspar/*.pfm"))

    if not isinstance(background, Iterable):
        bg = MOODS.tools.flat_bg(background)
    else:
        bg = background
    matrices = [MOODS.parsers.pfm_to_log_odds(pfm_file, bg, pseudocount) for pfm_file in files]

    return {"motifs": [os.path.basename(f).rstrip(".pfm") for f in files], "matrices": matrices}


def _prepare_motif_scanner(
    matrices=None, background: int | Iterable = 4, pvalue: float = 0.0001, window_size: int = 10
):
    try:
        import MOODS.scan
        import MOODS.tools
    except ImportError:
        raise ImportError(
            "MOODS is not available. Install MOODS from PyPI (`pip install MOODS-python`) or from GitHub (`pip install git+https://github.com/jhkorhonen/MOODS`)"
        ) from None

    if matrices is None:
        motifs_matrices = _parse_motif_matrices(files=None, background=background)
        matrices = motifs_matrices["matrices"]

    if not isinstance(background, Iterable):
        bg = MOODS.tools.flat_bg(background)
    else:
        bg = background
    thresholds = [MOODS.tools.threshold_from_p(m, bg, pvalue) for m in matrices]

    scanner = MOODS.scan.Scanner(window_size)
    scanner.set_motifs(matrices, bg, thresholds)

    return scanner


def scan_sequences(
    sequences: Iterable[str],
    motif_scanner=None,
    matrices=None,
    motifs: Iterable[str] | None = None,
    motif_meta: pd.DataFrame = None,
    background: int = 4,
    pvalue: float = 0.0001,
    window_size: int = 10,
) -> pd.DataFrame:
    """Scan sequences (e.g. peaks) searching for motifs (JASPAR by default).

    Args:
        sequences: The sequences to search.
        motif_scanner: An instance of a motif scanner class. Must provide a `scan` method whose interface is identical to
            `MOODS.scan.Scanner.scan <https://github.com/jhkorhonen/MOODS/wiki/Getting-Started#scanning>`__ :cite:p:`pmid28011774`.
        matrices: Log-odds motif matrices that can be passed to to `MOODS.scan.Scanner`. Used only if `motif_scanner` is `None`.
            Defaults to to the JASPAR database shipped with muon.
        motifs: Motif IDs corresponding to the motifs in `matrices`.
        motif_meta: Additional motif metadata. Will be part of the returned dataframe.
        background: Background distribution for MOODS. Used only if `motif_scanner` is `None`.
        pvalue: P-value threshold for MOODS. Used only if `motif_scanner` is `None`.
        window_size: The window size for MOODS. Used only if `motif_scanner` is `None`.

    Returns:
        Matched motifs and respective sequence IDs.
    """
    if motifs is None and matrices is not None:
        raise ValueError(
            "Both a list of matrices and a corresponding list of motif IDs should be provided — or none to use the built-in ones, unless a scanner is provided."
        )

    if motif_scanner is None:
        if matrices is None:
            motifs = _parse_motif_matrices(files=None, background=background)["motifs"]
        elif motifs is None:
            raise ValueError("A list of motif IDs should be provided if building a scanner from matrices")

        motif_scanner = _prepare_motif_scanner(
            matrices=matrices, background=background, pvalue=pvalue, window_size=window_size
        )

        if motif_meta is None:
            # For the default scanner, use the default metadata
            motif_meta = _parse_motif_ids()

    else:
        if motifs is None:
            raise ValueError(
                "A list of motif IDs should be provided that corresponds to the matrices that the motif scanner was built on."
            )

    matches = []
    for seq in sequences:
        results = motif_scanner.scan(seq)
        for rs, motif in zip(results, motifs, strict=True):
            for r in rs:
                matches.append((seq, motif, r.pos, r.score))

    matches = pd.DataFrame(matches)
    matches.columns = ["sequence", "motif_id", "position", "score"]

    if motif_meta is not None:
        matches = matches.set_index("motif_id").join(motif_meta, how="left").reset_index()

    return matches


def get_sequences(data: AnnData | MuData, bed: str, fasta_file: str, bed_file: str | None = None) -> list[str]:
    """Fetch nucleotide sequences for the regions in a BED string or file from a FASTA file."""
    try:
        import pybedtools
    except ImportError:
        raise ImportError(
            "Pybedtools is not available. Install pybedtools from PyPI (`pip install pybedtools`) or from GitHub (`pip install git+https://github.com/daler/pybedtools`)"
        ) from None

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if "files" not in adata.uns or "genome" not in adata.uns["files"]:
        if fasta_file is not None:
            locate_genome(adata, fasta_file)
        else:
            raise FileNotFoundError(
                "Genome file has to be provided with `fasta_file` \
                or located using `muon.atac.tl.locate_genome`."
            )
    else:
        # TODO: have a function to check validity of the file
        fasta_file = adata.uns["files"]["genome"]

    if bed_file is not None:
        assert bed is None
        bed = open(bed_file).read()
    else:
        if bed is None:
            # Use all the ATAC features,
            # expected to be named as chrX:NNN-NNN
            bed = "\n".join([i.replace(":", "-", 1).replace("-", "\t", 2) for i in adata.var.index.values])

    scanner = pybedtools.BedTool(bed, from_string=True)
    scanner = scanner.sequence(fi=fasta_file)
    sequences = []
    with open(scanner.seqfn, "rb") as f:
        for line in f:
            if not line.startswith(str.encode(">")):
                sequences.append(line.decode().strip())

    return sequences


def locate_file(data: AnnData | MuData, key: str, file: str) -> None:
    """Add path to the file to .uns["files"][key].

    The file to be added has to exist.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        key: A key to store the file (e.g. 'fragments')
        file: A path to the file (e.g. ./atac_fragments.tsv.gz).
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist")

    if "files" not in adata.uns:
        adata.uns["files"] = {}
    adata.uns["files"][key] = file


def locate_genome(data: AnnData | MuData, fasta_file: str) -> None:
    """Add path to the FASTA file with genome to .uns["files"]["genome"].

    Genome sequences can be downloaded from GENCODE:

    - GRCh38: ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_34/GRCh38.p13.genome.fa.gz
    - GRCm38: ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.p6.genome.fa.gz

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        fasta_file: A path to the file (e.g. ./atac_fragments.tsv.gz).
    """
    if not isinstance(data, AnnData) and not (isinstance(data, MuData) and "atac" in data.mod):
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    locate_file(data, "genome", fasta_file)


#
# Fragments
#
# Fragments file is a BED-like file describing individual fragments.
# A single record in such a file typically includes 5 tab-separated fields:
#
# chr1 10000 11000 GTCAGTCAGTCAGTCA-1 1
# ^    ^     ^     ^                  ^
# |    |     |     |                  |
# |    |     |     4: name (cell barcode)
# |    |     3: end (3' fragment position, exclusive)
# |    2: start (5' fragment position, inclusive)|
# 1: contig (chromosome)              5: score (number of cuts per fragment)
#
# Fragments file is compressed (.gz) and has to be indexed
# with Tabix in order to be used (.gz.tbi).
#


def locate_fragments(data: AnnData | MuData, fragments: str, return_fragments: bool = False) -> Any | None:
    """Parse fragments file and add a variable to access it to the .uns["files"]["fragments"].

    Fragments file is never read to memory, and connection to the file is closed
    upon function completion.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        fragments: A path to the compressed tab-separated fragments file (e.g. atac_fragments.tsv.gz).
        return_fragments: Whether to return the Tabix connection the fragments file. False by default.
    """
    frag = None
    try:
        if isinstance(data, AnnData):
            adata = data
        elif isinstance(data, MuData) and "atac" in data.mod:
            adata = data.mod["atac"]
        else:
            raise TypeError("Expected AnnData or MuData object with 'atac' modality")

        try:
            import pysam
        except ImportError:
            raise ImportError(
                "pysam is not available. It is required to work with the fragments file. \
                Install pysam from PyPI (`pip install pysam`) \
                or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
            ) from None

        # Here we make sure we can create a connection to the fragments file
        frag = pysam.TabixFile(fragments, parser=pysam.asBed())

        if adata.is_view:
            warn(
                "`data` is a view. The fragments file path will only be added to the view, not the original object.",
                stacklevel=1,
            )

        if "files" not in adata.uns:
            adata.uns["files"] = {}
        adata.uns["files"]["fragments"] = fragments

        if return_fragments:
            return frag
        # The connection has to be closed
        frag.close()

        return None

    except Exception:
        if frag is not None:
            frag.close()
        raise


def initialise_default_files(data: AnnData | MuData, path: str | Path) -> None:
    """Locate default files for ATAC-seq.

    - attempt to locate peak annotation file (atac_peak_annotation.tsv)
    - attempt to parse add peak annotation and store it as a DataFrame
    - attempt to locate fragments file (atac_fragments.tsv.gz)
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    # 2) Add peak annotation

    default_annotation = os.path.join(os.path.dirname(path), "atac_peak_annotation.tsv")
    if os.path.exists(default_annotation):
        try:
            add_peak_annotation(adata, default_annotation)
            print(f"Added peak annotation from {default_annotation} to .uns['atac']['peak_annotation']")

            if isinstance(data, MuData):
                try:
                    add_peak_annotation_gene_names(data)
                    print("Added gene names to peak annotation in .uns['atac']['peak_annotation']")
                except Exception:  # noqa: BLE001
                    pass
        except AttributeError:
            warn(
                f"Peak annotation from {default_annotation} could not be added. Please check the annotation file is formatted correctly.",
                stacklevel=2,
            )

    # 3) Locate fragments file

    default_fragments = os.path.join(os.path.dirname(path), "atac_fragments.tsv.gz")
    if os.path.exists(default_fragments):
        print(f"Located fragments file: {default_fragments}")
        try:
            locate_fragments(adata, default_fragments)
        except ImportError:
            warn(
                "Pysam is not installed. To work with the fragments file please install pysam (pip install pysam).",
                stacklevel=2,
            )
            if "files" not in adata.uns:
                adata.uns["files"] = {}
            adata.uns["files"]["fragments"] = default_fragments


def count_fragments_features(
    data: AnnData | MuData,
    features: pd.DataFrame | None = None,
    stranded: bool | None = None,
    extend_upstream: float = 2e3,
    extend_downstream: int = 0,
    count_reads: bool = False,
) -> AnnData:
    """Count fragments overlapping given features. Returns a cells x features matrix.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        features: A DataFrame with feature annotation, e.g. genes. Annotation should contain columns (case-insensitive):
            chr/chrom/chromosome (longer takes precedence), start, end.
        stranded: Use strand information for each feature. Has to be encoded as a "strand" (case-insensitive) column in features.
            When `stranded=True`, extend_upsteam and extend_downstream will be used according to each feature's strand information.
            If `None`, will use strand information if present and run unstranded otherwise.
        extend_upstream: Number of nucleotides to extend every gene upstream (2000 by default to extend gene coordinates to promoter regions)
        extend_downstream: Number of nucleotides to extend every gene downstream (0 by default)
        count_reads: Whether to count reads instead of fragments. If `True`, the number of reads (read support) per fragment will be used.
            This will also include duplicate read pairs. If `False`, `1` will be added for each fragment.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if features is None:
        # Try to gene gene annotation in the data.mod['rna']
        if isinstance(data, MuData) and "rna" in data.mod and "interval" in data.mod["rna"].var.columns:
            features = get_gene_annotation_from_rna(data)
        else:
            raise ValueError(
                "Argument `features` is required. It should be a BED-like DataFrame with gene coordinates and names."
            )

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise KeyError("There is no fragments file located yet. Run muon.atac.tl.locate_fragments first.")

    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        ) from None

    n = adata.n_obs
    n_features = features.shape[0]

    # TODO: refactor and reuse this code
    # TODO: write tests (see #59, #68, #110)

    f_cols = features.columns.str.lower()
    for col in ("start", "end"):
        if col not in f_cols:
            raise ValueError(f"No column with feature {col}s could be found")

    chrom_col: str | None = None
    for col in ("chromosome", "chrom", "chr"):
        if col in f_cols:
            chrom_col = col
            break
    if chrom_col is None:
        raise ValueError("No column with chromosome for features could be found")

    start_col = np.nonzero(f_cols == "start")[0][0]
    end_col = np.nonzero(f_cols == "end")[0][0]
    chr_col = np.nonzero(f_cols == chrom_col)[0][0]

    strand_col: str | None = None
    if stranded is None:
        stranded = "strand" in f_cols
    elif stranded and "strand" not in f_cols:
        raise ValueError("No column with strand for features could be found")

    if stranded:
        strand_col = np.nonzero(f_cols == "strand")[0][0]

    fragments = pysam.TabixFile(adata.uns["files"]["fragments"], parser=pysam.asBed())
    try:
        # List of lists matrix is quick and convenient to fill by row
        mx = lil_array((n_features, n), dtype=int)

        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Counting fragments in {n} cells for {features.shape[0]} features..."
        )

        for i in tqdm(range(n_features)):  # iterate over features (e.g. genes)
            if stranded and features.iloc[i, strand_col] == "-":
                f_from = features.iloc[i, start_col] - extend_downstream
                f_to = features.iloc[i, end_col] + extend_upstream
            else:
                f_from = features.iloc[i, start_col] - extend_upstream
                f_to = features.iloc[i, end_col] + extend_downstream

            for fr in fragments.fetch(features.iloc[i, chr_col], f_from, f_to):
                try:
                    ind = adata.obs.index.get_loc(fr.name)  # cell barcode (e.g. GTCAGTCAGTCAGTCA-1)
                    mx.rows[i].append(ind)
                    if count_reads:
                        # number of read pairs associated with the fragment
                        mx.data[i].append(int(fr.score))
                    else:
                        mx.data[i].append(1)
                except Exception:  # noqa: BLE001
                    pass

        # Faster to convert to csc first and then transpose
        mx = mx.tocsc().transpose()
        mx.sum_duplicates()

        return AnnData(X=mx, obs=adata.obs, var=features)
    finally:
        # The connection has to be closed
        fragments.close()


def tss_enrichment(
    data: AnnData | MuData,
    features: pd.DataFrame | None = None,
    extend_upstream: int = 1000,
    extend_downstream: int = 1000,
    n_tss: int = 2000,
    return_tss: bool = True,
    random_state: int | np.random.RandomState | None = 0,
    barcodes: str | None = None,
) -> AnnData | None:
    """Calculate TSS enrichment according to ENCODE guidelines.

    Adds a column ``tss_score`` to the ``.obs`` DataFrame and optionally returns a TSS score object.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with an 'atac' modality.
        features: A DataFrame with feature annotation, e.g. genes.
            Annotation has to contain columns: Chromosome, Start, End.
        extend_upstream: Number of nucleotides to extend every gene upstream
            (to extend gene coordinates to promoter regions).
        extend_downstream: Number of nucleotides to extend every gene downstream.
        n_tss: How many randomly chosen TSS sites to pile up. The fewer the faster.
        return_tss: Whether to return the TSS pileup matrix. Needed for enrichment plots.
        random_state: Random number generator seed.
        barcodes: Column name in the ``.obs`` of the AnnData with barcodes corresponding to
            the ones in the fragments file.

    Returns:
        The TSS pileup as an AnnData object with a ``tss_score`` column in its ``.obs`` slot
        when ``return_tss`` is set, ``None`` otherwise.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if features is None:
        # Try to gene gene annotation in the data.mod['rna']
        if isinstance(data, MuData) and "rna" in data.mod and "interval" in data.mod["rna"].var.columns:
            features = get_gene_annotation_from_rna(data)
        else:
            raise ValueError(
                "Argument `features` is required. It should be a BED-like DataFrame with gene coordinates and names."
            )

    if features.shape[0] > n_tss:
        # Only use n_tss randomly chosen sites to make function faster
        features = features.sample(n=n_tss, random_state=random_state)

    # Pile up tss regions
    tss_pileup = _tss_pileup(
        adata, features, extend_upstream=extend_upstream, extend_downstream=extend_downstream, barcodes=barcodes
    )

    flank_means, center_means = _calculate_tss_score(data=tss_pileup)

    tss_pileup.X = tss_pileup.X / flank_means[:, None]

    tss_scores = center_means / flank_means

    adata.obs["tss_score"] = tss_scores
    tss_pileup.obs["tss_score"] = tss_scores

    if isinstance(data, AnnData):
        logging.info('Added a "tss_score" column to the .obs slot of the AnnData object')
    else:
        logging.info("Added a \"tss_score\" column to the .obs slot of tof the 'atac' modality")

    if return_tss:
        return tss_pileup

    return None


def _tss_pileup(
    adata: AnnData,
    features: pd.DataFrame,
    extend_upstream: int = 1000,
    extend_downstream: int = 1000,
    barcodes: str | None = None,
) -> AnnData:
    """Pile up reads in TSS regions. Returns a cell x position matrix that can be used for QC.

    Args:
        adata: AnnData object with associated fragments file.
        features: A DataFrame with feature annotation, e.g. genes.
            Annotation has to contain columns: Chromosome, Start, End.
        extend_upstream: Number of nucleotides to extend every gene upstream
            (to extend gene coordinates to promoter regions).
        extend_downstream: Number of nucleotides to extend every gene downstream.
        barcodes: Column name in the .obs of the AnnData
            with barcodes corresponding to the ones in the fragments file.
    """
    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise KeyError("There is no fragments file located yet. Run muon.atac.tl.locate_fragments first.")

    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        ) from None

    n = adata.n_obs
    n_features = extend_downstream + extend_upstream + 1

    # Dictionary with matrix positions
    obs: pd.DataFrame = adata.obs
    if barcodes and barcodes in adata.obs.columns:
        d = dict(zip(obs.loc[:, barcodes], range(n), strict=True))
    else:
        d = dict(zip(obs.index, range(n), strict=True))

    # Not sparse since we expect most positions to be filled
    mx = np.zeros((n, n_features), dtype=int)

    fragments = pysam.TabixFile(adata.uns["files"]["fragments"], parser=pysam.asBed())

    # Subset the features to the chromosomes present in the fragments file
    chromosomes = fragments.contigs
    features = features[features.Chromosome.isin(chromosomes)]

    # logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Counting fragments in {n} cells for {features.shape[0]} features...")

    for i in tqdm(range(features.shape[0]), desc="Fetching Regions..."):  # iterate over features (e.g. genes)
        f = features.iloc[i]
        tss_start = f.Start - extend_upstream  # First position of the TSS region
        for fr in fragments.fetch(f.Chromosome, f.Start - extend_upstream, f.Start + extend_downstream):
            try:
                rowind = d[fr.name]  # cell barcode (e.g. GTCAGTCAGTCAGTCA-1)
                score = int(fr.score)  # number of cuts per fragment (e.g. 2)
                colind_start = max(fr.start - tss_start, 0)
                colind_end = min(fr.end - tss_start, n_features)  # ends are non-inclusive in bed
                mx[rowind, colind_start:colind_end] += score
            except Exception:  # noqa: BLE001
                pass

    fragments.close()

    anno = pd.DataFrame({"TSS_position": range(-extend_upstream, extend_downstream + 1)})
    anno.index = anno.index.astype(str)

    return AnnData(X=mx, obs=adata.obs, var=anno)


def _calculate_tss_score(data: AnnData, flank_size: int = 100, center_size: int = 1001):
    """Calculate TSS enrichment scores (defined by ENCODE) for each cell.

    Args:
        data: AnnData object with TSS positons as generated by `tss_pileup`.
        flank_size: Number of nucleotides in the flank on either side of the region (ENCODE standard: 100bp).
        center_size: Number of nucleotides in the center on either side of the region (ENCODE standard: 1001bp).
    """
    x = data.X
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected a dense TSS pileup in .X, got {type(x).__name__}.")

    region_size = x.shape[1]

    if center_size > region_size:
        raise ValueError(f"`center_size` ({center_size}) must smaller than the piled up region ({region_size}).")

    if center_size % 2 == 0:
        raise ValueError(f"`center_size` must be an uneven number, but is {center_size}.")

    # Calculate flank means
    flanks = np.hstack((x[:, :flank_size], x[:, -flank_size:]))
    flank_means = flanks.mean(axis=1)

    # Replace 0 means with population average (to not have 0 division after)
    flank_means[flank_means == 0] = flank_means.mean()

    # Calculate center means
    center_dist = (region_size - center_size) // 2  # distance from the edge of data region
    centers = x[:, center_dist:-center_dist]
    center_means = centers.mean(axis=1)

    return flank_means, center_means


def nucleosome_signal(
    data: AnnData | MuData,
    n: float | None = None,
    nucleosome_free_upper_bound: int = 147,
    mononuleosomal_upper_bound: int = 294,
    barcodes: str | None = None,
) -> None:
    """Computes the ratio of nucleosomal cut fragments to nucleosome-free fragments per cell.

    Nucleosome-free fragments are shorter than 147 bp while mono-mucleosomal fragments are between
    147 bp and 294 bp long.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        n: Number of fragments to count. If `None`, 1e4 fragments * number of cells.
        nucleosome_free_upper_bound: Number of bases up to which a fragment counts as nucleosome free.
        mononuleosomal_upper_bound: Number of bases up to which a fragment counts as mononuleosomal.
        barcodes: Column name in the .obs of the AnnData
            with barcodes corresponding to the ones in the fragments file.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise KeyError("There is no fragments file located yet. Run muon.atac.tl.locate_fragments first.")

    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        ) from None

    fragments = pysam.TabixFile(adata.uns["files"]["fragments"], parser=pysam.asBed())

    # Dictionary with matrix row indices
    obs: pd.DataFrame = adata.obs
    if barcodes and barcodes in adata.obs.columns:
        d = dict(zip(obs.loc[:, barcodes], range(adata.n_obs), strict=True))
    else:
        d = dict(zip(obs.index, range(adata.n_obs), strict=True))
    mat = np.zeros(shape=(adata.n_obs, 2), dtype=int)

    if n is None:
        n = int(adata.n_obs * 1e4)
    else:
        n = int(n)  # Cast n to int

    for f in tqdm(islice(fragments.fetch(), n), total=n, desc="Reading Fragments"):
        try:
            length = f.end - f.start
            row_ind = d[f.name]
            if length < nucleosome_free_upper_bound:
                mat[row_ind, 0] += 1
            elif length < mononuleosomal_upper_bound:
                mat[row_ind, 1] += 1
        except KeyError:
            pass

    # Prevent division by 0
    mat[mat[:, 0] == 0, :] += 1

    # Calculate nucleosome signal
    nucleosome_enrichment = mat[:, 1] / mat[:, 0]
    # nucleosome_enrichment[mat[:,0] == 0] = 0

    adata.obs["nucleosome_signal"] = nucleosome_enrichment

    # Message for the user
    if isinstance(data, AnnData):
        logging.info('Added a "nucleosome_signal" column to the .obs slot of the AnnData object')
    else:
        logging.info("Added a \"nucleosome_signal\" column to the .obs slot of tof the 'atac' modality")

    return None


def fetch_regions_to_df(
    fragment_path: str,
    features: pd.DataFrame | str,
    extend_upstream: int = 0,
    extend_downstream: int = 0,
    relative_coordinates=False,
) -> pd.DataFrame:
    """Parse peak annotation file and return it as DataFrame.

    Args:
        fragment_path: Location of the fragments file (must be tabix indexed).
        features: A DataFrame with feature annotation, e.g. genes or a string of format `chr1:1-2000000` or`chr1-1-2000000`.
            Annotation has to contain columns: Chromosome, Start, End.
        extend_upstream: Number of nucleotides to extend every gene upstream
            (to extend gene coordinates to promoter regions).
        extend_downstream: Number of nucleotides to extend every gene downstream.
        relative_coordinates: Return the coordinates with their relative position to the middle of the features.
    """
    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        ) from None

    if isinstance(features, str):
        features = utils.parse_region_string(features)

    fragments = pysam.TabixFile(fragment_path, parser=pysam.asBed())
    n_features = features.shape[0]

    dfs = []
    for i in tqdm(range(n_features), desc="Fetching Regions..."):  # iterate over features (e.g. genes)
        f = features.iloc[i]
        fr = fragments.fetch(f.Chromosome, f.Start - extend_upstream, f.End + extend_downstream)
        df = pd.DataFrame(
            [(x.contig, x.start, x.end, x.name, x.score) for x in fr],
            columns=["Chromosome", "Start", "End", "Cell", "Score"],
        )
        if df.shape[0] != 0:
            df["Feature"] = f.Chromosome + "_" + str(f.Start) + "_" + str(f.End)

            if relative_coordinates:
                middle = int(f.Start + (f.End - f.Start) / 2)
                df.Start = df.Start - middle
                df.End = df.End - middle

            dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df
