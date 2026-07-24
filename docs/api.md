# API

Import muon as:

```python
import muon as mu
```

The {class}`~mudata.MuData` container and `.h5mu` reading/writing (`mu.read`, `mu.write`, `mu.read_h5mu`, …)
are provided by [mudata](https://mudata.readthedocs.io/) and re-exported here; see its documentation for details.

## Input/Output

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    read_10x_h5
    read_10x_mtx
```

## Preprocessing

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    pp.filter_obs
    pp.filter_var
    pp.intersect_obs
    pp.sample_obs
    pp.l2norm
    pp.neighbors
```

## Tools

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    tl.mofa
    tl.snf
    tl.ica
    tl.umap
    tl.leiden
```

## Plotting

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    pl.embedding
    pl.histogram
    pl.scatter
    pl.umap
    pl.mofa
    pl.mofa_loadings
```

## ATAC

```{eval-rst}
.. currentmodule:: muon.atac

.. autosummary::
    :toctree: generated

    pp.tfidf
    pp.binarize
    pp.scopen
    tl.lsi
    tl.add_peak_annotation
    tl.add_peak_annotation_gene_names
    tl.add_genes_peaks_groups
    tl.rank_peaks_groups
    tl.count_fragments_features
    tl.locate_fragments
    tl.locate_genome
    tl.get_gene_annotation_from_rna
    tl.fetch_regions_to_df
    tl.get_sequences
    tl.scan_sequences
    tl.nucleosome_signal
    tl.tss_enrichment
    pl.tss_enrichment
    pl.fragment_histogram
    pl.dotplot
    pl.pca
    pl.lsi
```

## Protein (CITE-seq)

```{eval-rst}
.. currentmodule:: muon.prot

.. autosummary::
    :toctree: generated

    pp.clr
    pp.dsb
```
