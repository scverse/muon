# API

Import muon as:

```python
import muon as mu
```

## Input/Output

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    read
    read_10x_h5
    read_10x_mtx
    read_anndata
    read_h5ad
    read_h5mu
    read_zarr
    write
    write_anndata
    write_h5ad
    write_h5mu
    write_zarr
```

## Multimodal

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    MuData
```

### Preprocessing

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

### Tools

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    tl.mofa
    tl.snf
    tl.ica
    tl.umap
    tl.leiden
    tl.louvain
```

### Plotting

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

### Preprocessing

```{eval-rst}
.. currentmodule:: muon.atac

.. autosummary::
    :toctree: generated

    pp.tfidf
    pp.binarize
    pp.scopen
```

### Tools

```{eval-rst}
.. currentmodule:: muon.atac

.. autosummary::
    :toctree: generated

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
```

### Plotting

```{eval-rst}
.. currentmodule:: muon.atac

.. autosummary::
    :toctree: generated

    pl.tss_enrichment
    pl.fragment_histogram
    pl.dotplot
    pl.pca
    pl.lsi
    pl.umap
    pl.embedding
```

## Protein (CITE-seq)

```{eval-rst}
.. currentmodule:: muon.prot

.. autosummary::
    :toctree: generated

    pp.clr
    pp.dsb
```
