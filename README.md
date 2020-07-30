> Please note the project is experimental and no stable API is being provided as this point.

# <img src="./docs/img/muon_header.png" data-canonical-src="./docs/img/muon_header.png" width="500"/>

`muon` is a multimodal omics Python framework.

## Data structure

In the same vein as [scanpy](https://github.com/theislab/scanpy) and [AnnData](https://github.com/theislab/anndata) are designed to work with scRNA-seq data in Python, `muon` is designed to provide functionality to load, process, and store multimodal omics data.


```
muon
  .obs     -- annotation of observations (cells, samples)
  .var     -- annotation of features (genes, genomic loci, etc.)
  .mod
    AnnData
      .X    -- data matrix (cells x features)
      .var  -- annotation of features (genes, genomics sites or windows)
      .obs  -- cells metadata (assay-specific)
      .obsm -- multidimensional cell annotation, incl. "ammdata_map",
               which links cells from the assay to the global .obs 
      .varm -- multidimensional feature annotation, incl. "ammdata_map",
               which links features from the assay to the global .var 
      .uns
  .uns
```

By design, `muon` can incorporate disjoint multimodal experiments, i.e. the ones with different cells having different modalities measured. No redundant empty measurements are stored due to the distinct feature sets per assay as well as distinct cell sets mapped to a global set of observations.

### Individual assays

Individual assays are stored as AnnData object, which enables the use of all the default `scanpy` functionality per assay:

```py
import scanpy as sc

sc.pl.umap(mdata.mod["rna"])
```
