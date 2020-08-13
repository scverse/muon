> Please note the project is experimental and no stable API is being provided as this point.

<img src="./docs/img/muon_header.png" data-canonical-src="./docs/img/muon_header.png" width="700"/>

`muon` is a multimodal omics Python framework.

## Data structure

In the same vein as [scanpy](https://github.com/theislab/scanpy) and [AnnData](https://github.com/theislab/anndata) are designed to work with scRNA-seq data in Python, `muon` is designed to provide functionality to load, process, and store multimodal omics data.


```
muon
  .obs     -- annotation of observations (cells, samples)
  .var     -- annotation of features (genes, genomic loci, etc.)
  .obsm    -- multidimensional cell annotation, 
              incl. indices list for each modality
              that links .obs to the cells of that modality
  .varm    -- multidimensional feature annotation, 
              incl. indices list for each modality
              that links .var to the features of that modality
  .mod
    AnnData
      .X    -- data matrix (cells x features)
      .var  -- annotation of features (genes, genomics sites or windows)
      .obs  -- cells metadata (assay-specific)
      .uns
  .uns
```

By design, `muon` can incorporate disjoint multimodal experiments, i.e. the ones with different cells having different modalities measured. No redundant empty measurements are stored due to the distinct feature sets per assay as well as distinct cell sets mapped to a global set of observations.

### Input

For reading multimodal omics data, `muon` relies on the functionality available in scanpy. `muon` comes with `MuData` — a multimodal container, in which every modality is an AnnData object:

```py
from muon import MuData

mdata = MuData({'rna': adata_rna, 'atac': adata_atac})
```

If multimodal data from 10X Genomics is to be read, `muon` provides a reader that returns a `MuData` object with AnnData objects inside, each corresponding to its own modality:

```py
mu.read_10x_h5("filtered_feature_bc_matrix.h5")
# MuData object with n_obs × n_vars = 10000 × 80000 
# 2 modalities
#   rna:	10000 x 30000
#     var:	'gene_ids', 'feature_types', 'genome'
#   atac:	10000 x 50000
#     var:	'gene_ids', 'feature_types', 'genome'
```

### Individual assays

Individual assays are stored as AnnData object, which enables the use of all the default `scanpy` functionality per assay:

```py
import scanpy as sc

sc.pl.umap(mdata.mod["rna"])
```
