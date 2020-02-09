# muon

`muon` is a multimodal omics Python framework.

Please note the project is experimental and no stable API is being provided as this point.

## Data structure

In the same vein as [scanpy](https://github.com/theislab/scanpy) and [AnnData](https://github.com/theislab/anndata) are designed to work with scRNA-seq data in Python, `muon` is designed to provide functionality to load, process, and store multimodal omics data.


```
muon
  .obs -- annotation of observations (cells, samples)
  .var -- annotation of features (genes, genomic loci, etc.)
  .as
    "RNA" (e.g.)
      .X       -- data matrix (cells x features)
      .var     -- annotation of features (genes, genomics sites or windows)
      .obs_map -- mapping in the format of the long table (cell->index)
                  to link rows in X (index) to global metadata in .obs (cell)
                  as well as assay-specific metadata for these cells
      [.obs call would use global .obs and .obs_map of the assay to return assay-specific metadata]
      .var_map -- mapping in the format of the long table (feature->index)
                  to link columns in X (index) to global metadata in .var (feature)
                  as well as assay-specific feature information
      [.var call would use global .var and .var_map of the assay to return assay-specific feature information]
```

By design, `muon` can incorporate disjoint multimodal experiments, i.e. the ones with different cells having different modalities measured. No redundant empty measurements are stored due to the distinct feature sets per assay as well as distinct cell sets mapped to a global set of observations.
