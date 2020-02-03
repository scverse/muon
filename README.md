# muon

`muon` is a multimodal omics Python framework.

Please note the project is experimental and no stable API is being provided as this point.

## Data structure

In the same vein as [scanpy](https://github.com/theislab/scanpy) and [AnnData](https://github.com/theislab/anndata) are designed to work with scRNA-seq data in Python, `muon` is designed to provide functionality to load, process, and store multimodal omics data.


```
muon
  .obs -- annotation of observations (cells, samples)
  .assays
    "RNA"
      .X   -- data matrix (cells x features)
      .var -- annotation of features (genes, genomics sites or windows)
      .map -- mapping in the format of the long table (assay->cell->index)
              to link rows in X (assay and index) to global metadata in .obs (cell)
      [.obs call would use global .obs and .map of the assay to return assay-specific metadata]
```
