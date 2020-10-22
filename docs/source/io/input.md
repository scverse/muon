```eval_rst
.. muon documentation master file, created by
   sphinx-quickstart on Thu Oct 22 02:24:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
```

# Input data

A default way to import `muon` and omic-specific modules (when necessary) is the following:

```python
import muon as mu

from muon import atac as ac
from muon import prot as pt
```

There are various ways in which the data can be provided to `muon` to create a MuData object:

```eval_rst
.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10
```

## Count matrices

Read [Cell Ranger](https://support.10xgenomics.com/single-cell-multiome-atac-gex/software/pipelines/latest/what-is-cell-ranger-arc) HDF5 files, `raw_feature_bc_matrix.h5` and `filtered_feature_bc_matrix.h5`, which contain counts for features across all modalities:

```python
mdata = mu.read_10x_h5("filtered_feature_bc_matrix.h5")
# -> MuData object
```

If features counts for a particular omics are of interest, they can be fetched using functions from respective modules:

```python
atac = ac.read_10x_h5("rna-atac/filtered_feature_bc_matrix.h5")
# -> AnnData with peak counts ("Peaks")
```

```python
prot = pt.read_10x_h5("citeseq/filtered_feature_bc_matrix.h5")
# -> AnnData with protein counts ("Antibody capture")
```

`.read_10x_mtx()` methods work in the same way for directories with files `matrix.mtx`, `features.tsv.gz`, and `barcodes.tsv.gz`. Output from other tools can be formatted in the same way to be loaded with these functions.

## AnnData objects

MuData object can be constructed from a dictionary of existing AnnData objects:

```python
mdata = mu.MuData({'rna': adata_rna, 'atac': adata_atac})
```

AnnData objects themselves can be easily constructed from NumPy arrays and/or Pandas DataFrames annotating features (_variables_) and samples/cells (_observations_). This makes it a rather general data format to work with any type of high-dimensional data.

```python
import anndata
adata = anndata.AnnData(X=matrix, obs=metadata_df, var=features_df)
```

Please see more details on how to operate on AnnData objects [in the anndata documentation](https://anndata.readthedocs.io/).
