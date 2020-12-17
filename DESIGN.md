# Designing `muon`

This document outlines design considerations and some technical details about `muon`'s implementation.

## Multimodal data containers

`muon` provides an implementation of the data type for storing multimodal measurements. It is embodied in the `MuData` class and represents a particular way of thinking about multimodal objects as _containers_. That means the unimodal measurements are stored in fully functional objects that can be operated independently — those are `AnnData` instances in the case of `MuData`. These containers can also store information that only makes sense when all of its insides are considered together, e.g. embeddings or cell annotation generated on all modalities en masse.

Such design is transparent, builds on existing software, which has been widely adopted by some communities (e.g. `scanpy` for single-cell data analysis), as well as data formats (HDF5-based files), and also has potential to grow gaining container-level features without disrupting current code. The latter being particularly important when contrasting it to an alternative approach of extending existing data formats such as `AnnData`. See e.g. [this anndata-related discussion](https://github.com/theislab/anndata/issues/237) where multiple challenges has been raised such as [how to have](https://github.com/theislab/anndata/issues/237#issuecomment-562505701) both modality-specific and cross-modality APIs (e.g. to create and to store respective embeddings).

One of the great side effects of this _container_ approach is that AnnData objects can be directly read from and written to HDF5 files with multimodal data:

```py
import muon as mu

# Read from inside the .h5mu file
adata = mu.read("pbmc_10k.h5mu/rna")

# Write insides the .h5mu file
mu.write("pbmc_10k.h5mu/rna", adata)
```

One can verify in their terminal it's stored in the HDF5 file as expected:

```sh
h5ls pbmc_10k.h5mu/mod/rna
# X		Group
# obs		Group
# var		Group
# ...
```


## Multimodal methods

Multimodal methods do not change AnnData objects inside the container unless stated otherwise. They typically operate under the assumption that all the necessary preparations have been performed, e.g. the count matrix has been normalisation or the neighbours graph has been constructed for each modality. When possible, multimodal methods should throw a warning or an error describing the respective steps to be run first.


## Omic-specific modules

In order to enable users to perform analysis beyond prominent (sc)RNA-seq methods, which are available e.g. in `scanpy`, `muon` currently encompasses modality-specific modules. A module with functionality related to ATAC-seq is just an example for such a module:

```py
# Import ATAC-seq module
from muon import atac as ac

# Use a variable `atac` to refer to the ATAC-seq 
# modality inside the MuData object `mdata`
atac = mdata.mod["atac"]

# Run TF-IDF transformation on the atac.X count matrix
ac.pp.tfidf(atac)
```

These modules are named after respective sequencing protocols and comprise special functions that might come in handy. It is also handy to import them as two letter abbreviations:

```py
from muon import atac as ac
#                        ^
#                        |
#      ATAC module: e.g. ac.pp.tfidf()

from muon import protein as pt
#                           ^
#                           |
#                 protein (epitope) module
```

While sticking to the same API on their surface, the functions from these modules can be different in their implementation:

- Some provide new methods, e.g. `ac.tl.lsi()` for Latent Semantic Indexing. 

- Others mimic scanpy's API essentially using them under the hood but provide additional functionality, e.g. `ac.pl.umap()` allows to use gene names for plotting aggregated peak counts. 

- There's also a mix of both like `ac.tl.rank_peaks_groups()` that uses `sc.tl.rank_genes_groups()` and also calls `ac.tl.add_genes_peaks_groups()`.

There are some rules that functions in those modules try to follow:

1. Stay close to AnnData/MuData-centered worflow as much as possible. The default exprectation is to have it as the first argument of a function, and the function modifies the object in place by default.

1. Accept both AnnData and MuData with default keys for modalities such as `'rna'`, `'atac'`, and `'prot'`.

1. Small code overhead. If there's analogous functionality in AnnData or in scanpy, build on top of it. This is important both for matching users' expectations and for reducing the cost of supporting the code across its different versions.

1. Existing plotting functions should be reused unless specific plots are required for certain modalities.

