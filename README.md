<img src="./docs/img/muon_header.png" data-canonical-src="./docs/img/muon_header.png" width="700"/>

`muon` is a multimodal omics Python framework. 

[Documentation](https://muon.readthedocs.io/) | [Tutorials](https://muon-tutorials.readthedocs.io/) | [Preprint](https://www.biorxiv.org/content/10.1101/2021.06.01.445670v1) | [Discord](https://discord.com/invite/MMsgDhnSwQ)

[![Documentation Status](https://readthedocs.org/projects/muon/badge/?version=latest)](http://muon.readthedocs.io/?badge=latest)
[![PyPi version](https://img.shields.io/pypi/v/muon)](https://pypi.org/project/muon)

## Data structure

`muon` is designed around `MuData` (multimodal data) objects — in the same vein as [scanpy](https://github.com/theislab/scanpy) and [AnnData](https://github.com/theislab/anndata) are designed to work primarily with scRNA-seq data in Python. Individual modalities in `MuData` are naturally represented with `AnnData` objects.

`MuData` class and `.h5mu` files I/O operations are part of [the standalone mudata library](https://github.com/pmbio/mudata).

### Input

`MuData` class is implemented in the [mudata](https://github.com/pmbio/mudata) library and is exposed in `muon`:

```py
from muon import MuData

mdata = MuData({'rna': adata_rna, 'atac': adata_atac})
```

If [multimodal data from 10X Genomics](https://support.10xgenomics.com/single-cell-multiome-atac-gex/software/pipelines/latest/output/overview) is to be read, `muon` provides a reader that returns a `MuData` object with AnnData objects inside, each corresponding to its own modality:

```py
import muon as mu

mu.read_10x_h5("filtered_feature_bc_matrix.h5")
# MuData object with n_obs × n_vars = 10000 × 80000 
# 2 modalities
#   rna:	10000 x 30000
#     var:	'gene_ids', 'feature_types', 'genome', 'interval'
#   atac:	10000 x 50000
#     var:	'gene_ids', 'feature_types', 'genome', 'interval'
#     uns:	'atac', 'files'
```

### I/O with `.h5mu` files

Basic `.h5mu` files I/O functionality is implemented in [mudata](https://github.com/pmbio/mudata) and is exposed in `muon`. A `MuData` object represents modalities as collections of `AnnData` objects, and these collections can be saved on disk and retrieved using HDF5-based `.h5mu` files, which design is based on `.h5ad` file structure.

```py
mdata.write("pbmc_10k.h5mu")
mdata = mu.read("pbmc_10k.h5mu")
```

It allows to effectively use the hierarchical nature of HDF5 files and to read/write AnnData object directly from/to `.h5mu` files:

```py
adata = mu.read("pbmc_10k.h5mu/rna")
mu.write("pbmc_10k.h5mu/rna", adata)
```

## Multimodal omics analysis

`muon` incorporates a set of methods for multimodal omics analysis. These methods address the challenge of taking multimodal data as their input. For instance, while for a unimodal analysis one would use principal components analysis, `muon` comes with a method to run [multi-omics factor analysis](https://github.com/bioFAM/MOFA2):

```py
# Unimodal
import scanpy as sc
sc.tl.pca(adata)

# Multimodal
import muon as mu
mu.tl.mofa(mdata)
``` 

## Individual assays

Individual assays are stored as AnnData object, which enables the use of all the default `scanpy` functionality per assay:

```py
import scanpy as sc

sc.tl.umap(mdata.mod["rna"])
```

Typically, a modality inside a container can be referred to with a variable to make the code more concise:

```py
rna = mdata.mod["rna"]
sc.pl.umap(rna)
```

### Modules in `muon`

`muon` comes with a set of modules that can be used hand in hand with scanpy's API. These modules are named after respective sequencing protocols and comprise special functions that might come in handy. It is also handy to import them as two letter abbreviations:

```py
# ATAC module:
from muon import atac as ac

# Protein (epitope) module:
from muon import prot as pt
```

---

Some implementation details are noted in [DESIGN.md](./DESIGN.md). 

[Contributions](./CONTRIBUTING.md) in the form of [issues](https://github.com/PMBio/muon/issues), [pull requests](https://github.com/PMBio/muon/pulls) or [discussions](https://github.com/PMBio/muon/discussions) are welcome.
