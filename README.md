<img src="./docs/img/muon_header.png" data-canonical-src="./docs/img/muon_header.png" width="700"/>

`muon` is a multimodal omics Python framework. 

[Documentation](https://muon.readthedocs.io/) | [Tutorials](https://muon-tutorials.readthedocs.io/) | [Publication](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02577-8)

[![Documentation Status](https://readthedocs.org/projects/muon/badge/?version=latest)](http://muon.readthedocs.io/?badge=latest)
[![PyPi version](https://img.shields.io/pypi/v/muon)](https://pypi.org/project/muon)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)

## Data structure

`muon` is designed around `MuData` (multimodal data) objects — in the same vein as [scanpy](https://github.com/theislab/scanpy) and [AnnData](https://github.com/theislab/anndata) are designed to work primarily with scRNA-seq data in Python. Individual modalities in `MuData` are naturally represented with `AnnData` objects.

`MuData` class and `.h5mu` files I/O operations are part of [the standalone mudata library](https://github.com/scverse/mudata).

### Input

`MuData` class is implemented in the [mudata](https://github.com/scverse/mudata) library and is exposed in `muon`:

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

Basic `.h5mu` files I/O functionality is implemented in [mudata](https://github.com/scverse/mudata) and is exposed in `muon`. A `MuData` object represents modalities as collections of `AnnData` objects, and these collections can be saved on disk and retrieved using HDF5-based `.h5mu` files, which design is based on `.h5ad` file structure.

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

[Contributions](./CONTRIBUTING.md) in the form of [issues](https://github.com/scverse/muon/issues), [pull requests](https://github.com/scverse/muon/pulls) or [discussions](https://github.com/scverse/muon/discussions) are welcome.

## Citation

If you use `muon` in your work, please cite the `muon` publication as follows:

> **MUON: multimodal omics analysis framework**
> 
> Danila Bredikhin, Ilia Kats, Oliver Stegle
>
> _Genome Biology_ 2022 Feb 01. doi: [10.1186/s13059-021-02577-8](https://doi.org/10.1186/s13059-021-02577-8).

You can cite the scverse publication as follows:

> **The scverse project provides a computational ecosystem for single-cell omics data analysis**
>
> Isaac Virshup, Danila Bredikhin, Lukas Heumos, Giovanni Palla, Gregor Sturm, Adam Gayoso, Ilia Kats, Mikaela Koutrouli, Scverse Community, Bonnie Berger, Dana Pe’er, Aviv Regev, Sarah A. Teichmann, Francesca Finotello, F. Alexander Wolf, Nir Yosef, Oliver Stegle & Fabian J. Theis
>
> _Nat Biotechnol._ 2023 Apr 10. doi: [10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8).


[//]: # "numfocus-fiscal-sponsor-attribution"

`muon` is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

