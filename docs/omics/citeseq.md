# CITE-seq

`muon` features a module to work with protein measurements:

```python
from muon import prot as pt
```

CITE-seq is a method for cellular indexing of transcriptomes and epitopes by sequencing.
It's single-cell data comprising transcriptome-wide measurements for each cell (gene expression) as well as surface protein level information, typically for a few dozens of proteins.
The method is described in [Stoeckius et al., 2017](https://www.nature.com/articles/nmeth.4380) and also [on the cite-seq.com website](https://cite-seq.com/).

## Normalisation

### dsb

Various methods can be used to normalise protein counts in CITE-seq data.
`muon` brings one of the methods developed specifically for CITE-seq — _denoised and scaled by background_ — to Python CITE-seq workflows.
This method uses background droplets defined by low RNA content in order to estimate background protein signal and remove it from the data.
The method is described in [Korliarov, Sparks et al., 2020](https://www.nature.com/articles/s41591-020-0769-8) and its original implementation [is available on GitHub](https://github.com/niaid/dsb).

```python
pt.pp.dsb(adata_prot, adata_prot_raw, empty_counts_range=...)
# will use cell calling from the filtered matrix

# or

adata_prot = pt.pp.dsb(adata_prot_raw, cell_counts_range=..., empty_counts_range=...)
# will use provided cell_counts_range for cell calling
```

### CLR

The centered log ratio (CLR) transformation is one of the strategies to normalise protein counts (see e.g. [Stoeckius et al., 2017](https://www.nature.com/articles/nmeth.4380)):

```python
pt.pp.clr(mdata['prot'])
```
