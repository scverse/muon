ATAC-seq
========

``muon`` features a module to work with chromatin accessibility data:
::
	from muon import atac as ac

ATAC stands for an *assay for transposase-accessible chromatin*. Count matrices for this data type typically include transposase cuts counts aggregated in peaks as well as bins (windows along the genome) or features such as transcription start sites of genes.

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

Normalisation
-------------

There can be multiple options for ATAC-seq data normalisation.

One of them is constructing term-document matrix from the original count matrix. This is typically followed by singular value decomposition (SVD) — the same technique that convential principal component analysis (PCA) uses — to generate latent components, and these two steps together are referred to as latent semantic indexing (LSI). 

Note that there are different flavours of computing the term frequency — inverse document frequency (TF-IDF) matrix, optionally log-transforming individual terms (TF, IDF, or both).

TF-IDF transformation is implemented in the muon's ATAC module:
::
	ac.pp.tfidf(atac, scale_factor=1e4)


Another option is to use log-normalisation correcting for the total number of counts per cell and log-transforming the values. This is a typical normalisation for many scRNA-seq workflows, and it's typically followed by PCA to generate latent components. Analysing multimodal scRNA-seq & scATAC-seq datasets we notice this scATAC-seq normalisation yields PC & UMAP spaces similar to the ones generated on scRNA-seq counts.
::
	import scanpy as sc
	sc.pp.normalise_per_cell(atac, counts_per_cell_after=1e4)
	sc.pp.log1p(atac)


Since scATAC-seq count matrix is very sparse and most non-zero values in it are `1` and `2`, some workflows also binarise the matrix prior to its downstream analysis:
::
	ac.pp.binarize(atac)

