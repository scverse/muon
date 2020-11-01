CITE-seq
========

``muon`` features a module to work with protein measurements:
::
	from muon import prot as pt

CITE-seq is a method for cellular indexing of transcriptomes and epitopes by sequencing. It's single-cell data comprising transcriptome-wide measurements for each cell (gene expression) as well as surface protein level information, typically for a few dozens of proteins. The method is described in `Stoeckius et al., 2017 <https://www.nature.com/articles/nmeth.4380>`_ and also `on the cite-seq.com website <https://cite-seq.com/>`_.


.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

Normalisation
-------------

Various methods can be used to normalise protein counts in CITE-seq data. ``muon`` brings one of the methods developed specifically for CITE-seq — *denoised and scaled by background* — to Python CITE-seq workflows. This method uses background droplets defined by low RNA content in order to estimate background protein signal and remove it from the data. The method is described in `Korliarov, Sparks et al., 2020 <https://www.nature.com/articles/s41591-020-0769-8>`_ and its original implementation `is available on GitHub <https://github.com/niaid/dsb>`_.
::
	pt.pp.dsb(adata_prot, adata_prot_raw, empty_counts_range=...)
	# will use cell calling from the filtered matrix
	
	# or

	adata_prot = pt.pp.dsb(adata_prot_raw, cell_counts_range=..., empty_counts_range=...)
	# will use provided cell_counts_range for cell calling