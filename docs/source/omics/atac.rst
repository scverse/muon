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

Quality control
---------------

While typical quality control (QC) metrics are still relevant for ATAC-seq data, there are a few specific things about the data that one can check, such as fragment size and signal enrichment around transcription start sites.

Nucleosome signal
+++++++++++++++++

Fragment size distribution typically reflects nucleosome binding pattern showing enrichment around values corresponding to fragments bound to a single nucleosome (between 147 bp and 294 bp) as well as nucleosome-free fragments (shorter than 147 bp). The ratio of mono-nucleosome cut fragments to nucleosome-free fragments can be called *nucleosome signal*, and it can be estimated using a subset of fragments with :func:`muon.atac.tl.nucleosome_signal`.

Fragment size distribution can be estimated from a subset of data and plotted with :func:`muon.atac.pl.fragment_histogram`, and the *nucleosome signal* can visualised with a general function :func:`mu.pl.histogram`:
::
	# Plot fragment size distribution
	ac.pl.fragment_histogram(atac, regions="chr1:1-2000000")

	# Plot nucleosome signal distribution across cells
	ac.tl.nucleosome_signal(atac, n=1e6)
	mu.pl.histogram(atac, "nucleosome_signal")

TSS enrichment
++++++++++++++

Chromatin accessibility can be expected to be enriched around transcription start sites (TSS) compared to accessibility of flanking regions. Thus this measure averaged across multiple genes can serve as one more quality control metric. The positions of transcription start sites can be for instance obtained from the interval field of the gene annotation in the ``'rna'`` modality with :func:`muon.atac.tl.get_gene_annotation_from_rna`.

TSS enrichment function :func:`muon.atac.tl.tss_enrichment` will return an ``AnnData`` object with ``n_obs x bases`` dimensions where bases correspond to positions around TSS and are defined by ``extend_upstream`` and ``extend_downstream`` parameters, each of them being 1000 bp by default. It will also record ``tss_score`` in the ``.obs`` of the original ATAC object.
::
	# by default, features=ac.tl.get_gene_annotation_from_rna(mdata)
	tss = ac.tl.tss_enrichment(mdata, n_tss=1000)
	# => AnnData object with n_obs x 2001

	ac.pl.tss_enrichment(tss)


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


Differentially accessible peaks
-------------------------------

Peaks can be tested for differential accessibility in groups of cells with :func:`muon.atac.tl.rank_peaks_groups` in the same way as genes can be tested for differential expression with ``scanpy.tl.rank_genes_groups``. The former function is actually using the latter one under the hood so its behaviour is reproducible and familiar to the user but also automatically adds gene names with :func:`muon.atac.tl.add_genes_peaks_groups` so that the result is more interpretable:
::
	ac.tl.rank_peaks_groups(atac, 'leiden')
	# => adds 'rank_genes_groups' in .uns

	result = atac.uns['rank_genes_groups']
	groups = result['names'].dtype.names

	# One of the ways to format result as a table
	import pandas as pd
	pd.DataFrame(
	  {
	    group + "_" + key[:1]: result[key][group]
	      for group in groups
	      for key in ["names", "genes", "pvals"]
	  }
	)



Gene activity
-------------

ATAC counts can be aggregated in order to get an estimate of how open the chromatin is in certain genome regions, e.g. around transcription start sites (TSS) of genes. Counts for individual fragments from the fragments file would be usually used for this. While the common way to use this aggregation function, :func:`muon.atac.tl.count_fragments_features`, is to produce a gene activity matrix, it accepts any features annotation as long as the provided DataFrame contains ``'Chromosome'``, ``'Start'``, and ``'End'`` columns.
::
	# by default, features=ac.tl.get_gene_annotation_from_rna(mdata)
	adata = ac.tl.count_fragments_features(mdata)
	# => AnnData with n_obs x n_features

This data can be further interrogated in ways similar to (sc)RNA-seq analysis where genes are the features.
