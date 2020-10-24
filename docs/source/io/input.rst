.. muon documentation master file, created by
   sphinx-quickstart on Thu Oct 22 02:24:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Input data
============

A default way to import ``muon`` and omic-specific modules (when necessary) is the following:
::
	import muon as mu
	
	from muon import atac as ac
	from muon import prot as pt


There are various ways in which the data can be provided to ``muon`` to create a MuData object:


.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *


Count matrices
--------------

Read `Cell Ranger <https://support.10xgenomics.com/single-cell-multiome-atac-gex/software/pipelines/latest/what-is-cell-ranger-arc>`_ HDF5 files, ``raw_feature_bc_matrix.h5`` and ``filtered_feature_bc_matrix.h5``, which contain counts for features across all modalities:
::
	mdata = mu.read_10x_h5("filtered_feature_bc_matrix.h5")
	# -> MuData object


If feature counts for a particular omics are of interest, they can be fetched using functions from respective modules:
::
	atac = ac.read_10x_h5("rna-atac/filtered_feature_bc_matrix.h5")
	# -> AnnData with peak counts ("Peaks")

	prot = pt.read_10x_h5("citeseq/filtered_feature_bc_matrix.h5")
	# -> AnnData with protein counts ("Antibody capture")


``.read_10x_mtx()`` methods work in the same way for directories with files ``matrix.mtx``, ``features.tsv.gz``, and ``barcodes.tsv.gz``. Output from other tools can be formatted in the same way to be loaded with these functions.

AnnData objects
---------------

MuData object can be constructed from a dictionary of existing AnnData objects:
::
	mdata = mu.MuData({'rna': adata_rna, 'atac': adata_atac})


AnnData objects themselves can be easily constructed from NumPy arrays and/or Pandas DataFrames annotating features (*variables*) and samples/cells (*observations*). This makes it a rather general data format to work with any type of high-dimensional data.
::
	import anndata
	adata = anndata.AnnData(X=matrix, obs=metadata_df, var=features_df)


Please see more details on how to operate on AnnData objects `in the anndata documentation <https://anndata.readthedocs.io/>`_.


Snap files
----------

`Snap files <https://github.com/r3fang/SnapATAC/wiki/FAQs#whatissnap>`_ describe single-nucleus accessibility profiles. Those are HDF5 files that may contain multiple matrices (cells by bins, peaks, or genes). An individual matrix can be read from the snap file to create an AnnData object.
::
	atac = ac.read_snap("pbmc3k.snap", matrix="bins", bin_size=10000)
	# -> AnnData with cells-by-bins matrix

It can be then combined with AnnData with gene expression counts in one MuData object.
::
	mdata = mu.MuData({'atac': adata_atac, 'rna': adata_rna})