Unimodal omics
==============

``muon`` enhances efficiency and user experience when analysing individual omics by offering general functions for processing and plotting count data as well as functionality crafted for individual omics such as chromatin accessibility or antibody-derived tags. Omic-specific functions are grouped into respective modules inside ``muon``.

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *
   atac
   citeseq

In-place filtering
------------------

When subsetting ``AnnData`` objects with :func:`scanpy.pp.filter_cells` / :func:`scanpy.pp.filter_genes`, the ``AnnData`` object is being copied. Using the slicing syntax (e.g. ``adata[cell_ids]``) results in a ``View`` object, which has to be copied then for any modidying operations. While this behaviour can be useful in many cases, that nearly doubles the amount of required memory and introduces unnecessary challenges when handling exceptionally large datasets.

``muon`` introduces functions for in-place filtering: :func:`muon.pp.filter_obs` and :func:`muon.pp.filter_var`. These function directly modify the ``AnnData`` object that they are called on. Their syntax allows them to also be more general than the aforementioned filtering functions.
::
	mu.pp.filter_obs(adata, 'total_counts', lambda x: (x >= 10000) & (x <= 50000))
	# This is analogous to 
	#   sc.pp.filter_cells(atac, min_counts=10000)
	#   sc.pp.filter_cells(atac, max_counts=50000)
	# but does in-place filtering avoiding copying the object


Histograms
----------

In the same way as violin plots created with :func:`scanpy.pl.violin` are used to visualise quality control steps, :func:`muon.pl.histogram` allows to plot histograms for continuous measurements across cells:
::
	mu.pl.histogram(adata, ['n_counts', 'n_genes'])

It can also be called on the :class:`muon.MuData` object in the same way.