Multimodal data containers
==========================

``muon`` operates on multimodal objects derived from :class:`muon.MuData` class:
::
	from muon import MuData


``MuData`` objects comprise a dictionary with ``AnnData`` objects, one per modality, in their ``.mod`` attribute. As well as ``AnnData`` objects themselves, they also contain attributes like ``.obs`` with annotation of observations (samples or cells), ``.obsm`` with their multidimensional annotations such as embeddings, etc.

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *


.mod
----

Modalities are stored in a collection accessible via the ``.mod`` attribute of the ``MuData`` object with names of modalities as keys and ``AnnData`` objects as values.
::
	list(mdata.mod.keys())
	# => ['atac', 'rna']


Individual modalities can be accessed with their names via the ``.mod`` attribute or via the ``MuData`` object itself as a shorthand:
::
	mdata.mod['rna']
	# or
	mdata['rna']
	# => AnnData object


.obs & .var
-----------

Samples (cells) annotation is accessible via the ``.obs`` attribute and by default includes copies of columns from ``.obs`` data frames of individual modalities. Same goes for ``.var``, which contains annotation of variables (features). When those columns are changed in ``AnnData`` objects of modalities, the changes have to be fetched with the ``.update()`` method:
::
	mdata.update()



.obsm
-----

Multidimensional annotations of samples (cells) are accessible in the ``.obsm`` attribute. For instance, that can be UMAP coordinates that were learnt jointly on all modalities. Or `MOFA <https://biofam.github.io/MOFA2/>`_ embeddings — a generalisation of PCA to multiple omics.
::
	# mdata is a MuData object with CITE-seq data
	mdata.obsm  
	# => MuAxisArrays with keys: X_umap, X_mofa, prot, rna

As another multidimensional embedding, this slot contains boolean vectors, one per modality, indicating if samples (cells) are available in the respective modality. For instance, if all samples (cells) are the same across modalities, all values in those vectors are ``True``.