Multimodal data containers
==========================

``muon`` operates on multimodal objects derived from :class:`muon.MuData` class:
::
	from muon import MuData


``MuData`` objects comprise a dictionary with ``AnnData`` objects, one per modality, in their ``.mod`` attribute. Just as ``AnnData`` objects themselves, they also contain attributes like ``.obs`` with annotation of observations (samples or cells), ``.obsm`` with their multidimensional annotations such as embeddings, etc.

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

Container's attributes
----------------------

Key attributes & method of ``MuData`` objects as well as important concepts are described below. A full list of attributes and methods of multimodal containers can be found in the :class:`muon.MuData` documentation. 

.mod
^^^^

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
^^^^^^^^^^^

Samples (cells) annotation is accessible via the ``.obs`` attribute and by default includes copies of columns from ``.obs`` data frames of individual modalities. Same goes for ``.var``, which contains annotation of variables (features). 

Observations columns copied from individual modalities contain modality name as their prefix, e.g. ``rna:n_genes``. Same is true for variables columns however if there are columns with identical names in ``.var`` of multiple modalities — e.g. ``n_cells``, — these columns are merged across mdalities and no prefix is added.

When those slots are changed in ``AnnData`` objects of modalities, e.g. new columns are added or samples (cells) are filtered out, the changes have to be fetched with the ``.update()`` method:
::
	mdata.update()

.obsm
^^^^^

Multidimensional annotations of samples (cells) are accessible in the ``.obsm`` attribute. For instance, that can be UMAP coordinates that were learnt jointly on all modalities. Or `MOFA <https://biofam.github.io/MOFA2/>`_ embeddings — a generalisation of PCA to multiple omics.
::
	# mdata is a MuData object with CITE-seq data
	mdata.obsm  
	# => MuAxisArrays with keys: X_umap, X_mofa, prot, rna

As another multidimensional embedding, this slot contains boolean vectors, one per modality, indicating if samples (cells) are available in the respective modality. For instance, if all samples (cells) are the same across modalities, all values in those vectors are ``True``.


Container's shape
-----------------

The ``MuData`` object's shape is represented by two numbers calculated as a sum of the shapes of individual modalities — one for the number of observations and one for the number of variables.
::
        mdata.shape
        # => (9573, 132465)
        mdata.n_obs
        # => 9573
        mdata.n_vars
        # => 132465

By default, variables are always counted as belonging uniquely to a single modality while observations with the same name are counted as the same observation, which has variables across multiple modalities measured for.
::
        [ad.shape for ad in mdata.mod.values()]
        # => [(9500, 10100), (9573, 122364)]

If the shape of a modality is changed, :func:`muon.MuData.update` has to be run to bring the respective updates to the ``MuData`` object.

Keeping containers up to date
-----------------------------

Modalities inside the ``MuData`` container are full-fledged ``AnnData`` objects, which can be operated independently with any tool that works on ``AnnData`` objects. The shape of the ``MuData`` object as well as metadata fetched from individual modalities and boolean vectors of observations (in ``.obsm``) & variables (in ``.varm``) for each modality will then reflect the previous state of the data. To keep the container up to date, there is an ``.update()`` method that syncs the data.

Some functions in ``muon`` call ``.update()`` internally.


Backed containers
-----------------

To enable the backed mode for the count matrices in all the modalities, ``.h5mu`` files can be read with the relevant flag:
::
        mdata_b = mu.read("filename.h5mu", backed=True)
        mdata_b.isbacked
        # => True


When creating a copy of a backed ``MuData`` object, the filename has to be provided, and the copy of the object will be backed at a new location.
::
        mdata_copy = mdata_b.copy("filename_copy.h5mu")
        mdata_b.file.filename
        # => 'filename_copy.h5mu'


Container's views
-----------------

Analogous to the behaviour of ``AnnData`` objects, slicing ``MuData`` objects returns views of the original data.
::
        view = mdata[:100,:1000]
        view.is_view
        # => True

        # In the view, each modality is a view as well
        view["A"].is_view
        # => True

Subsetting ``MuData`` objects is special since it slices them across modalities. I.e. the slicing operation for a set of ``obs_names`` and/or ``var_names`` will be performed for each modality and not only for the global multimodal annotation.

This behaviour makes workflows memory-efficient, which is especially important when working with large datasets. If the object is to be modified however, a copy of it should be created, which is not a view anymore and has no dependance on the original object.
::
        mdata_sub = view.copy()
        mdata_sub.is_view
        # => False

If the original object is backed, the filename has to be provided to the ``.copy()`` call, and the resulting object will be backed at a new location.
::
        mdata_sub = backed_view.copy("mdata_sub.h5mu")
        mdata_sub.is_view
        # => False
        mdata_sub.isbacked
        # => True

