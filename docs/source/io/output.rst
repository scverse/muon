Output data
===========

In order to save & share multimodal data, ``.h5mu`` file format has been designed.

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *


.h5mu files
-----------

``.h5mu`` files are the default storage for ``MuData`` objects. These are HDF5 files with a standardised structure, which is similar to the one of ``.h5ad`` files where ``AnnData`` objects are stored. The most noticeable distinction is ``.mod`` group presence where individual modalities are stored — in the same way as they would be stored in the ``.h5ad`` files.
::
	# Python
	mdata.write("mudata.h5mu")

	# Shell
	❯ h5ls mudata.h5mu
	mod                      Group
	obs                      Group
	obsm                     Group
	var                      Group
	varm                     Group

	❯ h5ls data/mudata.h5mu/mod
	atac                     Group
	rna                      Group



AnnData inside .h5mu
--------------------

Individual modalities in the ``.h5mu`` file are stored in exactly the same way as AnnData objects. This, together with the hierarchical nature of HDF5 files, makes it possible to read individual modalities from ``.h5mu`` files as well as to save individual modalities to the ``.h5mu`` file:
::
	adata = mu.read("mudata.h5mu/rna")

	mu.write("mudata.h5mu/rna", adata)

The function :func:`muon.read` automatically decides based on the input if :func:`muon.read_h5mu` or rather :func:`muon.read_h5ad` should be called.