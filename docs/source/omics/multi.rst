Multi-omics
===========

``muon`` brings multimodal data objects and multimodal integration methods together.

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

Multi-omics factor analysis
---------------------------

Multi-omic factor analysis (MOFA) is a group factor analysis method that allows to learn an interpretable latent space jointly on multiple modalities. Intuitively, it can be viewed as a generalisation of PCA for multi-omics data. More information about this method can be found `on the MOFA website <https://biofam.github.io/MOFA2/>`_.

While there are quite a few options to configure the method for the task in question, running it with default options is simple with :func:`muon.tl.mofa`:

	>>> mu.tl.mofa(mdata)
	>>> "X_mofa" in mdata.obsm
	True

For example, the number of factors to learn can be adjusted with ``n_factors``, and training can be launched on the GPU when available with ``gpu_mode=True``. 

Feature selection
+++++++++++++++++

By default, only highly variable features are used with ``use_var='highly_variable'``. If there's no column ``.var['highly_variable']``, all features are used. Any other feature selection can be provided with ``use_var`` as long as it's defined for all the assays as a boolean value.

Grouping observations
+++++++++++++++++++++

If the variability inside groups of observations (samples or cells) is of interest, and *not* between them, ``groups_label`` can be provided to account for that during the training. For instance, the variability between batches can be accounted for in the MOFA framework. See more details about the multi-group fuctionality `in the MOFA+ FAQ <https://biofam.github.io/MOFA2/faq.html#faq-on-the-multi-group-functionality>`_.
::
	mu.tl.mofa(mdata, groups_label='batch')


Dealing with missing observations
+++++++++++++++++++++++++++++++++

Some observations might not be present in some modalities. While :func:`muon.pp.intersect_obs` can be used to make sure only common observations are preserved in all the modalities, MOFA+ also provides an interface to deal with missing data. There are two strategies: ``use_obs='intersection'`` to only use common observations for the training and ``use_obs='union'`` to use all the observations filling the missing piecies with missing values.
::
	mu.tl.mofa(mdata, use_obs='union')
	# or 
	mu.tl.mofa(mdata, use_obs='intersection')


Using GPU
+++++++++

Traning a factor model on a GPU would allow to reduce the computational time. MOFA+ uses `CuPy <https://cupy.dev/>`_ in order to take advantage of NVIDIA CUDA acceleration:
::
	mu.tl.mofa(mdata, gpu_mode=True)


Multiplex clustering
--------------------

Familiar clustering algorithms can be run based on neighbours information from different modalities with :func:`muon.tl.leiden` or :func:`muon.tl.louvain`. Resolution can be set for each modality individually. More than that, contribution of each modality can also be weighted.

	>>> mu.tl.leiden(mdata, resolution=[2., .5])
	>>> mu.tl.louvain(mdata, mod_weights=[1., .5])


See more information on the multiplex versions of `leiden <https://leidenalg.readthedocs.io/en/stable/multiplex.html>`_ and `louvain <https://louvain-igraph.readthedocs.io/en/stable/multiplex.html>`_ algorithms on their respective documentation pages.
