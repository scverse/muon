Get to know muon
================

``muon`` is a Python framework for multimodal omics analysis. While there are many features that ``muon`` brings to the table, there are three key areas that its functionality is focused on.

Multimodal data containers
--------------------------

``muon`` introduces multimodal data containers (:class:`muon.MuData` class) allowing Python users to work with increasigly complex datasets efficiently and to build new workflows and computational tools around it.
::
	MuData object with n_obs × n_vars = 10110 × 110101
	 2 modalities
	  atac: 10110 x 100001
	  rna: 10110 x 10100

``MuData`` objects enable multimodal information to be stored & accessed naturally, embrace `AnnData <https://github.com/theislab/anndata>`_ for the individual modalities, and can be serialized to ``.h5mu`` files. :doc:`Learn more about multimodal objects </io/mudata>` as well as :doc:`file formats for storing & sharing them </io/output>`. 

Multi-omics methods
-------------------

``muon`` brings multi-omics methods availability to a whole new level: state-of-the-art methods for multi-omics data integration are just a function call away.
::
	import muon as mu
	mu.tl.mofa(mdata)

:doc:`Learn more about variaous multimodal integration methods </omics/multi>` that can be readily applied to :class:`muon.MuData` objects.

Methods crafted for omics
-------------------------

``muon`` features methods for specific omics such as ATAC-seq and CITE-seq making it an extendable solution and enabling growth in an open-source environment.
::
	from muon import atac as ac
	ac.pp.tfidf(mdata.mod['atac'])
	
	from muon import prot as pt
	pt.pp.dsb(mdata.mod['prot'])

There is :doc:`atac module </omics/atac>` for chromatin accessibility data and :doc:`prot module </omics/citeseq>` for CITE-seq data as well as :doc:`additional functionality </omics/uni>` that make individual omics analysis easier.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   notebooks/quickstart_mudata.ipynb

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Documentation

   install
   io/input
   io/mudata
   omics/uni
   omics/multi
   io/output
   api/index
   changelog

