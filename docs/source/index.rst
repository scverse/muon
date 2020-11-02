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


Multi-omics methods
-------------------

``muon`` brings multi-omics methods availability to a whole new level: state-of-the-art methods for multi-omics data integration are just a function call away.
::
	import muon as mu
	mu.tl.mofa(mdata)


Methods crafted for omics
-------------------------

``muon`` features methods for specific omics such as ATAC-seq and CITE-seq making it an extendable solution and enabling growth in an open-source environment.
::
	from muon import atac as ac
	ac.pp.tfidf(mdata.mod['atac'])
	
	from muon import prot as pt
	pt.pp.dsb(mdata.mod['prot'])


.. toctree::
   :hidden:
   :maxdepth: 1

   io/input
   io/mudata
   omics/multi
   omics/atac
   omics/citeseq
   io/output
   api/index

