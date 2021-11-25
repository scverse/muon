Release notes
=============

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

v0.1.2
------

* In-place filtering functions (:func:`muon.pp.filter_obs` and :func:`muon.pp.filter_var`) can be now run one after another without requiring :func:`muon.MuData.update`).

v0.1.1
------

``MuData`` is now provided `as a separate package <https://mudata.readthedocs.io/>`_. Since this release, ``mudata`` is a hard dependency of ``muon``. Making the codebase more modular, this will also help developing tools around ``MuData`` that do not depend on ``scanpy`` or ``muon``.

We fixed a few things including reading ``.h5mu`` files in backed mode when modalities have ``.raw`` attributes (this is live in the ``mudata`` library), ``SNF`` functionality (:func:`muon.tl.snf`) and colouring plots by ``var_names`` that are present in ``.raw`` but not in the root ``AnnData`` object.

We also introduced some new features for the ATAC module including:
    
    * handling fragments files with barcodes different than ``obs_names`` and

    * supporting ``atac_peak_annotation.tsv`` files produced by Cell Ranger ARC 2.0.0.

v0.1.0
------

Initial ``muon`` release with ``MuData`` (:class:`muon.MuData`), ``atac`` and ``prot`` submodules (:mod:`muon.atac` and :mod:`muon.prot`), and multi-omics integration with ``MOFA`` (:func:`muon.tl.mofa`) and ``WNN`` (:func:`muon.pp.neighbors`).
