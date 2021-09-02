Release notes
=============

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *


v0.1.1
------

``MuData`` is now provided `as a separate package <https://mudata.readthedocs.io/>`_. Since this release, ``mudata`` is a hard dependency of ``muon``. Making the codebase more modular, this will also help developing tools around ``MuData`` that do not depend on ``scanpy`` or ``muon``.

* Fixes:

  * ``[mudata]`` Fix reading ``.h5mu`` files in backed mode when modalities have ``.raw`` attributes.

* New features:
  
  * ATAC module
    
    * Handle fragments files with barcodes different than ``obs_names``.

v0.1.0
------

Initial ``muon`` release with ``MuData`` (:class:`muon.MuData`), ``atac`` and ``prot`` submodules (:mod:`muon.atac` and :mod:`muon.prot`), and multi-omics integration with ``MOFA`` (:func:`muon.tl.mofa`) and ``WNN`` (:func:`muon.pp.neighbors`).
