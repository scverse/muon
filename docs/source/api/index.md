```eval_rst
.. muon documentation master file, created by
   sphinx-quickstart on Sun Sep 13 02:51:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
```

API reference
================================

```eval_rst
.. toctree::
   :maxdepth: 10
   :caption: Contents:
```

Multimodal omics
================

```eval_rst
.. module: muon
.. autosummary::
   :toctree: generated
   :recursive:

   muon.MuData
   muon.pp
   muon.tl
   muon.pl
   muon.utils
```

ATAC submodule
==============

```eval_rst
.. module:: muon.atac
.. currentmodule:: muon

.. autosummary::
   :toctree: generated

   atac.pl.pca
   atac.pl.umap
   atac.pl.embedding
```

Protein submodule
=================

```eval_rst
.. module:: muon.prot
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    prot.pp
```

Input/Output
================

```eval_rst

.. automodsumm:: muon
   :functions-only:
   :toctree: generated
```

