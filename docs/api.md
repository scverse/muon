# API

Import muon as:

```python
import muon as mu
```

The {class}`~mudata.MuData` container and `.h5mu` reading/writing (`mu.read`, `mu.write`, `mu.read_h5mu`, …)
are provided by [mudata](https://mudata.readthedocs.io/) and re-exported here; see its documentation for details.

## Input/Output

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated

    read_10x_h5
    read_10x_mtx
```

## Preprocessing

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated
    :recursive:

    pp
```

## Tools

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated
    :recursive:

    tl
```

## Plotting

```{eval-rst}
.. currentmodule:: muon

.. autosummary::
    :toctree: generated
    :recursive:

    pl
```

## ATAC

```{eval-rst}
.. autosummary::
    :toctree: generated
    :recursive:

    atac.pp
    atac.tl
    atac.pl
```

## Protein (CITE-seq)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :recursive:

    prot.pp
```
