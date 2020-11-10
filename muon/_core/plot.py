from typing import Union, List, Optional, Iterable, Sequence
import warnings

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from anndata import AnnData


from .mudata import MuData


def embedding(
    data: Union[AnnData, MuData],
    basis: str,
    color: Optional[Union[str, Sequence[str]]] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    """
    Scatter plot for .obs

    Produce a scatter plot in the define basis,
    which can also be a basis inside any modality,
    e.g. ``"rna:X_pca"``.

    See :func:`scanpy.pl.embedding` for details.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        MuData or AnnData object
    basis : str
        Name of the `obsm` basis to use
    color : Optional[Union[str, typing.Sequence[str]]], optional (default: None)
        Keys for variables or annotations of observations (.obs columns).
        Can be from any modality.
    use_raw : Optional[bool], optional (default: None)
        Use `.raw` attribute of the modality where a feature (from `color`) is derived from.
        If `None`, defaults to `True` if `.raw` is present and a valid `layer` is not provided.
    layer : Optional[str], optional (default: None)
        Name of the layer in the modality where a feature (from `color`) is derived from.
        No layer is used by default. If a valid `layer` is provided, this takes precedence
        over `use_raw=True`.
    """
    if isinstance(data, AnnData):
        return sc.pl.embedding(data, basis=basis, color=color, use_raw=use_raw, **kwargs)

    # `data` is MuData
    if basis not in data.obsm and "X_" + basis in data.obsm:
        basis = "X_" + basis

    if basis in data.obsm:
        adata = data
        basis_mod = basis
    else:
        # basis is not a joint embedding
        try:
            mod, basis_mod = basis.split(":")
        except ValueError:
            raise ValueError(f"Basis {basis} is not present in the MuData object (.obsm)")

        if mod not in data.mod:
            raise ValueError(
                f"Modality {mod} is not present in the MuData object with modalities {', '.join(data.mod)}"
            )

        adata = data.mod[mod]
        if basis_mod not in adata.obsm:
            if "X_" + basis_mod in adata.obsm:
                basis_mod = "X_" + basis_mod
            elif len(adata.obsm) > 0:
                raise ValueError(
                    f"Basis {basis_mod} is not present in the modality {mod} with embeddings {', '.join(adata.obsm)}"
                )
            else:
                raise ValueError(
                    f"Basis {basis_mod} is not present in the modality {mod} with no embeddings"
                )

    obs = data.obs.loc[adata.obs.index.values]

    if color is None:
        ad = AnnData(obs=obs, obsm=adata.obsm, obsp=adata.obsp)
        return sc.pl.embedding(ad, basis=basis_mod, **kwargs)

    # Some `color` has been provided
    if isinstance(color, str):
        keys = [color]
    elif isinstance(color, Iterable):
        keys = color
    else:
        raise TypeError("Expected color to be a string or an iterable.")

    # Fetch respective features from the
    if not all([key in obs for key in keys]):
        # {'rna': [True, False], 'prot': [False, True]}
        keys_in_mod = {m: [key in data.mod[m].var_names for key in keys] for m in data.mod}
        for m in data.mod:
            if np.sum(keys_in_mod[m]) > 0:
                mod_keys = np.array(keys)[keys_in_mod[m]]

                if use_raw is None or use_raw:
                    if data.mod[m].raw is not None:
                        fmod_adata = data.mod[m].raw[:, mod_keys]
                    else:
                        if use_raw:
                            warnings.warn(
                                f"Attibute .raw is None for the modality {m}, using .X instead"
                            )
                        fmod_adata = data.mod[m][:, mod_keys]
                else:
                    fmod_adata = data.mod[m][:, mod_keys]

                if layer is not None:
                    if layer in data.mod[m].layers:
                        fmod_adata.X = data.mod[m][:, mod_keys].layers[layer].X
                        if use_raw:
                            warnings.warn(f"Layer='{layer}' superseded use_raw={use_raw}")
                    else:
                        warnings.warn(
                            f"Layer {layer} is not present for the modality {m}, using count matrix instead"
                        )

                x = fmod_adata.X.toarray() if issparse(fmod_adata.X) else fmod_adata.X
                obs = obs.join(
                    pd.DataFrame(x, columns=mod_keys, index=fmod_adata.obs_names),
                    how="left",
                )

    ad = AnnData(obs=obs, obsm=adata.obsm, obsp=adata.obsp, uns=adata.uns)
    return sc.pl.embedding(ad, basis=basis_mod, color=color, **kwargs)


def mofa(mdata: MuData, **kwargs) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot in MOFA factors coordinates

    See :func:`muon.pl.embedding` for details.
    """
    return embedding(mdata, basis="mofa", **kwargs)


def umap(mdata: MuData, **kwargs) -> Union[Axes, List[Axes], None]:
    """
    UMAP Scatter plot

    See :func:`muon.pl.embedding` for details.
    """
    return embedding(mdata, basis="umap", **kwargs)


#
# Histogram
#


def histogram(
    data: Union[AnnData, MuData],
    keys: Union[str, Sequence[str]],
    groupby: Optional[Union[str]] = None,
    **kwags,
):
    """
    Plot Histogram of Fragment lengths within specified region.
    Parameters
    ----------
    data
        AnnData object with peak counts or multimodal MuData object.
    keys
        Keys to plot.
    groupby
        Column name(s) of .obs slot of the AnnData object according to which the plot is split.
    """

    if not isinstance(data, AnnData) and not isinstance(data, MuData):
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if isinstance(keys, str):
        keys = [keys]

    obs_keys = [i for i in keys if i in data.obs.columns]
    var_keys = [i for i in keys if i in data.var.index.values]
    assert len(obs_keys) + len(var_keys) == len(
        keys
    ), "Keys should be columns of .obs or some of .var_names"

    df = data.obs.loc[:, obs_keys]

    # Fetch respective features
    if len(var_keys) > 0:
        if isinstance(data, MuData):
            # Find the respective modality
            keys_in_mod = {m: [key in data.mod[m].var_names for key in keys] for m in data.mod}
        else:  # AnnData
            adata = data
            keys_in_mod = {"adata": [key in adata.var_names for key in keys]}

        for m, m_bool in keys_in_mod.items():
            if isinstance(data, MuData):
                adata = data.mod[m]
            if np.sum(m_bool) > 0:
                # Some keys in this modality
                mod_keys = np.array(keys)[keys_in_mod[m]]

                if adata.raw is not None:
                    x = adata.raw[:, mod_keys].X
                else:
                    x = adata[:, mod_keys].X

                x = x.toarray() if issparse(x) else x
                x_df = pd.DataFrame(x, index=adata.obs_names, columns=mod_keys)
                df = pd.concat([df, x_df], axis=1)

    # Handle sns.distplot deprecation and sns.histplot addition
    hist = sns.histplot if hasattr(sns, "histplot") else sns.distplot

    if groupby is None:
        df = df.melt()
        g = sns.FacetGrid(df, col="variable", sharey=False, sharex=False)
        g.map(hist, "value", **kwags)
        [x.set_xlabel(keys[i]) for i, x in enumerate(g.axes[0])]
        [x.set_title("") for i, x in enumerate(g.axes[0])]

    elif groupby is not None:
        if isinstance(groupby, str):
            groupby = [groupby]

        if len(groupby) > 2:
            raise ValueError("Maximum 2 categories in groupby")
        elif len(groupby) == 2 and len(keys) > 1:
            raise ValueError("Maximum 1 category in groupby with more than 1 key")

        if len(groupby) == 1:
            df = pd.concat((df, data.obs.loc[:, groupby]), axis=1)
            df = df.melt(id_vars=groupby[0], ignore_index=False)
            g = sns.FacetGrid(df, col=groupby[0], row="variable", sharey=False, sharex=False)
            g.map(hist, "value", **kwags)
            [
                x.set_xlabel(keys[row])
                for row in range(len(g.axes))
                for i, x in enumerate(g.axes[row])
            ]
            [
                x.set_title(f"{groupby[0]} {g.col_names[i]}")
                for row in range(len(g.axes))
                for i, x in enumerate(g.axes[row])
            ]

        else:
            # 1 key, 2 groupby arguments
            g = sns.FacetGrid(df, col=groupby[0], row=groupby[1], sharey=False, sharex=False)
            g.map(hist, keys[0], **kwags)
            [x.set_xlabel(keys[0]) for row in range(len(g.axes)) for i, x in enumerate(g.axes[row])]
            [
                x.set_title(f"{groupby[0]} {g.row_names[col]} | {groupby[1]} {g.row_names[row]}")
                for row in range(len(g.axes))
                for col, x in enumerate(g.axes[row])
            ]

    plt.show()

    return None
