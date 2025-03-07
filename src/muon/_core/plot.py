import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from mudata import MuData
from scipy.sparse import issparse

from .utils import _get_values

#
# Scatter
#


def scatter(
    data: AnnData | MuData,
    x: str | None = None,
    y: str | None = None,
    color: str | Sequence[str] | None = None,
    use_raw: bool | None = None,
    layers: str | Sequence[str] | None = None,
    **kwargs,
):
    """
    Scatter plot along observations or variables axes.
    Variables in each modality can be referenced,
    e.g. ``"rna:X_pca"``.

    See :func:`scanpy.pl.scatter` for details.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        MuData or AnnData object
    x : Optional[str]
        x coordinate
    y : Optional[str]
        y coordinate
    color : Optional[Union[str, Sequence[str]]], optional (default: None)
        Keys or a single key for variables or annotations of observations (.obs columns),
        or a hex colour specification.
    use_raw : Optional[bool], optional (default: None)
        Use `.raw` attribute of the modality where a feature (from `color`) is derived from.
        If `None`, defaults to `True` if `.raw` is present and a valid `layer` is not provided.
    layers : Optional[Union[str, Sequence[str]]], optional (default: None)
        Names of the layers where x, y, and color come from.
        No layer is used by default. A single layer value will be expanded to [layer, layer, layer].
    """
    if isinstance(data, AnnData):
        return sc.pl.scatter(data, x=x, y=y, color=color, use_raw=use_raw, layers=layers, **kwargs)

    if isinstance(layers, str) or layers is None:
        layers = [layers, layers, layers]

    obs = pd.DataFrame(
        {
            x: _get_values(data, x, use_raw=use_raw, layer=layers[0]),
            y: _get_values(data, y, use_raw=use_raw, layer=layers[1]),
        }
    )
    obs.index = data.obs_names
    if color is not None:
        # Workaround for scanpy#311, scanpy#1497
        if isinstance(color, str):
            color_obs = _get_values(data, color, use_raw=use_raw, layer=layers[2])
            color_obs = pd.DataFrame({color: color_obs})
        else:
            color_obs = _get_values(data, color, use_raw=use_raw, layer=layers[2])

        color_obs.index = data.obs_names
        obs = pd.concat([obs, color_obs], axis=1, ignore_index=False)

    ad = AnnData(obs=obs, uns=data.uns)

    # Note that use_raw and layers are not provided to the plotting function
    # as the corresponding values were fetched from individual modalities
    # and are now stored in .obs
    retval = sc.pl.scatter(ad, x=x, y=y, color=color, **kwargs)
    if color is not None:
        try:
            data.uns[f"{color}_colors"] = ad.uns[f"{color}_colors"]
        except KeyError:
            pass
    return retval


#
# Embedding
#


def embedding(
    data: AnnData | MuData,
    basis: str,
    color: str | Sequence[str] | None = None,
    use_raw: bool | None = None,
    layer: str | None = None,
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
        return sc.pl.embedding(
            data, basis=basis, color=color, use_raw=use_raw, layer=layer, **kwargs
        )

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
        keys = color = [color]
    elif isinstance(color, Iterable):
        keys = color
    else:
        raise TypeError("Expected color to be a string or an iterable.")

    # Fetch respective features
    if not all([key in obs for key in keys]):
        # {'rna': [True, False], 'prot': [False, True]}
        keys_in_mod = {m: [key in data.mod[m].var_names for key in keys] for m in data.mod}

        # .raw slots might have exclusive var_names
        if use_raw is None or use_raw:
            for i, k in enumerate(keys):
                for m in data.mod:
                    if not keys_in_mod[m][i] and data.mod[m].raw is not None:
                        keys_in_mod[m][i] = k in data.mod[m].raw.var_names

        # e.g. color="rna:CD8A" - especially relevant for mdata.axis == -1
        mod_key_modifier: dict[str, str] = dict()
        for i, k in enumerate(keys):
            mod_key_modifier[k] = k
            for m in data.mod:
                if not keys_in_mod[m][i]:
                    k_clean = k
                    if k.startswith(f"{m}:"):
                        k_clean = k.split(":", 1)[1]

                    keys_in_mod[m][i] = k_clean in data.mod[m].var_names
                    if keys_in_mod[m][i]:
                        mod_key_modifier[k] = k_clean
                    if use_raw is None or use_raw:
                        if not keys_in_mod[m][i] and data.mod[m].raw is not None:
                            keys_in_mod[m][i] = k_clean in data.mod[m].raw.var_names

        for m in data.mod:
            if np.sum(keys_in_mod[m]) > 0:
                mod_keys = np.array(keys)[keys_in_mod[m]]
                mod_keys = np.array([mod_key_modifier[k] for k in mod_keys])

                if use_raw is None or use_raw:
                    if data.mod[m].raw is not None:
                        keysidx = data.mod[m].raw.var.index.get_indexer_for(mod_keys)
                        fmod_adata = AnnData(
                            X=data.mod[m].raw.X[:, keysidx],
                            var=pd.DataFrame(index=mod_keys),
                            obs=data.mod[m].obs,
                        )
                    else:
                        if use_raw:
                            warnings.warn(
                                f"Attibute .raw is None for the modality {m}, using .X instead"
                            )
                        fmod_adata = data.mod[m][:, mod_keys]
                else:
                    fmod_adata = data.mod[m][:, mod_keys]

                if layer is not None:
                    if isinstance(layer, dict):
                        m_layer = layer.get(m, None)
                        if m_layer is not None:
                            x = data.mod[m][:, mod_keys].layers[m_layer]
                            fmod_adata.X = x.todense() if issparse(x) else x
                            if use_raw:
                                warnings.warn(f"Layer='{layer}' superseded use_raw={use_raw}")
                    elif layer in data.mod[m].layers:
                        x = data.mod[m][:, mod_keys].layers[layer]
                        fmod_adata.X = x.todense() if issparse(x) else x
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

        color = [mod_key_modifier[k] for k in keys]

    ad = AnnData(obs=obs, obsm=adata.obsm, obsp=adata.obsp, uns=adata.uns)
    retval = sc.pl.embedding(ad, basis=basis_mod, color=color, **kwargs)
    for key, col in zip(keys, color):
        try:
            adata.uns[f"{key}_colors"] = ad.uns[f"{col}_colors"]
        except KeyError:
            pass
    return retval


def mofa(mdata: MuData, **kwargs) -> Axes | list[Axes] | None:
    """
    Scatter plot in MOFA factors coordinates

    See :func:`muon.pl.embedding` for details.
    """
    return embedding(mdata, basis="mofa", **kwargs)


def umap(mdata: MuData, **kwargs) -> Axes | list[Axes] | None:
    """
    UMAP Scatter plot

    See :func:`muon.pl.embedding` for details.
    """
    return embedding(mdata, basis="umap", **kwargs)


#
# Histogram
#


def histogram(
    data: AnnData | MuData,
    keys: str | Sequence[str],
    groupby: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
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
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    """
    from scanpy.plotting._utils import savefig_or_show

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

    savefig_or_show("histogram_", show=show, save=save)


def mofa_loadings(
    mdata: MuData,
    factors: str | Sequence[int] | None = None,
    include_lowest: bool = True,
    n_points: int | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """\
    Rank genes according to contributions to MOFA factors.
    Mirrors the interface of scanpy.pl.pca_loadings.

    Parameters
    ----------
    mdata
        MuData objects with .obsm["X_mofa"] and .varm["LFs"].
    factors
        For example, ``'1,2,3'`` means ``[1, 2, 3]``, first, second, third factors.
    include_lowest
        Whether to show the variables with both highest and lowest loadings.
    n_points
        Number of variables to plot for each factor.
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    """
    from scanpy.plotting._anndata import ranking
    from scanpy.plotting._utils import savefig_or_show

    if factors is None:
        factors = [1, 2, 3]
    elif isinstance(factors, str):
        factors = [int(x) for x in factors.split(",")]
    factors = np.array(factors) - 1

    if np.any(factors < 0):
        raise ValueError("Component indices must be greater than zero.")

    if n_points is None:
        n_points = min(30, mdata.n_vars)
    elif mdata.n_vars < n_points:
        raise ValueError(
            f"Tried to plot {n_points} variables, but passed mudata only has {mdata.n_vars}."
        )

    for m in mdata.mod:
        ranking(
            mdata[:, mdata.varmap[m] != 0],
            "varm",
            "LFs",
            n_points=n_points,
            indices=factors,
            include_lowest=include_lowest,
        )

        savefig_or_show("mofa_loadings_", show=show, save=save)
