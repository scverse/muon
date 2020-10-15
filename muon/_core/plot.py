from typing import Union, List, Optional, Iterable, Sequence

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
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
    Scatter plot in the define basis, which can also be
    a basis inside any modality, e.g. "rna:X_pca".

    See sc.pl.embedding for details.

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
    if isinstance(data, AnnData) or basis in data.obsm:
        return sc.pl.embedding(data, basis=basis, color=color, use_raw=use_raw, **kwargs)

    # `data` is MuData
    # and basis is not a joint embedding
    try:
        mod, basis_mod = basis.split(":")
    except ValueError as e:
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

    obs = data.obs.loc[adata.obs_names]

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

                obs = obs.join(
                    pd.DataFrame(fmod_adata.X, columns=mod_keys, index=fmod_adata.obs_names),
                    how="left",
                )

    ad = AnnData(obs=obs, obsm=adata.obsm, obsp=adata.obsp)
    return sc.pl.embedding(ad, basis=basis_mod, color=color, **kwargs)


def mofa(mdata: MuData, **kwargs) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot in MOFA factors coordinates

    See sc.pl.embedding for details.
    """
    return embedding(mdata, basis="mofa", **kwargs)
