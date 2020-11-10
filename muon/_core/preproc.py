from typing import Union, Callable, Optional, Sequence
from functools import reduce
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix

from anndata import AnnData
from .._core.mudata import MuData

# Computational methods for preprocessing

# Utility functions: intersecting observations


def intersect_obs(mdata: MuData):
    """
    Subset observations (samples or cells) in-place
    taking observations present only in all modalities.

    This function is currently a draft, and it can be removed
    or its behaviour might be changed in future.

    Parameters
    ----------
    mdata: MuData
            MuData object
    """

    if mdata.isbacked:
        warn(
            "MuData object is backed. It might be required to re-read the object with `backed=False` to make the intersection work."
        )

    common_obs = reduce(np.intersect1d, [m.obs_names for m in mdata.mod.values()])

    for mod in mdata.mod:
        filter_obs(mdata.mod[mod], common_obs)

    mdata.update_obs()

    return


# Utility functions: filtering observations


def filter_obs(adata: AnnData, var: Union[str, Sequence[str]], func: Optional[Callable] = None):
    """
    Filter observations (samples or cells) in-place
    using any column in .obs or in .X.

    This function is currently a draft, and it can be removed
    or its behaviour might be changed in future.

    Parameters
    ----------
    adata: AnnData
            AnnData object
    var: str or Sequence[str]
            Column name in .obs or in .X to be used for filtering.
            Alternatively, obs_names can be provided directly.
    func
            Function to apply to the variable used for filtering.
            If the variable is of type boolean and func is an identity function,
            the func argument can be omitted.
    """

    if isinstance(var, str):
        if var in adata.obs.columns:
            if func is None:
                if adata.obs[var].dtypes.name == "bool":
                    func = lambda x: x
                else:
                    raise ValueError(f"Function has to be provided since {var} is not boolean")
            obs_subset = func(adata.obs[var].values)
        elif var in adata.var_names:
            obs_subset = func(adata.X[:, np.where(adata.var_names == var)[0]].reshape(-1))
        else:
            raise ValueError(
                f"Column name from .obs or one of the var_names was expected but got {var}."
            )
    else:
        if func is None:
            obs_subset = adata.obs_names.isin(var)
        else:
            raise ValueError(f"When providing obs_names directly, func has to be None.")

    # Subset .obs
    adata._obs = adata.obs[obs_subset]
    adata._n_obs = adata.obs.shape[0]

    # Subset .X
    adata._X = adata.X[obs_subset]

    # Subset layers
    for layer in adata.layers:
        adata.layers[layer] = adata.layers[layer][obs_subset]

    # Subset raw
    if adata.raw is not None:
        adata.raw._X = adata.raw.X[obs_subset]
        adata.raw._n_obs = adata.raw.X.shape[0]

    # Subset .obsm
    for k, v in adata.obsm.items():
        adata.obsm[k] = v[obs_subset]

    # Subset .obsp
    for k, v in adata.obsp.items():
        adata.obsp[k] = v[obs_subset][:, obs_subset]

    return


# Utility functions: filtering variables


def filter_var(adata: AnnData, var: Union[str, Sequence[str]], func: Optional[Callable] = None):
    """
    Filter variables (features, e.g. genes) in-place
    using any column in .var or row in .X.

    This function is currently a draft, and it can be removed
    or its behaviour might be changed in future.

    Parameters
    ----------
    adata: AnnData
            AnnData object
    var: str or Sequence[str]
            Column name in .var or row name in .X to be used for filtering.
            Alternatively, var_names can be provided directly.
    func
            Function to apply to the variable used for filtering.
            If the variable is of type boolean and func is an identity function,
            the func argument can be omitted.
    """

    if isinstance(var, str):
        if var in adata.var.columns:
            if func is None:
                if adata.var[var].dtypes.name == "bool":
                    func = lambda x: x
                else:
                    raise ValueError(f"Function has to be provided since {var} is not boolean")
            var_subset = func(adata.var[var].values)
        elif var in adata.obs_names:
            var_subset = func(adata.X[:, np.where(adata.obs_names == var)[0]].reshape(-1))
        else:
            raise ValueError(
                f"Column name from .var or one of the obs_names was expected but got {var}."
            )
    else:
        if func is None:
            var_subset = adata.var_names.isin(var)
        else:
            raise ValueError(f"When providing var_names directly, func has to be None.")

    # Subset .var
    adata._var = adata.var[var_subset]
    adata._n_vars = adata.var.shape[0]

    # Subset .X
    adata._X = adata.X[:, var_subset]

    # Subset layers
    for layer in adata.layers:
        adata.layers[layer] = adata.layers[layer][:, var_subset]

    # NOTE: .raw is not subsetted

    # Subset .varm
    for k, v in adata.varm.items():
        adata.varm[k] = v[var_subset]

    # Subset .varp
    for k, v in adata.varp.items():
        adata.varp[k] = v[var_subset][:, var_subset]

    return
