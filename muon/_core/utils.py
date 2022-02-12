from typing import Union, Optional, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from anndata import AnnData
import scanpy as sc

from mudata import MuData

# Utility functions


def _get_values(
    data: Union[AnnData, MuData],
    key: Optional[str] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
) -> Optional[Iterable]:
    """
    A helper function to get values
    for variables or annotations of observations (.obs columns).

    Strings like "rna:total_count", unless present in .var_names or .obs.columns,
    will be searched for in the modality "rna" if it's present.

    Strings like "X_umap:1" will be interpreted as .obsm["X_umap"][:,0]
    (indexing is 1-based due to how these are typically used).
    Respectively, value definition as "rna:X_umap:1" is supported as well.

    An AnnData object is returned with the requested variables
    available in .obs so that scanpy.pl interface can be reused.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        MuData or AnnData object
    key : Optional[str]
        String to search for
    use_raw : Optional[bool], optional (default: None)
        Use `.raw` attribute of the modality where a feature (from `color`) is derived from.
        If `None`, defaults to `True` if `.raw` is present and a valid `layer` is not provided.
    layer : Optional[str], optional (default: None)
        Name of the layer in the modality where a feature (from `color`) is derived from.
        No layer is used by default. If a valid `layer` is provided, this takes precedence
        over `use_raw=True`.
    """
    if key is None:
        return None

    # Handle multiple keys
    if isinstance(key, Iterable) and not isinstance(key, str):
        all_values = [_get_values(data, k, use_raw=use_raw, layer=layer) for k in key]
        df = pd.DataFrame(all_values).T
        df.columns = [k for k in key if k is not None]
        return df

    if not isinstance(key, str):
        raise TypeError("Expected key to be a string.")

    # .obs
    if key in data.obs.columns:
        return data.obs[key].values

    # Handle composite keys, e.g. rna:n_counts
    key_mod, mod_key = None, None
    if isinstance(data, MuData):
        if ":" in key:
            maybe_mod, maybe_key = key.split(":", 1)
            if maybe_mod in data.mod and key not in data.var_names and key not in data.obsm:
                key_mod = maybe_mod
                mod_key = maybe_key

    # Handle composite keys, e.g. X_umap:1
    obsm_key, obsm_index = None, None
    if ":" in key and key_mod is None:
        maybe_obsm_key, maybe_index = key.split(":", 1)
        if maybe_obsm_key in data.obsm and key not in data.var_names:
            try:
                maybe_index = int(maybe_index)
            except ValueError:
                pass
            if maybe_index == 0:
                raise ValueError(
                    f"Enumeration for the components in .obsm starts at 1, by convention."
                )
            obsm_key, obsm_index = maybe_obsm_key, maybe_index

    # .obsm
    if obsm_key:
        values = data.obsm[obsm_key][:, maybe_index - 1]

        if issparse(values):
            values = np.array(values.todense()).squeeze()
        return values

    # .var_names
    if isinstance(data, MuData):
        if key_mod and mod_key:
            return _get_values(data.mod[key_mod], key=mod_key, use_raw=use_raw, layer=layer)

        # {'rna': True, 'prot': False}
        key_in_mod = {m: key in data.mod[m].var_names for m in data.mod}

        # Check if the valid layer is requested
        if layer is not None:
            if sum(key_in_mod.values()) == 1:
                use_mod = [m for m, v in key_in_mod.items() if v][0]
                valid_layer = layer in data.mod[use_mod].layers
                if not valid_layer:
                    warnings.warn(
                        f"Layer {layer} is not present when searching for the key {key}, using count matrix instead"
                    )
                    layer = None

        # .raw slots might have exclusive var_names
        if (use_raw is None or use_raw) and layer is not None:
            for m in data.mod:
                if key_in_mod[m] == False and data.mod[m].raw is not None:
                    key_in_mod[m] = key in data.mod[m].raw.var_names
                    if key_in_mod[m] and data.mod[m].raw is None and layer is None:
                        warnings.warn(
                            f"Attibute .raw is None for the modality {m}, using .X instead"
                        )
                        use_raw = False

        if sum(key_in_mod.values()) == 0:
            pass  # not in var names
        elif sum(key_in_mod.values()) > 1:
            raise ValueError(
                f"var_name {key} is present in multiple modalities. Please make the var_names unique, e.g. by calling .var_names_make_unique()."
            )
        else:  # sum(key_in_mod.values()) == 1
            use_mod = [m for m, v in key_in_mod.items() if v][0]
            return _get_values(data.mod[use_mod], key=key, use_raw=use_raw, layer=layer)

    elif isinstance(data, AnnData):
        if (use_raw is None or use_raw) and data.raw is not None:
            keysidx = data.raw.var.index.get_indexer_for([key])
            if keysidx == -1:
                raise ValueError(f"Key {key} could not be found.")
            values = data.raw.X[:, keysidx[0]]
            if len(keysidx) > 1:
                warnings.warn(f"Key {key} is not unique in the index, using the first value...")

        elif layer is not None and layer in data.layers:
            if layer in data.layers:
                keysidx = data.var.index.get_indexer_for([key])
                if keysidx == -1:
                    raise ValueError(f"Key {key} could not be found.")
                values = data.layers[layer][:, keysidx[0]]
                if use_raw:
                    warnings.warn(f"Layer='{layer}' superseded use_raw={use_raw}")
                if len(keysidx) > 1:
                    warnings.warn(f"Key {key} is not unique in the index, using the first value...")

        else:
            if (use_raw is None or use_raw) and data.raw is None:
                warnings.warn(
                    f"Attibute .raw is None when searching for the key {key}, using .X instead"
                )
            if layer is not None and layer not in data.layers:
                warnings.warn(
                    f"Layer {layer} is not present when searching for the key {key}, using count matrix instead"
                )
            keysidx = data.var.index.get_indexer_for([key])
            if keysidx == -1:
                raise ValueError(f"Key {key} could not be found.")
            values = data.X[:, keysidx[0]]
            if len(keysidx) > 1:
                warnings.warn(f"Key {key} is not unique in the index, using the first value...")

        if issparse(values):
            values = np.array(values.todense()).squeeze()
        return values
    else:
        raise TypeError("Expected data to be MuData or AnnData")

    raise ValueError(f"Key {key} could not be found.")
