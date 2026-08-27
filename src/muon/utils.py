import warnings
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from scipy.sparse import sparray, spmatrix

# Utility functions


def _get_values(
    data: AnnData | MuData,
    key: str | Iterable[str] | None = None,
    use_raw: bool | Mapping[str, bool] = False,
    layer: str | Mapping[str, str | None] | None = None,
    gene_symbols: str | Mapping[str, str | None] | None = None,
    obsmap: np.ndarray | None = None,
) -> pd.DataFrame | np.ndarray | pd.api.extensions.ExtensionArray | None:
    """A helper function to get values for variables or annotations of observations (.obs columns).

    Strings like "rna:total_count", unless present in .var_names or .obs.columns,
    will be searched for in the modality "rna" if it's present.

    Strings like "X_umap:1" will be interpreted as .obsm["X_umap"][:,0]
    (indexing is 1-based due to how these are typically used).
    Respectively, value definition as "rna:X_umap:1" is supported as well.

    An AnnData object is returned with the requested variables
    available in .obs so that scanpy.pl interface can be reused.

    Args:
        data: MuData or AnnData object.
        key: String to search for.
        use_raw: Use `.raw` attribute of the modality where a feature (from `color`) is derived from.
            If a mapping is given, it must have one entry for each modality.
        layer: Name of the layer in the modality where a feature (from `color`) is derived from.
            No layer is used by default. If a mapping is given, it must have one entry for each modality.
        gene_symbols: Column of `.var` to search for `key` in. If a mapping is given, it must have
            one entry for each modality.
        obsmap: Provide a vector of the desired size were 0 are missing values and non-zero values
            correspond to the 1-based index of the value.
            This is used internally for when AnnData as a modality has less observations
            than MuData has globally (i.e. other modalities have other cells).
    """
    if key is None:
        return None

    def _maybe_apply_obsmap(vec, m):
        if m is not None:
            # Avoid numpy conversion of uint indices to float
            m = m.astype(int)
            values = pd.Series(dtype=pd.core.dtypes.cast.convert_dtypes(vec), index=pd.RangeIndex(len(m)))
            values.iloc[m[m > 0] - 1] = vec
            return values.array
        return vec

    # Handle multiple keys
    if isinstance(key, Iterable) and not isinstance(key, str):
        all_values = [
            _get_values(data, k, use_raw=use_raw, layer=layer, gene_symbols=gene_symbols, obsmap=obsmap) for k in key
        ]
        df = pd.DataFrame(all_values).T
        df.columns = [k for k in key if k is not None]
        return df

    if not isinstance(key, str):
        raise TypeError("Expected key to be a string.")

    # .obs
    if key in data.obs.columns:
        values = data.obs[key].values
        return _maybe_apply_obsmap(values, obsmap)

    # Handle composite keys, e.g. rna:n_counts
    key_mod, mod_key = None, None
    if (
        isinstance(data, MuData)
        and (
            gene_symbols is None
            and key not in data.var_names
            or isinstance(gene_symbols, str)
            and gene_symbols in data.var.columns
            and key not in data.var[gene_symbols]
        )
        and key not in data.obsm
    ):
        if ":" in key:
            maybe_mod, maybe_key = key.split(":", 1)
            if maybe_mod in data.mod:
                key_mod = maybe_mod
                mod_key = maybe_key

    # Handle composite keys, e.g. X_umap:1
    obsm_key = None
    obsm_index = 0
    if ":" in key and key_mod is None and key not in data.var_names:
        maybe_obsm_key, maybe_index = key.split(":", 1)
        if maybe_obsm_key in data.obsm:
            try:
                obsm_index = int(maybe_index)
            except ValueError:
                raise ValueError(
                    f"Expected an integer component index after ':' in '{key}', but got '{maybe_index}'."
                ) from None
            if obsm_index == 0:
                raise ValueError("Enumeration for the components in .obsm starts at 1, by convention.")
            obsm_key = maybe_obsm_key

    # .obsm
    if obsm_key:
        values = data.obsm[obsm_key][:, obsm_index - 1]
        if isinstance(values, spmatrix | sparray):
            values = values.toarray().squeeze()
        return _maybe_apply_obsmap(values, obsmap)

    # .var_names
    if isinstance(data, MuData):
        if key_mod and mod_key:
            if not data.obs_names.equals(data.mod[key_mod].obs_names) and obsmap is None:
                obsmap = data.obsmap[key_mod]
            if isinstance(gene_symbols, Mapping):
                gene_symbols = gene_symbols[key_mod]
            elif gene_symbols is not None and gene_symbols.startswith(f"{key_mod}:"):
                gene_symbols = gene_symbols[len(key_mod) + 1 :]
            if isinstance(layer, Mapping):
                layer = layer[key_mod]
            elif layer is not None and layer.startswith(f"{key_mod}:"):
                layer = layer[len(key_mod) + 1]
            return _get_values(
                data.mod[key_mod], key=mod_key, use_raw=use_raw, layer=layer, gene_symbols=gene_symbols, obsmap=obsmap
            )

        # {'rna': True, 'prot': False}
        key_in_mod = {}
        for m, mod in data.mod.items():
            if layer is not None and use_raw:
                raise ValueError("use_raw cannot be True when a layer is specified.")

            var = mod.var if not use_raw else mod.raw.var
            varidx = var.index if gene_symbols is None else var[gene_symbols]
            key_in_mod[m] = key in varidx

        if sum(key_in_mod.values()) == 0:
            pass  # not in var names
        elif sum(key_in_mod.values()) > 1:
            raise ValueError(
                f"var_name {key} is present in multiple modalities. Please make the var_names unique, e.g. by calling .var_names_make_unique()."
            )
        else:  # sum(key_in_mod.values()) == 1
            use_mod = [m for m, v in key_in_mod.items() if v][0]
            if not data.obs_names.equals(data.mod[use_mod].obs_names) and obsmap is None:
                obsmap = data.obsmap[use_mod]
            if isinstance(use_raw, Mapping):
                use_raw = use_raw[use_mod]
            if isinstance(layer, Mapping):
                layer = layer[use_mod]
            if isinstance(gene_symbols, Mapping):
                gene_symbols = gene_symbols[use_mod]
            return _get_values(
                data.mod[use_mod], key=key, use_raw=use_raw, layer=layer, gene_symbols=gene_symbols, obsmap=obsmap
            )

    elif isinstance(data, AnnData):
        if use_raw and layer is not None:
            raise ValueError("use_raw cannot be True when a layer is specified.")

        if layer is not None:
            keysidx = (
                data.var.index.get_indexer_for([key])
                if gene_symbols is None
                else np.nonzero(data.var[gene_symbols] == key)[0]
            )
            if len(keysidx) == 0 or keysidx == -1:
                raise ValueError(f"Key {key} could not be found.")
            values = data.layers[layer][:, keysidx[0]]  # type: ignore[index]
            if len(keysidx) > 1:
                warnings.warn(f"Key {key} is not unique in the index, using the first value...", stacklevel=2)
        elif use_raw:
            keysidx = (
                data.raw.var.index.get_indexer_for([key])
                if gene_symbols is None
                else np.nonzero(data.raw.var[gene_symbols] == key)[0]
            )
            if len(keysidx) == 0 or keysidx == -1:
                raise ValueError(f"Key {key} could not be found.")
            values = data.raw.X[:, keysidx[0]]
            if len(keysidx) > 1:
                warnings.warn(f"Key {key} is not unique in the index, using the first value...", stacklevel=2)
        else:
            keysidx = (
                data.var.index.get_indexer_for([key])
                if gene_symbols is None
                else np.nonzero(data.var[gene_symbols] == key)[0]
            )
            if len(keysidx) == 0 or keysidx == -1:
                raise ValueError(f"Key {key} could not be found.")
            x = data.X
            if x is None:
                raise ValueError(f"Cannot retrieve the key {key}: .X is None.")
            values = x[:, keysidx[0]]
            if len(keysidx) > 1:
                warnings.warn(f"Key {key} is not unique in the index, using the first value...", stacklevel=2)

        if isinstance(values, sparray | spmatrix):
            values = values.toarray().squeeze()
        values = _maybe_apply_obsmap(values, obsmap)

        return values
    else:
        raise TypeError("Expected data to be MuData or AnnData")

    raise ValueError(f"Key {key} could not be found.")
