from typing import List, Tuple, Union, Optional, Mapping, Iterable, Sequence, Any
from numbers import Integral
from collections import abc
from functools import reduce
import warnings
from copy import deepcopy
from os import PathLike

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_categorical_dtype
import anndata
from anndata import AnnData
from anndata.utils import convert_to_dict
from anndata._core.aligned_mapping import (
    AxisArrays,
    AlignedViewMixin,
    AxisArraysBase,
    PairwiseArrays,
    PairwiseArraysView,
)


class MuAxisArraysView(AlignedViewMixin, AxisArraysBase):
    def __init__(self, parent_mapping: AxisArraysBase, parent_view: "MuData", subset_idx: Any):
        self.parent_mapping = parent_mapping
        self._parent = parent_view
        self.subset_idx = subset_idx
        self._axis = parent_mapping._axis
        self.dim_names = None


class MuAxisArrays(AxisArrays):
    _view_class = MuAxisArraysView


class MuData:
    """
    Multimodal data object

    MuData represents modalities as collections of AnnData objects
    as well as includes multimodal annotations
    such as embeddings and neighbours graphs learned jointly
    on multiple modalities and generalised sample
    and feature metadata tables.
    """

    def __init__(
        self,
        data: Union[AnnData, Mapping[str, AnnData], "MuData"] = None,
        feature_types_names: Optional[dict] = {
            "Gene Expression": "rna",
            "Peaks": "atac",
            "Antibody Capture": "prot",
        },
        as_view: bool = False,
        index: Optional[
            Union[Tuple[Union[slice, Integral], Union[slice, Integral]], slice, Integral]
        ] = None,
        **kwargs,
    ):
        self._init_common()
        if as_view:
            self._init_as_view(data, index)
            return

        # Add all modalities to a MuData object
        self.mod = dict()
        if isinstance(data, abc.Mapping):
            for k, v in data.items():
                self.mod[k] = v
        elif isinstance(data, AnnData):
            # Get the list of modalities
            if "feature_types" in data.var.columns:
                if data.var.feature_types.dtype.name == "category:":
                    mod_names = data.var.feature_types.cat.categories.values
                else:
                    mod_names = data.var.feature_types.unique()

                for name in mod_names:
                    alias = name
                    if feature_types_names is not None:
                        if name in feature_types_names.keys():
                            alias = feature_types_names[name]
                    self.mod[alias] = data[:, data.var.feature_types == name].copy()
        else:
            raise TypeError("Expected AnnData object or dictionary with AnnData objects as values")

        # When creating from a dictionary with _init_from_dict_
        if len(kwargs) > 0:
            # Get global observations
            self._obs = kwargs.get("obs", None)
            if isinstance(self._obs, abc.Mapping):
                self._obs = pd.DataFrame(self._obs)

            # Get global variables
            self._var = kwargs.get("var", None)
            if isinstance(self._var, abc.Mapping):
                self._var = pd.Dataframe(self._var)

            # Get global obsm
            self._obsm = MuAxisArrays(self, 0, kwargs.get("obsm", {}))
            # Get global varm
            self._varm = MuAxisArrays(self, 1, kwargs.get("varm", {}))

            self._obsp = PairwiseArrays(self, 0, kwargs.get("obsp", {}))
            self._varp = PairwiseArrays(self, 1, kwargs.get("varp", {}))

            # Restore proper .obs and .var
            self.update()

            self.uns = kwargs.get("uns", {})

            return

        # Initialise global observations
        self._obs = pd.concat(
            [a.obs.add_prefix(m + ":") for m, a in self.mod.items()],
            join="outer",
            axis=1,
            sort=False,
        )

        # Initialise global variables
        self._var = pd.concat(
            [a.var.add_prefix(m + ":") for m, a in self.mod.items()],
            join="outer",
            axis=0,
            sort=False,
        )

        # Make obs map for each modality
        self._obsm = dict()
        for k, v in self.mod.items():
            self._obsm[k] = self.obs.index.isin(v.obs.index)
        self._obsm = MuAxisArrays(self, 0, self._obsm)

        self._obsp = PairwiseArrays(self, 0, dict())

        # Make var map for each modality
        self._varm = dict()
        for k, v in self.mod.items():
            self._varm[k] = self.var.index.isin(v.var.index)
        self._varm = MuAxisArrays(self, 1, self._varm)

        self._varp = PairwiseArrays(self, 1, dict())

    def _init_common(self):
        self._mudata_ref = None

        # Unstructured annotations
        # NOTE: this is dict in contract to OrderedDict in anndata
        #       due to favourable performance and lack of need to preserve the insertion order
        self.uns = dict()

        # For compatibility with calls requiring AnnData slots
        self.raw = None
        self.X = None
        self.layers = None
        self.file = None
        self.filename = None
        self.filemode = None
        self.is_view = False

    def _init_as_view(self, mudata_ref: "MuData", index):
        def slice_mapping(mapping, obsnames, varnames):
            mp = {}
            for n, v in mapping.items():
                obsidx = v.obs.index.isin(obsnames)
                varidx = v.var.index.isin(varnames)
                mp[n] = v[obsidx, varidx]
            return mp

        from anndata._core.index import _normalize_indices

        obsidx, varidx = _normalize_indices(index, mudata_ref.obs.index, mudata_ref.var.index)
        if isinstance(
            obsidx, Integral
        ):  # to handle single-element subsets, otherwise pd.Index[int] returns
            obsidx = slice(
                obsidx, obsidx + 1
            )  # a str and pd.Index.intersection throws an exception
        if isinstance(varidx, Integral):
            varidx = slice(varidx, varidx + 1)

        self.mod = slice_mapping(
            mudata_ref.mod, mudata_ref.obs.index[obsidx], mudata_ref.var.index[varidx]
        )
        self._obs = mudata_ref.obs.iloc[obsidx, :]
        self._obsm = mudata_ref.obsm._view(self, (obsidx,))
        self._obsp = mudata_ref.obsp._view(self, obsidx)
        self._var = mudata_ref.var.iloc[varidx, :]
        self._varm = mudata_ref.varm._view(self, (varidx,))
        self._varp = mudata_ref.varp._view(self, varidx)

        self.is_view = True
        self.file = mudata_ref.file
        self.filename = mudata_ref.filename
        self.filemode = mudata_ref.filemode
        self._mudata_ref = mudata_ref

    def _init_as_actual(self, data: "MuData"):
        self._init_common()
        self.mod = data.mod
        self._obs = data.obs
        self._var = data.var
        self._obsm = MuAxisArrays(self, 0, convert_to_dict(data.obsm))
        self._obsp = PairwiseArrays(self, 0, convert_to_dict(data.obsp))
        self._varm = MuAxisArrays(self, 1, convert_to_dict(data.varm))
        self._varp = PairwiseArrays(self, 1, convert_to_dict(data.varp))
        self.uns = data.uns

    @classmethod
    def _init_from_dict_(
        cls,
        mod: Optional[Mapping[str, Union[Mapping, AnnData]]] = None,
        obs: Optional[Union[pd.DataFrame, Mapping[str, Iterable[Any]]]] = None,
        var: Optional[Union[pd.DataFrame, Mapping[str, Iterable[Any]]]] = None,
        uns: Optional[Mapping[str, Any]] = None,
        obsm: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        varm: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        obsp: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        varp: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
    ):

        return cls(
            data={k: (v if isinstance(v, AnnData) else AnnData(**v)) for k, v in mod.items()},
            obs=obs,
            var=var,
            uns=uns,
            obsm=obsm,
            varm=varm,
            obsp=obsp,
            varp=varp,
        )

    def copy(self, filename: Optional[PathLike] = None) -> "MuData":
        if not self.isbacked:
            mod = {}
            for k, v in self.mod.items():
                mod[k] = v.copy()
            return self._init_from_dict_(
                mod,
                self.obs.copy(),
                self.var.copy(),
                deepcopy(self.uns),  # this should always be an empty dict
                self.obsm.copy(),
                self.varm.copy(),
                deepcopy(self.obsp),
                deepcopy(self.varp),
            )
        else:
            if filename is None:
                raise ValueError(
                    "To copy a MuData object in backed mode, pass a filename: `copy(filename='myfilename.h5mu')`"
                )
            from .io import read_h5mu

            self.write_h5mu(filename)
            return read_h5mu(filename, self.filemode)

    def strings_to_categoricals(self, df: Optional[pd.DataFrame] = None):
        """
        Transform string columns in .var and .obs slots of MuData to categorical
        as well as of .var and .obs slots in each AnnData object

        This keeps it compatible with AnnData.strings_to_categoricals() method.
        """
        AnnData.strings_to_categoricals(self, df)

        # Call the same method on each modality
        for k in self.mod:
            self.mod[k].strings_to_categoricals(df)

    # To increase compatibility with scanpy methods
    _sanitize = strings_to_categoricals

    def __getitem__(self, index) -> Union["MuData", AnnData]:
        if isinstance(index, str):
            return self.mod[index]
        else:
            return MuData(self, as_view=True, index=index)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of data, all variables and observations combined (:attr:`n_obs`, :attr:`n_var`)."""
        return self.n_obs, self.n_vars

    # # Currently rely on AnnData's interface for setting .obs / .var
    # # This code implements AnnData._set_dim_df for another namespace
    # def _set_dim_df(self, value: pd.DataFrame, attr: str):
    #     if not isinstance(value, pd.DataFrame):
    #         raise ValueError(f"Can only assign pd.DataFrame to {attr}.")
    #     value_idx = AnnData._prep_dim_index(self, value.index, attr)
    #     setattr(self, f"_{attr}", value)
    #     AnnData._set_dim_index(self, value_idx, attr)

    def _update_attr(self, attr: str, join_common: bool = False):
        """
        Update global observations/variables with observations/variables for each modality
        """
        prev_index = getattr(self, attr).index

        # Check if the are same obs_names/var_names in different modalities
        # If there are, join_common=True request can not be satisfied
        if any(
            list(
                map(
                    lambda x: len(np.intersect1d(*x)) > 0,
                    [
                        (
                            getattr(self.mod[mod_i], attr + "_names").values,
                            getattr(self.mod[mod_j], attr + "_names").values,
                        )
                        for i, mod_i in enumerate(self.mod)
                        for j, mod_j in enumerate(self.mod)
                        if j > i
                    ],
                )
            )
        ):
            if join_common:
                warnings.warn(
                    f"Cannot join columns with the same name because {attr}_names are intersecting."
                )
                join_common = False

        # Figure out which global columns exist
        columns_global = getattr(self, attr).columns[
            list(
                map(
                    all,
                    zip(
                        *list(
                            [
                                [
                                    not col.startswith(mod + ":")
                                    for col in getattr(self, attr).columns
                                ]
                                for mod in self.mod
                            ]
                        )
                    ),
                )
            )
        ]

        # If all modalities have a column with the same name, it is not global
        columns_common = reduce(
            np.intersect1d, [getattr(self.mod[mod], attr).columns for mod in self.mod]
        )
        columns_global = [i for i in columns_global if i not in columns_common]

        # Keep data from global .obs/.var columns
        data_global = getattr(self, attr).loc[:, columns_global]

        # Join modality .obs/.var tables
        if join_common:
            data_mod = pd.concat(
                [
                    getattr(a, attr).drop(columns_common, axis=1).add_prefix(m + ":")
                    for m, a in self.mod.items()
                ],
                join="outer",
                axis=1,
                sort=False,
            )
            data_common = pd.concat(
                [getattr(a, attr)[columns_common] for m, a in self.mod.items()],
                join="outer",
                axis=0,
                sort=False,
            )
            data_mod = data_mod.join(data_common, how="left")
        else:
            data_mod = pd.concat(
                [getattr(a, attr).add_prefix(m + ":") for m, a in self.mod.items()],
                join="outer",
                axis=1,
                sort=False,
            )

        # Add data from global .obs/.var columns
        # This might reduce the size of .obs/.var if observations/variables were removed
        setattr(self, "_" + attr, data_mod.join(data_global, how="left", sort=False))

        # Update .obsm/.varm
        for k, v in self.mod.items():
            getattr(self, attr + "m")[k] = getattr(self, attr).index.isin(getattr(v, attr).index)

        keep_index = prev_index.isin(getattr(self, attr).index)

        if keep_index.sum() != len(prev_index):
            for mx_key, mx in getattr(self, attr + "m").items():
                if mx_key not in self.mod.keys():  # not a modality name
                    getattr(self, attr + "m")[mx_key] = getattr(self, attr + "m")[mx_key][
                        keep_index, :
                    ]

            # Update .obsp/.varp (size might have changed)
            for mx_key, mx in getattr(self, attr + "p").items():
                if mx_key not in self.mod.keys():  # not a modality name
                    getattr(self, attr + "p")[mx_key] = getattr(self, attr + "p")[mx_key][
                        keep_index, keep_index
                    ]

    def _shrink_attr(self, attr: str):
        """
        Remove observations/variables for each modality from the global observations/variables table
        """
        # Figure out which global columns exist
        columns_global = list(
            map(
                all,
                zip(
                    *list(
                        [
                            [not col.startswith(mod + ":") for col in getattr(self, attr).columns]
                            for mod in self.mod
                        ]
                    )
                ),
            )
        )
        # Only keep data from global .obs/.var columns
        setattr(self, attr, getattr(self, attr).loc[:, columns_global])

    @property
    def n_mod(self) -> int:
        return len(self.mod)

    @property
    def isbacked(self) -> bool:
        return self.filename is not None

    @property
    def obs(self) -> pd.DataFrame:
        """
        Annotation of observation
        """
        return self._obs

    @obs.setter
    def obs(self, value: pd.DataFrame):
        # self._set_dim_df(value, "obs")
        if len(value) != self.shape[0]:
            raise ValueError(
                f"The length of provided annotation {len(value)} does not match the length {self.shape[0]} of MuData.obs."
            )
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obs = value

    @property
    def n_obs(self) -> int:
        """
        Total number of observations
        """
        return self._obs.shape[0]

    def obs_vector(self, key: str, layer: Optional[str] = None) -> np.ndarray:
        """
        Return an array of values for the requested key of length n_obs
        """
        if key not in self.obs.columns:
            for m, a in self.mod.items():
                if key in a.obs.columns:
                    raise KeyError(
                        f"There is no {key} in MuData .obs but there is one in {m} .obs. Consider running `mu.update_obs()` to update global .obs."
                    )
            raise KeyError(f"There is no key {key} in MuData .obs or in .obs of any modalities.")
        return self.obs[key].values

    def update_obs(self):
        """
        Update .obs slot of MuData with the newest .obs data from all the modalities
        """
        self._update_attr("obs")

    @property
    def obs_names(self) -> pd.Index:
        """
        Names of variables (alias for `.obs.index`)

        This property is read-only.
        To be modified, obs_names of individual modalities
        should be changed, and .update_obs() should be called then.
        """
        return self.obs.index

    @property
    def var(self) -> pd.DataFrame:
        """
        Annotation of variables
        """
        return self._var

    @var.setter
    def var(self, value: pd.DataFrame):
        if len(value) != self.shape[1]:
            raise ValueError(
                f"The length of provided annotation {len(value)} does not match the length {self.shape[1]} of MuData.var."
            )
        if self.is_view:
            self._init_as_actual(self.copy())
        self._var = value

    @property
    def n_var(self) -> int:
        """
        Total number of variables
        """
        return self._var.shape[0]

    # API legacy from AnnData
    n_vars = n_var

    def var_vector(self, key: str, layer: Optional[str] = None) -> np.ndarray:
        """
        Return an array of values for the requested key of length n_var
        """
        if key not in self.var.columns:
            for m, a in self.mod.items():
                if key in a.var.columns:
                    raise KeyError(
                        f"There is no {key} in MuData .var but there is one in {m} .var. Consider running `mu.update_var()` to update global .var."
                    )
            raise KeyError(f"There is no key {key} in MuData .var or in .var of any modalities.")
        return self.var[key].values

    def update_var(self):
        """
        Update .var slot of MuData with the newest .var data from all the modalities
        """
        self._update_attr("var", join_common=True)

    def var_names_make_unique(self):
        """
        Call .var_names_make_unique() method on each AnnData object.

        If there are var_names, which are the same for multiple modalities,
        append modality name to all var_names.
        """
        mod_var_sum = np.sum([a.n_vars for a in self.mod.values()])
        if mod_var_sum != self.n_vars:
            self.update_var()

        for k in self.mod:
            self.mod[k].var_names_make_unique()

        # Check if there are variables with the same name in different modalities
        common_vars = []
        mods = list(self.mod.keys())
        for i in range(len(self.mod) - 1):
            ki = mods[i]
            for j in range(i + 1, len(self.mod)):
                kj = mods[j]
                common_vars.append(
                    np.intersect1d(self.mod[ki].var_names.values, self.mod[kj].var_names.values)
                )
        if any(map(lambda x: len(x) > 0, common_vars)):
            warnings.warn(
                "Modality names will be prepended to var_names since there are identical var_names in different modalities."
            )
            for k in self.mod:
                self.mod[k].var_names = k + ":" + self.mod[k].var_names.astype(str)

        # Update .var.index in the MuData
        var_names = [var for a in self.mod.values() for var in a.var_names.values]
        self._var.index = var_names

    @property
    def var_names(self) -> pd.Index:
        """
        Names of variables (alias for `.var.index`)

        This property is read-only.
        To be modified, var_names of individual modalities
        should be changed, and .update_var() should be called then.
        """
        return self.var.index

    # Multi-dimensional annotations (.obsm and .varm)

    @property
    def obsm(self) -> Union[MuAxisArrays, MuAxisArraysView]:
        """
        Multi-dimensional annotation of observation
        """
        return self._obsm

    @obsm.setter
    def obsm(self, value):
        obsm = MuAxisArrays(self, 0, vals=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obsm = obsm

    @obsm.deleter
    def obsm(self):
        self.obsm = dict()

    @property
    def obsp(self) -> Union[PairwiseArrays, PairwiseArraysView]:
        """
        Pairwise annotatation of observations
        """
        return self._obsp

    @obsp.setter
    def obsp(self, value):
        obsp = PairwiseArrays(self, 0, vals=convert_to_dict(value))
        if self.is_vew:
            self._init_as_actual(self.copy())
        self._obsp = obsp

    @obsp.deleter
    def obsp(self):
        self.obsp = dict()

    @property
    def varm(self) -> Union[MuAxisArrays, MuAxisArraysView]:
        """
        Multi-dimensional annotation of variables
        """
        return self._varm

    @varm.setter
    def varm(self, value):
        varm = MuAxisArrays(self, 1, vals=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._varm = varm

    @varm.deleter
    def varm(self):
        self.varm = dict()

    @property
    def varp(self) -> Union[PairwiseArrays, PairwiseArraysView]:
        """
        Pairwise annotatation of variables
        """
        return self._varp

    @varp.setter
    def varp(self, value):
        varp = PairwiseArrays(self, 0, vals=convert_to_dict(value))
        if self.is_vew:
            self._init_as_actual(self.copy())
        self._varp = varp

    @varp.deleter
    def varp(self):
        self.varp = dict()

    # _keys methods to increase compatibility
    # with calls requiring those AnnData methods

    def obs_keys(self) -> List[str]:
        """List keys of observation annotation :attr:`obs`."""
        return self._obs.keys().tolist()

    def var_keys(self) -> List[str]:
        """List keys of variable annotation :attr:`var`."""
        return self._var.keys().tolist()

    def obsm_keys(self) -> List[str]:
        """List keys of observation annotation :attr:`obsm`."""
        return list(self._obsm.keys())

    def varm_keys(self) -> List[str]:
        """List keys of variable annotation :attr:`varm`."""
        return list(self._varm.keys())

    def uns_keys(self) -> List[str]:
        """List keys of unstructured annotation."""
        return list(self._uns.keys())

    def update(self):
        """
        Update both .obs and .var of MuData with the data from all the modalities
        """
        self.update_var()
        self.update_obs()

    def write_h5mu(self, filename: str, *args, **kwargs):
        """
        Write MuData object to an HDF5 file
        """
        from .io import write_h5mu

        write_h5mu(filename=filename, mdata=self, *args, **kwargs)

    write = write_h5mu

    def _gen_repr(self, n_obs, n_vars, extensive: bool = False) -> str:
        backed_at = f" backed at {str(self.filename)!r}" if self.isbacked else ""
        view_of = "View of " if self.is_view else ""
        descr = f"{view_of}MuData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in ["obs", "var", "obsm", "varm", "obsp", "varp"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                keys = list(getattr(self, attr).keys())
                if len(keys) > 0:
                    mod_sep = ":" if isinstance(getattr(self, attr), pd.DataFrame) else ""
                    global_keys = list(
                        map(
                            all,
                            zip(
                                *list(
                                    [
                                        [
                                            not col.startswith(mod + mod_sep)
                                            for col in getattr(self, attr).keys()
                                        ]
                                        for mod in self.mod
                                    ]
                                )
                            ),
                        )
                    )
                    if any(global_keys):
                        descr += f"\n  {attr}:\t{str([keys[i] for i in range(len(keys)) if global_keys[i]])[1:-1]}"
        descr += f"\n  {len(self.mod)} modalities"
        for k, v in self.mod.items():
            descr += f"\n    {k}:\t{v.n_obs} x {v.n_vars}"
            for attr in [
                "obs",
                "var",
                "uns",
                "obsm",
                "varm",
                "layers",
                "obsp",
                "varp",
            ]:
                keys = getattr(v, attr).keys()
                if len(keys) > 0:
                    descr += f"\n      {attr}:\t{str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        return self._gen_repr(self.n_obs, self.n_vars, extensive=True)
