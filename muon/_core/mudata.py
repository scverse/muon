from typing import List, Tuple, Union, Optional, Mapping, Iterable, Sequence, Any
import collections
from functools import reduce
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_categorical_dtype
import anndata
from anndata import AnnData


class MuData():
    """
    Multimodal data object

    MuData represents modalities as collections of AnnData objects
    as well as includes multimodal annotations 
    such as embeddings and neighbours graphs learned jointly 
    on multiple modalities and generalised sample 
    and feature metadata tables.
    """
    def __init__(self,
                 data: Union[AnnData, Mapping[str, AnnData]] = None,
                 feature_types_names: Optional[dict] = {"Gene Expression": "rna",
                                                        "Peaks": "atac",
                                                        "Antibody Capture": "cite"},
                 **kwargs):

        # Add all modalities to a MuData object
        self.mod = dict()
        if isinstance(data, collections.Mapping):
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
                    self.mod[alias] = data[:,data.var.feature_types == name]
        else:
            raise TypeError("Expected AnnData object or dictionary with AnnData objects as values")

        self.n_mod = len(self.mod)

        # When creating from a dictionary with _init_from_dict_
        if len(kwargs) > 0:
            # Get global observations
            self._obs = kwargs.get("obs", None)
            self._n_obs = self.obs.shape[0] if self.obs is not None else None

            # Get global obsm
            self.obsm = kwargs.get("obsm", {})

            # Get global obsp
            self.obsp = kwargs.get("obsp", None)

            # Get global variables
            self._var = kwargs.get("var", None)

            # Get global varm
            self.varm = kwargs.get("varm", {})
            
            # Get global varp
            self.varp = kwargs.get("varp", None)

            # Unstructured annotations
            self.uns = kwargs.get("uns", {})
            if self.uns is None:
                self.uns = dict()

            # For compatibility with calls requiring AnnData slots
            self.raw = None
            self.X = None
            self.layers = None
            self.isbacked = False
            self.is_view = False

            # Restore proper .obs and .var
            self.update()

            return

        # Initialise global observations
        self._obs = pd.concat([a.obs.add_prefix(m+':') for m, a in self.mod.items()], join='outer', axis=1, sort=False)

        # Make obs map for each modality
        self.obsm = dict()
        for k, v in self.mod.items():
            self.obsm[k] = self.obs.index.isin(v.obs.index)

        # Initialise global variables
        self._var = pd.concat([a.var.add_prefix(m+':') for m, a in self.mod.items()], join="outer", axis=0, sort=False)

        # Make var map for each modality
        self.varm = dict()
        for k, v in self.mod.items():
            self.varm[k] = self.var.index.isin(v.var.index)

        # Unstructured annotations
        # NOTE: this is dict in contract to OrderedDict in anndata
        #       due to favourable performance and lack of need to preserve the insertion order
        self.uns = dict()

        # For compatibility with calls requiring AnnData slots
        self.raw = None
        self.X = None
        self.layers = None
        self.isbacked = False
        self.is_view = False

        # TODO
        self.obsp = None
        self.varp = None


    @classmethod
    def _init_from_dict_(cls,
        mod: Optional[Mapping[str, Union[Mapping, AnnData]]] = None,
        obs: Optional[Union[pd.DataFrame, Mapping[str, Iterable[Any]]]] = None,
        var: Optional[Union[pd.DataFrame, Mapping[str, Iterable[Any]]]] = None,
        uns: Optional[Mapping[str, Any]] = None,
        obsm: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        varm: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        obsp: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        varp: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        ):

        return cls(data = {k:(v if isinstance(v, AnnData) else AnnData(**v)) for k, v in mod.items()},
                   obs=obs,
                   var=var,
                   uns=uns,
                   obsm=obsm,
                   varm=varm,
                   obsp=obsp,
                   varp=varp)


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
        raise NotImplementedError("MuData slicing is not implemented yet")

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

        # Check if the are same obs_names/var_names in different modalities
        # If there are, join_common=True request can not be satisfied
        if any(list(map(lambda x: len(np.intersect1d(*x)) > 0, 
            [(getattr(self.mod[mod_i], attr+'_names').values, 
                getattr(self.mod[mod_j], attr+'_names').values) 
            for i, mod_i in enumerate(self.mod) 
            for j, mod_j in enumerate(self.mod) 
            if j>i]))
        ):
            if join_common:
                warnings.warn(f"Cannot join columns with the same name because {attr}_names are intersecting.")
                join_common = False

        # Figure out which global columns exist
        columns_global = list(map(all, zip(*list([[not col.startswith(mod+":") for col in getattr(self, attr).columns] for mod in self.mod]))))

        if join_common:
            # If all modalities have a column with the same name, it is not global
            columns_common = reduce(np.intersect1d, [getattr(self.mod[mod], attr).columns for mod in self.mod])
            columns_global = [i for i in columns_global if i not in columns_common]

        # Keep data from global .obs/.var columns
        data_global = getattr(self, attr).loc[:,columns_global]

        # Join modality .obs/.var tables
        if join_common:
            data_mod = pd.concat([getattr(a, attr).drop(columns_common, axis=1).add_prefix(m + ':') for m, a in self.mod.items()], 
                join='outer', axis=1, sort=False)
            data_common = pd.concat([getattr(a, attr)[columns_common] for m, a in self.mod.items()],
                join='outer', axis=0, sort=False)
            data_mod = data_mod.join(data_common, how='left')
        else:
            data_mod = pd.concat([getattr(a, attr).add_prefix(m + ':') for m, a in self.mod.items()], join='outer', axis=1, sort=False)

        # Add data from global .obs/.var columns
        # This might reduce the size of .obs/.var if observations/variables were removed
        setattr(self, '_'+attr, data_mod.join(data_global, how='left'))
        
        # Update .obsm/.varm
        for k, v in self.mod.items():
            getattr(self, attr+'m')[k] = getattr(self, attr).index.isin(v.obs.index)

        # TODO: update .obsp/.varp (size might have changed)

    def _shrink_attr(self, attr: str):
        """
        Remove observations/variables for each modality from the global observations/variables table
        """
        # Figure out which global columns exist
        columns_global = list(map(all, zip(*list([[not col.startswith(mod+":") for col in getattr(self, attr).columns] for mod in self.mod]))))
        # Only keep data from global .obs/.var columns
        setattr(self, attr, getattr(self, attr).loc[:,columns_global])

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
            raise ValueError(f"The length of provided annotation {len(value)} does not match the length {self.shape[0]} of MuData.obs.")
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
                    raise KeyError(f"There is no {key} in MuData .obs but there is one in {m} .obs. Consider running `mu.update_obs()` to update global .obs.")
            raise KeyError(f"There is no key {key} in MuData .obs or in .obs of any modalities.")
        return self.obs[key].values

    def update_obs(self):
        """
        Update .obs slot of MuData with the newest .obs data from all the modalities
        """
        self._update_attr('obs')

    @property
    def var(self) -> pd.DataFrame:
        """
        Annotation of variables
        """
        return self._var

    @var.setter
    def var(self, value: pd.DataFrame):
        if len(value) != self.shape[1]:
            raise ValueError(f"The length of provided annotation {len(value)} does not match the length {self.shape[1]} of MuData.var.")
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
                    raise KeyError(f"There is no {key} in MuData .var but there is one in {m} .var. Consider running `mu.update_var()` to update global .var.")
            raise KeyError(f"There is no key {key} in MuData .var or in .var of any modalities.")
        return self.var[key].values

    def update_var(self):
        """
        Update .var slot of MuData with the newest .var data from all the modalities
        """
        self._update_attr('var', join_common=True)

    def var_names_make_unique(self):
        """
        Call .var_names_make_unique() method on each AnnData object
        """
        for k in self.mod:
            self.mod[k].var_names_make_unique()

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
        self._update_attr('obs')
        self._update_attr('var')

    def write_h5mu(self, filename: str, *args, **kwargs):
        """
        Write MuData object to an HDF5 file
        """
        from .io import write_h5mu

        write_h5mu(filename=filename, mdata=self, *args, **kwargs)

    write = write_h5mu

    def _gen_repr(self, n_obs, n_vars, extensive: bool = False) -> str:
        if self.isbacked:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"MuData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in [
            "obs",
            "var",
            "obsm",
            "varm",
            "obsp",
            "varp"
        ]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                keys = list(getattr(self, attr).keys())
                if len(keys) > 0:
                    mod_sep = ":" if isinstance(getattr(self, attr), pd.DataFrame) else ""
                    global_keys = list(map(all, zip(*list([[not col.startswith(mod+mod_sep) 
                        for col in getattr(self, attr).keys()] 
                        for mod in self.mod]))))
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
