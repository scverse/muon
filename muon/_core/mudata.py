from typing import Tuple, Union, Optional, Mapping, Iterable, Sequence, Any
import collections
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_categorical_dtype
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
                                                        "Peaks": "atac"},
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
            self.obs = kwargs.get("obs", None)
            self.n_obs = self.obs.shape[0] if self.obs is not None else None

            # Get global obsm
            self.obsm = kwargs.get("obsm", {})

            # Get global obsp
            self.obsp = kwargs.get("obsp", None)

            # Get global variables
            self.var = kwargs.get("var", None)
            self.n_var = self.var.shape[0] if self.var is not None else None
            # API legacy from AnnData
            self.n_vars = self.n_var

            # Get global varm
            self.varm = kwargs.get("varm", {})
            
            # Get global varp
            self.varp = kwargs.get("varp", None)

            # Unstructured annotations
            self.uns = kwargs.get("uns", {})

            # For compatibility with calls requiring AnnData slots
            self.raw = None
            self.X = None
            self.layers = None
            self.isbacked = False
            self.is_view = False

            return

        self.n_obs = 0
        self.n_vars = 0
    
        # Initialise global observations
        self.obs = pd.concat([a.obs for m, a in self.mod.items()], join='outer', axis=1, sort=False)
        self.n_obs = self.obs.shape[0]

        # Make obs map for each modality
        self.obsm = dict()
        self.obs['ix'] = range(len(self.obs))
        for k, v in self.mod.items():
            self.obsm[k] = self.obs.index.isin(v.obs.index)

        # Initialise global variables
        self.var = pd.concat([a.var for a in self.mod.values()], join="outer", axis=0, sort=False)
        self.n_var = self.var.shape[0]
        # API legacy from AnnData
        self.n_vars = self.n_var

        # Make var map for each modality
        self.varm = dict()
        self.var['ix'] = range(len(self.var))
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
        mod: Optional[Mapping[str, Mapping]] = None,
        obs: Optional[Union[pd.DataFrame, Mapping[str, Iterable[Any]]]] = None,
        var: Optional[Union[pd.DataFrame, Mapping[str, Iterable[Any]]]] = None,
        uns: Optional[Mapping[str, Any]] = None,
        obsm: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        varm: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        obsp: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        varp: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        ):

        return cls(data = {k:AnnData(**v) for k, v in mod.items()},
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

    def __getitem__(self, modality: str) -> AnnData:
        return self.mod[modality]

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of data, all variables and observations combined (:attr:`n_obs`, :attr:`n_var`)."""
        return self.n_obs, self.n_var

    def update_obs(self):
        """
        Update global observations from observations for each modality
        """
        # TODO: preserve columns unique to global obs
        self.obs = pd.concat([a.obs for m, a in self.mod.items()], join='outer', axis=1, sort=False)
        self.n_obs = self.obs.shape[0]

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

    def update_var(self):
        """
        Update global variables from variables for each modality
        """
        # TODO: preserve columns unique to global var
        self.var = pd.concat([a.var for a in self.mod.values()], join="outer", axis=0, sort=False)
        self.n_vars = self.var.shape[0]

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

    def var_names_make_unique(self):
        """
        Call .var_names_make_unique() method on each AnnData object
        """
        for k in self.mod:
            self.mod[k].var_names_make_unique()

    def write_h5mu(self, filename: str, *args, **kwargs):
        """
        Write MuData object to an HDF5 file
        """
        from .io import write_h5mu

        write_h5mu(filename=filename, mdata=self, *args, **kwargs)

    write = write_h5mu

    def _gen_repr(self, n_obs, n_vars, extensive: bool = False) -> str:
        if self.isbacked:
            backed_at = f"backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"MuData object with n_obs × n_vars = {n_obs} × {n_vars} {backed_at}"
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
