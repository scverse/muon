from functools import reduce
from typing import Tuple, Union, Optional, Mapping, MutableMapping
import collections
import numpy as np
import pandas as pd
from anndata import AnnData

class MuData():
    def __init__(self,
                 data: Union[AnnData, Mapping[str, AnnData]] = None,
                 feature_types_names: Optional[dict] = {"Gene Expression": "rna",
                                                        "Peaks": "atac"}):

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

                for k in mod_names:
                    if feature_types_names is not None:
                        if k in feature_types_names.keys():
                            k = feature_types_names[k]
                    self.mod[k] = data[:,data.var.feature_types == k]
        else:
            raise TypeError("Expected AnnData object or dictionary with AnnData objects as values")

        self.n_obs = 0
        self.n_vars = 0
        self.n_mod = len(self.mod)
        self.isbacked = False
        
        # Initialise global observations
        self.obs = pd.concat([a.obs for m, a in self.mod.items()], join='outer', axis=1, sort=False)
        self.n_obs = self.obs.shape[0]

        # Make obs map for each modality
        self.obsm = dict()
        self.obs['ix'] = range(len(self.obs))
        for k, v in self.mod.items():
            self.obsm[k] = self.obs.loc[v.obs.index.values].ix.values

        # Initialise global variables
        self.var = pd.concat([a.var for a in self.mod.values()], join="outer", axis=0, sort=False)
        self.n_var = self.var.shape[0]
        # API legacy from AnnData
        self.n_vars = self.n_var

        # Make var map for each modality
        self.varm = dict()
        self.var['ix'] = range(len(self.var))
        for k, v in self.mod.items():
            self.varm[k] = self.var.loc[v.var.index.values].ix.values

        # Unstructured annotations
        # NOTE: this is dict in contract to OrderedDict in anndata
        #       due to favourable performance and lack of need to preserve the insertion order
        self.uns = dict()

        print(self)

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
        self.obs = pd.concat([a.obs for m, a in self.mod.items()], join='outer', axis=1, sort=False)
        self.n_obs = self.obs.shape[0]

    def update_var(self):
        """
        Update global variables from variables for each modality
        """
        self.var = pd.concat([a.var for a in self.mod.values()], join="outer", axis=0, sort=False)
        self.n_vars = self.var.shape[0]

    def var_names_make_unique(self):
        """
        Call .var_names_make_unique() method on each AnnData object
        """
        for k in self.mod:
            self.mod[k].var_names_make_unique()

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
        return self._gen_repr(self.n_obs, self.n_vars, extensive = True)
