from typing import Tuple, Union, Mapping, MutableMapping
import collections
import numpy as np
import pandas as pd
from anndata import AnnData

class AmmData():
    def __init__(self,
                 data: Union[AnnData, Mapping[str, AnnData]] = None):

        # Add all modalities to the AmmData object
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
                    self.mod[k] = data[:,data.var.feature_types == k]
        else:
            raise TypeError("Expected AnnData object or dictionary with AnnData objects as values")

        self.n_obs = 0
        self.n_vars = 0
        self.isbacked = False
        
        # Initialise global observations
        self.obs = pd.concat([a.obs.add_suffix(f"_{m}") for m, a in self.mod.items()], join='outer', axis=1)
        self.n_obs = self.obs.shape[0]

        # Make obs map for each modality
        self.obsm = dict()
        for k, v in self.mod.items():
            global_obs_indices = [self.obs.index.get_loc(i) for i in v.obs.index.values]
            self.obsm[k] = np.array(global_obs_indices)

        # Initialise global variables
        self.var = pd.concat([a.var.add_suffix(f"_{m}") for m, a in self.mod.items()], join='outer', axis=1, sort=False)
        self.n_var = self.var.shape[0]
        # API legacy from AnnData
        self.n_vars = self.n_var

        # Make var map for each modality
        self.varm = dict()
        for k, v in self.mod.items():
            global_var_indices = [self.var.index.get_loc(i) for i in v.var.index.values]
            self.varm[k] = np.array(global_var_indices)

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

    def _gen_repr(self, n_obs, n_vars, extensive: bool = False) -> str:
        if self.isbacked:
            backed_at = f"backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"AmmData object with n_obs × n_vars = {n_obs} × {n_vars} {backed_at}"
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
