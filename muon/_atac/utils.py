import re
import pandas as pd
from anndata import AnnData
from mudata import MuData
from typing import Union


def fetch_atac_mod(data: Union[AnnData, MuData]):
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
        # TODO: check that ATAC-seq slot is present with this name
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")
    return adata


def parse_region_string(region: str) -> pd.DataFrame:
    feat_list = re.split("-|:", region)
    feature_df = pd.DataFrame(columns=["Chromosome", "Start", "End"])
    feature_df.loc[0] = feat_list
    feature_df = feature_df.astype({"Start": int, "End": int})

    return feature_df
