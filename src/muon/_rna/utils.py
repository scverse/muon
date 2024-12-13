import pandas as pd
from anndata import AnnData
from .. import MuData
from typing import Union


def get_gene_annotation_from_rna(data: Union[AnnData, MuData]) -> pd.DataFrame:
    """
    Get data frame with start and end positions from interval
    column of the 'rna' layers .var.

    Parameters
    ----------
    mdata: MuData
            MuData object
    """

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "rna" in data.mod:
        adata = data.mod["rna"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'rna' modality")

    if "interval" in adata.var.columns:
        features = pd.DataFrame([s.replace(":", "-", 1).split("-") for s in adata.var.interval])
        features.columns = ["Chromosome", "Start", "End"]
        features["gene_id"] = adata.var.gene_ids.values
        features["gene_name"] = adata.var.index.values
        features.index = adata.var.index
        # Remove genes with no coordinates indicated
        features = features.loc[~features.Start.isnull()]
        features.Start = features.Start.astype(int)
        features.End = features.End.astype(int)
    else:
        raise ValueError(".var object does not have a column named interval")
    return features
