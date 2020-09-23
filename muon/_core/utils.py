import pandas as pd
from .._core.mudata import MuData


def get_gene_annotation_from_rna(mdata: MuData):
    """
	Get data frame with start and end positions from interval
	column of the 'rna' layers .var.

	Parameters
	----------
	mdata: MuData
		MuData object
	"""

    if "rna" in mdata.mod:
        if "interval" in mdata.mod["rna"].var.columns:
            features = pd.DataFrame(
                [
                    s.replace(":", "-", 1).split("-")
                    for s in mdata.mod["rna"].var.interval
                ]
            )
            features.columns = ["Chromosome", "Start", "End"]
            features["gene_id"] = mdata.mod["rna"].var.gene_ids.values
            features["gene_name"] = mdata.mod["rna"].var.index.values
            # Remove genes with no coordinates indicated
            features = features.loc[~features.Start.isnull()]
            features.Start = features.Start.astype(int)
            features.End = features.End.astype(int)
        else:
            raise ValueError(".var object does not have a column named interval")
    else:
        raise ValueError("No rna layer found")
    return features
