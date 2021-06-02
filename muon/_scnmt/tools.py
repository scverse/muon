from collections import OrderedDict
from anndata import AnnData
from .._core.mudata import MuData

from typing import List, Union, Optional, Callable, Iterable
from . import utils

from .._scnmt.utils import GenomeRegion
import pandas as pd
import numpy as np


def locate_fragments(data: Union[AnnData, MuData], fragments: str, return_fragments: bool = False):
    """
    Parse fragments file and add a variable to access it to the .uns["files"]["fragments"]

    Fragments file is never read to memory, and connection to the file is closed
    upon function completion.

    Parameters
    ----------
    data
            AnnData object with peak counts or multimodal MuData object with 'met' modality.
    fragments
            A path to the compressed tab-separated fragments file (e.g. met_fragments.tsv.gz).
    return_fragments
            If return the Tabix connection the fragments file. False by default.
    """
    frag = None
    try:
        adata = utils.get_modality(data, modality="met")
        pysam = utils.import_pysam()
        # Here we make sure we can create a connection to the fragments file
        frag = pysam.TabixFile(fragments, parser=pysam.asTuple())

        # if "connections" not in adata.uns:
        #     adata.uns["connections"] = OrderedDict()
        # adata.uns["connections"]["fragments"] = frag

        if "files" not in adata.uns:
            adata.uns["files"] = OrderedDict()
        adata.uns["files"]["fragments"] = fragments

        if return_fragments:
            return frag

    except Exception as e:
        print(e)

    finally:
        if frag is not None and not return_fragments:
            # The connection has to be closed
            frag.close()


def fetch_region(
    data: Union[AnnData, MuData], region: Union[str, GenomeRegion], return_region=True
):
    """
    Fetch region from tabix file and return in data frame
    Parameters
    ----------
    data
            AnnData object with peak counts or multimodal MuData object with 'met' modality.
    region
        Genomic Region to fetch. Can be specified as region string or as GenomeRegion object.
    return_region
        Whether to return a DataFrame with the methylation events or just store it in data.uns['active_region']
    """
    # Names and dtypes of the resulting DataFrame
    columns = {
        "chromosome": "category",
        "pos": np.int64,
        "met": np.int8,
        "unmet": np.int8,
        "rate": np.float32,
        "id_met": "category",
    }

    if not isinstance(region, GenomeRegion):
        region = GenomeRegion(region)  # Parse the region input
        
    adata = utils.get_modality(data, modality="met")

    pysam = utils.import_pysam()
    with pysam.TabixFile(adata.uns["files"]["fragments"], parser=pysam.asTuple()) as tabix:
        # Try to switch chromosome notations
        if region.chrom not in tabix.contigs:
            region.change_chrom_notation()
        fetched = tabix.fetch(
            region.chrom,
            region.start,
            region.end,
            parser=pysam.asTuple(),
        )

        region_df = pd.DataFrame(fetched, columns=columns.keys())
        region_df = region_df.astype(columns)
    adata.uns["active_region"] = region_df
    if return_region:
        return region_df
