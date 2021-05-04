import re
import itertools
import numpy as np
from anndata import AnnData
from .._core.mudata import MuData
from typing import Union


def get_modality(data: Union[AnnData, MuData], modality: str):
    """
    Accepts a mudata object and returns the specified modality or just returns an anndata object.
    Parameters
    ----------
    data
            AnnData object with peak counts or multimodal MuData object with the right modality.
    modality
        Name of the modality to fetch
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and modality in data.mod:
        adata = data.mod[modality]
    else:
        raise TypeError(f"Expected AnnData or MuData object with '{modality}' modality")
    return adata


def import_pysam():
    """Print helpful message if pysam not available"""
    try:
        import pysam

        return pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. \
            Install pysam from PyPI (`pip install pysam`) \
            or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        )


class GenomeRegion:
    """
    A genomic region. A region consists of reference name (‘chr1’), start (10000), and end (20000).
    Attributes
    ----------
    chrom : str
        chromosome in Ensembl notation ('chr1')
    start : int
        start position
    end : int
        end position
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], str):
                chrom, start, end = self.parse_region_str(args[0])
            else:
                chrom, start, end = args[0]
        elif len(args) == 3:
            chrom, start, end = args
        else:
            raise ValueError(
                "Cannot parse genomic region.\n"
                "Genomic ranges can be specified as follows:\n"
                "Region string as used by samtools: 'chr1:1000-2000'\n"
                "Tuple or list like ['chr1', 1000, 2000]"
            )

        # Check the formatting
        if "chrom_notation" in kwargs:
            chrom = GenomeRegion.change_chrom_notation(chrom, notation=kwargs["chrom_notation"])
        start, end = int(start), int(end)
        chrom = str(chrom)
        if end < start:
            raise ValueError(f"End position({end}) was smaller than start position({start})")
        self.chrom, self.start, self.end = chrom, start, end
        self.notation = "Ensembl"

    def __repr__(self):
        return f"GenomeRegion object: {self.chrom}:{self.start}-{self.end}"

    @staticmethod
    def parse_region_str(region: str):
        try:
            chrom, start, end = re.split("-|:", region)
            return chrom, start, end

        except Exception:
            raise ValueError(f"Cannot parse {region}. Please specify as:" f"chr1:2000-3000")

    @staticmethod
    def change_chrom_notation(chrom: str, notation: str = "Ensembl"):
        """
        Changes chromosome notation between UCSC and Ensembl
        Parameters
        ----------
        chrom: str
            chromosome
        notation: str
            Either UCSC ('1', '2', 'X', 'Y'), or ENSEMBL ('chr1', 'chr2', 'chrX', 'chrY')
        """
        chrom = chrom.lstrip("chr")
        if notation.lower() == "ensembl":
            chrom = "chr" + chrom
        elif notation.lower() != "ucsc":
            raise ValueError(f"notation must either be ucsc or ensembl, but was {notation}")
        return chrom


def center_colum(arr):
    """Center columns such that the maximum and minimum value have equal distance from 0"""
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    rng = maxs - mins
    return arr - mins - rng / 2


def make_grid(num_points=100, lower=-1, upper=1, input_dim=2):

    # Make sure bounds have the same dimensions as input dim
    if input_dim > 1:
        if np.isscalar(lower):
            lower = np.repeat(lower, input_dim)
        elif len(lower) != input_dim:
            raise ValueError("lower must be a scalar or a list with input_dim elements")
        if np.isscalar(upper):
            upper = np.repeat(upper, input_dim)
        elif len(upper) != input_dim:
            raise ValueError("upper must be a scalar or a list with input_dim elements")
        if np.isscalar(num_points):
            num_points = np.repeat(num_points, input_dim)
        elif len(num_points) != input_dim:
            raise ValueError("num_points must be a scalar or a list with input_dim elements")

        a = [np.linspace(lower[i], upper[i], num_points[i]) for i in range(input_dim)]
        grid = np.array([x for x in itertools.product(*a)])
    else:
        grid = np.linspace(lower, upper, num_points)[:, None]
    return grid


def make_grid_along_input(X, num_points):
    input_dim = X.shape[1]
    minima = np.amin(X, axis=0)
    maxima = np.amax(X, axis=0)
    grid = make_grid(num_points=num_points, lower=minima, upper=maxima, input_dim=input_dim)
    return grid
