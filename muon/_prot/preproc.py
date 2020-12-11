from typing import Optional, Iterable, Tuple, Union
from numbers import Integral, Real
from warnings import warn

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from anndata import AnnData

from .. import MuData


def dsb(
    data: Union[AnnData, MuData],
    data_raw: Optional[Union[AnnData, MuData]] = None,
    pseudocount: Integral = 10,
    denoise_counts: bool = True,
    isotype_controls: Optional[Iterable[str]] = None,
    empty_counts_range: Optional[Tuple[Real, Real]] = None,
    cell_counts_range: Optional[Tuple[Real, Real]] = None,
    add_layer: bool = False,
    random_state: Optional[Union[int, np.random.RandomState, None]] = None,
) -> Union[None, MuData]:
    """
    Normalize protein expression with DSB (Denoised and Scaled by Background)

    Normalized data will be written to ``data`` (if it is an AnnData object) or ``data.mod['prot']``
    (if it is a MuData object) as an X matrix or as a new layer named ``dsb``.

    References:
        Mulè et al, 2020 (`doi:10.1101/2020.02.24.963603 <https://dx.doi.org/10.1101/2020.02.24.963603>`_)

    Args:
        data: AnnData object with protein expression counts or MuData object with ``prot`` modality.
            If ``data_raw`` is ``None``, must be a ``MuData`` object containing raw (unfiltered,
            including empty droplets) data for both ``prot`` and ``rna`` modalities. If ``data_raw``
            is not ``None``, must contain filtered (non-empty droplets) data.
        data_raw: AnnData object with protein expression counts or MuData object with 'prot' modality
            containing raw (unfiltered, including empty droplets) data.
        pseudocount: Pseudocount to add before log-transform.
        denoise_counts: Whether to perform denoising.
        isotype_controls: Names of the isotype controls. If ``None``, isotype controls will not be used.
        empty_counts_range: If ``data_raw`` is ``None``, i.e. ``data`` contains the unfiltered data,
            this specifies the minimum and maximum log10-counts for a droplet to be considered empty.
        cell_counts_range: If ``data_raw`` is ``None``, i.e. ``data`` contains the unfiltered data,
            this specifies the minimum and maximum log10-counts for a droplet to be considered not empty.
        add_layer: Whether to add a ``'dsb'`` layer instead of assigning to the X matrix.
        random_state: Random seed.

    Returns:
        ``None`` if ``data_raw`` is not ``None`` (in this case the normalized data are written directly
        to ``data``), otherwise a ``MuData`` object containing filtered data (non-empty droplets).
    """
    toreturn = None
    if data_raw is None:
        if empty_counts_range is None or cell_counts_range is None:
            raise ValueError(
                "data_raw is None, assuming data is the unfiltered object, but no count ranges provided"
            )
        if max(*empty_counts_range) > min(*cell_counts_range):
            raise ValueError("overlapping count ranges")
        if not isinstance(data, MuData) or "prot" not in data.mod or "rna" not in data.mod:
            raise TypeError(
                "No data_raw given, assuming data is the unfiltered object, but data is not MuData"
                " or does not contain 'prot' and 'rna' modalities"
            )
        if data.mod["rna"].n_obs != data.mod["prot"].n_obs:
            raise ValueError("different numbers of cells in 'rna' and 'prot' modalities.")

        log10umi = np.log10(np.asarray(data.mod["rna"].X.sum(axis=1)).squeeze() + 1)
        empty_idx = np.where(
            (log10umi >= min(*empty_counts_range)) & (log10umi < max(*empty_counts_range))
        )[0]
        cell_idx = np.where(
            (log10umi >= min(*cell_counts_range)) & (log10umi < max(*cell_counts_range))
        )[0]
        cellidx = data.mod["prot"].obs_names[cell_idx]
        empty = data.mod["prot"][empty_idx, :]

        data = data[cellidx, :].copy()
        cells = data.mod["prot"]

        toreturn = data

    elif isinstance(data_raw, AnnData):
        empty = data_raw
    elif isinstance(data_raw, MuData) and "prot" in data_raw.mod:
        empty = data_raw["prot"]
    else:
        raise TypeError("data_raw must be an AnnData or a MuData object with 'prot' modality")

    if isinstance(data, AnnData):
        cells = data
    elif isinstance(data, MuData) and "prot" in data.mod:
        cells = data["prot"]
    else:
        raise TypeError("data must be an AnnData or a MuData object with 'prot' modality")

    if pseudocount < 0:
        raise ValueError("pseudocount cannot be negative")

    if cells.shape[1] != empty.shape[1]:  # this should only be possible if mudata_raw != None
        raise ValueError("data and data_raw have different numbers of proteins")

    if empty_counts_range is None:  # mudata_raw != None
        warn(
            "empty_counts_range values are not provided, treating all the non-cells as empty droplets"
        )
        empty = empty[~empty.obs_names.isin(cells.obs_names)]
    else:
        if data_raw is not None:
            if not isinstance(data_raw, MuData) or "rna" not in data_raw.mod:
                warn(
                    "data_raw must be a MuData object with 'rna' modality, ignoring empty_counts_range and treating all the non-cells as empty droplets"
                )
                empty = empty[~empty.obs_names.isin(cells.obs_names)]
            else:
                # data_raw is a MuData with 'rna' modality and empty_counts_range values are provided
                log10umi = np.log10(np.asarray(data_raw.mod["rna"].X.sum(axis=1)).squeeze() + 1)
                bc_umis = pd.DataFrame({"log10umi": log10umi}, index=data_raw.mod["rna"].obs_names)
                empty_droplets = bc_umis.query(
                    f"log10umi >= {min(*empty_counts_range)} & log10umi < {max(*empty_counts_range)}"
                ).index.values

                empty_len_orig = len(empty_droplets)
                empty_droplets = np.array([i for i in empty_droplets if i not in cells.obs_names])
                empty_len = len(empty_droplets)
                if empty_len != empty_len_orig:
                    warn(
                        f"Dropping {empty_len_orig - empty_len} empty droplets as they are already defined as cells"
                    )
                empty = empty[empty_droplets].copy()

    if data_raw is not None and cell_counts_range is not None:
        warn("cell_counts_range values are ignored since cells are provided in data")

    empty_scaled = (
        np.log(empty.X + pseudocount)
        if not issparse(empty.X)
        else np.log(empty.X.toarray() + pseudocount)
    )
    cells_scaled = (
        np.log(cells.X + pseudocount)
        if not issparse(cells.X)
        else np.log(cells.X.toarray() + pseudocount)
    )

    cells_scaled = (cells_scaled - empty_scaled.mean(axis=0)) / empty_scaled.std(axis=0)

    if denoise_counts:
        bgmeans = np.empty(cells_scaled.shape[0], np.float32)
        # init_params needs to be random, otherwise fitted variance for one of the n_components
        # sometimes goes to 0
        sharedvar = GaussianMixture(
            n_components=2, covariance_type="tied", init_params="random", random_state=random_state
        )
        separatevar = GaussianMixture(
            n_components=2, covariance_type="full", init_params="random", random_state=random_state
        )
        for c in range(cells_scaled.shape[0]):
            sharedvar.fit(cells_scaled[c, :, np.newaxis])
            separatevar.fit(cells_scaled[c, :, np.newaxis])

            if sharedvar.bic(cells_scaled[c, :, np.newaxis]) < separatevar.bic(
                cells_scaled[c, :, np.newaxis]
            ):
                bgmeans[c] = np.min(sharedvar.means_)
            else:
                bgmeans[c] = np.min(separatevar.means_)

        if isotype_controls is not None:
            pca = PCA(n_components=1, whiten=True)
            ctrl_idx = np.where(cells.var_names.isin(set(isotype_controls)))[0]
            if len(ctrl_idx) < len(isotype_controls):
                warn("Some isotype controls are not present in the data.")
            covar = pca.fit_transform(
                np.hstack((cells_scaled[:, ctrl_idx], bgmeans.reshape(-1, 1)))
            )
        else:
            covar = bgmeans[:, np.newaxis]

        reg = LinearRegression(fit_intercept=True, copy_X=False)
        reg.fit(covar, cells_scaled)

        cells_scaled -= reg.predict(covar) - reg.intercept_
    if add_layer:
        cells.layers["dsb"] = cells_scaled
    else:
        cells.X = cells_scaled
    return toreturn


def clr(adata: AnnData, inplace: bool = True) -> Union[None, AnnData]:
    """
    Apply the centered log ratio (CLR) transformation
    to normalize counts in adata.X.

    Args:
        data: AnnData object with protein expression counts.
        inplace: Whether to update adata.X inplace.
    """
    sparse = False
    if issparse(adata.X):
        sparse = True
    # Geometric mean of ADT counts
    x = adata.X
    g_mean = np.exp(np.log1p(x).sum(axis=0) / x.shape[0])
    # Centered log ratio
    clr = np.log1p(x / g_mean)

    if sparse:
        clr = csr_matrix(clr)

    if not inplace:
        adata = adata.copy()

    adata.X = clr

    return None if inplace else adata
