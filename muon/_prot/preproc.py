from typing import Optional, Iterable, Tuple, Union
from numbers import Integral, Real
from warnings import warn

import numpy as np
from scipy.sparse import issparse
from anndata import AnnData

from .. import MuData


def dsb(
    mdata: Union[AnnData, MuData],
    mdata_raw: Optional[Union[AnnData, MuData]] = None,
    pseudocount: Integral = 10,
    denoise_counts: bool = True,
    isotype_controls: Optional[Iterable[str]] = None,
    empty_counts_range: Optional[Tuple[Real, Real]] = None,
    cell_counts_range: Optional[Tuple[Real, Real]] = None,
    random_state: Optional[Union[int, np.random.RandomState, None]] = None,
) -> Union[None, MuData]:
    """
    Normalize protein expression with DSB (Denoised and Scaled by Background)

    Normalized data will be written to ``mdata`` (if it is an AnnData object) or ``mdata.mod['cite']``
    (if it is a MuData object) as a new layer named ``dsb``.

    References:
        Mulè et al, 2020 (`doi:10.1101/2020.02.24.963603 <https://dx.doi.org/10.1101/2020.02.24.963603>`_)

    Args:
        mdata: AnnData object with protein expression counts or MuData object with ``cite`` modality.
            If ``mdata_raw`` is ``None``, must be a ``MuData`` object containing raw (unfiltered,
            including empty droplets) data for both ``cite`` and ``rna`` modalities. If ``mdata_raw``
            is not ``None``, must contain filtered (non-empty droplets) data.
        mdata_raw: AnnData object with protein expression counts or MuData object with 'cite' modality
            containing raw (unfiltered, including empty droplets) data.
        pseudocount: Pseudocount to add before log-transform.
        denoise_counts: Whether to perform denoising.
        isotype_controls: Names of the isotype controls. If ``None``, isotype controls will not be used.
        empty_counts_range: If ``mdata_raw`` is ``None``, i.e. ``mdata`` contains the unfiltered data,
            this specifies the minimum and maximum log10-counts for a droplet to be considered empty.
        cell_counts_range: If ``mdata_raw`` is ``None``, i.e. ``mdata`` contains the unfiltered data,
            this specifies the minimum and maximum log10-counts for a droplet to be considered not empty.
        random_state: Random seed.

    Returns:
        ``None`` if ``mdata_raw`` is not ``None`` (in this case the normalized data are written directly
        to ``mdata``), otherwise a ``MuData`` object containing filtered data (non-empty droplets).
    """
    toreturn = None
    if mdata_raw is None:
        if empty_counts_range is None or cell_counts_range is None:
            raise ValueError(
                "mdata_raw is None, assuming mdata is the unfiltered object, no count ranges provided"
            )
        if max(*empty_counts_range) > min(*cell_counts_range):
            raise ValueError("overlapping count ranges")
        if not isinstance(mdata, MuData) or "cite" not in mdata.mod or "rna" not in mdata.mod:
            raise TypeError(
                "No mdata_raw given, assuming mdata is the unfiltered object, but mdata is not MuData"
                " or does not contain 'cite' and 'rna' modalities"
            )
        if mdata.mod["rna"].n_obs != mdata.mod["cite"].n_obs:
            raise ValueError("different numbers of cells in 'rna' and 'cite' modalities.")

        log10umi = np.log10(np.asarray(mdata.mod["rna"].X.sum(axis=1)).squeeze() + 1)
        empty_idx = np.where(
            (log10umi >= min(*empty_counts_range)) & (log10umi < max(*empty_counts_range))
        )[0]
        cell_idx = np.where(
            (log10umi >= min(*cell_counts_range)) & (log10umi < max(*cell_counts_range))
        )[0]
        cellidx = mdata.mod["cite"].obs_names[cell_idx]
        empty = mdata.mod["cite"][empty_idx, :]

        mdata = mdata[cellidx, :].copy()
        cells = mdata.mod["cite"]

        toreturn = mdata

    elif isinstance(mdata_raw, AnnData):
        empty = mdata_raw
    elif isinstance(mdata_raw, MuData) and "cite" in mdata_raw.mod:
        empty = mdata_raw["cite"]
    else:
        raise TypeError("mdata_raw must be an AnnData or a MuData object with 'cite' modality")

    if isinstance(mdata, AnnData):
        cells = mdata
    elif isinstance(mdata, MuData) and "cite" in mdata.mod:
        cells = mdata["cite"]
    else:
        raise TypeError("mdata must be an AnnData or a MuData object with 'cite' modality")

    if pseudocount < 0:
        raise ValueError("pseudocount cannot be negative")

    if cells.shape[1] != empty.shape[1]:  # this should only be possible if mudata_raw != None
        raise ValueError("mdata and mdata_raw have different numbers of proteins")

    empty = empty[~empty.obs_names.isin(cells.obs_names)]

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
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LinearRegression
        except ImportError:
            raise ImportError("sklearn package not found. Install the sklearn package to denoise.")

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
            covar = pca.fit_transform(np.hstack(cells_scaled[:, ctrl_idx], bgmeans))
        else:
            covar = bgmeans[:, np.newaxis]

        reg = LinearRegression(fit_intercept=True, copy_X=False)
        reg.fit(covar, cells_scaled)

        cells_scaled -= reg.predict(covar) - reg.intercept_
    cells.layers["dsb"] = cells_scaled
    return toreturn
