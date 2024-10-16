from typing import Union, Optional
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, dia_matrix, issparse

from anndata import AnnData
from mudata import MuData

from scanpy._utils import view_to_actual


# Computational methods for preprocessing


def tfidf(
    data: Union[AnnData, MuData],
    log_tf: bool = True,
    log_idf: bool = True,
    log_tfidf: bool = False,
    scale_factor: Union[int, float] = 1e4,
    inplace: bool = True,
    copy: bool = False,
    from_layer: Optional[str] = None,
    to_layer: Optional[str] = None,
):
    """
    Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).

    TF: peak counts are normalised by total number of counts per cell
    DF: total number of counts for each peak
    IDF: number of cells divided by DF

    By default, log(TF) * log(IDF) is returned.

    Parameters
    ----------
    data
            AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    log_idf
            Log-transform IDF term (True by default).
    log_tf
            Log-transform TF term (True by default).
    log_tfidf
            Log-transform TF*IDF term (False by default).
            Can only be used when log_tf and log_idf are False.
    scale_factor
            Scale factor to multiply the TF-IDF matrix by (1e4 by default).
    inplace
            If to modify counts in the AnnData object (True by default).
    copy
            If to return a copy of the AnnData object or the 'atac' modality (False by default).
            Not compatible with inplace=False.
    from_layer
            Layer to use counts (AnnData.layers[from_layer])
            instead of AnnData.X used by default.
    to_layer
            Layer to save transformed counts to (AnnData.layers[to_layer])
            instead of AnnData.X used by default.
            Not compatible with inplace=False.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if log_tfidf and (log_tf or log_idf):
        raise AttributeError(
            "When returning log(TF*IDF), \
            applying neither log(TF) nor log(IDF) is possible."
        )

    if copy and not inplace:
        raise ValueError("`copy=True` cannot be used with `inplace=False`.")

    if to_layer is not None and not inplace:
        raise ValueError(f"`to_layer='{str(to_layer)}'` cannot be used with `inplace=False`.")

    if copy:
        adata = adata.copy()

    view_to_actual(adata)

    counts = adata.X if from_layer is None else adata.layers[from_layer]

    # Check before the computation
    if to_layer is not None and to_layer in adata.layers:
        warn(f"Existing layer '{str(to_layer)}' will be overwritten")

    if issparse(counts):
        n_peaks = np.asarray(counts.sum(axis=1)).reshape(-1)
        n_peaks = dia_matrix((1.0 / n_peaks, 0), shape=(n_peaks.size, n_peaks.size))
        # This prevents making TF dense
        tf = np.dot(n_peaks, counts)
    else:
        n_peaks = np.asarray(counts.sum(axis=1)).reshape(-1, 1)
        tf = adata.X / n_peaks

    if scale_factor is not None and scale_factor != 0 and scale_factor != 1:
        tf = tf * scale_factor
    if log_tf:
        tf = np.log1p(tf)

    idf = np.asarray(adata.shape[0] / adata.X.sum(axis=0)).reshape(-1)
    if log_idf:
        idf = np.log1p(idf)

    if issparse(tf):
        idf = dia_matrix((idf, 0), shape=(idf.size, idf.size))
        tf_idf = np.dot(tf, idf)
    else:
        tf_idf = np.dot(csr_matrix(tf), csr_matrix(np.diag(idf)))

    if log_tfidf:
        tf_idf = np.log1p(tf_idf)

    res = np.nan_to_num(tf_idf, nan=0.0)
    if not inplace:
        return res

    if to_layer is not None:
        adata.layers[to_layer] = res
    else:
        adata.X = res

    if copy:
        return adata


def binarize(data: Union[AnnData, MuData]):
    """
    Transform peak counts to the binary matrix (all the non-zero values become 1).

    Parameters
    ----------
    data
            AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    if issparse(adata.X):
        # Sparse matrix
        adata.X.data[adata.X.data != 0] = 1
    else:
        adata.X[adata.X != 0] = 1


def scopen(
    data: Union[AnnData, MuData],
    n_components: int = 30,
    max_iter: int = 500,
    min_rho: float = 0.0,
    max_rho: float = 0.5,
    alpha: int = 1,
    verbose: bool = False,
):
    """
    Run scOpen (Li et al., 2019, https://doi.org/10.1101/865931) on the count matrix

    This function follows the original implementation of the main method
    (https://github.com/CostaLab/scopen/blob/master/scopen/Main.py)
    adapting it for AnnDaata and MuData formats.

    Parameters
    ----------
    data
            AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    try:
        import time
        from scopen.MF import non_negative_factorization
    except ImportError:
        raise ImportError(
            "scOpen is not available. Install scOpen from PyPI (`pip install scopen`) \
            or from GitHub (`pip install git+https://github.com/CostaLab/scopen`)"
        )

    start = time.time()

    data = adata.X.T
    # Make a dense matrix if it's sparse
    if callable(getattr(data, "toarray", None)):
        data = data.toarray()

    data = np.greater(data, 0)

    (m, n) = data.shape

    n_open_regions = np.log10(data.sum(axis=0))
    max_n_open_regions = np.max(n_open_regions)
    min_n_open_regions = np.min(n_open_regions)

    print(f"Number of peaks: {m}\nNumber of cells: {n}")
    print(f"Number of non-zeros before imputation: {np.count_nonzero(data)}")

    rho = min_rho + (max_rho - min_rho) * (max_n_open_regions - n_open_regions) / (
        max_n_open_regions - min_n_open_regions
    )

    data = data[:, :] * (1 / (1 - rho))

    # Run bounded non-negative matrix factorisation
    w_hat, h_hat, _ = non_negative_factorization(
        X=data, n_components=n_components, alpha=alpha, max_iter=max_iter, verbose=int(verbose)
    )

    del data

    # Calculate imputed matrix
    m_hat = np.dot(w_hat, h_hat)
    np.clip(m_hat, 0, 1, out=m_hat)

    # Save results in the AnnData object
    adata.obsm["X_scopen"] = h_hat.T
    adata.varm["scopen"] = w_hat
    adata.X = m_hat.T

    # Output time stats
    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    print("[total time: ", "%dh %dm %ds" % (h, m, s), "]")
