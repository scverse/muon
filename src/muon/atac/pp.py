from warnings import warn

import numpy as np
from anndata import AnnData
from mudata import MuData
from scanpy._utils import view_to_actual
from scipy.sparse import csr_array, dia_array, issparse, sparray, spmatrix

# Computational methods for preprocessing


def tfidf(
    data: AnnData | MuData,
    log_tf: bool = True,
    log_idf: bool = True,
    log_tfidf: bool = False,
    scale_factor: float = 1e4,
    inplace: bool = True,
    copy: bool = False,
    from_layer: str | None = None,
    to_layer: str | None = None,
) -> np.ndarray | spmatrix | AnnData | None:
    """Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).

    TF: peak counts are normalised by total number of counts per cell
    DF: total number of counts for each peak
    IDF: number of cells divided by DF

    By default, log(TF) * log(IDF) is returned.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        log_idf: Log-transform IDF term.
        log_tf: Log-transform TF term.
        log_tfidf: Log-transform TF*IDF term. Can only be used when log_tf and log_idf are `False`.
        scale_factor: Scale factor to multiply the TF-IDF matrix by.
        inplace: Whether to modify counts in the AnnData object.
        copy: Whether to return a copy of the AnnData object or the 'atac' modality. Not compatible
            with `inplace=False`.
        from_layer: Layer to use as input (`AnnData.layers[from_layer]`). Defaults to `AnnData.X`.
        to_layer: Layer to save transformed counts to (`AnnData.layers[to_layer]`). Defaults to `AnnData.X`.
            Not compatible with `inplace=False`.

    Returns:
        A matrix of transformed values if `inplace=False`, otherwise an AnnData object if `copy=True` or `None`.
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

    counts: np.ndarray | spmatrix = adata.X if from_layer is None else adata.layers[from_layer]
    if counts is None:
        raise ValueError("Expected a count matrix, but none was found")

    # Check before the computation
    if to_layer is not None and to_layer in adata.layers:
        warn(f"Existing layer '{str(to_layer)}' will be overwritten", stacklevel=2)

    if issparse(counts):
        n_peaks: np.ndarray | dia_array = np.asarray(counts.sum(axis=1)).reshape(-1)
        # This prevents making TF dense
        n_peaks = dia_array((1.0 / n_peaks, 0), shape=(n_peaks.size, n_peaks.size))
        tf = n_peaks @ counts
    else:
        n_peaks = np.asarray(counts.sum(axis=1)).reshape(-1, 1)
        tf = counts / n_peaks

    if scale_factor is not None and scale_factor != 0 and scale_factor != 1:
        tf = tf * scale_factor
    if log_tf:
        tf = np.log1p(tf)

    idf = np.asarray(adata.shape[0] / counts.sum(axis=0)).reshape(-1)
    if log_idf:
        idf = np.log1p(idf)

    if issparse(tf):
        idf = dia_array((idf, 0), shape=(idf.size, idf.size))
        tf_idf = tf @ idf
    else:
        tf_idf = csr_array(tf) @ csr_array(np.diag(idf))

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

    return None


def binarize(data: AnnData | MuData) -> None:
    """Transform peak counts to the binary matrix (all the non-zero values become 1).

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    counts: np.ndarray | spmatrix = adata.X
    if counts is None:
        raise ValueError("Expected a count matrix, but none was found")

    if issparse(counts):
        # Sparse matrix
        counts.data[counts.data != 0] = 1
    else:
        counts[counts != 0] = 1


def scopen(
    data: AnnData | MuData,
    n_components: int = 30,
    max_iter: int = 500,
    min_rho: float = 0.0,
    max_rho: float = 0.5,
    alpha: int = 1,
    verbose: bool = False,
) -> None:
    """Run scOpen (Li et al., 2019, https://doi.org/10.1101/865931) on the count matrix.

    This function follows the original implementation of the main method
    (https://github.com/CostaLab/scopen/blob/master/scopen/Main.py)
    adapting it for AnnDaata and MuData formats.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        n_components: Number of components of the matrix factorisation.
        max_iter: Number of iterations for the optimisation.
        min_rho: Lower bound of the per-cell dropout rate that the number of open regions is scaled to.
        max_rho: Upper bound of the per-cell dropout rate that the number of open regions is scaled to.
        alpha: Parameter for model regularisation to prevent from over-fitting.
        verbose: If to print the progress of the matrix factorisation.
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
        ) from None

    start = time.time()

    x: np.ndarray | spmatrix = adata.X
    if x is None:
        raise ValueError("Expected a count matrix, but none was found")

    # Make a dense matrix if it's sparse
    counts = x.toarray() if isinstance(x, spmatrix | sparray) else x
    counts = np.greater(counts, 0).T

    (m, n) = counts.shape

    n_open_regions = np.log10(counts.sum(axis=0))
    max_n_open_regions = np.max(n_open_regions)
    min_n_open_regions = np.min(n_open_regions)

    print(f"Number of peaks: {m}\nNumber of cells: {n}")
    print(f"Number of non-zeros before imputation: {np.count_nonzero(counts)}")

    rho = min_rho + (max_rho - min_rho) * (max_n_open_regions - n_open_regions) / (
        max_n_open_regions - min_n_open_regions
    )

    counts = counts[:, :] * (1 / (1 - rho))

    # Run bounded non-negative matrix factorisation
    w_hat, h_hat, _ = non_negative_factorization(
        X=counts, n_components=n_components, alpha=alpha, max_iter=max_iter, verbose=int(verbose)
    )

    del counts

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
    print(f"[total time: {int(h)}h {int(m)}m {int(s)}s]")
