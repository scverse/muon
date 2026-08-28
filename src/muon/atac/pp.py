from typing import Literal
from warnings import warn

import numpy as np
from anndata import AnnData
from mudata import MuData
from scanpy._utils import view_to_actual
from scipy.sparse import csr_array, csr_matrix, dia_array, issparse, spmatrix

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
        log_tf: Log-transform TF term.
        log_idf: Log-transform IDF term.
        log_tfidf: Log-transform TF*IDF term. Can only be used when log_tf and log_idf are `False`.
        scale_factor: Scale factor to multiply the TF matrix by.
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
    copy: bool = False,
    from_layer: str | None = None,
    to_layer: str | None = None,
    impute: bool = True,
    random_state: int | np.random.RandomState | None = 0,
    alpha: float = 1,
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar"] = "nndsvd",
    verbose: bool = False,
) -> None:
    """Run scOpen :cite:p:`pmid34737275` on the count matrix.

    This function follows the original implementation of the main method
    (https://github.com/CostaLab/scopen/blob/master/scopen/Main.py)
    adapting it for AnnData and MuData formats.

    Args:
        data: AnnData object with peak counts or multimodal MuData object with 'atac' modality.
        n_components: Number of components of the matrix factorisation.
        max_iter: Number of iterations for the optimisation.
        copy: Whether to return a copy of the AnnData object or the 'atac' modality.
        from_layer: Layer to use as input. Defaults to the `.X` matrix.
        to_layer: Layer to save imputed counts to. Defaults to the `.X` matrix. Ignored if `impute=False`.
        impute: Whether to impute the data based on the NMF results. Setting this to `True` may cause excessive memory use.
        random_state: Random number generator seed.
        alpha: Parameter for model regularisation to prevent from over-fitting.
        init: Method used to initialize the procedure.
        verbose: Whether to print the progress of the matrix factorisation.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    try:
        import time

        from scopen.Main import run_nmf, tf_idf_transform
    except ImportError:
        raise ImportError(
            "scOpen is not available. Install scOpen from PyPI (`pip install scopen`) \
            or from GitHub (`pip install git+https://github.com/CostaLab/scopen`)"
        ) from None

    start = time.time()
    if copy:
        adata = adata.copy()

    if to_layer is not None and to_layer in adata.layers:
        warn(f"Existing layer '{str(to_layer)}' will be overwritten", stacklevel=2)

    x: np.ndarray | spmatrix = adata.X if from_layer is None else adata.layers[from_layer]
    if x is None:
        raise ValueError("Expected a count matrix, but none was found")

    x = csr_matrix((x > 0).T)  # scopen not compatible with sparray

    (m, n) = x.shape
    nnz = x.count_nonzero()

    print(f"Number of peaks: {m}\nNumber of cells: {n}")
    print(f"Number of non-zeros before imputation: {nnz}")
    print(f"Sparsity: {1 - nnz / (m * n)}")

    x = tf_idf_transform(x)
    w_hat, h_hat, _ = run_nmf((x, n_components, alpha, max_iter, int(verbose), random_state, init))
    del x

    # Save results in the AnnData object
    adata.obsm["X_scopen"] = h_hat.T
    adata.varm["scopen"] = w_hat

    if impute:
        # Calculate imputed matrix
        m_hat = np.dot(w_hat, h_hat).astype(np.float32).T
        np.clip(m_hat, 0, 1, out=m_hat)
        if to_layer is None:
            adata.X = m_hat
        else:
            adata.layers[to_layer] = m_hat

    # Output time stats
    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    print(f"[total time: {int(h)}h {int(m)}m {int(s)}s]")
