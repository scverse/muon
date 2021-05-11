from typing import Union, Callable, Optional, Sequence, Dict, Iterable
from functools import reduce
import warnings
from collections import OrderedDict
from itertools import repeat

import numpy as np
from scipy.sparse import csr_matrix, issparse, SparseEfficiencyWarning
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.utils import check_random_state

from anndata import AnnData
from scanpy import logging
from scanpy.tools._utils import _choose_representation
from scanpy.neighbors import _compute_connectivities_umap
from scanpy._compat import Literal
from umap.distances import euclidean
from umap.sparse import sparse_euclidean, sparse_jaccard
from umap.umap_ import nearest_neighbors
from numba import njit, prange

from .._core.mudata import MuData

# Computational methods for preprocessing

_euclidean = njit(euclidean.py_func, inline="always", fastmath=True)
_sparse_euclidean = njit(sparse_euclidean.py_func, inline="always")
_sparse_jaccard = njit(sparse_jaccard.py_func, inline="always")


@njit
def _jaccard_euclidean_metric(
    x: int,
    y: int,
    X: np.ndarray,
    neighbors_indices: np.ndarray,
    neighbors_indptr: np.ndarray,
    neighbors_data: np.ndarray,
    N: int,
    bbox_norm: float,
):
    x = int(x[0])  # this is for compatibility with pynndescent
    y = int(y[0])  # pynndescent converts the data to float32
    if x == y:
        return N + 1.0

    from_inds = neighbors_indices[neighbors_indptr[x] : neighbors_indptr[x + 1]]
    from_data = neighbors_data[neighbors_indptr[x] : neighbors_indptr[x + 1]]
    to_inds = neighbors_indices[neighbors_indptr[y] : neighbors_indptr[y + 1]]
    to_data = neighbors_data[neighbors_indptr[y] : neighbors_indptr[y + 1]]
    jac = _sparse_jaccard(from_inds, from_data, to_inds, to_data)

    if jac < 1.0:
        return (N - jac * N) + (bbox_norm - _euclidean(X[x, :], X[y, :])) / bbox_norm
    else:
        return N + 1.0


@njit
def _jaccard_sparse_euclidean_metric(
    x: int,
    y: int,
    X_indices: np.ndarray,
    X_indptr: np.ndarray,
    X_data: np.ndarray,
    neighbors_indices: np.ndarray,
    neighbors_indptr: np.ndarray,
    neighbors_data: np.ndarray,
    N: int,
    bbox_norm: float,
):
    x = int(x[0])  # this is for compatibility with pynndescent
    y = int(y[0])  # pynndescent converts the data to float32
    if x == y:
        return N + 1.0

    from_inds = X_indices[X_indptr[x] : X_indptr[x + 1]]
    from_data = X_data[X_indptr[x] : X_indptr[x + 1]]
    to_inds = X_indices[X_indptr[y] : X_indptr[y + 1]]
    to_data = X_data[X_indptr[y] : X_indptr[y + 1]]
    jac = _sparse_jaccard(from_inds, from_data, to_inds, to_data)

    if jac < 1.0:
        euclidean = _sparse_euclidean(from_inds, from_data, to_inds, to_data)
        return (N - jac * N) + (bbox_norm - euclidean) / bbox_norm
    else:
        return N + 1.0


@njit(parallel=True)
def _sparse_csr_fast_knn_(
    N: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n_neighbors: int
):
    knn_indptr = np.arange(N * n_neighbors, step=n_neighbors, dtype=indptr.dtype)
    knn_indices = np.zeros((N * n_neighbors,), dtype=indices.dtype)
    knn_data = np.zeros((N * n_neighbors,), dtype=data.dtype)

    for i in prange(indptr.size - 1):
        start = indptr[i]
        end = indptr[i + 1]
        cols = indices[start:end]
        rowdata = data[start:end]

        # would like to use argpartition, but not supported by numba
        idx = np.argsort(rowdata)
        startidx = i * n_neighbors
        endidx = (i + 1) * n_neighbors
        # numba's parallel loops only support reductions, not assignment
        knn_indices[startidx:endidx] += cols[idx[:n_neighbors]]
        knn_data[startidx:endidx] += rowdata[idx[:n_neighbors]]
    return knn_data, knn_indices, knn_indptr


# numba doesn't know about SciPy
def _sparse_csr_fast_knn(X: csr_matrix, n_neighbors: int):
    data, indices, indptr = _sparse_csr_fast_knn_(
        X.shape[0], X.indptr, X.indices, X.data, n_neighbors
    )
    indptr = np.concatenate((indptr, (indices.size,)))
    return csr_matrix((data, indices, indptr), X.shape)


@njit(parallel=True)
def _sparse_csr_ptp_(N: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray):
    minelems = np.zeros((N,), dtype=data.dtype)
    maxelems = np.zeros((N,), dtype=data.dtype)
    for row in range(indptr.size - 1):
        cols = indices[indptr[row] : indptr[row + 1]]
        minelems[cols] = np.minimum(minelems[cols], data[indices[indptr[row] : indptr[row + 1]]])
        maxelems[cols] = np.maximum(maxelems[cols], data[indices[indptr[row] : indptr[row + 1]]])
    return maxelems - minelems


def _sparse_csr_ptp(X: csr_matrix):
    return _sparse_csr_ptp_(X.shape[1], X.indptr, X.indices, X.data)


def _make_slice_intervals(idx, maxsize=10000):
    bins = np.concatenate(((-1,), np.where(np.diff(idx) > 1)[0], (idx.size - 1,)))
    allstarts, allstops = [], []
    for start, stop in zip(bins[:-1] + 1, bins[1:]):
        size = stop - start
        if size > maxsize:
            nbins = size // maxsize
            starts = np.arange(nbins) * maxsize + start
            stops = np.concatenate((starts[1:], (size,)))
            allstarts.append(starts)
            allstops.append(stops)
        else:
            allstarts.append((start,))
            allstops.append((stop,))
    return np.concatenate(allstarts), np.concatenate(allstops)


def _l2norm(
    adata: AnnData, rep: Optional[Union[Iterable[str], str]] = None, n_pcs: Optional[int] = 0
):
    X = _choose_representation(adata, rep, n_pcs)
    norm = X / np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    norm[~np.isfinite(norm)] = 0
    X.astype(norm.dtype, copy=False)
    X[:] = norm


def l2norm(
    mdata: Union[MuData, AnnData],
    mod: Optional[Union[Iterable[str], str]] = None,
    rep: Optional[Union[Iterable[str], str]] = None,
    n_pcs: Optional[Union[Iterable[int], int]] = 0,
    copy: bool = False,
) -> Optional[Union[MuData, AnnData]]:
    """
    Normalize observations to unit L2 norm.

    Args:
        mdata: The MuData or AnnData object to normalize.
        mod: If ``mdata`` is a MuData object, this specifies the modalities to normalize.
            ``None`` indicates all modalities.
        rep: The representation to normalize. ``X`` or any key for ``.obsm`` is valid. If
            ``None``, the representation is chosen automatically. If ``mdata`` is a MuData
            object and this is not an iterable, the given representation will be used for
            all modalities.
        n_pcs: The number of principal components to use. This affects the result only if
            a PCA representation is being normalized. If ``mdata`` is a MuData object and
            this is not an iterable, the given number will be used for all modalities.
        copy: Return a copy instead of writing to `mdata`.

    Returns: Depending on ``copy``, returns or updates ``mdata``.
    """
    if isinstance(mdata, AnnData):
        if rep is not None and not isinstance(rep, str):
            it = iter(rep)
            rep = next(it)
            try:
                next(it)
            except StopIteration as e:
                pass
            else:
                raise RuntimeError("If 'rep' is an Iterable, it must have length 1")
        if n_pcs is not None and isinstance(n_pcs, Iterable):
            it = iter(n_pcs)
            n_pcs = next(it)
            try:
                next(it)
            except StopIteration as e:
                pass
            else:
                raise RuntimeError("If 'n_pcs' is an Iterable, it must have length 1")
        if copy:
            mdata = mdata.copy()
        _l2norm(mdata, rep, n_pcs)
    else:
        if mod is None:
            mod = mdata.mod.keys()
        elif isinstance(mod, str):
            mod = [mod]
        if rep is None or isinstance(rep, str):
            rep = repeat(rep)
        if n_pcs is None or isinstance(n_pcs, int):
            n_pcs = repeat(n_pcs)
        if copy:
            mdata = mdata.copy()
        for m, r, n in zip(mod, rep, n_pcs):
            _l2norm(mdata.mod[m], r, n)

    return mdata if copy else None


def neighbors(
    mdata: MuData,
    n_neighbors: Optional[int] = None,
    n_bandwidth_neighbors: int = 20,
    n_multineighbors: int = 200,
    neighbor_keys: Optional[Dict[str, Optional[str]]] = None,
    metric: Literal[
        "euclidean",
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "hamming",
        "jaccard",
        "jensenshannon",
        "kulsinski",
        "mahalanobis",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "wminkowski",
        "yule",
    ] = "euclidean",
    low_memory: Optional[bool] = None,
    key_added: Optional[str] = None,
    weight_key: Optional[str] = "mod_weight",
    add_weights_to_modalities: bool = False,
    eps: float = 1e-4,
    copy: bool = False,
    random_state: Optional[Union[int, np.random.RandomState]] = 42,
) -> Optional[MuData]:
    """
    Multimodal nearest neighbor search.

    This implements the multimodal nearest neighbor method of Hao et al. and Swanson et al. The neighbor search
    efficiency on this heavily relies on UMAP. In particular, you may want to decrease n_multineighbors for large
    data set to avoid excessive peak memory use. Note that to achieve results as close as possible to the Seurat
    implementation, observations must be normalized to unit L2 norm (see :func:`l2norm`) prior to running per-modality
    nearest-neighbor search.

    References:
        Hao et al, 2020 (`doi:10.1101/2020.10.12.335331 <https://dx.doi.org/10.1101/2020.10.12.335331>`_)
        Swanson et al, 2020 (`doi:10.1101/2020.09.04.283887 <https://dx.doi.org/10.1101/2020.09.04.283887>`_)

    Args:
        mdata: MuData object. Per-modality nearest neighbor search must have already been performed for all modalities
            that are to be used for multimodal nearest neighbor search.
        n_neighbors: Number of nearest neighbors to find. If ``None``, will be set to the arithmetic mean of per-modality
            neighbors.
        n_bandwidth_neighbors: Number of nearest neighbors to use for bandwidth selection.
        n_multineighbors: Number of nearest neighbors in each modality to consider as candidates for multimodal nearest
            neighbors. Only points in the union of per-modality nearest neighbors are candidates for multimodal nearest
            neighbors. This will use the same metric that was used for the nearest neighbor search in the respective modality.
        neighbor_keys: Keys in .uns where per-modality neighborhood information is stored. Defaults to ``"neighbors"``.
            If set, only the modalities present in ``neighbor_keys`` will be used for multimodal nearest neighbor search.
        metric: Distance measure to use. This will only be used in the final step to search for nearest neighbors in the set
            of candidates.
        low_memory: Whether to use the low-memory implementation of nearest-neighbor descent. If not set, will default to True
            if the data set has more than 50 000 samples.
        key_added: If not specified, the multimodal neighbors data is stored in ``.uns["neighbors"]``, distances and
            connectivities are stored in ``.obsp["distances"]`` and ``.obsp["connectivities"]``, respectively. If specified, the
            neighbors data is added to ``.uns[key_added]``, distances are stored in ``.obsp[key_added + "_distances"]`` and
            connectivities in ``.obsp[key_added + "_connectivities"]``.
        weight_key: Weight key to add to each modality's ``.obs`` or to ``mdata.obs``. By default, it is ``"mod_weight"``.
        add_weights_to_modalities: If to add weights to individual modalities. By default, it is ``False``
            and the weights will be added to ``mdata.obs``.
        eps: Small number to avoid numerical errors.
        copy: Return a copy instead of writing to ``mdata``.
        random_state: Random seed.

    Returns: Depending on ``copy``, returns or updates ``mdata``. Cell-modality weights will be stored in
        ``.obs["modality_weight"]`` separately for each modality.
    """
    randomstate = check_random_state(random_state)
    mdata = mdata.copy() if copy else mdata
    if neighbor_keys is None:
        modalities = mdata.mod.keys()
        neighbor_keys = {}
    else:
        modalities = neighbor_keys.keys()
    neighbors_params = {}
    reps = {}
    observations = mdata.obs.index

    if low_memory or low_memory is None and observations.size > 50000:
        sparse_matrix_assign_splits = 10000
    else:
        sparse_matrix_assign_splits = 30000

    mod_neighbors = np.empty((len(modalities),), dtype=np.uint16)
    mod_reps = {}
    mod_n_pcs = {}
    for i, mod in enumerate(modalities):
        nkey = neighbor_keys.get(mod, "neighbors")
        try:
            nparams = mdata.mod[mod].uns[nkey]
        except KeyError:
            raise ValueError(
                f'Did not find .uns["{nkey}"] for modality "{mod}". Run `sc.pp.neighbors` on all modalities first.'
            )

        use_rep = nparams["params"].get("use_rep", None)
        n_pcs = nparams["params"].get("n_pcs", None)
        mod_neighbors[i] = nparams["params"].get("n_neighbors", 0)

        neighbors_params[mod] = nparams
        reps[mod] = _choose_representation(mdata.mod[mod], use_rep, n_pcs)
        mod_reps[mod] = (
            use_rep if use_rep is not None else -1
        )  # otherwise this is not saved to h5mu
        mod_n_pcs[mod] = n_pcs if n_pcs is not None else -1

    if n_neighbors is None:
        mod_neighbors = mod_neighbors[mod_neighbors > 0]
        n_neighbors = int(round(np.mean(mod_neighbors), 0))

    ratios = np.full((len(observations), len(modalities)), -np.inf, dtype=np.float64)
    sigmas = {}

    for i1, mod1 in enumerate(modalities):
        observations1 = observations.intersection(mdata.mod[mod1].obs.index)
        ratioidx = np.where(observations.isin(observations1))[0]
        nparams1 = neighbors_params[mod1]
        X = reps[mod1]
        neighbordistances = mdata.mod[mod1].obsp[nparams1["distances_key"]]
        nndistances = np.empty((neighbordistances.shape[0],), neighbordistances.dtype)
        # neighborsdistances is a sparse matrix, we can either convert to dense, or loop
        for i in range(neighbordistances.shape[0]):
            nndist = neighbordistances[i, :].data
            if nndist.size == 0:
                raise ValueError(
                    f"Cell {i} in modality {mod1} does not have any neighbors. "
                    "This could be due to subsetting after nearest neighbors calculation. "
                    "Make sure to subset before calculating nearest neighbors."
                )
            nndistances[i] = nndist.min()

        # We want to get the k-nn with the largest Jaccard distance, but break ties using
        # Euclidean distance. Largest Jaccard distance corresponds to lowest Jaccard index,
        # i.e. 1 - Jaccard distance. The naive version would be to compute pairwise Jaccard and
        # Euclidean distances for all points, but this is low and needs lots of memory. We
        # want to use an efficient k-nn algorithm, however no package that I know of supports
        # tie-breaking k-nn, so we use a custom distance. Jaccard index is always between 0 and 1
        # and has discrete increments of at least 1/N, where N is the number of data points.
        # If we scale the Jaccard indices by N, the minimum Jaccard index will be 1. If we scale
        # all Euclidean distances to be less than one, we can define a combined distance as the
        # sum of the scaled Jaccard index and one minus the Euclidean distances. This is not a
        # proper metric, but UMAP's nearest neighbor search uses NN-descent, which works with
        # arbitrary similarity measures.
        # The scaling factor for the Euclidean distance is given by the length of the diagonal
        # of the bounding box of the data. This can be computed in linear time by just taking
        # the minimal and maximal coordinates of each dimension.
        N = X.shape[0]
        bbox_norm = np.linalg.norm(_sparse_csr_ptp(X) if issparse(X) else np.ptp(X, axis=0), ord=2)
        lmemory = low_memory if low_memory is not None else N > 50000
        if issparse(X):
            X = X.tocsr()
            cmetric = _jaccard_sparse_euclidean_metric
            metric_kwds = OrderedDict(
                X_indices=X.indices,
                X_indptr=X.indptr,
                X_data=X.data,
                neighbors_indices=neighbordistances.indices,
                neighbors_indptr=neighbordistances.indptr,
                neighbors_data=neighbordistances.data,
                N=N,
                bbox_norm=bbox_norm,
            )
        else:
            cmetric = _jaccard_euclidean_metric
            metric_kwds = OrderedDict(
                X=X,
                neighbors_indices=neighbordistances.indices,
                neighbors_indptr=neighbordistances.indptr,
                neighbors_data=neighbordistances.data,
                N=N,
                bbox_norm=bbox_norm,
            )

        logging.info(f"Calculating kernel bandwidth for '{mod1}' modality...")
        nn_indices, _, _ = nearest_neighbors(
            np.arange(N)[:, np.newaxis],
            n_neighbors=n_bandwidth_neighbors,
            metric=cmetric,
            metric_kwds=metric_kwds,
            random_state=randomstate,
            angular=False,
            low_memory=lmemory,
        )

        csigmas = np.empty((N,), dtype=neighbordistances.dtype)
        if issparse(X):
            for i, neighbors in enumerate(nn_indices):
                csigmas[i] = cdist(
                    X[i : (i + 1), :].toarray(), X[neighbors, :].toarray(), metric="euclidean"
                ).mean()
        else:
            for i, neighbors in enumerate(nn_indices):
                csigmas[i] = cdist(X[i : (i + 1), :], X[neighbors, :], metric="euclidean").mean()

        currtheta = None
        thetas = np.full(
            (len(observations1), len(modalities) - 1), -np.inf, dtype=neighbordistances.dtype
        )

        lasti = 0

        logging.info(f"Calculating cell affinities for '{mod1} modality...")
        for i2, mod2 in enumerate(modalities):
            nparams2 = neighbors_params[mod2]
            neighbordistances = mdata.mod[mod2].obsp[nparams2["distances_key"]]
            observations2 = observations1.intersection(mdata.mod[mod2].obs.index)
            Xidx = np.where(observations1.isin(observations2))[0]
            r = np.empty(shape=(len(observations2), X.shape[1]), dtype=neighbordistances.dtype)
            # alternative to the loop would be broadcasting, but this requires converting the sparse
            # connectivity matrix to a dense ndarray and producing a temporary 3d array of size
            # n_cells x n_cells x n_genes => requires a lot of memory
            for i, cell in enumerate(Xidx):
                r[i, :] = np.asarray(
                    np.mean(X[neighbordistances[cell, :].nonzero()[1], :], axis=0)
                ).squeeze()

            theta = np.exp(
                -np.maximum(np.linalg.norm(X[Xidx, :] - r, ord=2, axis=-1) - nndistances[Xidx], 0)
                / (csigmas[Xidx] - nndistances[Xidx])
            )
            if i1 == i2:
                currtheta = theta
            else:
                thetas[:, lasti] = theta
                lasti += 1
        ratios[ratioidx, i1] = currtheta / (np.max(thetas, axis=1) + eps)
        sigmas[mod1] = csigmas

    weights = softmax(ratios, axis=1)
    neighbordistances = csr_matrix((mdata.n_obs, mdata.n_obs), dtype=np.float64)
    largeidx = mdata.n_obs ** 2 > np.iinfo(np.int32).max
    if largeidx:  # work around scipy bug https://github.com/scipy/scipy/issues/13155
        neighbordistances.indptr = neighbordistances.indptr.astype(np.int64)
        neighbordistances.indices = neighbordistances.indices.astype(np.int64)
    for i, m in enumerate(modalities):
        cmetric = neighbors_params[m].get("metric", "euclidean")
        observations1 = observations.intersection(mdata.mod[m].obs.index)

        rep = reps[m]
        lmemory = low_memory if low_memory is not None else rep.shape[0] > 50000
        logging.info(f"Calculating nearest neighbor candidates for '{m}' modality...")
        logging.debug(f"Using low_memory={lmemory} for '{m}' modality")
        nn_indices, distances, _ = nearest_neighbors(
            rep,
            n_neighbors=n_multineighbors + 1,
            metric=cmetric,
            metric_kwds={},
            random_state=randomstate,
            angular=False,
            low_memory=lmemory,
        )
        graph = csr_matrix(
            (
                distances[:, 1:].reshape(-1),
                nn_indices[:, 1:].reshape(-1),
                np.concatenate((nn_indices[:, 0] * n_multineighbors, (nn_indices[:, 1:].size,))),
            ),
            shape=(rep.shape[0], rep.shape[0]),
        )
        with warnings.catch_warnings():
            # CSR is faster here than LIL, no matter what SciPy says
            warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
            if observations1.size == observations.size:
                if neighbordistances.size == 0:
                    neighbordistances = graph
                else:
                    neighbordistances += graph
            # the naive version of neighbordistances[idx[:, np.newaxis], idx[np.newaxis, :]] += graph
            else:
                # uses way too much memory
                if largeidx:
                    graph.indptr = graph.indptr.astype(np.int64)
                    graph.indices = graph.indices.astype(np.int64)
                fullstarts, fullstops = _make_slice_intervals(
                    np.where(observations.isin(observations1))[0], sparse_matrix_assign_splits
                )
                modstarts, modstops = _make_slice_intervals(
                    np.where(mdata.mod[m].obs.index.isin(observations1))[0],
                    sparse_matrix_assign_splits,
                )

                for fullidxstart1, fullidxstop1, modidxstart1, modidxstop1 in zip(
                    fullstarts, fullstops, modstarts, modstops
                ):
                    for fullidxstart2, fullidxstop2, modidxstart2, modidxstop2 in zip(
                        fullstarts, fullstops, modstarts, modstops
                    ):
                        neighbordistances[
                            fullidxstart1:fullidxstop1, fullidxstart2:fullidxstop2
                        ] += graph[modidxstart1:modidxstop1, modidxstart2:modidxstop2]

    neighbordistances.data[:] = 0
    logging.info("Calculating multimodal nearest neighbors...")
    for i, m in enumerate(modalities):
        observations1 = observations.intersection(mdata.mod[m].obs.index)
        fullidx = np.where(observations.isin(observations1))[0]

        if weight_key:
            if add_weights_to_modalities:
                mdata.mod[m].obs[weight_key] = weights[fullidx, i]
            else:
                # mod_weight -> mod:mod_weight
                mdata.obs[":".join([m, weight_key])] = weights[fullidx, i]

        rep = reps[m]
        csigmas = sigmas[m]
        if issparse(rep):

            def neighdist(cell, nz):
                return -cdist(rep[cell, :].toarray(), rep[nz, :].toarray(), metric=metric)

        else:

            def neighdist(cell, nz):
                return -cdist(rep[np.newaxis, cell, :], rep[nz, :], metric=metric)

        for cell, j in enumerate(fullidx):
            row = slice(neighbordistances.indptr[cell], neighbordistances.indptr[cell + 1])
            nz = neighbordistances.indices[row]
            neighbordistances.data[row] += (
                np.exp(neighdist(cell, nz) / csigmas[cell]).squeeze() * weights[cell, i]
            )
    neighbordistances.data = np.sqrt(0.5 * (1 - neighbordistances.data))

    neighbordistances = _sparse_csr_fast_knn(neighbordistances, n_neighbors + 1)

    logging.info("Calculating connectivities...")
    _, connectivities = _compute_connectivities_umap(
        knn_indices=neighbordistances.indices.reshape(
            (neighbordistances.shape[0], n_neighbors + 1)
        ),
        knn_dists=neighbordistances.data.reshape((neighbordistances.shape[0], n_neighbors + 1)),
        n_obs=neighbordistances.shape[0],
        n_neighbors=n_neighbors + 1,
    )

    if key_added is None:
        key_added = "neighbors"
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = key_added + "_connectivities"
        dists_key = key_added + "_distances"
    neighbors_dict = {"connectivities_key": conns_key, "distances_key": dists_key}
    neighbors_dict["params"] = {
        "n_neighbors": n_neighbors,
        "n_multineighbors": n_multineighbors,
        "metric": metric,
        "eps": eps,
        "random_state": random_state,
        "use_rep": mod_reps,
        "n_pcs": mod_n_pcs,
        "method": "umap",
    }
    mdata.obsp[dists_key] = neighbordistances
    mdata.obsp[conns_key] = connectivities
    mdata.uns[key_added] = neighbors_dict

    mdata.update_obs()

    return mdata if copy else None


# Utility functions: intersecting observations


def intersect_obs(mdata: MuData):
    """
    Subset observations (samples or cells) in-place
    taking observations present only in all modalities.

    This function is currently a draft, and it can be removed
    or its behaviour might be changed in future.

    Parameters
    ----------
    mdata: MuData
            MuData object
    """

    if mdata.isbacked:
        warnings.warn(
            "MuData object is backed. It might be required to re-read the object with `backed=False` to make the intersection work."
        )

    common_obs = reduce(np.intersect1d, [m.obs_names for m in mdata.mod.values()])

    for mod in mdata.mod:
        filter_obs(mdata.mod[mod], common_obs)

    mdata.update_obs()

    return


# Utility functions: filtering observations


def filter_obs(
    data: Union[AnnData, MuData], var: Union[str, Sequence[str]], func: Optional[Callable] = None
) -> None:
    """
    Filter observations (samples or cells) in-place
    using any column in .obs or in .X.

    This function is currently a draft, and it can be removed
    or its behaviour might be changed in future.

    Parameters
    ----------
    data: AnnData or MuData
            AnnData or MuData object
    var: str or Sequence[str]
            Column name in .obs or in .X to be used for filtering.
            Alternatively, obs_names can be provided directly.
    func
            Function to apply to the variable used for filtering.
            If the variable is of type boolean and func is an identity function,
            the func argument can be omitted.
    """

    if data.is_view:
        raise ValueError(
            "The provided adata is a view. In-place filtering does not operate on views."
        )
    if data.isbacked:
        if isinstance(data, AnnData):
            warnings.warn(
                "AnnData object is backed. The requested subset of the matrix .X will be read into memory, and the object will not be backed anymore."
            )
        else:
            warnings.warn(
                "MuData object is backed. The requested subset of the .X matrices of its modalities will be read into memory, and the object will not be backed anymore."
            )

    if isinstance(var, str):
        if var in data.obs.columns:
            if func is None:
                if data.obs[var].dtypes.name == "bool":

                    def func(x):
                        return x

                else:
                    raise ValueError(f"Function has to be provided since {var} is not boolean")
            obs_subset = func(data.obs[var].values)
        elif var in data.var_names:
            obs_subset = func(data.X[:, np.where(data.var_names == var)[0]].reshape(-1))
        else:
            raise ValueError(
                f"Column name from .obs or one of the var_names was expected but got {var}."
            )
    else:
        if func is None:
            if np.array(var).dtype == np.bool:
                obs_subset = np.array(var)
            else:
                obs_subset = data.obs_names.isin(var)
        else:
            raise ValueError(f"When providing obs_names directly, func has to be None.")

    # Subset .obs
    data._obs = data.obs[obs_subset]
    data._n_obs = data.obs.shape[0]

    # Subset .obsm
    for k, v in data.obsm.items():
        data.obsm[k] = v[obs_subset]

    # Subset .obsp
    for k, v in data.obsp.items():
        data.obsp[k] = v[obs_subset][:, obs_subset]

    if isinstance(data, AnnData):
        # Subset .X
        try:
            data._X = data.X[obs_subset, :]
        except TypeError:
            data._X = data.X[np.where(obs_subset)[0], :]
            # For some h5py versions, indexing arrays must have integer dtypes
            # https://github.com/h5py/h5py/issues/1847
        if data.isbacked:
            data.file.close()
            data.filename = None

        # Subset layers
        for layer in data.layers:
            data.layers[layer] = data.layers[layer][obs_subset, :]

        # Subset raw
        if data.raw is not None:
            data.raw._X = data.raw.X[obs_subset, :]
            data.raw._n_obs = data.raw.X.shape[0]

    else:
        # filter_obs() for each modality
        for m, mod in data.mod.items():
            obsmap = data.obsmap[m][obs_subset]
            obsmap = obsmap[obsmap != 0] - 1
            filter_obs(mod, mod.obs_names[obsmap])
        # Subset .obsmap
        # This would return wrong indices however:
        # for k, v in data.obsmap.items():
        #     data.obsmap[k] = v[obs_subset]
        #
        # A fix is to get rid of obsmaps
        # for them to be reconstructed later.
        obsmap_keys = list(data.obsmap.keys())
        for k in obsmap_keys:
            del data.obsmap[k]

    return


# Utility functions: filtering variables


def filter_var(
    data: Union[AnnData, MuData], var: Union[str, Sequence[str]], func: Optional[Callable] = None
):
    """
    Filter variables (features, e.g. genes) in-place
    using any column in .var or row in .X.

    This function is currently a draft, and it can be removed
    or its behaviour might be changed in future.

    Parameters
    ----------
    data: AnnData or MuData
            AnnData or MuData object
    var: str or Sequence[str]
            Column name in .var or row name in .X to be used for filtering.
            Alternatively, var_names can be provided directly.
    func
            Function to apply to the variable used for filtering.
            If the variable is of type boolean and func is an identity function,
            the func argument can be omitted.
    """

    if data.is_view:
        raise ValueError(
            "The provided adata is a view. In-place filtering does not operate on views."
        )
    if data.isbacked:
        if isinstance(data, AnnData):
            warnings.warn(
                "AnnData object is backed. The requested subset of the matrix .X will be read into memory, and the object will not be backed anymore."
            )
        else:
            warnings.warn(
                "MuData object is backed. The requested subset of the .X matrices of its modalities will be read into memory, and the object will not be backed anymore."
            )

    if isinstance(var, str):
        if var in data.var.columns:
            if func is None:
                if data.var[var].dtypes.name == "bool":

                    def func(x):
                        return x

                else:
                    raise ValueError(f"Function has to be provided since {var} is not boolean")
            var_subset = func(data.var[var].values)
        elif var in data.obs_names:
            var_subset = func(data.X[:, np.where(data.obs_names == var)[0]].reshape(-1))
        else:
            raise ValueError(
                f"Column name from .var or one of the obs_names was expected but got {var}."
            )
    else:
        if func is None:
            if np.array(var).dtype == np.bool:
                var_subset = var
            else:
                var_subset = data.var_names.isin(var)
        else:
            raise ValueError(f"When providing var_names directly, func has to be None.")

    # Subset .var
    data._var = data.var[var_subset]
    data._n_vars = data.var.shape[0]

    # Subset .varm
    for k, v in data.varm.items():
        data.varm[k] = v[var_subset]

    # Subset .varp
    for k, v in data.varp.items():
        data.varp[k] = v[var_subset][:, var_subset]

    if isinstance(data, AnnData):
        # Subset .X
        try:
            data._X = data.X[:, var_subset]
        except TypeError:
            data._X = data.X[:, np.where(var_subset)[0]]
            # For some h5py versions, indexing arrays must have integer dtypes
            # https://github.com/h5py/h5py/issues/1847
        if data.isbacked:
            data.file.close()
            data.filename = None

        # Subset layers
        for layer in data.layers:
            data.layers[layer] = data.layers[layer][:, var_subset]

        # NOTE: .raw is not subsetted

    else:
        # filter_var() for each modality
        for m, mod in data.mod.items():
            varmap = data.varmap[m][var_subset]
            varmap = varmap[varmap != 0] - 1
            filter_var(mod, mod.var_names[varmap])
        # Subset .varmap
        # This would return wrong indices however:
        # for k, v in data.varmap.items():
        #     data.varmap[k] = v[var_subset]
        #
        # A fix is to get rid of varmaps
        # for them to be reconstructed later.
        varmap_keys = list(data.varmap.keys())
        for k in varmap_keys:
            del data.varmap[k]

    return
