from typing import Union, Callable, Optional, Sequence, Dict
from functools import reduce
import warnings
from collections import OrderedDict

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, issparse, SparseEfficiencyWarning
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.utils import check_random_state

from anndata import AnnData
from scanpy.tools._utils import _choose_representation
from scanpy.neighbors import _compute_connectivities_umap
from scanpy._compat import Literal
import umap
from umap.umap_ import nearest_neighbors
from numba import njit, prange

from .._core.mudata import MuData

# Computational methods for preprocessing

_euclidean = njit(umap.distances.euclidean.py_func, inline="always", fastmath=True)
_sparse_euclidean = njit(umap.sparse.sparse_euclidean.py_func, inline="always")


@njit(inline="always")
def _sparse_csr_jaccard_metric(x, y, indices, indptr, data):
    xrowidx = indices[indptr[x] : (indptr[x + 1])]
    xrowidx = xrowidx[data[xrowidx] != 0]
    yrowidx = indices[indptr[y] : (indptr[y + 1])]
    yrowidx = yrowidx[data[yrowidx] != 0]

    # numba does not have np.isin yet... https://github.com/numba/numba/pull/4815
    # assumes that indices are sorted and unique
    intersect = 0
    lasty = yrowidx[0]
    lastyidx = 0
    for xidx in xrowidx:
        if xidx > lasty:
            if lastyidx < yrowidx.size:
                lastyidx += 1
                lasty = yrowidx[lastyidx]
            else:
                break
        if xidx == lasty:
            intersect += 1

    union = xrowidx.size + yrowidx.size - intersect
    return (union - intersect) / union


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
    return (
        0
        if x == y
        else _sparse_csr_jaccard_metric(x, y, neighbors_indices, neighbors_indptr, neighbors_data)
        * N
        + (bbox_norm - _euclidean(X[x, :], X[y, :])) / bbox_norm
    )


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
        return 0

    from_inds = X_indices[X_indptr[x] : X_indptr[x + 1]]
    from_data = X_data[X_indptr[x] : X_indptr[x + 1]]

    to_inds = X_indices[X_indptr[y] : X_indptr[y + 1]]
    to_data = X_data[X_indptr[y] : X_indptr[y + 1]]

    euclidean = _sparse_euclidean(from_inds, from_data, to_inds, to_data)
    jac = _sparse_csr_jaccard_metric(x, y, neighbors_indices, neighbors_indptr, neighbors_data)

    return jac * N + (bbox_norm - euclidean) / bbox_norm


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

        idx = np.argsort(rowdata)  # would like to use argpartition, but not supported by numba
        startidx = i * n_neighbors
        endidx = (i + 1) * n_neighbors
        # numba's parallel loops only support reductions, not assignment
        knn_indices[startidx:endidx] += cols[idx[:n_neighbors]]
        knn_data[startidx:endidx] += rowdata[idx[:n_neighbors]]
    return knn_data, knn_indices, knn_indptr


def _sparse_csr_fast_knn(X: csr_matrix, n_neighbors: int):  # numba doesn't know about SciPy
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
    eps: float = 1e-4,
    copy: bool = False,
    random_state: Optional[Union[int, np.random.RandomState]] = 42,
) -> Optional[MuData]:
    """
    Multimodal nearest neighbor search.

    This implements the multimodal nearest neighbor method of Hao et al. and Swanson et al. The neighbor search
    efficiency on this heavily relies on UMAP. In particular, you may want to decrease n_multineighbors for large
    data set to avoid excessive peak memory use.

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
        eps: Small number to avoid numerical errors.
        copy: Return a copy instead of writing to ``mdata``.
        random_state: Random seed.

    Returns: Depending on ``copy``, returns or updates ``mdata``.
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
        mod_reps[mod] = use_rep
        mod_n_pcs[mod] = n_pcs

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

        # We want to get the k-nn by Jaccard distance, but break ties using Euclidean distance.
        # The naive version would be to compute pairwise Jaccard and Euclidean distances for
        # all points, but this is low and needs lots of memory. We want to use an efficient
        # k-nn algorithm, however no package that I know of supports tie-breaking k-nn, so we
        # use a custom distance. Jaccard distance is always between 0 and 1 and has discrete
        # increments of at least 1/N, where N is the number of data points. If we scale the
        # Jaccard distances by N, the minimum Jaccard distance will be 1. If we scale all Euclidean
        # distances to be less than one, we can define a combined distance as the sum of the
        # scaled Jaccard and one minus the Euclidean distances. This is not a proper metric, but
        # UMAP's nearest neighbor search uses NN-descent, which works with arbitrary similarity
        # measures.
        # The scaling factor for the Euclidean distance is given by the length of the diagonal
        # of the bounding box of the data. This can be computed in linear time by just taking
        # the minimal and maximal coordinates of each dimension.
        N = X.shape[0]
        bbox_norm = np.linalg.norm(_sparse_csr_ptp(X) if issparse(X) else np.ptp(X, axis=0), ord=2)
        neighbordistances.sort_indices()
        lmemory = low_memory if low_memory is not None else N > 50000
        if issparse(X):
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
        nn_indices, _, _ = nearest_neighbors(
            np.arange(N)[:, np.newaxis],
            n_neighbors=n_bandwidth_neighbors + 1,
            metric=cmetric,
            metric_kwds=metric_kwds,
            random_state=randomstate,
            angular=False,
            low_memory=lmemory,
        )
        nn_indices = nn_indices[:, 1:]  # the point itself is its nearest neighbor

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
    neighbordistances = lil_matrix((mdata.n_obs, mdata.n_obs), dtype=np.float64)
    for i, m in enumerate(modalities):
        cmetric = neighbors_params[m].get("metric", "euclidean")
        observations1 = observations.intersection(mdata.mod[m].obs.index)

        rep = reps[m]
        lmemory = low_memory if low_memory is not None else rep.shape[0] > 50000
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
        if observations1.size == mdata.n_obs and neighbordistances.size == 0:
            neighbordistances = graph.tolil()
        else:  # the naive version of neighbordistances[idx[:, np.newaxis], idx[np.newaxis, :]] += graph
            # uses way too much memory
            fullidx = np.where(observations.isin(observations1))[0]
            modidx = np.where(mdata.mod[m].obs.index.isin(observations1))[0]
            fullidx = fullidx[
                np.concatenate(
                    (np.where(np.diff(fullidx, prepend=fullidx[0] - 2) > 1)[0], (fullidx.size - 1,))
                )
            ]
            modidx = modidx[
                np.concatenate(
                    (np.where(np.diff(modidx, prepend=modidx[0] - 2) > 1)[0], (modidx.size - 1,))
                )
            ]
            for fullidxstart1, fullidxstop1, modidxstart1, modidxstop1 in zip(
                fullidx[:-1], fullidx[1:], modidx[:-1], modidx[1:]
            ):
                for fullidxstart2, fullidxstop2, modidxstart2, modidxstop2 in zip(
                    fullidx[:-1], fullidx[1:], modidx[:-1], modidx[1:]
                ):
                    neighbordistances[
                        fullidxstart1:fullidxstop1, fullidxstart2:fullidxstop2
                    ] += graph[modidxstart1:modidxstop1, modidxstart2:modidxstop2]
    neighbordistances = neighbordistances.tocsr()

    neighbordistances.data[:] = 0
    for i, m in enumerate(modalities):
        observations1 = observations.intersection(mdata.mod[m].obs.index)
        fullidx = np.where(observations.isin(observations1))[0]
        rep = reps[m]
        csigmas = sigmas[m]
        if issparse(rep):
            neighdist = lambda cell, nz: -cdist(
                rep[cell, :].toarray(), rep[nz, :].toarray(), metric=metric
            )
        else:
            neighdist = lambda cell, nz: -cdist(rep[np.newaxis, cell, :], rep[nz, :], metric=metric)
        for cell, j in enumerate(fullidx):
            row = slice(neighbordistances.indptr[cell], neighbordistances.indptr[cell + 1])
            nz = neighbordistances.indices[row]
            neighbordistances.data[row] += (
                np.exp(neighdist(cell, nz) / csigmas[cell]).squeeze() * weights[cell, i]
            )
    neighbordistances.data = np.sqrt(0.5 * (1 - neighbordistances.data))

    neighbordistances = _sparse_csr_fast_knn(neighbordistances, n_neighbors + 1)
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


def filter_obs(adata: AnnData, var: Union[str, Sequence[str]], func: Optional[Callable] = None):
    """
    Filter observations (samples or cells) in-place
    using any column in .obs or in .X.

    This function is currently a draft, and it can be removed
    or its behaviour might be changed in future.

    Parameters
    ----------
    adata: AnnData
            AnnData object
    var: str or Sequence[str]
            Column name in .obs or in .X to be used for filtering.
            Alternatively, obs_names can be provided directly.
    func
            Function to apply to the variable used for filtering.
            If the variable is of type boolean and func is an identity function,
            the func argument can be omitted.
    """

    if isinstance(var, str):
        if var in adata.obs.columns:
            if func is None:
                if adata.obs[var].dtypes.name == "bool":
                    func = lambda x: x
                else:
                    raise ValueError(f"Function has to be provided since {var} is not boolean")
            obs_subset = func(adata.obs[var].values)
        elif var in adata.var_names:
            obs_subset = func(adata.X[:, np.where(adata.var_names == var)[0]].reshape(-1))
        else:
            raise ValueError(
                f"Column name from .obs or one of the var_names was expected but got {var}."
            )
    else:
        if func is None:
            obs_subset = adata.obs_names.isin(var)
        else:
            raise ValueError(f"When providing obs_names directly, func has to be None.")

    # Subset .obs
    adata._obs = adata.obs[obs_subset]
    adata._n_obs = adata.obs.shape[0]

    # Subset .X
    adata._X = adata.X[obs_subset]

    # Subset layers
    for layer in adata.layers:
        adata.layers[layer] = adata.layers[layer][obs_subset]

    # Subset raw
    if adata.raw is not None:
        adata.raw._X = adata.raw.X[obs_subset]
        adata.raw._n_obs = adata.raw.X.shape[0]

    # Subset .obsm
    for k, v in adata.obsm.items():
        adata.obsm[k] = v[obs_subset]

    # Subset .obsp
    for k, v in adata.obsp.items():
        adata.obsp[k] = v[obs_subset][:, obs_subset]

    return


# Utility functions: filtering variables


def filter_var(adata: AnnData, var: Union[str, Sequence[str]], func: Optional[Callable] = None):
    """
    Filter variables (features, e.g. genes) in-place
    using any column in .var or row in .X.

    This function is currently a draft, and it can be removed
    or its behaviour might be changed in future.

    Parameters
    ----------
    adata: AnnData
            AnnData object
    var: str or Sequence[str]
            Column name in .var or row name in .X to be used for filtering.
            Alternatively, var_names can be provided directly.
    func
            Function to apply to the variable used for filtering.
            If the variable is of type boolean and func is an identity function,
            the func argument can be omitted.
    """

    if isinstance(var, str):
        if var in adata.var.columns:
            if func is None:
                if adata.var[var].dtypes.name == "bool":
                    func = lambda x: x
                else:
                    raise ValueError(f"Function has to be provided since {var} is not boolean")
            var_subset = func(adata.var[var].values)
        elif var in adata.obs_names:
            var_subset = func(adata.X[:, np.where(adata.obs_names == var)[0]].reshape(-1))
        else:
            raise ValueError(
                f"Column name from .var or one of the obs_names was expected but got {var}."
            )
    else:
        if func is None:
            var_subset = adata.var_names.isin(var)
        else:
            raise ValueError(f"When providing var_names directly, func has to be None.")

    # Subset .var
    adata._var = adata.var[var_subset]
    adata._n_vars = adata.var.shape[0]

    # Subset .X
    adata._X = adata.X[:, var_subset]

    # Subset layers
    for layer in adata.layers:
        adata.layers[layer] = adata.layers[layer][:, var_subset]

    # NOTE: .raw is not subsetted

    # Subset .varm
    for k, v in adata.varm.items():
        adata.varm[k] = v[var_subset]

    # Subset .varp
    for k, v in adata.varp.items():
        adata.varp[k] = v[var_subset][:, var_subset]

    return
