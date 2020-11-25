from typing import Union, Callable, Optional, Sequence, Dict
from functools import reduce
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, issparse
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors

from anndata import AnnData
from scanpy.tools._utils import _choose_representation
from scanpy import Neighbors
from .._core.mudata import MuData

# Computational methods for preprocessing


def weighted_neighbors(
    mdata: MuData,
    n_neighbors: Optional[int] = None,
    n_multineighbors: int = 200,
    neighbor_keys: Optional[Dict[str, Optional[str]]] = None,
    key_added: Optional[str] = None,
    eps=1e-4,
    copy=False,
):
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
    for i, mod in enumerate(modalities):
        nkey = neighbor_keys.get(mod, "neighbors")
        try:
            nparams = mdata[mod].uns[nkey]
        except KeyError:
            raise ValueError(
                f'Did not find .uns["{nkey}"] for modality "{mod}". Run `sc.pp.neighbors` on all modalities first.'
            )

        use_rep = nparams["params"].get("use_rep", None)
        n_pcs = nparams["params"].get("n_pcs", None)
        mod_neighbors[i] = nparams["params"].get("n_neighbors", 0)

        neighbors_params[mod] = nparams
        reps[mod] = _choose_representation(mdata[mod], use_rep, n_pcs)
        observations = observations.intersection(mdata[mod].obs.index)

    if n_neighbors is None:
        mod_neighbors = mod_neighbors[mod_neighbors > 0]
        n_neighbors = int(round(np.mean(mod_neighbors), 0))

    ratios = np.full((len(observations), len(modalities)), -np.inf, dtype=np.float64)
    sigmas = {}
    for i1, mod1 in enumerate(modalities):
        observations1 = observations.intersection(mdata[mod1].obs.index)
        ratioidx = np.where(observations.isin(observations1))[0]
        nparams1 = neighbors_params[mod1]
        X = reps[mod1]
        distances = squareform(pdist(X if not issparse(X) else X.toarray(), "euclidean"))
        if nparams["params"].get("metric") in ("euclidean", "l2"):
            neighbordistances = mdata[mod1].obsp[nparams1["distances_key"]]
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
        else:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="precomputed")
            nbrs.fit(distances)
            neighbordistances = nbrs.kneighbors_graph(distances, mode="distance")
            np.fill_diagonal(distances, np.inf)
            nndistances = np.min(distances, axis=1)
            np.fill_diagonal(distances, 0)

        # the following code is somewhat non-intuitive in an attempt to keep the memory usage at
        # a manageable level
        distances = np.stack(
            (distances, squareform(pdist((neighbordistances > 0).toarray(), "jaccard"))), axis=-1
        )
        distances[distances[..., -1] == 1, -1] = -np.inf
        distances = np.partition(
            distances.view([("distance", distances.dtype), ("jaccard", distances.dtype)]),
            kth=21,
            axis=1,
            order=("jaccard", "distance"),
        ).view(distances.dtype)[:, :21, 0]
        distances = np.partition(distances, kth=0, axis=1)[:, 1:]
        csigmas = distances.mean(axis=1)

        currtheta = None
        thetas = np.full(
            (len(observations1), len(modalities) - 1), -np.inf, dtype=neighbordistances.dtype
        )

        lasti = 0
        for i2, mod2 in enumerate(modalities):
            nparams2 = neighbors_params[mod2]
            neighbordistances = mdata[mod2].obsp[nparams2["distances_key"]]
            observations2 = observations1.intersection(mdata[mod2].obs.index)
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
        observations1 = observations.intersection(mdata[m].obs.index)
        idx = np.where(observations.isin(observations1))[0]
        rep = reps[m]
        nbrs = NearestNeighbors(n_neighbors=n_multineighbors, metric="euclidean")
        nbrs.fit(rep)
        neighbordistances[idx[:, np.newaxis], idx[np.newaxis, :]] += nbrs.kneighbors_graph()
    neighbordistances = neighbordistances.tocsr()

    neighbordistances.data[:] = 0
    for i, m in enumerate(modalities):
        observations1 = observations.intersection(mdata[m].obs.index)
        idx = np.where(observations.isin(observations1))[0]
        rep = reps[m]
        csigmas = sigmas[m]
        if issparse(rep):
            neighdist = lambda cell, nz: -cdist(
                rep[cell, :].toarray(), rep[nz, :].toarray(), metric="euclidean"
            )
        else:
            neighdist = lambda cell, nz: -cdist(
                rep[np.newaxis, cell, :], rep[nz, :], metric="euclidean"
            )
        for cell, j in enumerate(idx):
            row = slice(neighbordistances.indptr[cell], neighbordistances.indptr[cell + 1])
            nz = neighbordistances.indices[row]
            neighbordistances.data[row] += (
                np.exp(neighdist(cell, nz) / csigmas[cell]).squeeze() * weights[cell, i]
            )
    neighbordistances.data = np.sqrt(0.5 * (1 - neighbordistances.data))

    # NearestNeighbors wants sorted data, it will copy the matrix if it's not sorted
    for i, m in enumerate(modalities):
        observations1 = observations.intersection(mdata[m].obs.index)
        idx = np.where(observations.isin(observations1))[0]
        for cell, j in enumerate(idx):
            rowidx = slice(neighbordistances.indptr[cell], neighbordistances.indptr[cell + 1])
            row = neighbordistances.data[rowidx]
            nz = neighbordistances.indices[rowidx]
            idx = np.argsort(row)
            row[:] = row[idx]
            nz[:] = nz[idx]

    mnn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed")
    mnn.fit(neighbordistances)
    connectivities = mnn.kneighbors_graph(mode="connectivity")
    neighbordistances = mnn.kneighbors_graph(mode="distance")

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
        "metric": "euclidean",
        "eps": eps,
    }
    if use_rep is not None:
        neighbors_dict["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        neighbors_dict["params"]["n_pcs"] = n_pcs
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
        warn(
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
