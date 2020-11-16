import sys
import os
from functools import reduce

import logging
from datetime import datetime
from time import strftime
from warnings import warn

import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from natsort import natsorted
from anndata import AnnData
from .mudata import MuData

from typing import Union, Optional, List, Iterable, Mapping, Sequence, Type, Any
from types import MappingProxyType

try:
    from louvain.VertexPartition import MutableVertexPartition as LouvainMutableVertexPartition
except ImportError:

    class LouvainMutableVertexPartition:
        pass

    LouvainMutableVertexPartition.__module__ = "louvain.VertexPartition"

try:
    from leidenalg.VertexPartition import MutableVertexPartition as LeidenMutableVertexPartition
except ImportError:

    class LeidenMutableVertexPartition:
        pass

    LeidenMutableVertexPartition.__module__ = "leidenalg.VertexPartition"


#
# Multi-omics factor analysis (MOFA)
#


def _set_mofa_data_from_mudata(
    model,
    mdata,
    groups_label=None,
    use_raw=False,
    use_layer=None,
    likelihoods=None,
    features_subset=None,
    save_metadata=None,
    use_obs=None,
):
    """
    Input the data in MuData format

    PARAMETERS
    ----------
    model: MOFA+ model entry point object
    mdata: a MuData object
    groups_label : optional: a column name in adata.obs for grouping the samples
    use_raw : optional: use raw slot of AnnData as input values
    use_layer : optional: use a specific layer of AnnData as input values (supersedes use_raw option)
    likelihoods : optional: likelihoods to use (guessed from the data if not provided)
    features_subset : optional: .var column with a boolean value to select genes (e.g. "highly_variable"), None by default
    save_metadata : optional: save metadata for samples (cells) and for features
    use_obs : optional: 'union' or 'intersection', relevant when not all samples (cells) are the same across views
    """

    try:
        from mofapy2.build_model.utils import process_data
        from mofapy2.build_model.utils import guess_likelihoods
    except ImportError:
        raise ImportError(
            "MOFA+ is not available. Install MOFA+ from PyPI (`pip install mofapy2`) or from GitHub (`pip install git+https://github.com/bioFAM/MOFA2`)"
        )

    # Sanity checks
    if not hasattr(model, "data_opts"):
        # print("Data options not defined before setting the data, using default values...")
        model.set_data_options()

    # Use the union or the intersection of observations
    obs_names = mdata.obs.index.values
    if use_obs:
        if use_obs == "intersection":
            common_obs = reduce(np.intersect1d, [v.obs_names.values for k, v in mdata.mod.items()])
            mdata = mdata[common_obs]
        elif use_obs == "union":  # expand matrices later
            pass
        else:
            raise ValueError("Expected `use_obs` to be 'union' or 'intersection")

    # Check groups_label is defined properly
    n_groups = 1  # no grouping by default
    if groups_label is not None:
        if not isinstance(groups_label, str):
            print("Error: groups_label should be a string present in the observations column names")
            sys.stdout.flush()
            sys.exit()
        if groups_label not in mdata.obs.columns:
            print("Error: {} is not in observations names".format(groups_label))
            sys.stdout.flush()
            sys.exit()
        n_groups = mdata.obs[groups_label].unique().shape[0]

    # Get the respective data slot
    data = []
    if use_layer:
        for m in mdata.mod.keys():
            adata = mdata.mod[m]
            if use_layer in adata.layers.keys():
                if callable(getattr(adata.layers[use_layer], "todense", None)):
                    data.append(np.array(adata.layers[use_layer].todense()))
                else:
                    data.append(adata.layers[use_layer].copy())
            else:
                print("Error: Layer {} does not exist".format(use_layer))
                sys.stdout.flush()
                sys.exit()
    elif use_raw:
        for m in mdata.mod.keys():
            adata = mdata.mod[m]
            adata_raw_dense = np.array(adata.raw[:, adata.var_names].X.todense())
            data.append(adata_raw_dense)
    else:
        for m in mdata.mod.keys():
            adata = mdata.mod[m]
            if callable(getattr(adata.X, "todense", None)):
                data.append(np.array(adata.X.todense()))
            else:
                data.append(adata.X.copy())

    # Expand samples (cells) if union is required
    if use_obs == "union":
        n = mdata.shape[0]
        mods = tuple(mdata.mod.keys())
        obs_names = mdata.obs.index.values
        n = len(obs_names)
        for i, x in enumerate(data):
            m = mods[i]

            # empty matrix of right dimensions
            x = np.empty(shape=(n, x.shape[1]))
            x[:] = np.nan

            # use pandas data frames indices to expand obs
            orig = mdata[m].obs.loc[:, []].copy()
            expanded = orig.reindex(obs_names)

            # find indices in the expanded matrix to copy orig data
            expanded["ix"] = range(n)
            ix = expanded.join(orig, how="right").ix.values

            # set expanded data with part of the orig data
            x[ix, :] = mdata[m].X
            data[i] = x

    # Subset features if required
    if features_subset is not None:
        for i, m in enumerate(mdata.mod.keys()):
            if features_subset not in mdata.mod[m].var.columns:
                raise KeyError(f"There is no column {features_subset} in .var for modality {m}")
            data[i] = data[i][:, mdata.mod[m].var[features_subset].values]

    # Save dimensionalities
    M = model.dimensionalities["M"] = len(mdata.mod)
    G = model.dimensionalities["G"] = n_groups
    N = model.dimensionalities["N"] = mdata.shape[0]
    D = model.dimensionalities["D"] = [
        data[m].shape[1] for m in range(M)
    ]  # Feature may have been filtered
    n_grouped = [mdata.shape[0]] if n_groups == 1 else mdata.obs.groupby(groups_label).size().values

    # Define views names and features names and metadata
    model.data_opts["views_names"] = list(mdata.mod.keys())

    if features_subset is not None:
        model.data_opts["features_names"] = [
            adata.var_names[adata.var[features_subset].values] for adata in mdata.mod.values()
        ]
    else:
        model.data_opts["features_names"] = [adata.var_names for adata in mdata.mod.values()]

    if save_metadata and mdata.var.shape[1] > 0:
        if features_subset is not None:
            model.data_opts["features_metadata"] = [
                adata.var[adata.var[features_subset].values] for adata in mdata.mod.values()
            ]
        else:
            model.data_opts["features_metadata"] = [adata.var for adata in mdata.mod.values()]

    # Define groups and samples names and metadata
    if groups_label is None:
        model.data_opts["groups_names"] = ["group1"]
        model.data_opts["samples_names"] = [mdata.obs.index.values.tolist()]
        model.data_opts["samples_groups"] = ["group1"] * N
        if save_metadata and mdata.obs.shape[1] > 0:
            model.data_opts["samples_metadata"] = [mdata.obs]
    else:
        # While grouping the pandas.DataFrame, the group_label would be sorted.
        # Hence the naive implementation `mdata.obs[groups_label].unique()` to get group names
        # wouldn't match samples_names if the samples are not ordered according to their group beforehand.
        samples_groups = (
            mdata.obs.reset_index(drop=False)
            .groupby(groups_label, sort=False)[groups_label]
            .apply(list)
        )

        # List of names of groups, i.e. [group1, group2, ...]
        model.data_opts["groups_names"] = [str(g) for g in samples_groups.index.values]
        # Nested list of names of samples, one inner list per group, i.e. [[group1_sample1, group1_sample2, ...], ...]
        model.data_opts["samples_names"] = (
            mdata.obs.reset_index(drop=False)
            .rename(columns={mdata.obs.index.name: "index"})
            .groupby(groups_label, sort=False)["index"]
            .apply(list)
            .tolist()
        )
        # List of names of groups for samples ordered as they are when split according to their group
        model.data_opts["samples_groups"] = np.concatenate(samples_groups.values)
        if save_metadata:
            # List of metadata tables for each group of samples
            model.data_opts["samples_metadata"] = [
                g for _, g in mdata.obs.groupby(groups_label, sort=False)
            ]

    # Use the order of samples according to the groups in the model.data_opts["samples_groups"]
    if groups_label:
        for m in range(M):
            data[m] = data[m][
                np.concatenate(
                    mdata.obs.reset_index(drop=False)
                    .reset_index(drop=False)
                    .groupby(groups_label, sort=False)["index"]
                    .apply(np.array)
                    .tolist()
                ),
                :,
            ]

    # If everything successful, print verbose message
    for m in range(M):
        for g in range(G):
            print(
                "Loaded view='%s' group='%s' with N=%d samples and D=%d features..."
                % (
                    model.data_opts["views_names"][m],
                    model.data_opts["groups_names"][g],
                    n_grouped[g],
                    D[m],
                )
            )
    print("\n")

    # Store intercepts (it is for one view only)
    model.intercepts = [[] for _ in range(M)]

    # Define likelihoods
    if likelihoods is None:
        likelihoods = guess_likelihoods(data)
    assert (
        len(likelihoods) == model.dimensionalities["M"]
    ), "Please specify one likelihood for each view"
    assert set(likelihoods).issubset(
        set(["gaussian", "bernoulli", "poisson"])
    ), "Available likelihoods are 'gaussian', 'bernoulli', 'poisson'"
    model.likelihoods = likelihoods

    # Process the data (center, scaling, etc.)
    for m in range(M):
        for g in model.data_opts["groups_names"]:
            samples_idx = np.where(np.array(model.data_opts["samples_groups"]) == g)[0]
            model.intercepts[m].append(np.nanmean(data[m][samples_idx, :], axis=0))
    model.data = process_data(data, likelihoods, model.data_opts, model.data_opts["samples_groups"])


def mofa(
    data: Union[AnnData, MuData],
    groups_label: bool = None,
    use_raw: bool = False,
    use_layer: bool = None,
    use_var: Optional[str] = "highly_variable",
    use_obs: Optional[str] = None,
    likelihoods: Optional[Union[str, List[str]]] = None,
    n_factors: int = 10,
    scale_views: bool = False,
    scale_groups: bool = False,
    ard_weights: bool = True,
    ard_factors: bool = True,
    spikeslab_weights: bool = True,
    spikeslab_factors: bool = False,
    n_iterations: int = 1000,
    convergence_mode: str = "fast",
    gpu_mode: bool = False,
    Y_ELBO_TauTrick: bool = True,
    mefisto_covariate: Optional[str] = None,
    mefisto_covariates_names: Optional[str] = None,
    mefisto_scale_cov: bool = False,
    mefisto_start_opt: int = 20,
    mefisto_n_grid: int = 20,
    mefisto_opt_freq: int = 10,
    mefisto_model_groups: bool = True,
    mefisto_warping: bool = False,
    mefisto_warping_freq: int = 20,
    mefisto_warping_ref: int = 0,
    mefisto_warping_open_begin: bool = True,
    mefisto_warping_open_end: bool = True,
    mefisto_sparseGP: bool = False,
    mefisto_frac_inducing: Optional[float] = None,
    save_parameters: bool = False,
    save_data: bool = True,
    save_metadata: bool = True,
    seed: int = 1,
    outfile: Optional[str] = None,
    expectations: Optional[List[str]] = None,
    save_interrupted: bool = True,
    verbose: bool = False,
    quiet: bool = True,
    copy: bool = False,
):
    """
    Run Multi-Omics Factor Analysis

    PARAMETERS
    ----------
    data
            an MuData object
    groups_label : optional
            a column name in adata.obs for grouping the samples
    use_raw : optional
            use raw slot of AnnData as input values
    use_layer : optional
            use a specific layer of AnnData as input values (supersedes use_raw option)
    use_var : optional
            .var column with a boolean value to select genes (e.g. "highly_variable"), None by default
    use_obs : optional
            strategy to deal with samples (cells) not being the same across modalities ("union" or "intersection", throw error by default)
    likelihoods : optional
            likelihoods to use, default is guessed from the data
    n_factors : optional
            number of factors to train the model with
    scale_views : optional
            scale views to unit variance
    scale_groups : optional
            scale groups to unit variance
    ard_weights : optional
            use view-wise sparsity
    ard_factors : optional
            use group-wise sparsity
    spikeslab_weights : optional
            use feature-wise sparsity (e.g. gene-wise)
    spikeslab_factors : optional
            use sample-wise sparsity (e.g. cell-wise)
    n_iterations : optional
            upper limit on the number of iterations
    convergence_mode : optional
            fast, medium, or slow convergence mode
    gpu_mode : optional
            if to use GPU mode
    Y_ELBO_TauTrick : optional
            if to use ELBO Tau trick to speed up computations
    save_parameters : optional
            if to save training parameters
    save_data : optional
            if to save training data
    save_metadata : optional
            if to load metadata from the AnnData object (.obs and .var tables) and save it, False by default
    seed : optional
            random seed
    outfile : optional
            path to HDF5 file to store the model
    expectations : optional
            which nodes should be used to save expectations for (will save only W and Z by default);
    possible expectations names
            nclude Y, W, Z, Tau, AlphaZ, AlphaW, ThetaW, ThetaZ
    save_interrupted : optional
            if to save partially trained model when the training is interrupted
    verbose : optional
            print verbose information during traing
    quiet : optional
            silence messages during training procedure
    copy : optional
            return a copy of AnnData instead of writing to the provided object
    """

    try:
        from mofapy2.run.entry_point import entry_point
    except ImportError:
        raise ImportError(
            "MOFA+ is not available. Install MOFA+ from PyPI (`pip install mofapy2`) or from GitHub (`pip install git+https://github.com/bioFAM/MOFA2`)"
        )

    if isinstance(data, AnnData):
        logging.info("Wrapping an AnnData object into an MuData container")
        mdata = MuData(data)
    elif isinstance(data, MuData):
        mdata = data
    else:
        raise TypeError("Expected an MuData object")

    if outfile is None:
        outfile = os.path.join("/tmp", "mofa_{}.hdf5".format(strftime("%Y%m%d-%H%M%S")))

    if use_var:
        if use_var not in data.var.columns:
            warn(f"There is no column {use_var} in the provided object")
            use_var = None

    if isinstance(data, MuData):
        common_obs = reduce(np.intersect1d, [v.obs_names.values for k, v in mdata.mod.items()])
        if len(common_obs) != mdata.n_obs:
            if not use_obs:
                raise IndexError(
                    "Not all the observations are the same across modalities. Please run `mdata.intersect_obs()` to subset the data or devise a strategy with `use_obs` ('union' or 'intersection')"
                )
            elif use_obs not in ["union", "intersection"]:
                raise ValueError(
                    f"Expected `use_obs` argument to be 'union' or 'intersection', not '{use_obs}'"
                )
        else:
            use_obs = None

    ent = entry_point()

    lik = likelihoods
    if lik is not None:
        if isinstance(lik, str) and isinstance(lik, Iterable):
            lik = [lik for _ in range(len(mdata.mod))]

    ent.set_data_options(scale_views=scale_views, scale_groups=scale_groups)
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Setting data from MuData object..."
    )
    _set_mofa_data_from_mudata(
        model=ent,
        mdata=mdata,
        groups_label=groups_label,
        use_raw=use_raw,
        use_layer=use_layer,
        likelihoods=lik,
        features_subset=use_var,
        save_metadata=save_metadata,
        use_obs=use_obs,
    )
    ent.set_model_options(
        ard_factors=ard_factors,
        ard_weights=ard_weights,
        spikeslab_weights=spikeslab_weights,
        spikeslab_factors=spikeslab_factors,
        factors=n_factors,
    )
    ent.set_train_options(
        iter=n_iterations,
        convergence_mode=convergence_mode,
        gpu_mode=gpu_mode,
        Y_ELBO_TauTrick=Y_ELBO_TauTrick,
        seed=seed,
        verbose=verbose,
        quiet=quiet,
        outfile=outfile,
        save_interrupted=save_interrupted,
    )

    if mefisto_covariate is not None:
        if mefisto_covariates_names is None:
            if isinstance(mefisto_covariate, str) or (
                isinstance(mefisto_covariate, list) and isinstance(mefisto_covariate[0], str)
            ):
                mefisto_covariates_names = mefisto_covariate
        ent.set_covariates(mefisto_covariate, covariates_names=mefisto_covariates_names)
        ent.set_smooth_options(
            scale_cov=mefisto_scale_cov,
            start_opt=mefisto_start_opt,
            n_grid=mefisto_n_grid,
            opt_freq=mefisto_opt_freq,
            model_groups=mefisto_model_groups,
            warping=mefisto_warping,
            warping_freq=mefisto_warping_freq,
            warping_ref=mefisto_warping_ref,
            warping_open_begin=mefisto_warping_open_begin,
            warping_open_end=mefisto_warping_open_end,
            sparseGP=mefisto_sparseGP,
            frac_inducing=mefisto_frac_inducing,
        )

    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Building the model...")
    ent.build()
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running the model...")
    ent.run()

    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving the model...")
    ent.save(
        outfile, save_data=save_data, save_parameters=save_parameters, expectations=expectations
    )

    f = h5py.File(outfile)
    if copy:
        data = data.copy()

    # Factors
    z = np.concatenate([v[:, :] for k, v in f["expectations"]["Z"].items()], axis=1).T

    # Samples are grouped per sample group
    # so the rows of the Z matrix have to be re-ordered
    if groups_label:
        zs = np.concatenate([v[:] for k, v in f["samples"].items()], axis=0).astype(str)

    if use_obs and use_obs == "intersection":  # data is MuData and common_obs is available
        if groups_label:
            z = pd.DataFrame(z, index=zs).loc[common_obs].to_numpy()
        # Set factor values outside of the obs intersection to nan
        data.obsm["X_mofa"] = np.empty(shape=(data.n_obs, z.shape[1]))
        data.obsm["X_mofa"][:] = np.nan
        # Samples
        data.obsm["X_mofa"][data.obs.index.isin(common_obs)] = z
    else:
        if groups_label:
            z = pd.DataFrame(z, index=zs).loc[mdata.obs.index.values].to_numpy()
        data.obsm["X_mofa"] = z

    # Weights
    w = np.concatenate([v[:, :] for k, v in f["expectations"]["W"].items()], axis=1).T
    if use_var:
        # Set the weights of features that were not used to zero
        data.varm["LFs"] = np.zeros(shape=(data.n_vars, w.shape[1]))
        data.varm["LFs"][data.var[use_var]] = w
    else:
        data.varm["LFs"] = w

    # aligned times
    if mefisto_covariate is not None and mefisto_warping:
        for c in range(ent.dimensionalities["C"]):
            cnm = ent.smooth_opts["covariates_names"][c] + "_warped"
            data.obs[cnm] = ent.model.getNodes()["Sigma"].sample_cov_transformed[:, c]

    if copy:
        return data
    else:
        print("Saved MOFA embeddings in .obsm['X_mofa'] slot and their loadings in .varm['LFs'].")

    return None


#
# Similarity network fusion (SNF)
#


def snf(mdata: MuData, key: str = "connectivities", k: int = 20, iterations: int = 20):
    """
    Similarity network fusion (SNF)

    See Wang et al., 2014 (DOI: 10.1038/nmeth.2810).

    Reference implementation can be found in the SNFtool R package:
    https://github.com/cran/SNFtool/blob/master/R/SNF.R

    PARAMETERS
    ----------
    mdata:
            MuData object
    key: str (default: 'connectivities')
            Key in .obsp to be used as SNF algorithm input.
            Has to exist in all modalities.
    k: int (default: 20)
            Number of neighbours to be used in the K-nearest neighbours step
    iterations: int (default: 20)
            Number of iterations for the diffusion process
    """
    wall = []
    for mod in mdata.mod:
        # TODO: check the key exists in every modality
        wall.append(mdata.mod[mod].obsp[key])

    def _normalize(x):
        row_sum_mdiag = x.sum(axis=1) - x.diagonal()
        row_sum_mdiag[row_sum_mdiag == 0] = 1
        x = x / (2 * row_sum_mdiag)
        np.fill_diagonal(x, 0.5)
        x = (x + x.T) / 2
        return x

    def _dominateset(x, k=20):
        def _zero(arr):
            arr[np.argsort(arr)[: (len(arr) - k)]] = 0
            return arr

        x = np.apply_along_axis(_zero, 0, wall[0])
        return x / x.sum(axis=1)

    for i in range(len(wall)):
        wall[i] = _normalize(wall[i])

    new = []
    for i in range(len(wall)):
        new.append(_dominateset(wall[i], k))

    nextW = [None] * len(wall)

    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting {iterations} iterations..."
    )
    for ti in range(iterations):
        for j in range(len(wall)):
            sumWJ = np.zeros(shape=(wall[j].shape[0], wall[j].shape[1]))
            for ki in range(len(wall)):
                if ki != j:
                    sumWJ = sumWJ + wall[ki]
            nextW[j] = new[j] * (sumWJ / (len(wall) - 1)) * new[j].T
        for j in range(len(wall)):
            wall[j] = _normalize(nextW[j])
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done: iteration {ti} of {iterations}."
        )

    # Sum diffused matrices
    w = np.sum(wall, axis=0)
    w = w / len(wall)
    w = _normalize(w)

    mdata.obsp[key] = w


#
# Clustering: Louvain and Leiden
#


def _cluster(
    data: Union[MuData, AnnData],
    resolution: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
    mod_weights: Optional[Union[Sequence[float], Mapping[str, float]]] = None,
    random_state: int = 0,
    key_added: str = "louvain",
    neighbors_key: str = None,
    directed: bool = True,
    partition_type: Optional[
        Union[Type[LeidenMutableVertexPartition], Type[LouvainMutableVertexPartition]]
    ] = None,
    partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
    algorithm: str = "leiden",  # Literal["leiden", "louvain"]
    **kwargs,
):
    """
    Cluster cells using the Leiden or Louvain algorithm.

    See :func:`scanpy.tl.leiden` and :func:`scanpy.tl.louvain` for details.
    """

    from scanpy.tools._utils import _choose_graph
    from scanpy._utils import get_igraph_from_adjacency

    if algorithm == "louvain":
        import louvain

        alg = louvain
    elif algorithm == "leiden":
        import leidenalg

        alg = leidenalg
    else:
        raise ValueError(f"Algorithms should be either 'louvain' or 'leiden', not '{algorithm}'")

    if isinstance(data, AnnData):
        sc_tl_cluster = sc.tl.leiden if algorithm == "leiden" else sc.tl.louvain
        return sc_tl_cluster(
            data,
            resolution=resolution,
            random_state=random_state,
            key_added=key_added,
            neighbors_key=neighbors_key,
            **kwargs,
        )
    elif isinstance(data, MuData):
        mdata = data
    else:
        raise TypeError("Expected a MuData object")

    partition_kwargs = dict(partition_kwargs)

    gs = {}

    for mod in mdata.mod:
        adjacency = _choose_graph(mdata.mod[mod], None, neighbors_key)
        g = get_igraph_from_adjacency(adjacency, directed=directed)

        gs[mod] = g

    if mod_weights:
        if isinstance(mod_weights, Mapping):
            layer_weights = [mod_weights.get(mod, 1) for mod in mdata.mod]
        elif isinstance(mod_weights, Sequence) and not isinstance(mod_weights, str):
            assert len(mod_weights) == len(
                mdata.mod
            ), f"Length of layers_weights ({len(mod_weights)}) does not match the number of modalities ({len(mdata.mod)})"
            layer_weights = mod_weights
        else:
            layer_weights = [mod_weights for _ in mdata.mod]
    else:
        layer_weights = None

    if partition_type is None:
        partition_type = alg.RBConfigurationVertexPartition

    optimiser = alg.Optimiser()
    if random_state:
        optimiser.set_rng_seed(random_state)

    # The same as leiden.find_partition_multiplex() (louvain.find_partition_multiplex())
    # but allows to specify resolution for each modality
    if resolution:
        if isinstance(resolution, Mapping):
            # Specific resolution for each modality
            parts = [
                partition_type(gs[mod], resolution_parameter=resolution[mod], **partition_kwargs)
                for mod in mdata.mod
            ]
        elif isinstance(resolution, Sequence) and not isinstance(resolution, str):
            assert len(resolution) == len(
                mdata.mod
            ), f"Length of resolution ({len(resolution)}) does not match the number of modalities ({len(mdata.mod)})"
            parts = [
                partition_type(gs[mod], resolution_parameter=resolution[i], **partition_kwargs)
                for i, mod in enumerate(mdata.mod)
            ]
        else:
            # Single resolution for all modalities
            parts = [
                partition_type(gs[mod], resolution_parameter=resolution, **partition_kwargs)
                for mod in mdata.mod
            ]
    else:
        parts = [partition_type(gs[mod], **partition_kwargs) for mod in mdata.mod]

    improv = optimiser.optimise_partition_multiplex(
        partitions=parts,
        layer_weights=layer_weights,
        **kwargs,
    )

    # All partitions are the same
    groups = np.array(parts[0].membership)

    mdata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    mdata.uns[algorithm] = {}
    mdata.uns[algorithm]["params"] = dict(
        resolution=resolution,
        random_state=random_state,
        partition_improvement=improv,
    )

    return None


def leiden(
    data: Union[MuData, AnnData],
    resolution: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
    mod_weights: Optional[Union[Sequence[float], Mapping[str, float]]] = None,
    random_state: int = 0,
    key_added: str = "leiden",
    neighbors_key: str = None,
    directed: bool = True,
    partition_type: Optional[Type[LeidenMutableVertexPartition]] = None,
    partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs,
):
    """
    Cluster cells using the Leiden algorithm.

    This runs only the multiplex Leiden algorithm on the MuData object
    using connectivities of individual modalities
    (see `documentation <https://leidenalg.readthedocs.io/en/stable/multiplex.html>`_ for more details).
    For that, :func:`scanpy.pp.neighbors` should be run first for each modality.

    For taking use of ``mdata.obsp['connectivities']``, it's :func:`scanpy.tl.leiden` that should be used.
    See :func:`scanpy.tl.leiden` for details.

    Parameters
    ----------
    data
        :class:`muon.MuData` object.
    resolution
        Resolution parameter controlling coarseness of the clustering
        (higher values -> more clusters).
        To use different resolution per modality, dictionary ``{mod: value}``
        or list/tuple ``[value_mod1, value_mod2, ...]``.
        Single value to use the same resolution for all modalities.
    mod_weights
        Weight each modality controlling its contribution
        (higher values -> more important).
        To use different weight per modality, dictionary ``{mod: value}``
        or list/tuple ``[value_mod1, value_mod2, ...]``.
        Single value to use the same weight for all modalities.
    random_state
        Random seed for the optimization.
    key_added
        `mdata.obs` key where cluster labels to be added.
    neighbors_key
        Use neighbors connectivities as adjacency.
        If not specified, look for ``.obsp['connectivities']`` in each modality.
        If specified, look for
        ``.obsp[.uns[neighbors_key]['connectivities_key']]`` in each modality
        for connectivities.
    directed
        Treat the graph as directed or undirected.
    partition_type
        Type of partition to use,
        :class:`~leidenalg.RBConfigurationVertexPartition` by default.
        See :func:`~leidenalg.find_partition` for more details.
    partition_kwargs
        Arguments to be passed to the ``partition_type``.
    **kwargs
        Arguments to be passed to ``optimizer.optimise_partition_multiplex()``.
    """

    return _cluster(
        data=data,
        resolution=resolution,
        mod_weights=mod_weights,
        random_state=random_state,
        key_added=key_added,
        neighbors_key=neighbors_key,
        directed=directed,
        partition_type=partition_type,
        partition_kwargs=partition_kwargs,
        algorithm="leiden",
        **kwargs,
    )


def louvain(
    data: Union[MuData, AnnData],
    resolution: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
    mod_weights: Optional[Union[Sequence[float], Mapping[str, float]]] = None,
    random_state: int = 0,
    key_added: str = "louvain",
    neighbors_key: str = None,
    directed: bool = True,
    partition_type: Optional[Type[LouvainMutableVertexPartition]] = None,
    partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs,
):
    """
    Cluster cells using the Louvain algorithm.

    This runs only the multiplex Louvain algorithm on the MuData object
    using connectivities of individual modalities
    (see `documentation <https://louvain-igraph.readthedocs.io/en/latest/multiplex.html>`_ for more details).
    For that, :func:`scanpy.pp.neighbors` should be run first for each modality.

    For taking use of ``mdata.obsp['connectivities']``, it's :func:`scanpy.tl.louvain` that should be used.
    See :func:`scanpy.tl.louvain` for details.

    Parameters
    ----------
    data
        :class:`muon.MuData` object.
    resolution
        Resolution parameter controlling coarseness of the clustering
        (higher values -> more clusters).
        To use different resolution per modality, dictionary ``{mod: value}``
        or list/tuple ``[value_mod1, value_mod2, ...]``.
        Single value to use the same resolution for all modalities.
    mod_weights
        Weight each modality controlling its contribution
        (higher values -> more important).
        To use different weight per modality, dictionary ``{mod: value}``
        or list/tuple ``[value_mod1, value_mod2, ...]``.
        Single value to use the same weight for all modalities.
    random_state
        Random seed for the optimization.
    key_added
        `mdata.obs` key where cluster labels to be added.
    neighbors_key
        Use neighbors connectivities as adjacency.
        If not specified, look for ``.obsp['connectivities']`` in each modality.
        If specified, look for
        ``.obsp[.uns[neighbors_key]['connectivities_key']]`` in each modality
        for connectivities.
    directed
        Treat the graph as directed or undirected.
    partition_type
        Type of partition to use,
        :class:`~louvain.RBConfigurationVertexPartition` by default.
        See :func:`~louvain.find_partition` for more details.
    partition_kwargs
        Arguments to be passed to the ``partition_type``.
    **kwargs
        Arguments to be passed to ``optimizer.optimise_partition_multiplex()``.
    """

    return _cluster(
        data=data,
        resolution=resolution,
        mod_weights=mod_weights,
        random_state=random_state,
        key_added=key_added,
        neighbors_key=neighbors_key,
        directed=directed,
        partition_type=partition_type,
        partition_kwargs=partition_kwargs,
        algorithm="louvain",
        **kwargs,
    )
