import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from functools import reduce
from pathlib import Path
from time import strftime
from types import MappingProxyType

# from scanpy.neighbors import _compute_connectivities_umap
from typing import (
    Any,
    Literal,
)
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from natsort import natsorted
from scanpy import logging
from scanpy.tools._utils import _choose_representation
from scipy.sparse import csr_matrix, issparse

from .preproc import _sparse_csr_fast_knn

try:
    from louvain.VertexPartition import (
        MutableVertexPartition as LouvainMutableVertexPartition,
    )
except ImportError:

    class LouvainMutableVertexPartition:
        pass

    LouvainMutableVertexPartition.__module__ = "louvain.VertexPartition"

try:
    from leidenalg.VertexPartition import (
        MutableVertexPartition as LeidenMutableVertexPartition,
    )
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
        from mofapy2.build_model.utils import guess_likelihoods, process_data
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
            print(f"Error: {groups_label} is not in observations names")
            sys.stdout.flush()
            sys.exit()
        n_groups = mdata.obs[groups_label].unique().shape[0]

    # Get the respective data slot
    data = []
    if use_layer:
        for m in mdata.mod.keys():
            adata = mdata.mod[m]
            if use_layer in adata.layers.keys():
                if issparse(adata.layers[use_layer]):
                    data.append(np.array(adata.layers[use_layer].todense()))
                else:
                    data.append(adata.layers[use_layer].copy())
            else:
                print(f"Error: Layer {use_layer} does not exist")
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
            if issparse(adata.X):
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
            if issparse(mdata[m].X):
                x[ix, :] = np.array(mdata[m].X.todense())
            else:
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
        model.data_opts["samples_groups"] = np.concatenate(samples_groups.values).astype(str)
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
                    mdata.obs.reset_index(drop=True)
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
            view_name = model.data_opts["views_names"][m]
            group_name = model.data_opts["groups_names"][g]
            print(
                f"Loaded view='{view_name}' group='{group_name}' with N={str(n_grouped[g])} samples and D={str(D[m])} features..."
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
    data: AnnData | MuData,
    groups_label: bool = None,
    use_raw: bool = False,
    use_layer: str = None,
    use_var: str | None = "highly_variable",
    use_obs: str | None = None,
    likelihoods: str | list[str] | None = None,
    n_factors: int = 10,
    scale_views: bool = False,
    scale_groups: bool = False,
    center_groups: bool = True,
    ard_weights: bool = True,
    ard_factors: bool = True,
    spikeslab_weights: bool = True,
    spikeslab_factors: bool = False,
    n_iterations: int = 1000,
    convergence_mode: str = "fast",
    use_float32: bool = False,
    gpu_mode: bool = False,
    gpu_device: bool | None = None,
    svi_mode: bool = False,
    svi_batch_size: float = 0.5,
    svi_learning_rate: float = 1.0,
    svi_forgetting_rate: float = 0.5,
    svi_start_stochastic: int = 1,
    smooth_covariate: str | None = None,
    smooth_warping: bool = False,
    smooth_kwargs: Mapping[str, Any] | None = None,
    save_parameters: bool = False,
    save_data: bool = True,
    save_metadata: bool = True,
    seed: int = 1,
    outfile: str | None = None,
    expectations: list[str] | None = None,
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
    center_groups : optional
            center groups to zero mean (True by default)
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
    use_float32 : optional
            use reduced precision (float32)
    gpu_mode : optional
            if to use GPU mode
    gpu_mode : optional
            which GPU device to use
    svi_mode : optional
            if to use Stochastic Variational Inference (SVI)
    svi_batch_size : optional
            batch size as a fraction (only applicable when svi_mode=True, 0.5 by default)
    svi_learning_rate : optional
            learning rate (only applicable when svi_mode=True, 1.0 by default)
    svi_forgetting_rate : optional
            forgetting_rate (only applicable when svi_mode=True, 0.5 by default)
    svi_start_stochastic : optional
            first iteration to start SVI (only applicable when svi_mode=True, 1 by default)
    smooth_covariate : optional
            use a covariate (column in .obs) to learn smooth factors (MEFISTO)
    smooth_warping : optional
            if to learn the alignment of covariates (e.g. time points) from different groups;
            by default, the first group is used as a reference, which can be adjusted by setting
            the REF_GROUP in smooth_kwargs = { "warping_ref": REF_GROUP } (MEFISTO)
    smooth_kwargs : optional
            additional arguments for MEFISTO (covariates_names, scale_cov, start_opt, n_grid, opt_freq,
            warping_freq, warping_ref, warping_open_begin, warping_open_end,
            sparseGP, frac_inducing, model_groups, new_values)
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
        # Modality name is used as a prefix by default
        mdata.obs = data.obs
    elif isinstance(data, MuData):
        mdata = data
    else:
        raise TypeError("Expected an MuData object")

    if outfile is None:
        outfile = str(Path("/tmp") / "mofa_{}.hdf5".format(strftime("%Y%m%d-%H%M%S")))

    if use_var and use_var not in data.var.columns:
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
    if lik is not None and (isinstance(lik, str) and isinstance(lik, Iterable)):
        lik = [lik for _ in range(len(mdata.mod))]

    ent.set_data_options(
        scale_views=scale_views,
        scale_groups=scale_groups,
        center_groups=center_groups,
        use_float32=use_float32,
    )
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
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Setting model options...")
    ent.set_model_options(
        ard_factors=ard_factors,
        ard_weights=ard_weights,
        spikeslab_weights=spikeslab_weights,
        spikeslab_factors=spikeslab_factors,
        factors=n_factors,
    )
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Setting training options...")

    try:
        ent.set_train_options(
            iter=n_iterations,
            convergence_mode=convergence_mode,
            gpu_mode=gpu_mode,
            gpu_device=gpu_device,
            seed=seed,
            verbose=verbose,
            quiet=quiet,
            outfile=outfile,
            save_interrupted=save_interrupted,
        )
    except TypeError:
        # mofapy2 <0.7 does not have a gpu_device argument
        if gpu_device is not None:
            logging.warning(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Upgrade mofapy2 to 0.7 or above in order to support custrom gpu_device setting."
            )
        ent.set_train_options(
            iter=n_iterations,
            convergence_mode=convergence_mode,
            gpu_mode=gpu_mode,
            seed=seed,
            verbose=verbose,
            quiet=quiet,
            outfile=outfile,
            save_interrupted=save_interrupted,
        )

    if svi_mode:
        logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Setting up SVI...")
        ent.set_stochastic_options(
            learning_rate=svi_learning_rate,
            forgetting_rate=svi_forgetting_rate,
            batch_size=svi_batch_size,
            start_stochastic=svi_start_stochastic,
        )

    # MEFISTO options

    smooth_kwargs_default = dict(
        covariates_names=smooth_covariate,
        scale_cov=False,
        start_opt=20,
        n_grid=20,
        opt_freq=10,
        model_groups=True,
        warping_freq=20,
        warping_ref=0,
        warping_open_begin=True,
        warping_open_end=True,
        sparseGP=False,
        frac_inducing=None,
        new_values=None,
    )

    if not smooth_kwargs:
        smooth_kwargs = {}

    # warping_ref has to be an integer
    if "warping_ref" in smooth_kwargs:
        warping_ref = smooth_kwargs["warping_ref"]
        if not (isinstance("warping_ref", int)):
            warping_ref = np.where(np.array(ent.data_opts["groups_names"]) == warping_ref)[0]
            if len(warping_ref) == 0:
                raise KeyError(
                    f"Expected 'warping_ref' for be a group name but there is no group {warping_ref}"
                )
            smooth_kwargs["warping_ref"] = warping_ref[0]

    # Add default options where they are not provided
    smooth_kwargs = {**smooth_kwargs_default, **smooth_kwargs}

    if smooth_covariate is not None:
        logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adding smooth options...")
        ent.set_covariates(smooth_covariate, covariates_names=smooth_kwargs["covariates_names"])
        ent.set_smooth_options(
            scale_cov=smooth_kwargs["scale_cov"],
            start_opt=smooth_kwargs["start_opt"],
            n_grid=smooth_kwargs["n_grid"],
            opt_freq=smooth_kwargs["opt_freq"],
            model_groups=smooth_kwargs["model_groups"],
            warping=smooth_warping,
            warping_freq=smooth_kwargs["warping_freq"],
            warping_ref=smooth_kwargs["warping_ref"],
            warping_open_begin=smooth_kwargs["warping_open_begin"],
            warping_open_end=smooth_kwargs["warping_open_end"],
            sparseGP=smooth_kwargs["sparseGP"],
            frac_inducing=smooth_kwargs["frac_inducing"],
        )

    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Building the model...")
    ent.build()
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running the model...")
    ent.run()

    if (
        smooth_kwargs is not None
        and "new_values" in smooth_kwargs
        and smooth_kwargs["new_values"]
        and smooth_covariate
    ):
        logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Interpolating factors...")
        new_values = np.array(smooth_kwargs["new_values"])
        if new_values.ndim == 1:
            new_values = new_values.reshape(-1, 1)
        ent.predict_factor(new_covariates=new_values)

    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving the model...")
    ent.save(
        outfile, save_data=save_data, save_parameters=save_parameters, expectations=expectations
    )

    f = h5py.File(outfile, "r")
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
    expectations_w = f["expectations"]["W"]
    if hasattr(data, "mod"):
        w = np.concatenate([expectations_w[m][:, :] for m in data.mod], axis=1).T
    else:
        w = expectations_w["data"][:, :].T

    if use_var:
        # Set the weights of features that were not used to zero
        data.varm["LFs"] = np.zeros(shape=(data.n_vars, w.shape[1]))
        data.varm["LFs"][data.var[use_var].astype(bool)] = w
    else:
        data.varm["LFs"] = w

    # Aligned times
    if smooth_covariate is not None and smooth_warping:
        for c in range(ent.dimensionalities["C"]):
            cnm = ent.smooth_opts["covariates_names"][c] + "_warped"
            cval = ent.model.getNodes()["Sigma"].sample_cov_transformed[:, c]
            if groups_label:
                cval = pd.DataFrame(cval, index=zs).loc[common_obs].to_numpy()
            data.obs[cnm] = cval

    # Parameters
    data.uns["mofa"] = {
        "params": {
            "data": {
                "groups_label": groups_label,
                "use_raw": use_raw,
                "use_layer": use_layer,
                "likelihoods": f["model_options"]["likelihoods"][:].astype(str),
                "features_subset": use_var,
                "use_obs": use_obs,
                "scale_views": scale_views,
                "scale_groups": scale_groups,
                "center_groups": center_groups,
                "use_float32": use_float32,
            },
            "model": {
                "ard_factors": ard_factors,
                "ard_weights": ard_weights,
                "spikeslab_weights": spikeslab_weights,
                "spikeslab_factors": spikeslab_factors,
                "n_factors": n_factors,
            },
            "training": {
                "n_iterations": n_iterations,
                "convergence_mode": convergence_mode,
                "gpu_mode": gpu_mode,
                "seed": seed,
            },
        }
    }

    # Variance explained
    try:
        views = f["views"]["views"][:].astype(str)
        variance_per_group = f["variance_explained"]["r2_per_factor"]
        variance = {m: {} for m in views}

        groups = f["groups"]["groups"][:].astype(str)
        if len(groups) > 1:
            for group in list(variance_per_group.keys()):
                for i, view in enumerate(views):
                    variance[view][group] = variance_per_group[group][i, :]
        else:
            for i, view in enumerate(views):
                variance[view] = variance_per_group[groups[0]][i, :]
        data.uns["mofa"]["variance"] = variance
    except Exception:
        warn("Cannot save variance estimates")

    f.close()

    if copy:
        return data
    else:
        print("Saved MOFA embeddings in .obsm['X_mofa'] slot and their loadings in .varm['LFs'].")

    return None


#
# Similarity network fusion (SNF)
#


def snf(
    mdata: MuData,
    n_neighbors: int = 20,
    neighbor_keys: str | dict[str, str | None] | None = None,
    key_added: str | None = None,
    n_iterations: int = 20,
    sigma: float = 0.5,
    eps: float = np.finfo(np.float64).eps,
    copy: bool = False,
) -> MuData | None:
    """
    Similarity network fusion (SNF)

    See Wang et al., 2014 (DOI: 10.1038/nmeth.2810).

    Reference implementation can be found in the SNFtool R package:
    https://github.com/cran/SNFtool/blob/master/R/SNF.R

    PARAMETERS
    ----------
    mdata:
            MuData object
    n_neighbors: int (default: 20)
            Number of neighbours to be used in the K-nearest neighbours step
    neighbor_keys: Keys in .uns where per-modality neighborhood information is stored. Defaults to ``"neighbors"``.
            If set as a dictionary, only the modalities present in ``neighbor_keys`` will be used for multimodal nearest neighbor search.
            If set as a string, has to exist in all modalities.
    key_added: If not specified, the multimodal neighbors data is stored in ``.uns["neighbors"]``, distances and
            connectivities are stored in ``.obsp["distances"]`` and ``.obsp["connectivities"]``, respectively. If specified, the
            neighbors data is added to ``.uns[key_added]``, distances are stored in ``.obsp[key_added + "_distances"]`` and
            connectivities in ``.obsp[key_added + "_connectivities"]``.
    n_iterations: int (default: 20)
            Number of iterations for the diffusion process
    sigma: float (default: 0.5)
            Variance for the local model when calculating affinity matrices
    eps: Small number to avoid numerical errors.
    copy: Return a copy instead of writing to ``mdata``.
    """
    import scipy.stats as stats

    mdata = mdata.copy() if copy else mdata

    if neighbor_keys is None:
        modalities = mdata.mod.keys()
        neighbor_keys = {}
    elif isinstance(neighbor_keys, str):
        modalities = mdata.mod.keys()
        neighbor_keys = {m: neighbor_keys for m in modalities}
    else:
        modalities = neighbor_keys.keys()

    mod_neighbors = np.empty((len(modalities),), dtype=np.uint16)
    mod_reps = {}
    reps = {}
    mod_n_pcs = {}
    neighbors_params = {}
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
        reps[mod] = _choose_representation(adata=mdata.mod[mod], use_rep=use_rep, n_pcs=n_pcs)
        mod_reps[mod] = (
            use_rep if use_rep is not None else -1
        )  # otherwise this is not saved to h5mu
        mod_n_pcs[mod] = n_pcs if n_pcs is not None else -1

    def _affinity_matrix(dist, k, sigma):
        """
        Compute the affinity matrix for a distance matrix

        Reference implementation can be found in the SNFtool R package:
        https://github.com/cran/SNFtool/blob/master/R/affinityMatrix.R

        PARAMETERS
        ----------
        mdata:
                MuData object
        k: int (default: 20)
                Number of neighbours to be used in the K-nearest neighbours step
        sigma: float (default: 0.5)
                Variance for the local model when calculating affinity matrices
        """
        dist = (dist + dist.T) / 2
        if issparse(dist):
            dist.setdiag(0)
            dist.eliminate_zeros()
        else:
            np.fill_diagonal(dist, 0)

        # FIXME: adopt for sparse matrices
        if issparse(dist):
            logging.warning(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using dense distance matrix when computing affinity matrix..."
            )
            dist = dist.todense()
        sorted_columns = np.apply_along_axis(np.sort, 1, dist)

        def finite_mean(x, *args, **kwargs):
            return np.mean(x[~np.isinf(x)], *args, **kwargs)

        means = np.apply_along_axis(finite_mean, 1, sorted_columns[:, 1 : k + 1]) + eps
        sig = np.add.outer(means, means) / 3 + dist / 3 + eps
        densities = stats.norm(0, sigma * sig).pdf(dist)

        w = (densities + densities.T) / 2
        return w

    wall = []
    for mod in mdata.mod:
        nkey = neighbor_keys.get(mod, "neighbors")
        # the key has to exists in every modality
        if nkey not in mdata.mod[mod].uns:
            raise ValueError(f"The key '{nkey}' is missing from the .uns slot of modality '{mod}'")
        mod_distances = mdata.mod[mod].obsp[neighbors_params[mod]["distances_key"]]
        w = _affinity_matrix(mod_distances, k=n_neighbors, sigma=sigma)
        wall.append(w)

    def _normalize(x):
        row_sum_mdiag = x.sum(axis=1) - x.diagonal()
        row_sum_mdiag[row_sum_mdiag == 0] = 1
        x = x / (2 * row_sum_mdiag[:, None])
        np.fill_diagonal(x, 0.5)
        x = (x + x.T) / 2
        return x

    def _dominateset(x, k=20):
        def _zero(arr):
            if k >= len(arr):
                raise ValueError("'n_neighbors' seems to be too high.")
            arr = arr.copy()
            arr[np.argsort(arr)[: (len(arr) - k)]] = 0
            return arr

        x = np.apply_along_axis(_zero, 0, x)
        return x / x.sum(axis=1)

    for i in range(len(wall)):
        wall[i] = _normalize(wall[i])

    new = []
    for i in range(len(wall)):
        new.append(_dominateset(wall[i], n_neighbors))

    nextW = [None] * len(wall)

    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting {n_iterations} iterations..."
    )
    for ti in range(n_iterations):
        for j in range(len(wall)):
            sumWJ = np.zeros(shape=(wall[j].shape[0], wall[j].shape[1]))
            for ki in range(len(wall)):
                if ki != j:
                    sumWJ = sumWJ + wall[ki]
            nextW[j] = np.dot(np.dot(new[j], (sumWJ / (len(wall) - 1))), new[j].T)
        for j in range(len(wall)):
            wall[j] = _normalize(nextW[j])
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done: iteration {ti} of {n_iterations}."
        )

    # Sum diffused matrices
    w = np.sum(wall, axis=0)
    w = w / len(wall)
    w = _normalize(w)

    # Keep n neighbours
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculating distances...")
    neighbordistances = _sparse_csr_fast_knn(csr_matrix(0.5 - w), n_neighbors)

    # TODO: use _compute_connectivities_umap?
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculating connectivities...")
    connectivities = _sparse_csr_fast_knn(csr_matrix(w), n_neighbors)

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
        "eps": eps,
        "use_rep": mod_reps,
        "n_pcs": mod_n_pcs,
        "method": "snf",
    }
    mdata.obsp[conns_key] = connectivities
    mdata.obsp[dists_key] = neighbordistances
    mdata.uns[key_added] = neighbors_dict

    return mdata if copy else None


#
# Clustering: Louvain and Leiden
#


def _cluster(
    data: MuData | AnnData,
    resolution: float | Sequence[float] | Mapping[str, float] | None = None,
    mod_weights: Sequence[float] | Mapping[str, float] | None = None,
    random_state: int = 0,
    key_added: str = "louvain",
    neighbors_key: str = None,
    directed: bool = True,
    partition_type: (
        type[LeidenMutableVertexPartition] | type[LouvainMutableVertexPartition] | None
    ) = None,
    partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
    algorithm: str = "leiden",  # Literal["leiden", "louvain"]
    **kwargs,
):
    """
    Cluster cells using the Leiden or Louvain algorithm.

    See :func:`scanpy.tl.leiden` and :func:`scanpy.tl.louvain` for details.
    """

    from scanpy._utils import get_igraph_from_adjacency
    from scanpy.tools._utils import _choose_graph

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
    data: MuData | AnnData,
    resolution: float | Sequence[float] | Mapping[str, float] | None = None,
    mod_weights: Sequence[float] | Mapping[str, float] | None = None,
    random_state: int = 0,
    key_added: str = "leiden",
    neighbors_key: str = None,
    directed: bool = True,
    partition_type: type[LeidenMutableVertexPartition] | None = None,
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
    data: MuData | AnnData,
    resolution: float | Sequence[float] | Mapping[str, float] | None = None,
    mod_weights: Sequence[float] | Mapping[str, float] | None = None,
    random_state: int = 0,
    key_added: str = "louvain",
    neighbors_key: str = None,
    directed: bool = True,
    partition_type: type[LouvainMutableVertexPartition] | None = None,
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


def umap(
    mdata: MuData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Literal["spectral", "random"] | np.ndarray | None = "spectral",
    random_state: int | np.random.RandomState | None = 42,
    a: float | None = None,
    b: float | None = None,
    copy: bool = False,
    method: Literal["umap", "rapids"] = "umap",
    neighbors_key: str | None = None,
) -> MuData | None:
    """
    Embed the multimodal neighborhood graph using UMAP (McInnes et al, 2018).

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. We use ScanPy's
    implementation.

    References:
        McInnes et al, 2018 (`arXiv:1802.03426` <https://arxiv.org/abs/1802.03426>`_)

    Args:
        mdata: MuData object. Multimodal nearest neighbor search must have already
            been performed.
        min_dist: The effective minimum distance between embedded points. Smaller
            values will result in a more clustered/clumped embedding where nearby points
            on the manifold are drawn closer together, while larger values will result
            on a more even dispersal of points. The value should be set relative to
            the ``spread`` value, which determines the scale at which embedded points
            will be spread out. The default of in the ``umap-learn`` package is 0.1.
        spread: The effective scale of embedded points. In combination with ``min_dist``
            this determines how clustered/clumped the embedded points are.
        n_components: The number of dimensions of the embedding.
        maxiter: The number of iterations (epochs) of the optimization. Called ``n_epochs``
            in the original UMAP.
        alpha: The initial learning rate for the embedding optimization.
        gamma: Weighting applied to negative samples in low dimensional embedding
            optimization. Values higher than one will result in greater weight
            being given to negative samples.
        negative_sample_rate: The number of negative edge/1-simplex samples to use per
            positive edge/1-simplex sample in optimizing the low dimensional embedding.
        init_pos: How to initialize the low dimensional embedding. Called ``init`` in the
            original UMAP. Options are:
            - 'spectral': use a spectral embedding of the graph.
            - 'random': assign initial embedding positions at random.
            - A numpy array of initial embedding positions.
        random_state: Random seed.
        a: More specific parameters controlling the embedding. If ``None`` these
            values are set automatically as determined by ``min_dist`` and
            ``spread``.
        b: More specific parameters controlling the embedding. If ``None`` these
            values are set automatically as determined by ``min_dist`` and
            ``spread``.
        copy: Return a copy instead of writing to mdata.
        method: Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
        neighbors_key: If not specified, umap looks in ``.uns['neighbors']`` for neighbors
            settings and ``.obsp['connectivities']`` for connectivities (default storage
            places for ``pp.neighbors``). If specified, umap looks ``.uns[neighbors_key]``
            for neighbors settings and ``.obsp[.uns[neighbors_key]['connectivities_key']]``
            for connectivities.
    Returns: Depending on ``copy``, returns or updates ``adata`` with the following fields.

        **X_umap** : ``mdata.obsm`` field holding UMAP coordinates of data.
    """
    if isinstance(mdata, AnnData):
        return sc.tl.umap(
            adata=mdata,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            maxiter=maxiter,
            alpha=alpha,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            init_pos=init_pos,
            random_state=random_state,
            a=a,
            b=b,
            copy=copy,
            method=method,
            neighbors_key=neighbors_key,
        )

    if neighbors_key is None:
        neighbors_key = "neighbors"

    try:
        neighbors = mdata.uns[neighbors_key]
    except KeyError:
        raise ValueError(f'Did not find .uns["{neighbors_key}"]. Run `muon.pp.neighbors` first.')

    from copy import deepcopy

    from scanpy.tools._utils import _choose_representation
    from scipy.sparse import issparse

    # we need a data matrix. This is used only for initialization and only if init_pos=="spectral"
    # and the graph has many connected components, so we can do very simple imputation
    reps = {}
    nfeatures = 0
    nparams = neighbors["params"]
    use_rep = {k: (v if v != -1 else None) for k, v in nparams["use_rep"].items()}
    n_pcs = {k: (v if v != -1 else None) for k, v in nparams["n_pcs"].items()}
    observations = mdata.obs.index
    for mod, rep in use_rep.items():
        rep = _choose_representation(adata=mdata.mod[mod], use_rep=rep, n_pcs=n_pcs[mod])
        nfeatures += rep.shape[1]
        reps[mod] = rep
    rep = np.empty((len(observations), nfeatures), np.float32)
    nfeatures = 0
    for mod, crep in reps.items():
        cnfeatures = nfeatures + crep.shape[1]
        idx = observations.isin(mdata.mod[mod].obs.index)
        rep[idx, nfeatures:cnfeatures] = crep.toarray() if issparse(crep) else crep
        if np.sum(idx) < rep.shape[0]:
            imputed = crep.mean(axis=0)
            if issparse(crep):
                imputed = np.asarray(imputed).squeeze()
            rep[~idx, nfeatures : crep.shape[1]] = imputed
        nfeatures = cnfeatures
    adata = AnnData(X=rep, obs=mdata.obs)
    adata.uns[neighbors_key] = deepcopy(neighbors)
    adata.uns[neighbors_key]["params"]["use_rep"] = "X"
    del adata.uns[neighbors_key]["params"]["n_pcs"]
    adata.obsp[neighbors["connectivities_key"]] = mdata.obsp[neighbors["connectivities_key"]]
    adata.obsp[neighbors["distances_key"]] = mdata.obsp[neighbors["distances_key"]]

    sc.tl.umap(
        adata=adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        maxiter=maxiter,
        alpha=alpha,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        init_pos=init_pos,
        random_state=random_state,
        a=a,
        b=b,
        copy=False,
        method=method,
        neighbors_key=neighbors_key,
    )

    mdata = mdata.copy() if copy else mdata
    mdata.obsm["X_umap"] = adata.obsm["X_umap"]
    mdata.uns["umap"] = adata.uns["umap"]
    return mdata if copy else None


def ica(
    data: AnnData | MuData,
    basis="X_pca",
    n_components=None,
    *,
    random_state=None,
    scale=False,
    copy=False,
    **kwargs,
):
    """Run Independent component analysis"""
    from sklearn.decomposition import FastICA

    ica = FastICA(random_state=random_state, n_components=n_components, **kwargs)
    x_ica = ica.fit_transcrotm(data.obsm[basis])

    if scale:
        x_ica /= x_ica.std(axis=0)

    data = data.copy() if copy else data
    data.obsm["X_ica"] = x_ica
    return data if copy else None
