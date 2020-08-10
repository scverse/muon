import sys
from typing import Union, Optional, List, Iterable
import logging

import numpy as np
from anndata import AnnData
from .mudata import MuData

def _set_mofa_data_from_mudata(model, mdata, groups_label=None, use_raw=False, use_layer=None, likelihoods=None, features_subset=None, save_metadata=None):
	""" Method to input the data in AnnData format

	PARAMETERS
	----------
	model: MOFA+ model entry point object
	mdata: a MuData object
	groups_label (optional): a column name in adata.obs for grouping the samples
	use_raw (optional): use raw slot of AnnData as input values
	use_layer (optional): use a specific layer of AnnData as input values (supersedes use_raw option)
	likelihoods (optional): likelihoods to use (guessed from the data if not provided)
	features_subset (optional): .var column with a boolean value to select genes (e.g. "highly_variable"), None by default
	"""

	# Sanity checks
	if not hasattr(model, 'data_opts'): 
		# print("Data options not defined before setting the data, using default values...")
		model.set_data_options()

	# Check groups_label is defined properly
	n_groups = 1  # no grouping by default
	if groups_label is not None:
		if not isinstance(groups_label, str):
			print("Error: groups_label should be a string present in the observations column names"); sys.stdout.flush(); sys.exit()
		if groups_label not in mdata.obs.columns:
			print("Error: {} is not in observations names".format(groups_label)); sys.stdout.flush(); sys.exit()
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
					data.append(adata.layers[use_layer])
			else:
				print("Error: Layer {} does not exist".format(use_layer)); sys.stdout.flush(); sys.exit()
	elif use_raw:
		for m in mdata.mod.keys():
			adata = mdata.mod[m]
			adata_raw_dense = np.array(adata.raw[:,adata.var_names].X.todense())
			data.append(adata_raw_dense)
	else:
		for m in mdata.mod.keys():
			adata = mdata.mod[m]
			if callable(getattr(adata.X, "todense", None)):
			    data.append(np.array(adata.X.todense()))
			else:
			    data.append(adata.X)
	
	# Subset features if required
	if features_subset is not None:
		for i, m in enumerate(mdata.mod.keys()):
			data[i] = data[i][:,mdata.mod[m].var[features_subset].values]

	# Save dimensionalities
	M = model.dimensionalities["M"] = len(mdata.mod)
	G = model.dimensionalities["G"] = n_groups
	N = model.dimensionalities["N"] = mdata.shape[0]
	D = model.dimensionalities["D"] = [mdata[m].shape[1] for m in mdata.mod]  # Feature may have been filtered
	n_grouped = [mdata.shape[0]] if n_groups == 1 else mdata.obs.groupby(groups_label).size().values

	# Define views names and features names and metadata
	model.data_opts['views_names'] = list(mdata.mod.keys())
	
	if features_subset is not None:
		model.data_opts['features_names'] = [adata.var_names[adata.var[features_subset].values] for adata in mdata.mod.values()]
	else:
		model.data_opts['features_names'] = [adata.var_names for adata in mdata.mod.values()]

	if save_metadata:
		if features_subset is not None:
			model.data_opts['features_metadata'] = [adata.var[adata.var[features_subset].values] for adata in mdata.mod.values()]
		else:
			model.data_opts['features_metadata'] = [adata.var]

	# Define groups and samples names and metadata
	if groups_label is None:
		model.data_opts['groups_names'] = ["group1"]
		model.data_opts['samples_names'] = [mdata.obs.index.values.tolist()]
		model.data_opts['samples_groups'] = ["group1"] * N
		if save_metadata:
			model.data_opts['samples_metadata'] = [mdata.obs]
	else:
		# While grouping the pandas.DataFrame, the group_label would be sorted.
		# Hence the naive implementation `mdata.obs[groups_label].unique()` to get group names
		# wouldn't match samples_names if the samples are not ordered according to their group beforehand.

		# List of names of groups, i.e. [group1, group2, ...]
		model.data_opts['groups_names'] = [str(g) for g in mdata.obs.reset_index(drop=False).groupby(groups_label)[groups_label].apply(list).index.values]
		# Nested list of names of samples, one inner list per group, i.e. [[group1_sample1, group1_sample2, ...], ...]
		model.data_opts['samples_names'] = mdata.obs.reset_index(drop=False).rename(columns={mdata.obs.index.name:'index'}).groupby(groups_label)["index"].apply(list).tolist()
		# List of names of groups for samples ordered as they are in the original data, i.e. [group2, group1, group1, ...]
		model.data_opts['samples_groups'] = mdata.obs[groups_label].values.astype(str)
		if save_metadata:
			# List of metadata tables for each group of samples
			model.data_opts['samples_metadata'] = [g for _, g in mdata.obs.groupby(groups_label)]


	# If everything successful, print verbose message
	for m in range(M):
		for g in range(G):
			print("Loaded view='%s' group='%s' with N=%d samples and D=%d features..." % (model.data_opts['views_names'][m], model.data_opts['groups_names'][g], n_grouped[g], D[m]))
	print("\n")


	# Store intercepts (it is for one view only)
	model.intercepts = [[]]

	# Define likelihoods
	if likelihoods is None:
		likelihoods = guess_likelihoods(data)
	assert len(likelihoods) == model.dimensionalities["M"], "Please specify one likelihood for each view"
	assert set(likelihoods).issubset(set(["gaussian", "bernoulli", "poisson"])), "Available likelihoods are 'gaussian', 'bernoulli', 'poisson'"
	model.likelihoods = likelihoods

	# Process the data (center, scaling, etc.)
	for g in model.data_opts['groups_names']:
		samples_idx = np.where(np.array(model.data_opts['samples_groups']) == g)[0]
		model.intercepts[0].append(np.nanmean(data[0][samples_idx,:], axis=0))
	model.data = process_data(data, likelihoods, model.data_opts, model.data_opts['samples_groups'])


def mofa(data: Union[AnnData, MuData], groups_label: bool = None, 
		 use_raw: bool = False, use_layer: bool = None, 
		 features_subset: Optional[str] = None,
		 likelihoods: Optional[Union[str, List[str]]] = None, n_factors: int = 10,
		 scale_views: bool = False, scale_groups: bool = False,
		 ard_weights: bool = True, ard_factors: bool = True,
		 spikeslab_weights: bool = True, spikeslab_factors: bool = False,
		 n_iterations: int = 1000, convergence_mode: str = "fast",
		 gpu_mode: bool = False, Y_ELBO_TauTrick: bool = True, 
		 save_parameters: bool = False, save_data: bool = True, save_metadata: bool = True,
		 seed: int = 1, outfile: Optional[str] = None,
		 expectations: Optional[List[str]] = None,
		 save_interrupted: bool = True,
		 verbose: bool = False, quiet: bool = True, copy: bool = False):
	"""
	Run Multi-Omics Factor Analysis

	PARAMETERS
	----------
	data: an MuData object
	groups_label (optional): a column name in adata.obs for grouping the samples
	use_raw (optional): use raw slot of AnnData as input values
	use_layer (optional): use a specific layer of AnnData as input values (supersedes use_raw option)
	features_subset (optional): .var column with a boolean value to select genes (e.g. "highly_variable"), None by default
	likelihoods (optional): likelihoods to use, default is guessed from the data
	n_factors (optional): number of factors to train the model with
	scale_views (optional): scale views to unit variance
	scale_groups (optional): scale groups to unit variance
	ard_weights (optional): use view-wise sparsity
	ard_factors (optional): use group-wise sparsity
	spikeslab_weights (optional): use feature-wise sparsity (e.g. gene-wise)
	spikeslab_factors (optional): use sample-wise sparsity (e.g. cell-wise)
	n_iterations (optional): upper limit on the number of iterations
	convergence_mode (optional): fast, medium, or slow convergence mode
	gpu_mode (optional): if to use GPU mode
	Y_ELBO_TauTrick (optional): if to use ELBO Tau trick to speed up computations
	save_parameters (optional): if to save training parameters
	save_data (optional): if to save training data
	save_metadata (optional): if to load metadata from the AnnData object (.obs and .var tables) and save it, False by default
	seed (optional): random seed
	outfile (optional): path to HDF5 file to store the model
	expectations (optional): which nodes should be used to save expectations for (will save only W and Z by default);
	possible expectations names include Y, W, Z, Tau, AlphaZ, AlphaW, ThetaW, ThetaZ
	outfile (optional): output file name
	save_interrupted (optional): if to save partially trained model when the training is interrupted
	verbose (optional): print verbose information during traing
	quiet (optional): silence messages during training procedure
	copy (optional): return a copy of AnnData instead of writing to the provided object
	"""

	try:
		from mofapy2.run.entry_point import entry_point
		from mofapy2.build_model.utils import process_data
		from mofapy2.build_model.utils import guess_likelihoods
    except ImportError:
    	raise ImportError(
    		"MOFA+ is not available. Install MOFA+ from PyPI (`pip install mofapy2`) or from GitHub (`pip install git+https://github.com/bioFAM/MOFA2`)'
    		)
	
	if isinstance(data, AnnData):
		logging.info("Wrapping an AnnData object into an MuData container")
		mdata = MuData(data)
	elif isinstance(data, MuData):
		mdata = data
	else:
		raise TypeError("Expected an MuData object")


	ent = entry_point()

	lik = likelihoods
	if lik is not None:
		if isinstance(lik, str) and isinstance(lik, Iterable):
			lik = [lik for _ in range(len(mdata.mod))]

	ent.set_data_options(scale_views=scale_views, scale_groups=scale_groups)
	_set_mofa_data_from_mudata(model=ent, mdata=mdata, groups_label=groups_label, use_raw=use_raw, use_layer=use_layer,
			          		  likelihoods=lik, features_subset=features_subset, save_metadata=save_metadata)
	ent.set_model_options(ard_factors=ard_factors, ard_weights=ard_weights, 
						  spikeslab_weights=spikeslab_weights, spikeslab_factors=spikeslab_factors, 
						  factors=n_factors)
	ent.set_train_options(iter=n_iterations, convergence_mode=convergence_mode, 
						  gpu_mode=gpu_mode, Y_ELBO_TauTrick=Y_ELBO_TauTrick,
						  seed=seed, verbose=verbose, quiet=quiet, outfile=outfile, save_interrupted=save_interrupted)

	ent.build()
	ent.run()

	ent.save(outfile, save_data=save_data, save_parameters=save_parameters, expectations=expectations)

	try:
		import h5py
	except ImportError:
		h5py = None

	if h5py:
		f = h5py.File(outfile)
		if copy:
			mdata = mdata.copy()
		mdata.obsm['X_mofa'] = np.concatenate([v[:,:] for k, v in f['expectations']['Z'].items()], axis=1).T
		if features_subset is None:
			# Loadings can be saved only if all the features were used in training
			mdata.varm['LFs'] = np.concatenate([v[:,:] for k, v in f['expectations']['W'].items()], axis=1).T
		if copy:
			return mdata
		else:
			if features_subset is None:
				print("Saved MOFA embeddings in mdata.obsm['X_mofa'] slot and their loadings in mdata.varm['LFs'].")
			else:
				print("Saved MOFA embeddings in mdata.obsm['X_mofa'] slot.")
	else:
		print("Can not add embeddings and loadings to MuData object since h5py is not installed.")

	return None