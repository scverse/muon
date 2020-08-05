from typing import Union
import logging
import numpy as np
from scipy.sparse.linalg import svds
from anndata import AnnData
from .._core.ammdata import AmmData


def lsi(data: Union[AnnData, AmmData], scale_embeddings=True, n_comps=50):
	"""
	Run Latent Semantic Indexing
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, AmmData):
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or AmmData object with 'atac' modality")

	# In an unlikely scnenario when there are less 50 features, set n_comps to that value
	n_comps = min(n_comps, adata.X.shape[1])

	logging.info("Performing SVD")
	cell_embeddings, svalues, peaks_loadings = svds(adata.X, k = n_comps)

	# Re-order components in the descending order
	cell_embeddings = cell_embeddings[:,::-1]
	svalues = svalues[::-1]
	peaks_loadings = peaks_loadings[::-1,:]

	if scale_embeddings:
		cell_embeddings = (cell_embeddings - cell_embeddings.mean(axis=0)) / cell_embeddings.std(axis=0)
	
	stdev = svalues / np.sqrt(adata.X.shape[0] - 1)

	adata.obsm['X_lsi'] = cell_embeddings
	adata.uns['lsi'] = {}
	adata.uns['lsi']['stdev'] = stdev
	adata.varm['LSI'] = peaks_loadings.T

	return None


