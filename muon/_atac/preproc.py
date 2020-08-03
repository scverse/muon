from typing import Union
import numpy as np
from anndata import AnnData

def tfidf(data: Union[AnnData, AmmData], log_idf=True, log_tf=False):
	if isinstance(data, Anndata):
		adata = data
	elif isinstance(data, AmmData):
		adata = data.mod['atac']
		# TODO: check that ATAC-seq slot is present with this name
	else:
		raise TypeError("Expected AnnData or AmmData object with 'atac' modality")
	
	n_peaks = adata.X.sum(axis=1)
	tf = adata.X / n_peaks
	if log_tf:
		tf = np.log(tf + 1)
	idf = (adata.shape[0] / adata.X.sum(axis=0))
	if log_idf:
		idf = np.log(idf + 1)
	tf_idf = np.dot(tf, np.diag(np.array(idf)[0]))

	adata.X = tf_idf

