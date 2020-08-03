from typing import Union
import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData

def tfidf(data: Union[AnnData, AmmData], log_idf=True, log_tf=False):
	"""
	Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).

	TF: peak counts are normalised by total number of counts per cell
	DF: total number of counts for each peak
	IDF: number of cells divided by DF

	By default, TF * log(IDF) is returned.
	"""
	if isinstance(data, Anndata):
		adata = data
	elif isinstance(data, AmmData):
		adata = data.mod['atac']
		# TODO: check that ATAC-seq slot is present with this name
	else:
		raise TypeError("Expected AnnData or AmmData object with 'atac' modality")

	n_peaks = adata.X.sum(axis=1)
	tf = np.asarray(adata.X / n_peaks)
	if log_tf:
		tf = np.log(tf + 1)
	idf = np.asarray(adata.shape[0] / adata.X.sum(axis=0))
	if log_idf:
		idf = np.log(idf + 1)
	tf_idf = np.dot(tf, np.diag(np.array(idf)[0]))

	adata.X = csr_matrix(tf_idf)

