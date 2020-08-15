from collections import OrderedDict
from typing import Union
import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from anndata import AnnData
from .._core.mudata import MuData

def tfidf(data: Union[AnnData, MuData], log_tf=True, log_idf=True, scale_factor=1e4):
	"""
	Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).

	TF: peak counts are normalised by total number of counts per cell
	DF: total number of counts for each peak
	IDF: number of cells divided by DF

	By default, log(TF) * log(IDF) is returned.

	Parameters
	----------
	data
		AnnData object with peak counts or multimodal MuData object with 'atac' modality.
	log_idf
		Log-transform IDF term (True by default)
	log_tf
		Log-transform TF term (True by default)
	scale_factor
	    Scale factor to multiply the TF-IDF matrix by (1e4 by default)
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData):
		adata = data.mod['atac']
		# TODO: check that ATAC-seq slot is present with this name
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	n_peaks = adata.X.sum(axis=1).reshape(-1, 1)
	tf = np.asarray(adata.X / n_peaks)
	if scale_factor is not None and scale_factor != 0 and scale_factor != 1:
		tf = tf * scale_factor
	if log_tf:
		tf = np.log1p(tf)

	idf = np.asarray(adata.shape[0] / adata.X.sum(axis=0)).reshape(-1)
	if log_idf:
		idf = np.log1p(idf)

	tf_idf = np.dot(csr_matrix(tf), csr_matrix(np.diag(idf)))

	adata.X = np.nan_to_num(tf_idf, 0)

	return None

def binarize(data: Union[AnnData, MuData]):
	"""
	Transform peak counts to the binary matrix (all the non-zero values become 1).

	Parameters
	----------
	data
		AnnData object with peak counts or multimodal MuData object with 'atac' modality.
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData):
		adata = data.mod['atac']
		# TODO: check that ATAC-seq slot is present with this name
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	if callable(getattr(adata.X, "todense", None)):
		# Sparse matrix
		adata.X.data = np.where(adata.X.data > 0, 1, 0)
	else:
		adata.X = np.where(adata.X > 0, 1, 0)


def add_peak_annotation(data: Union[AnnData, MuData],
						annotation: Union[str, pd.DataFrame],
						sep: str = "\t",
						return_annotation: bool = False):
	"""
	Parse peak annotation file and add it to the .uns["atac"]["peak_annotation"]

	Parameters
	----------
	data
		AnnData object with peak counts or multimodal MuData object with 'atac' modality.
	annotation
		A path to the peak annotation file (e.g. peak_annotation.tsv) or DataFrame with it.
		Annotation has to contain columns: peak, gene, distance, peak_type.
	sep
		Separator for the peak annotation file. Only used if the file name is provided. 
		Tab by default.
	return_annotation
		If return adata.uns['atac']['peak_annotation']. False by default.
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData):
		adata = data.mod['atac']
		# TODO: check that ATAC-seq slot is present with this name
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	if isinstance(annotation, str):
		pa = pd.read_csv(annotation, sep=sep)
	else:
		pa = annotation

	# Convert null values to empty strings
	pa.gene[pa.gene.isnull()] = ""
	pa.distance[pa.distance.isnull()] = ""
	pa.peak_type[pa.peak_type.isnull()] = ""

	# Split genes, distances, and peaks into individual records
	pa_g = pd.DataFrame(pa.gene.str.split(';').tolist(), index=pa.peak).stack()
	pa_d = pd.DataFrame(pa.distance.str.split(';').tolist(), index=pa.peak).stack()
	pa_p = pd.DataFrame(pa.peak_type.str.split(';').tolist(), index=pa.peak).stack()

	# Make a long dataframe indexed by gene
	pa_long = pd.concat([pa_g.reset_index()[["peak", 0]],
	                     pa_d.reset_index()[[0]],
	                     pa_p.reset_index()[[0]]], axis=1)
	pa_long.columns = ["peak", "gene", "distance", "peak_type"]
	pa_long = pa_long.set_index("gene")

	# chrX_NNNNN_NNNNN -> chrX:NNNNN-NNNNN
	pa_long.peak = [peak.replace("_", ":", 1).replace("_", "-", 1) for peak in pa_long.peak]

	if 'atac' not in data.uns:
		adata.uns["atac"] = OrderedDict()
	adata.uns['atac']['peak_annotation'] = pa_long

	if return_annotation:
		return pa_long

