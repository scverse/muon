import sys
from collections import defaultdict
from typing import Union, Optional, List, Iterable
import logging
import warnings

import numpy as np
from matplotlib.axes import Axes
import scanpy as sc
from anndata import AnnData
from .._core.mudata import MuData

def embedding(data: Union[AnnData, MuData], 
			  basis: str, 
			  color: Optional[Union[str, List[str]]] = None, 
			  average: Optional[str] = 'total',
			  **kwargs):
	"""
	Scatter plot in the define basis

	See sc.pl.embedding for details.
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData):
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	if color is not None:
		if isinstance(color, str):
			keys = [color]
		elif isinstance(color, Iterable):
			keys = color
		else:
			raise TypeError("Expected color to be a string or an iterable.")

		# New keys will be placed here
		attr_names = []
		tmp_names = []
		for key in keys:
			if key not in adata.var_names and key not in adata.var.columns:
				if 'atac' not in adata.uns or 'peak_annotation' not in adata.uns['atac']:
					raise KeyError(f"There is no feature or feature annotation {key}. If it is a gene name, load peak annotation with muon.atac.pp.add_peak_annotation first.")
				peak_sel = adata.uns["atac"]["peak_annotation"].loc[[key]]

				# only use peaks that are in the object (e.g. haven't been filtered out)
				peak_sel = peak_sel[peak_sel.peak.isin(adata.var_names.values)]

				peaks = peak_sel.peak
				
				if average == 'total' or average == 'all':
					attr_name = f"{key} (all peaks)"
					attr_names.append(attr_name)
					tmp_names.append(attr_name)

					if attr_name not in adata.obs.columns:
						# TODO: raw and layer options
						adata.obs[attr_name] = np.asarray(adata.raw[:,peaks].X.mean(axis=1)).reshape(-1)

				elif average == 'peak_type':
					peak_types = peak_sel.peak_type

					# {'promoter': ['chrX:NNN_NNN', ...], 'distal': ['chrX:NNN_NNN', ...]}
					peak_dict = defaultdict(list)
					for k, v in zip(peak_types, peaks):
						peak_dict[k].append(v)

					# 'CD4 (promoter peaks)', 'CD4 (distal peaks)'
					for t, p in peak_dict.items():
						attr_name = f"{key} ({t} peaks)"
						attr_names.append(attr_name)
						tmp_names.append(attr_name)

						if attr_name not in adata.obs.columns:
							# TODO: raw and layer options
							adata.obs[attr_name] = np.asarray(adata.raw[:,p].X.mean(axis=1)).reshape(-1)

				else:
					# No averaging, one plot per peak
					if average is not None and average != False and average != -1:
						wargnings.warn(f"Plotting individual peaks since {average} was not recognised. Try using 'total' or 'peak_type'.")
					attr_names += peak_sel.peak.values
			
			else:
				attr_names.append(key)

		sc.pl.embedding(adata, basis=basis, color=attr_names, **kwargs)

		# Remove temporary names
		for name in tmp_names:
			del adata.obs[name]

		return None

	else:
		return sc.pl.embedding(adata, basis=basis, color=peak_sel.peak.values, **kwargs)

	return None


def pca(data: Union[AnnData, MuData], **kwargs) -> Union[Axes, List[Axes], None]:
	"""
	Scatter plot for principle components

	See sc.pl.embedding for details.
	"""
	return embedding(data, basis='pca', **kwargs)


def umap(data: Union[AnnData, MuData], **kwargs) -> Union[Axes, List[Axes], None]:
	"""
	Scatter plot in UMAP space

	See sc.pl.embedding for details.
	"""
	return embedding(data, basis='umap', **kwargs)


def mofa(mdata: MuData, **kwargs) -> Union[Axes, List[Axes], None]:
	"""
	Scatter plot in MOFA factors coordinates

	See sc.pl.embedding for details.
	"""
	return sc.pl.embedding(mdata, 'mofa', **kwargs)