from typing import Union
from pathlib import Path
import os

import numpy as np
import scanpy as sc
from .mudata import MuData

from .._atac.tools import add_peak_annotation, locate_fragments, add_peak_annotation_gene_names

def read_10x_h5(filename: Union[str, Path],
				extended: bool = True,
				*args, **kwargs) -> MuData:
	
	adata = sc.read_10x_h5(filename, gex_only=False, *args, **kwargs)

	# Patches sc.read_10x_h5 behaviour to:
	# - attempt to read `interval` field for features from the HDF5 file
	# - attempt to add peak annotation
	# - attempt to locate fragments file

	if extended:

		# 1) Read interval field from the HDF5 file

		try:
			import h5py
		except ImportError:
			raise ImportError(
				"h5py is not available. Install h5py from PyPI (`pip install pypi`) or from GitHub (`pip install git+https://github.com/h5py/h5py`)"
				)

		h5file = h5py.File(filename, 'r')

		if 'interval' in h5file["matrix"]["features"]:
			intervals = np.array(h5file["matrix"]["features"]["interval"]).astype(str)

		h5file.close()

		adata.var["interval"] = intervals

		print(f"Added `interval` annotation for features from {filename}")

	mdata = MuData(adata)

	if extended:
		if 'atac' in mdata.mod:
		
			# 2) Add peak annotation

			default_annotation = os.path.join(os.path.dirname(filename), "atac_peak_annotation.tsv")
			if os.path.exists(default_annotation):
				add_peak_annotation(mdata.mod['atac'], default_annotation)
				print(f"Added peak annotation from {default_annotation} to .uns['atac']['peak_annotation']")

			try:
				add_peak_annotation_gene_names(mdata)
				print(f"Added gene names to peak annotation in .uns['atac']['peak_annotation']")
			except Exception as e:
				pass

			# 3) Locate fragments file

			default_fragments = os.path.join(os.path.dirname(filename), "atac_fragments.tsv.gz")
			if os.path.exists(default_annotation):
				locate_fragments(mdata.mod['atac'], default_fragments)
				print(f"Located fragments file: {default_fragments}")

	return mdata
