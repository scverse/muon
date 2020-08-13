import io
import os
from glob import glob
import pkgutil
from typing import Union, Optional
import logging

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from anndata import AnnData
from .._core.mudata import MuData


def lsi(data: Union[AnnData, MuData], scale_embeddings=True, n_comps=50):
	"""
	Run Latent Semantic Indexing
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData):
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

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


def _parse_motif_ids(filename: Optional[str] = None):
	if filename is None:
		# Use a file from the embedded JASPAR database
		filename = io.BytesIO(pkgutil.get_data(__name__, "_ref/jaspar/motif_to_gene.txt"))
	motifs = pd.read_csv(filename, sep="\t", header=None)
	motifs.columns = ["motif_id", "tf_gene_name"]
	motifs = motifs.set_index("motif_id")

	return motifs

def _parse_motif_matrices(files: Optional[str] = None,
						  background: int = 4,
						  pseudocount: float = 0.0001,
						  ):
	try:
		import MOODS.tools
		import MOODS.parsers
	except ImportError:
		raise ImportError(
			"MOODS is not available. Install MOODS from PyPI (`pip install MOODS-python`) or from GitHub (`pip install git+https://github.com/jhkorhonen/MOODS`)"
			)

	if files is None:
		# Use pfm files from the embedded JASPAR database
		files = glob(os.path.join(os.path.dirname(__file__), "_ref/jaspar/*.pfm"))
	
	bg = MOODS.tools.flat_bg(background)
	matrices = [MOODS.parsers.pfm_to_log_odds(pfm_file, bg, pseudocount) for pfm_file in files]

	return {'motifs': [os.path.basename(f).rstrip(".pfm") for f in files],
			'matrices': matrices}

def _prepare_motif_scanner(matrices,
						   background: int = 4,
						   pvalue: float = 0.0001,
						   max_hits: int = 10):
	try:
		import MOODS.tools
		import MOODS.scan
	except ImportError:
		raise ImportError(
			"MOODS is not available. Install MOODS from PyPI (`pip install MOODS-python`) or from GitHub (`pip install git+https://github.com/jhkorhonen/MOODS`)"
			)

	if matrices is None:
		motifs_matrices = _parse_motif_matrices(files=None, background=background)
		matrices = motifs_matrices['matrices']

	bg = MOODS.tools.flat_bg(background)
	thresholds = [MOODS.tools.threshold_from_p(m, bg, pvalue) for m in matrices]

	scanner = MOODS.scan.Scanner(max_hits)
	scanner.set_motifs(matrices, bg, thresholds)

	return scanner

def scan_sequences(sequences,
				   motif_scanner = None,
				   matrices = None,
				   motifs = None,
				   motif_meta: pd.DataFrame = None,
				   background: int = 4,
				   pvalue: float = 0.0001,
				   max_hits: int = 10):
	try:
		import MOODS.tools
		import MOODS.scan
	except ImportError:
		raise ImportError(
			"MOODS is not available. Install MOODS from PyPI (`pip install MOODS-python`) or from GitHub (`pip install git+https://github.com/jhkorhonen/MOODS`)"
			)

	if matrices is None or motifs is None:
		assert matrices is None and motifs is None, "Both a list of matrices and a corresponding list of motif IDs should be provided — or none to use the built-in ones."

	if motif_scanner is None:
		if matrices is None:
			motifs = _parse_motif_matrices(files=None, background=background)['motifs']
		else:
			assert motifs is not None, "A list of motif IDs should be provided if building a scanner from matrices"
		
		motif_scanner = _prepare_motif_scanner(matrices = matrices,
			background = background, pvalue = pvalue, max_hits = max_hits)

		if motif_meta is None:
			# For the default scanner, use the default metadata
			motif_meta = _parse_motif_ids()

	else:
		assert motifs is not None, "A list of motif IDs should be provided that corresponds to the matrices that the motif scanner was built on."

	matches = []
	for seq in sequences:
		results = motif_scanner.scan(seq)
		for i, rs in enumerate(results):
			for r in rs:
				matches.append((seq, motifs[i], r.pos, r.score))

	matches = pd.DataFrame(matches)
	matches.columns = ["sequence", "motif_id", "position", "score"]

	if motif_meta is not None:
		matches = matches.set_index("motif_id").join(motif_meta, how="left").reset_index()

	return matches



def get_sequences(bed: str,
				  fasta_file: str,
				  bed_file: str = None):

	try:
		import pybedtools
	except ImportError:
		raise ImportError(
			"Pybedtools is not available. Install pybedtools from PyPI (`pip install pybedtools`) or from GitHub (`pip install git+https://github.com/daler/pybedtools`)"
			)

	if bed_file is not None:
		assert bed is None
		bed = open(bed_file).read()

	scanner = pybedtools.BedTool(bed, from_string=True)
	scanner = scanner.sequence(fi=fasta_file)
	sequences = []
	with open(scanner.seqfn, 'rb') as f:
	    for line in f:
	        if not line.startswith(str.encode(">")):
	            sequences.append(line.decode().strip())

	return sequences