import io
import os
from glob import glob
import pkgutil
from collections import OrderedDict
from typing import List, Union, Optional, Callable, Iterable
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from anndata import AnnData
from .._core.mudata import MuData

#
# Computational methods for transforming and analysing count data
#

def lsi(data: Union[AnnData, MuData], scale_embeddings=True, n_comps=50):
	"""
	Run Latent Semantic Indexing

	PARAMETERS
	----------
	data: 
		AnnData object or MuData object with 'atac' modality
	scale_embeddings: bool (default: True)
		Scale embeddings to zero mean and unit variance 
	n_comps: int (default: 50)
		Number of components to calculate with SVD
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData) and 'atac' in data.mod:
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	# In an unlikely scnenario when there are less 50 features, set n_comps to that value
	n_comps = min(n_comps, adata.X.shape[1])

	logging.info("Performing SVD")
	cell_embeddings, svalues, peaks_loadings = svds(adata.X, k=n_comps)

	# Re-order components in the descending order
	cell_embeddings = cell_embeddings[:,::-1]
	svalues = svalues[::-1]
	peaks_loadings = peaks_loadings[::-1,:]

	if scale_embeddings:
		cell_embeddings = (cell_embeddings - cell_embeddings.mean(axis=0)) / cell_embeddings.std(axis=0)
	
	stdev = svalues / np.sqrt(adata.X.shape[0] - 1)

	adata.obsm['X_lsi'] = cell_embeddings
	adata.uns['lsi'] = {'stdev': stdev}
	adata.varm['LSI'] = peaks_loadings.T

	return None

#
# Peak annotation
# 
# Peak annotation can include peak type (e.g. promoter, distal, intergenic),
# genes that the peak can be linked to (by proximity),
# as well as distances to these genes.
# 

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
	elif isinstance(data, MuData) and 'atac' in data.mod:
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

	# Make distance values integers with 0 for intergenic peaks
	# DEPRECATED: Make distance values nullable integers
	# See https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
	null_distance = pa_long.distance == ""
	pa_long.distance[null_distance] = 0
	pa_long.distance = pa_long.distance.astype(int)
	# DEPRECATED: Int64 is not recognized when saving HDF5 files with scanpy.write
	# pa_long.distance = pa_long.distance.astype(int).astype("Int64")
	# pa_long.distance[null_distance] = np.nan

	if 'atac' not in adata.uns:
		adata.uns["atac"] = OrderedDict()
	adata.uns['atac']['peak_annotation'] = pa_long

	if return_annotation:
		return pa_long


def add_peak_annotation_gene_names(data: Union[AnnData, MuData],
								   gene_names: Optional[pd.DataFrame] = None,
								   join_on: str = "gene_ids",
								   return_annotation: bool = False):
	"""
	Add gene names to peak annotation table in .uns["atac"]["peak_annotation"]

	Parameters
	----------
	data
		AnnData object with peak counts or multimodal MuData object with 'atac' modality.
	gene_names
		A DataFrame indexed on the gene name
	join_on
		Name of the column in the gene_names DataFrame corresponding to the peak annotation index
	return_annotation
		If return adata.uns['atac']['peak_annotation']. False by default.
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData) and 'atac' in data.mod:
		adata = data.mod['atac']
		# TODO: check that ATAC-seq slot is present with this name

		if gene_names is None:
			if 'rna' in data.mod:
				gene_names = data.mod['rna'].var
			else:
				raise ValueError("There is no .mod['rna'] modality. Provide `gene_names` as a pd.DataFrame.")
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	if 'atac' not in adata.uns or 'peak_annotation' not in adata.uns['atac']:
		raise KeyError("There is no peak annotation yet. Run muon.atac.tl.add_peak_annotation first.")

	# Extract a table with gene IDs and gene names only
	gene_id_name = gene_names.loc[:,[join_on]].rename_axis("gene_name").reset_index(drop=False).set_index(join_on)

	# Add gene names to the peak annotatoin table, then reset the index on gene IDs
	ann = adata.uns['atac']['peak_annotation']

	# Check whether the annotation index is not gene IDs
	if len(np.intersect1d(ann.index.values, gene_id_name.index.values)) == 0:
		if return_annotation:
			return ann
		return

	ann = ann.join(gene_id_name).rename_axis("gene").reset_index(drop=False)

	# Use empty strings for intergenic peaks when there is no gene
	ann.loc[ann.gene_name.isnull(),"gene_name"] = ""

	# Finally, set the index to gene name
	ann = ann.set_index("gene_name")
	adata.uns['atac']['peak_annotation'] = ann

	if return_annotation:
		return ann


# Gene names for peaks
def add_genes_peaks_groups(data: Union[AnnData, MuData],
						   peak_type: Optional[str] = None,
						   distance_filter: Optional[Callable[[int], bool]] = None):
	"""
	Add gene names to peaks ranked by clustering group

	To add gene names to ranked peaks, peaks have to be ranked first.
	For that, run `sc.tl.rank_genes_groups`.

	Gene names are picked as indices of the peak annotation table.
	To create annotation table, first run `muon.atac.tl.add_peak_annotation`.
	To add gene names instead of gene IDs, consider
	running `muon.atac.tl.add_peak_annotation_gene_names` then.
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData) and 'atac' in data.mod:
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	if 'rank_genes_groups' not in adata.uns:
		raise KeyError("There is no .uns['rank_genes_groups'] yet. Run sc.tl.rank_genes_groups first.")

	if 'atac' not in adata.uns or 'peak_annotation' not in adata.uns['atac']:
		raise KeyError("There is no peak annotation yet. Run muon.atac.pp.add_peak_annotation first.")

	def choose_peak_annotations(annotation: pd.DataFrame,
								peak: str,
								peak_type: Optional[str] = None,
								distance_filter: Optional[Callable[[int], bool]] = None):
		"""
		annotation (adata.uns['atac']['peak_annotation']), peak_type, and distance_filter
		are fetched from the outer scope (add_genes_peaks_groups)
		"""
		if 'peak' not in annotation.columns:
			raise KeyError("Peak annotation has to contain 'peak' column.")

		# Choose all annotations for the peak
		annotation = annotation[annotation.peak == peak]
		
		# Pick required peak types
		if peak_type is not None:
			if 'peak_type' not in annotation.columns:
				raise KeyError("Peak annotation has to contain 'peak_type' column.")
			annotation[annotation.peak_type == peak_type]

		# Pick annotations at required distance
		if distance_filter is not None:
			if 'distance' not in annotation.columns:
				raise KeyError("Peak annotation has to contain 'distance' column.")
			annotation = annotation[distance_filter(annotation.distance)]

		return annotation

	annotation = adata.uns['atac']['peak_annotation']

	adata.uns['rank_genes_groups']['genes'] = {}
	for i in adata.uns['rank_genes_groups']['names'].dtype.names:
		group = adata.uns['rank_genes_groups']['names'][i]
		genes = [', '.join(choose_peak_annotations(annotation, value, peak_type, distance_filter).index.values) for value in group]
		adata.uns['rank_genes_groups']['genes'][i] = genes

	# Convert to rec.array to match 'names', 'scores', and 'pvals'
	adata.uns['rank_genes_groups']['genes'] = pd.DataFrame(adata.uns['rank_genes_groups']['genes']).to_records()


def rank_peaks_groups(data: Union[AnnData, MuData],
					  groupby: str, 
					  peak_type: Optional[str] = None, 
					  distance_filter: Optional[Callable[[int], bool]] = None,
					  **kwargs):
	"""
	Rank peaks in clusters groups.

	Shorthand for running sc.tl.rank_genes_groups
	followed by muon.atac.tl.add_genes_peaks_groups.

	See sc.tl.rank_genes_groups for details.
	"""

	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData):
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	sc.tl.rank_genes_groups(adata, groupby, **kwargs)

	add_genes_peaks_groups(adata, peak_type=peak_type, distance_filter=distance_filter)


#
# Sequences and motifs
#

def _parse_motif_ids(filename: Optional[str] = None):
	if filename is None:
		# Use a file from the embedded JASPAR database
		filename = io.BytesIO(pkgutil.get_data(__name__, "_ref/jaspar/motif_to_gene.txt"))
	motifs = pd.read_csv(filename, sep="\t", header=None)
	motifs.columns = ["motif_id", "tf_gene_name"]
	motifs = motifs.set_index("motif_id")

	return motifs


def _parse_motif_matrices(files: Optional[str] = None,
						  background: Union[int, List] = 4,
						  pseudocount: float = 0.0001,
						  ):
	try:
		import MOODS.tools
		import MOODS.parsers
	except ImportError:
		raise ImportError(
			"MOODS is not available. Install MOODS from PyPI (`pip install MOODS-python`) \
			or from GitHub (`pip install git+https://github.com/jhkorhonen/MOODS`)"
			)

	if files is None:
		# Use pfm files from the embedded JASPAR database
		files = glob(os.path.join(os.path.dirname(__file__), "_ref/jaspar/*.pfm"))

	if not isinstance(background, Iterable):
		bg = MOODS.tools.flat_bg(background)
	else:
		bg = background
	matrices = [MOODS.parsers.pfm_to_log_odds(pfm_file, bg, pseudocount) for pfm_file in files]

	return {'motifs': [os.path.basename(f).rstrip(".pfm") for f in files],
			'matrices': matrices}


def _prepare_motif_scanner(matrices=None,
						   background: Union[int, Iterable] = 4,
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

	if not isinstance(background, Iterable):
		bg = MOODS.tools.flat_bg(background)
	else:
		bg = background
	thresholds = [MOODS.tools.threshold_from_p(m, bg, pvalue) for m in matrices]

	scanner = MOODS.scan.Scanner(max_hits)
	scanner.set_motifs(matrices, bg, thresholds)

	return scanner


def scan_sequences(sequences,
				   motif_scanner=None,
				   matrices=None,
				   motifs=None,
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

	if motifs is None:
		assert matrices is None, "Both a list of matrices and a corresponding list of motif IDs should be provided — or none to use the built-in ones, unless a scanner is provided."

	if motif_scanner is None:
		if matrices is None:
			motifs = _parse_motif_matrices(files=None, background=background)['motifs']
		else:
			assert motifs is not None, "A list of motif IDs should be provided if building a scanner from matrices"
		
		motif_scanner = _prepare_motif_scanner(matrices=matrices,
			background=background, pvalue=pvalue, max_hits=max_hits)

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


def get_sequences(data: Union[AnnData, MuData],
				  bed: str,
				  fasta_file: str,
				  bed_file: str = None):

	try:
		import pybedtools
	except ImportError:
		raise ImportError(
			"Pybedtools is not available. Install pybedtools from PyPI (`pip install pybedtools`) or from GitHub (`pip install git+https://github.com/daler/pybedtools`)"
			)

	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData) and 'atac' in data.mod:
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	if 'files' not in adata.uns or 'genome' not in adata.uns['files']:
		if fasta_file is not None:
			locate_genome(adata, fasta_file)
		else:
			raise FileNotFoundError("Genome file has to be provided with `fasta_file` \
				or located using `muon.atac.tl.locate_genome`.")
	else:
		# TODO: have a function to check validity of the file
		fasta_file = adata.uns['files']['genome']

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


def locate_file(data: Union[AnnData, MuData],
				key: str,
				file: str):
	"""
	Add path to the file to .uns["files"][key]

	The file to be added has to exist.

	Parameters
	----------
	data
		AnnData object with peak counts or multimodal MuData object with 'atac' modality.
	key
		A key to store the file (e.g. 'fragments')
	file
		A path to the file (e.g. ./atac_fragments.tsv.gz).
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData) and 'atac' in data.mod:
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	if not os.path.exists(file):
		raise FileNotFoundError(f"File {file} does not exist")

	if 'files' not in adata.uns:
		adata.uns["files"] = OrderedDict()
	adata.uns['files'][key] = file


def locate_genome(data: Union[AnnData, MuData],
				  fasta_file: str):
	"""
	Add path to the FASTA file with genome to .uns["files"]["genome"]

	Genome sequences can be downloaded from GENCODE:

	- GRCh38: ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_34/GRCh38.p13.genome.fa.gz
	- GRCm38: ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.p6.genome.fa.gz

	Parameters
	----------
	data
		AnnData object with peak counts or multimodal MuData object with 'atac' modality.
	fasta_file
		A path to the file (e.g. ./atac_fragments.tsv.gz).
	"""
	if not isinstance(data, AnnData) and not (isinstance(data, MuData) and 'atac' in data.mod):
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	locate_file(data, "genome", fasta_file)


# 
# Fragments
# 
# Fragments file is a BED-like file describing individual fragments.
# A single record in such a file typically includes 5 tab-separated fields: 
# 
# chr1 10000 11000 GTCAGTCAGTCAGTCA-1 1
# ^    ^     ^     ^                  ^
# |    |     |     |                  |
# |    |     |     4: name (cell barcode)
# |    |     3: end (3' fragment position)
# |    2: start (5' fragment position)|
# 1: contig (chromosome)              5: score (number of cuts per fragment)
# 
# Fragments file is compressed (.gz) and has to be indexed 
# with Tabix in order to be used (.gz.tbi).
# 


def locate_fragments(data: Union[AnnData, MuData],
					 fragments: str,
					 return_fragments: bool = False):
	"""
	Parse fragments file and add a variable to access it to the .uns["files"]["fragments"]

	Fragments file is never read to memory, and connection to the file is closed
	upon function completion.

	Parameters
	----------
	data
		AnnData object with peak counts or multimodal MuData object with 'atac' modality.
	fragments
		A path to the compressed tab-separated fragments file (e.g. atac_fragments.tsv.gz).
	return_fragments
		If return the Tabix connection the fragments file. False by default.
	"""
	try:
		if isinstance(data, AnnData):
			adata = data
		elif isinstance(data, MuData) and 'atac' in data.mod:
			adata = data.mod['atac']
		else:
			raise TypeError("Expected AnnData or MuData object with 'atac' modality")

		try:
			import pysam
		except ImportError:
			raise ImportError(
				"pysam is not available. It is required to work with the fragments file. \
				Install pysam from PyPI (`pip install pysam`) \
				or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
				)

		# Here we make sure we can create a connection to the fragments file
		frag = pysam.TabixFile(fragments, parser=pysam.asBed())

		if 'files' not in adata.uns:
			adata.uns["files"] = OrderedDict()
		adata.uns['files']['fragments'] = fragments

		if return_fragments:
			return frag

	except Exception as e:
		print(e)

	finally:
		if not return_fragments:
			# The connection has to be closed
			frag.close()


def count_fragments_features(data: Union[AnnData, MuData],
						     features: Optional[pd.DataFrame] = None,
						     extend_upstream: int = 2e3,
						     extend_downstream: int = 0,
						     average: str = 'sum') -> AnnData:
	"""
	Parse peak annotation file and add it to the .uns["atac"]["peak_annotation"]

	Parameters
	----------
	data
		AnnData object with peak counts or multimodal MuData object with 'atac' modality.
	features
		A DataFrame with feature annotation, e.g. genes.
		Annotation has to contain columns: Chromosome, Start, End.
	extend_upsteam
		Number of nucleotides to extend every gene upstream (2000 by default to extend gene coordinates to promoter regions)
	extend_downstream
		Number of nucleotides to extend every gene downstream (0 by default)
	average
		Name of the function to aggregate fragments per gene ('sum' by default to sum scores, 'count' to calculate number of fragments)
	"""
	if isinstance(data, AnnData):
		adata = data
	elif isinstance(data, MuData) and 'atac' in data.mod:
		adata = data.mod['atac']
	else:
		raise TypeError("Expected AnnData or MuData object with 'atac' modality")

	if features is None:
		# Try to gene gene annotation in the data.mod['rna']
		if isinstance(data, MuData) and 'rna' in data.mod and 'interval' in data.mod['rna'].var.columns:
			features = pd.DataFrame([s.replace(":", "-", 1).split("-") for s in data.mod['rna'].var.interval])
			features.columns = ["Chromosome", "Start", "End"]
			features['gene_id'] = data.mod['rna'].var.gene_ids
			features['gene_name'] = data.mod['rna'].var.index
			# Remove genes with no coordinates indicated
			features = features.loc[~features.Start.isnull()]
			features.Start = features.Start.astype(int)
			features.End = features.End.astype(int)
		else:
			raise ValueError("Argument `features` is required. It should be a BED-like DataFrame with gene coordinates and names.")

	if 'files' not in adata.uns or 'fragments' not in adata.uns['files']:
		raise KeyError("There is no fragments file located yet. Run muon.atac.tl.locate_fragments first.")

	try:
		import pysam
	except ImportError:
		raise ImportError(
			"pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
			)

	n = adata.n_obs
	cells_df = pd.DataFrame(index=adata.obs.index)
	cells_df["cell_index"] = range(n)

	fragments = pysam.TabixFile(adata.uns['files']['fragments'], parser=pysam.asBed())
	try:
		# Construct an empty sparse matrix with the right amount of cells
		mx = csr_matrix(([], ([], [])), shape=(n, 0), dtype=np.int8)

		logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Counting fragments in {n} cells for {features.shape[0]} features...")
		# Gene order is determined
		sparse_columns = []
		for i in range(features.shape[0]):  # iterate over features (e.g. genes)
			f = features.iloc[i]
			barcodes = []
			scores = []
			for fr in fragments.fetch(f.Chromosome, f.Start - extend_upstream, f.End + extend_downstream):
				barcodes.append(fr.name)      # cell barcode (e.g. GTCAGTCAGTCAGTCA-1)
				scores.append(int(fr.score))  # number of cuts per fragment (e.g. 2)

			# Note: This will also discard barcodes not present in the original data
			feature_df = pd.DataFrame({"score": scores}, index=barcodes, dtype=int)\
						   .join(cells_df, how="inner")\
						   .groupby("cell_index")\
						   .agg({"score": average})

			feature_mx = csr_matrix((feature_df.score.values,
								    (feature_df.index.values,
								     [0] * feature_df.shape[0])),
								    shape=(n, 1),
								    dtype=np.int8)

			sparse_columns.append(feature_mx)

			if i > 0 and i % 1000 == 0:
				logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed {i} features")

		mx = hstack(sparse_columns).tocsr()

		return AnnData(X=mx, obs=adata.obs, var=features)

	except Exception as e:
		logging.error(e)
		raise e

	finally:
		# The connection has to be closed
		fragments.close()
