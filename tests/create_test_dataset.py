# Create a minimal dataset that can be used for testing functions that are difficult to test with generated data


import muon as mu
import scanpy as sc
import pandas as pd
import os


data_dir = "/home/max/projects/pbmc_multimodal/PBMC_rep1/"
# muon_dir = "/home/max/code/muon/"

outdir = "data/atac/"


mdata = mu.read_10x_h5(os.path.join(data_dir, "filtered_feature_bc_matrix.h5"))

rna = mdata.mod["rna"]
atac = mdata.mod["atac"]


###########################
# RNA
###########################

# Filter cells to 1000 quality cells
rna.var["mt"] = rna.var_names.str.startswith(
    "MT-"
)  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)


mu.pp.filter_obs(rna, "n_genes_by_counts", lambda x: (x >= 200) & (x < 5000))
mu.pp.filter_obs(rna, "pct_counts_mt", lambda x: x < 20)

mu.pp.filter_obs(rna, "total_counts", lambda x: x < 15000)

filter_cells = rna.obs.index[:1000]
mu.pp.filter_obs(rna, filter_cells)

# Filter marker genes
marker_genes = [
    "IL7R",
    "TRAC",
    "ITGB1",
    "CD2",
    "SLC4A10",
    "CD8A",
    "CD8B",
    "CCL5",
    "GNLY",
    "NKG7",
    "CD79A",
    "MS4A1",
    "IGHM",
    "IGHD",
    "IL4R",
    "TCL1A",
    "KLF4",
    "LYZ",
    "S100A8",
    "ITGAM",
    "CD14",
    "FCGR3A",
    "MS4A7",
    "CST3",
    "CLEC10A",
    "IRF8",
    "TCF4",
    "INPP4B",
    "IL32",
    "LTB",
    "SYNE2",
    "ANK3",
    "CDC14A",
    "IL7R",
    "ITGB1",
    "BCL11B",
    "LEF1",
    "SLC8A1",
    "VCAN",
    "BANK1",
    "NEAT1",
    "TCF7L2",
    "CD74",
    "RPS27",
    "CDK6",
    "MAML3",
    "SOX4",
]

mu.pp.filter_var(rna, marker_genes)
rna.obs = pd.DataFrame(index=rna.obs.index)
rna.var = rna.var[["gene_ids", "feature_types", "genome", "interval"]]


# Quick check that data still has some signal
# sc.pp.normalize_total(rna, target_sum=1e4)
# sc.pp.log1p(rna)
# rna.raw = rna
# sc.pp.scale(rna, max_value=10)

# sc.tl.pca(rna, svd_solver='arpack')
# sc.pl.pca(rna, color=['CD2', 'CD79A', 'KLF4', 'IRF8'])
# sc.pl.pca_variance_ratio(rna, log=True)

# sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)
# sc.tl.leiden(rna, resolution=0.5)
# sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)
# sc.pl.umap(rna, color="leiden", legend_loc="on data")
# sc.pl.umap(rna, color=['CD2', 'CD79A', 'KLF4', 'IRF8'], legend_loc="on data")

###########################
## ATAC
###########################

# # Filter cells to 1000 quality cells
mu.pp.filter_obs(atac, filter_cells)  # From RNA


# Filter peaks around the interesting genes
extension = 5000

regions = mu.atac.tl.get_gene_annotation_from_rna(rna)
regions.Start = regions.Start - extension
regions.End = regions.End + extension

import pyranges as pr

peaks = pd.DataFrame([s.replace(":", "-", 1).split("-") for s in atac.var.interval])
peaks.columns = ["Chromosome", "Start", "End"]
peaks["id"] = atac.var.gene_ids.values

peaks = pr.PyRanges(peaks)
genes = pr.PyRanges(regions)
genes = genes.slack(extension)
p2 = peaks.overlap(genes)

mu.pp.filter_var(atac, p2.id)
peakann = atac.uns["atac"]["peak_annotation"]
atac.uns["atac"]["peak_annotation"] = peakann[peakann.peak.isin(atac.var.index)]

# Filter fragments around the interesting genes
# and write subsetted fragments file
fragments_file = atac.uns["files"]["fragments"]
if not os.path.isdir(outdir):
    os.makedirs(outdir)
outfile = os.path.join(outdir, "test_rna_atac_fragments.tsv")


import pysam

tbx = pysam.TabixFile(fragments_file)


with open(outfile, "w") as file:
    for region in regions.itertuples():
        for f in tbx.fetch(region.Chromosome, region.Start, region.End):
            # print(str(f))
            file.writelines(f"{f}\n")

# Compress and create tabix index
pysam.tabix_index(outfile, force=True, seq_col=0, start_col=1, end_col=2)

atac.uns["files"]["fragments"] = str(outfile + ".gz")

mdata.update()


print(rna.obs.index)
mdata.write(os.path.join(outdir, "test_rna_atac.h5mu"))
rna.write(os.path.join(outdir, "test_rna.h5ad"))

# Make sure file can be read

# mu2 = mu.read_h5mu(os.path.join(outdir, "test_rna_atac.h5mu"))
