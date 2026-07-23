# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Changed

- Adopted the [cookiecutter-scverse](https://github.com/scverse/cookiecutter-scverse) project template
  (`src/` layout, `hatchling` build backend, `ruff`, updated CI and documentation).

## [0.1.9]

- Fix a regression in `muon.prot.pp.clr` introduced in v0.1.8.

## [0.1.8]

- Enable compatibility of in-place filtering with anndata 0.13.
- `muon.atac.tl.add_peak_annotation` no longer crashes when given empty distance values.
- `muon.pp.neighbors` now works when given sparse matrices.
- `muon.pp.neighbors` no longer crashes when multiple cells have identical coordinates.
- `muon.prot.pp.dsb` implements `scale_factor` and `quantile_clipping` options, matching the R package.
- `muon.prot.pp.dsb` uses `ddof=1` in standard deviation calculation to match the R behavior.
- `muon.prot.pp.dsb` no longer overflows for large datasets.
- `muon.prot.pp.clr` now supports multiple flavors, matching different implementations of CLR used in Seurat and publications.

## [0.1.7]

- Enable compatibility of in-place filtering with the latest anndata releases.
- `muon.pp.tfidf` when using data from a layer.
- Fix custom chromosome names in `muon.atac.tl.count_fragments_features`.
- Prepare to count unique fragments in `muon.atac.tl.count_fragments_features` from the next version.
- Improve `muon.pl.scatter`.

## [0.1.6]

- Compatibility with scanpy 1.10.
- Extend `_l2norm` to sparse inputs.

## [0.1.5]

- Fix handling and saving colour palettes in MuData for categorical and continuous variables in `muon.pl.embedding`.
- Fix using sparse matrices in the MOFA interface to combine modalities with missing samples in `muon.tl.mofa`.
- Fix error messages and mixing metadata and features when plotting across modalities with `muon.pl.embedding`.

## [0.1.4]

- `muon.pp.intersect_obs` now works for modalities that have no `.X`.
- `muon.pl.embedding` now saves the colour palette in `.uns`.
- `muon.atac.pl.fragment_histogram` and `muon.pl.histogram` now have save/show arguments.
- `muon.atac.tl.count_fragments_features` now has a `stranded` argument.
- `muon.atac.tl.nucleosome_signal` now works on more `pysam` setups.
- Support for numpy 1.24 and newer scanpy versions.

## [0.1.3]

- MOFA can now be run in the stochastic mode (SVI) using the new arguments for `muon.tl.mofa`.
- MOFA model weights can be visualised with `muon.pl.mofa_loadings`.
- New plots such as `muon.pl.scatter`.
- It is now possible to define layers as `{modality: layer}` in `muon.pl.embedding`.
- Improvements to the TF-IDF normalisation interface including view handling.
- Better dependency handling such as `pysam` and `scikit-learn`.

## [0.1.2]

- In-place filtering functions (`muon.pp.filter_obs` and `muon.pp.filter_var`) can now be run one
  after another without requiring `muon.MuData.update`.

## [0.1.1]

- `MuData` is now provided [as a separate package](https://mudata.readthedocs.io/) and is a hard dependency of `muon`.
- Fix reading `.h5mu` files in backed mode when modalities have `.raw` attributes, `SNF` functionality
  (`muon.tl.snf`), and colouring plots by `var_names` present in `.raw` but not in the root `AnnData`.
- ATAC module: handle fragments files with barcodes different from `obs_names` and support
  `atac_peak_annotation.tsv` files produced by Cell Ranger ARC 2.0.0.

## [0.1.0]

- Initial `muon` release with `MuData`, `atac` and `prot` submodules, and multi-omics integration with
  MOFA (`muon.tl.mofa`) and WNN (`muon.pp.neighbors`).
