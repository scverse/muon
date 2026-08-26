# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [0.2.0] (Unreleased)

### Added

- `pl.scatter` and `pl.embedding` have a new `gene_symbols` argument with the same functionality as in scanpy.
- `tl.mofa` has a new `train_kwargs` argument for more detailed control over training.

### Changed

- muon now requires Python 3.12 or newer.
- `pp`, `tl`, `pl`, `atac.pp`, `atac.tl`, `atac.pl` and `prot.pp` are now regular submodules,
  so they can be imported directly (e.g. `from muon.pp import neighbors`).
- The `use_raw` argument of `muon.pl.scatter` and `muon.pl.embedding` can no longer be `None`, it must be either `True` or `False`.
  This leads to more predictable plotting behavior.
- The arguments `layer` and `use_raw` of `muon.pl.scatter` and `muon.pl.embedding` can be specified individually for each modality by passing dictionaries.

### Deprecated

- `muon.tl.louvain`. Use `muon.tl.leiden` instead.

### Fixed

- `muon.atac.tl.tss_enrichment` no longer fails with anndata 0.13, which removed the `dtype` argument of `AnnData`.
- `muon.atac.tl.add_peak_annotation_gene_names` now raises a clear error instead of failing with an
  `AttributeError` when called on an `AnnData` without `gene_names`.
- `muon.pp.filter_obs` and `muon.pp.filter_var` now raise a clear error when a function is required but not provided.
- Selecting an `.obsm` component with a non-integer index (e.g. `X_umap:abc`) now raises a clear error.
- Avoid a `FutureWarning` on import by querying the scanpy version via `importlib.metadata.version`
  instead of the deprecated `scanpy.__version__`.
- `muon.tl.leiden` now correctly sets the seed with `random_state=0`.
- `muon.pl.embedding` no longer mutates its input when a layer is used.

## [0.1.9]

### Fixed

- Fix a regression in `muon.prot.pp.clr` introduced in v0.1.8.

## [0.1.8]

### Added

- `muon.prot.pp.dsb` implements `scale_factor` and `quantile_clipping` options, matching the R package.
- `muon.prot.pp.clr` now supports multiple flavors, matching different implementations of CLR used in Seurat and publications.

### Changed

- `muon.prot.pp.dsb` uses `ddof=1` in standard deviation calculation to match the R behavior.
- Enable compatibility of in-place filtering with anndata 0.13.

### Fixed

- `muon.atac.tl.add_peak_annotation` no longer crashes when given empty distance values.
- `muon.pp.neighbors` now works when given sparse matrices.
- `muon.pp.neighbors` no longer crashes when multiple cells have identical coordinates.
- `muon.prot.pp.dsb` no longer overflows for large datasets.

## [0.1.7]

### Added

- Prepare to count unique fragments in `muon.atac.tl.count_fragments_features` from the next version.

### Changed

- Enable compatibility of in-place filtering with the latest anndata releases.
- Improve `muon.pl.scatter`.

### Fixed

- `muon.pp.tfidf` when using data from a layer.
- Fix custom chromosome names in `muon.atac.tl.count_fragments_features`.

## [0.1.6]

### Changed

- Compatibility with scanpy 1.10.
- Extend `l2norm` to sparse inputs.

## [0.1.5]

### Fixed

- Handling and saving colour palettes in MuData for categorical and continuous variables in `muon.pl.embedding`.
- Using sparse matrices in the MOFA interface to combine modalities with missing samples in `muon.tl.mofa`.
- Error messages and mixing metadata and features when plotting across modalities with `muon.pl.embedding`.

## [0.1.4]

### Added

- `muon.atac.pl.fragment_histogram` and `muon.pl.histogram` now have save/show arguments.
- `muon.atac.tl.count_fragments_features` now has a `stranded` argument.

### Changed

- `muon.pl.embedding` now saves the colour palette in `.uns`.
- Support for numpy 1.24 and newer scanpy versions.

### Fixed

- `muon.pp.intersect_obs` now works for modalities that have no `.X`.
- `muon.atac.tl.nucleosome_signal` now works on more `pysam` setups.

## [0.1.3]

### Added

- MOFA can now be run in the stochastic mode (SVI) using the new arguments for `muon.tl.mofa`.
- MOFA model weights can be visualized with `muon.pl.mofa_loadings`.
- New plots such as `muon.pl.scatter`.
- Layers can be defined as `{modality: layer}` in `muon.pl.embedding`.

### Changed

- Improvements to the TF-IDF normalisation interface including view handling.
- Dependencies such as `pysam` and `scikit-learn` are handled better.

## [0.1.2]

### Fixed

- In-place filtering functions (`muon.pp.filter_obs` and `muon.pp.filter_var`) can now be run one
  after another without requiring `mudata.MuData.update`.

## [0.1.1]

### Added

- The ATAC module can now handle fragments files with barcodes different from `obs_names`.
- Support for `atac_peak_annotation.tsv` files produced by Cell Ranger ARC 2.0.0.

### Changed

- `MuData` is now provided [as a separate package](https://mudata.readthedocs.io/) and is a hard dependency of `muon`.

### Fixed

- Reading `.h5mu` files in backed mode when modalities have `.raw` attributes.
- `SNF` functionality (`muon.tl.snf`).
- coloring plots by `var_names` present in `.raw` but not in the root `AnnData`.

## [0.1.0]

### Added

- Initial `muon` release with `MuData`, `atac` and `prot` submodules, and multi-omics integration with
  MOFA (`muon.tl.mofa`) and WNN (`muon.pp.neighbors`).

[0.2.0]: https://github.com/scverse/muon/releases/tag/v0.2.0
[0.1.9]: https://github.com/scverse/muon/releases/tag/v0.1.9
[0.1.8]: https://github.com/scverse/muon/releases/tag/v0.1.8
[0.1.7]: https://github.com/scverse/muon/releases/tag/v0.1.7
[0.1.6]: https://github.com/scverse/muon/releases/tag/v0.1.6
[0.1.5]: https://github.com/scverse/muon/releases/tag/v0.1.5
[0.1.4]: https://github.com/scverse/muon/releases/tag/v0.1.4
[0.1.3]: https://github.com/scverse/muon/releases/tag/v0.1.3
[0.1.2]: https://github.com/scverse/muon/releases/tag/v0.1.2
[0.1.1]: https://github.com/scverse/muon/releases/tag/v0.1.1
[0.1.0]: https://github.com/scverse/muon/releases/tag/v0.1.0
