# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `tools/scatac_pseudobulk.py`: Default Tn5 shift now converts 10x +4/-5
  fragment coordinates to the newer +4/-4 convention (`--shift-right` default
  changed from 0 to 1). This better models Tn5 sequence bias symmetry per
  Mao et al. 2024. Added `--no-shift` flag to keep original 10x coordinates.
  Added Tn5 shift argument group with detailed help text.

### Added
- `kidney_scatac` dataset in `download_dataset()`: human kidney 10x scATAC-seq
  from CellxGene (27,034 cells, 14 cell types, 5 donors, GRCh38). Downloads
  tabix-indexed fragment file, index, and gene activity h5ad.
- `tools/scatac_pseudobulk.py`: SnapATAC2-based pseudobulk BigWig generation
  and MACS3 peak calling from scATAC-seq fragments. Uses Rust-backed fragment
  import, multiprocessing-based parallel pipeline, and per-group narrowPeak BED
  output. Features: per-cell-type and bulk modes (`--bulk`), peak calling
  (`--call-peaks`), built-in genomes (hg38/hg19/mm10/mm39), all counting
  strategies (insertion/fragment/paired-insertion), `--merge` to collapse all
  peaks into a single `merged.narrowPeak.bed.gz` with median summits,
  `--n-jobs` budget with `--sequential` fallback.
- `examples/scatac_kidney_pseudobulk.sh`: example script for generating
  pseudobulk BigWigs and peaks from the kidney scATAC-seq dataset.
- `snapatac2` added to dev dependencies.

## [0.9.3] - 2026-03-03

### Added
- `prepare_data()` method on `CerberusDataModule` that pre-computes complexity
  metrics on rank 0 and caches them to disk, eliminating redundant FASTA reads
  across DDP ranks.
- `cache_dir` parameter on `CerberusDataModule` for configurable cache location
  (defaults to `$XDG_CACHE_HOME/cerberus` or `~/.cache/cerberus`).
- `src/cerberus/cache.py` module with cache directory resolution, serialization,
  and loading utilities.
- `prepare_cache` parameter threaded through `CerberusDataset`, `create_sampler()`,
  and `PeakSampler` to `ComplexityMatchedSampler`.
- `seed` parameter on `train_single()` and `train_multi()` (default: 42),
  propagated to `CerberusDataModule` for deterministic caching across DDP ranks.
- `--seed` CLI argument on all training tool scripts (`train_pomeranian.py`,
  `train_bpnet.py`, `train_asap.py`, `train_gopher.py`).
- Comprehensive test suite (`test_since_092.py`) covering all changes since v0.9.2.

### Fixed
- `CerberusDataModule.seed` is now `int` (default 42) instead of `int | None`.
  Previously, `seed=None` caused `random.Random(None)` to seed from system time,
  producing different cache directories across DDP ranks and defeating the
  `prepare_data()` cache.
- All sampler `__init__` seed parameters tightened from `int | None` to `int`
  (default 42), preventing accidental non-deterministic initialization.
- `generate_sub_seeds()` now requires `int` seed (removed `None` branch that
  returned `[None] * n`).

## [0.9.2] - 2026-03-03

### Added
- Extractor registry and enhanced `UniversalExtractor` functionality.
- Validation and tests for reverse complement functionality.
- Documentation system using mkdocs.

## [0.9.1] - 2026-03-01

### Added
- Stable training mode for BPNet using weight normalization and GELU activation.

### Fixed
- Removed unused imports from `bpnet.py`.

## [0.9.0] - 2026-03-01

### Added
- Initial tracked release.