# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `kidney_scatac` dataset in `download_dataset()`: human kidney 10x scATAC-seq
  from CellxGene (27,034 cells, 14 cell types, 5 donors, GRCh38). Downloads
  tabix-indexed fragment file, index, and gene activity h5ad.
- `tools/pseudobulk_bigwig.py`: generates per-cell-type BigWig files from
  scATAC-seq fragments. Supports configurable coverage mode (insertion/fragment),
  normalization (cpm/rpkm/raw), fragment size filtering, and Tn5 shift.

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