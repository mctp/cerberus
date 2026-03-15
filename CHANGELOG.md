# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **PWMBiasModel** moved from `cerberus.models` to `debug/pwm_model/` as a
  self-contained debug package. No longer part of the cerberus public API.
  Includes `RegularizedProfileCountOutput` (output with optional reg_loss),
  `RegularizedMSEMultinomialLoss` (loss wrapper that consumes reg_loss), and
  cosine similarity decorrelation penalty (`decorrelation_weight` parameter)
  for encouraging distinct PWM filters.

### Added
- **Depthwise-only PGC mode** (`expansion=0`): Setting `expansion=0` in
  Pomeranian/Dalmatian model args skips all pointwise projections and gating
  in PGC tower blocks, producing a pure depthwise-convolution tower. Useful
  for lightweight baselines and ablation studies.
- **k562_chrombpnet dataset** (`cerberus.download`): New downloadable dataset
  from Zenodo (DOI: 10.5281/zenodo.15713376) with K562 ATAC-seq narrowPeak
  and unstranded BigWig files for ChromBPNet benchmarking.
- **`examples/atac_k562_dalmatian.sh`**: Dalmatian training example for K562
  ATAC-seq using the k562_chrombpnet Zenodo dataset (auto-downloaded).
- **NegativePeakSampler** (`cerberus.samplers`): Background-only sampler that
  generates complexity-matched non-peak intervals, excluding peak regions.
  Used for training bias-only models (Tn5 sequence bias) on non-peak regions.
  Config type: `"negative_peak"`, same args as `"peak"`.
- **Dalmatian model** (`cerberus.models.Dalmatian`): End-to-end bias-factorized
  sequence-to-function model for ATAC-seq. Composes two Pomeranian sub-networks
  (BiasNet ~147bp RF, SignalNet ~1089bp RF) with zero-initialized signal outputs,
  logit addition for profiles, and logsumexp for counts. Per-channel count
  prediction by default (ATAC-seq channels are independent samples).
- **FactorizedProfileCountOutput** (`cerberus.output`): Extends
  `ProfileCountOutput` with decomposed bias/signal logits and log_counts.
  Compatible with all existing losses and metrics expecting `ProfileCountOutput`.
- **DalmatianLoss** (`cerberus.loss`): Peak-conditioned loss wrapping any
  profile+count base loss. Adds bias-only reconstruction on background regions
  and L1 signal suppression on non-peak examples. Requires `peak_status` batch
  context.
- All loss classes now accept `**kwargs` for batch context forwarding.
  `CerberusModule._shared_step` passes all non-input/target batch fields as
  keyword arguments to the loss function.
- Input length validation (center-crop or `ValueError`) added to all models:
  Pomeranian, BPNet, ConvNeXtDCNN, GlobalProfileCNN.
- **`tools/train_dalmatian.py`**: Generic training tool for Dalmatian models.
  Supports MSE and Poisson base losses, BiasNet/SignalNet filter overrides,
  and all standard training options (multi-fold, precision, etc.).
- **`examples/scatac_kidney_dalmatian.sh`**: Example training script for
  Dalmatian on the kidney scATAC-seq pseudobulk dataset (renal
  beta-intercalated cell by default).

### Fixed
- **`negative_peak` sampler disk caching**: `NegativePeakSampler` was missing
  from the cacheable sampler types in `CerberusDataModule`, causing complexity
  metrics to be recomputed from scratch on every run instead of loading from
  the `~/.cache/cerberus/` disk cache.

### Changed
- `tools/scatac_pseudobulk.py`: Reworked CLI flags and output naming:
  - `--overwrite` flag (off by default) skips stages with existing outputs.
  - Replaced `bgzip`/`tabix` binaries with `pysam` (no htslib dependency).
  - Bulk BigWig (`bulk.bw`) is now always exported; `--bulk` replaced by
    `--bulk-peaks` (off by default) for the slow bulk peak calling step.
  - Peak merging on by default with `--call-peaks`; use `--no-merge` to
    disable. Single-group case copies the file instead of failing.
  - Output files renamed: merged peaks → `bulk_merge.narrowPeak.bed.gz`,
    bulk called peaks → `bulk_call.narrowPeak.bed.gz`. Bulk called peaks
    are excluded from the merge.
  - Default Tn5 shift converts 10x +4/-5 to +4/-4 (`--shift-right` default
    changed from 0 to 1, per Mao et al. 2024). `--no-shift` to keep
    original coordinates.

### Added
- `kidney_scatac` dataset in `download_dataset()`: human kidney 10x scATAC-seq
  from CellxGene (27,034 cells, 14 cell types, 5 donors, GRCh38). Downloads
  tabix-indexed fragment file, index, and gene activity h5ad.
- `tools/scatac_pseudobulk.py`: SnapATAC2-based pseudobulk BigWig generation
  and MACS3 peak calling from scATAC-seq fragments. Uses Rust-backed fragment
  import, multiprocessing-based parallel pipeline, and per-group narrowPeak BED
  output. Features: per-cell-type BigWigs and always-on bulk BigWig, peak
  calling (`--call-peaks`) with automatic merge into
  `bulk_merge.narrowPeak.bed.gz`, optional bulk peak calling (`--bulk-peaks`),
  built-in genomes (hg38/hg19/mm10/mm39), all counting strategies
  (insertion/fragment/paired-insertion), `--overwrite` for re-runs,
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