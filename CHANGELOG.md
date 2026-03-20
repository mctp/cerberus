# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Interval manifests saved at training time**: After training, each fold directory
  contains `intervals_{train,val,test}.bed` files recording the exact intervals and
  their source sampler (via `interval_source` column). This eliminates the need to
  reconstruct the sampler pipeline for evaluation and guarantees reproducibility even
  if sampler code changes. Saved from rank 0 only in multi-GPU training.
- **`Interval.to_bed_row()`**: New method for BED-format serialization.
- **`write_intervals_bed()` / `load_intervals_bed()`**: I/O utilities for interval
  manifest files with source labels.
- **`CerberusDataModule.save_interval_manifests()`**: Writes interval manifests for
  all splits to a given directory.

### Changed
- **`get_peak_status()` replaced with `get_interval_source()`**: `MultiSampler.get_peak_status()`
  (returning binary `0`/`1`) is replaced by `get_interval_source()` which returns the class name
  of the sub-sampler that produced each interval (e.g. `"IntervalSampler"`,
  `"ComplexityMatchedSampler"`). The batch dict key is renamed from `peak_status` (int) to
  `interval_source` (str). `DalmatianLoss` converts `interval_source` to a peak mask internally.
  `get_interval_source()` is now defined on `BaseSampler` (returns own class name) and overridden
  in `MultiSampler` (returns sub-sampler class name).
- **Strict type hints across `src/cerberus/`**: Added missing parameter and return
  type annotations to all `__init__`, `forward`, `loss_components`,
  `_compute_profile_loss`, `compute`, and `update` methods in `layers.py`,
  `loss.py`, `metrics.py`, `module.py`, `datamodule.py`, `models/gopher.py`,
  and `models/asap.py`. Added return types to private methods in `samplers.py`,
  `sequence.py`, `mask.py`, `dataset.py`, `output.py`, and `download.py`.
  Narrowed `Any` to concrete union types for `query` parameters in
  `dataset.py` and bin-index dicts in `samplers.py`. Narrowed
  `instantiate_metrics_and_loss` return type from `tuple[Any, Any]` to
  `tuple[MetricCollection, CerberusLoss]`.
- Added `pyrightconfig.json` with `reportMissingParameterType` and
  `reportMissingReturnType` as warnings to catch future regressions.

### Added
- **`uses_count_pseudocount` class attribute** on all loss classes: declares
  whether the loss trains log_counts in `log(count + pseudocount)` space.
  Eliminates `isinstance` checks against specific loss classes throughout
  inference code.
- **`get_log_count_params(model_config)`** in `config.py`: reads
  `uses_count_pseudocount` from the loss class and returns
  `(log_counts_include_pseudocount, count_pseudocount)`. Single source of
  truth for pseudocount transform parameters at inference time.
- **`compute_obs_log_counts`** in `output.py`: utility for computing observed
  total log-counts from raw targets, matching the loss function's transform.

### Changed
- `count_pseudocount` validation relaxed from `> 0` to `>= 0` — allows `0.0`
  for Poisson/NB losses that do not use a pseudocount offset.
- `propagate_pseudocount` now warns when `count_pseudocount > 0` is paired
  with a loss that ignores it.
- `predict_to_bigwig` no longer takes a `count_pseudocount` parameter —
  auto-detects from the model config via `get_log_count_params`.
- `export_bigwig.py` CLI: removed `--count-pseudocount` argument (now automatic).
- `export_predictions.py`: replaced `isinstance` + `getattr` pseudocount
  detection with `get_log_count_params`; uses `compute_obs_log_counts`.

### Fixed
- **Multi-channel log-count metric aggregation**: `LogCountsMeanSquaredError` and
  `LogCountsPearsonCorrCoef` now correctly aggregate multi-channel
  `ProfileCountOutput` predictions in offset-log space. Previously used plain
  `logsumexp` which gave `log(total + C*pseudocount)` instead of the correct
  `log(total + pseudocount)`. Added `log_counts_include_pseudocount` parameter
  to both metrics and `DefaultMetricCollection`; `propagate_pseudocount` now
  injects the flag automatically.

## [0.9.4] - 2026-03-19

### Added
- **`CerberusLoss` protocol**: All loss classes now implement a uniform
  `loss_components(outputs, targets, **kwargs) -> dict[str, Tensor]` interface
  that returns named, unweighted loss components. Enables generic per-component
  logging in `CerberusModule._shared_step` without type-checking branches.
- **Per-component loss logging**: `_shared_step` now logs each named component
  (e.g. `train_profile_loss`, `val_count_loss`, `train_recon_loss`) as separate
  Lightning metrics alongside the combined `loss`, for all loss types.
- `export_predictions.py`: `--eval-split` argument (`test`/`val`/`train`/`all`, default `test`) restricts evaluation to the correct held-out chromosome set, preventing data leakage from training chromosomes into reported metrics.
- `export_predictions.py`: `--include-background` flag adds complexity-matched background intervals alongside peaks, replicating the training evaluation setup. The output TSV gains a `peak_status` column (1=peak, 0=background); background is fold-restricted to match the requested `--eval-split`. Also adds `--background-ratio` and `--seed`.
- **`tools/export_bigwig.py`**: CLI tool for exporting genome-wide model
  predictions to BigWig format. Loads a model ensemble, slides windows across
  all allowed chromosomes, and streams predictions to a BigWig file via
  `pybigtools`. Skips loading target extractors (only sequence inputs are
  needed). Supports custom stride, fold selection, batch size, and device.

### Fixed
- **`predict_bigwig` OOM on whole chromosomes**: Replaced per-window tensor list
  with a streaming accumulator in `_process_island`. Memory is now bounded by
  the accumulator array size, not the number of prediction windows.
- **`predict_bigwig` pseudocount inversion**: Added `count_pseudocount` parameter
  to `predict_to_bigwig` and `_reconstruct_linear_signal`. MSE-trained models
  (BPNet/Pomeranian/Dalmatian) that predict `log(count + pseudocount)` now
  correctly subtract the pseudocount before signal reconstruction.
  `tools/export_bigwig.py` auto-reads the value from the model config.
- **`predict_bigwig` resource leak**: BigWig file handle now closed via
  try/finally.
- **`predict_bigwig` dead `aggregation` parameter**: Removed from
  `predict_to_bigwig` and `_process_island` — per-window reconstruction
  requires "model" aggregation, which is now hardcoded internally.
- **`predict_bigwig` dead code**: Removed unused `while ndim > 2` squeeze loop
  and unused `numpy` import (now used by streaming accumulator).
- **`unbatch_modeloutput` preserved nested types**: Replaced `dataclasses.asdict()`
  with shallow field extraction via `dataclasses.fields()`. Fixes a latent bug
  where `Interval` objects were silently converted to plain dicts, and avoids
  unnecessary deep-copying of tensors. Same fix applied to `aggregate_models`.
- **`compute_total_log_counts` simplified**: Removed redundant single-channel vs
  multi-channel branching for `ProfileLogRates` — both paths computed the same
  `logsumexp` over all channels and positions.

### Changed
- **Model loading prefers `model.pt`**: `ModelEnsemble` now loads the clean
  `model.pt` state dict when available (written by training since the
  pretrained-weight-loading update), falling back to Lightning `.ckpt` files
  for backward compatibility. This eliminates checkpoint filename parsing and
  prefix stripping overhead for new training runs.

### Added
- **Pretrained weight loading** (`cerberus.pretrained`): Generic system for loading
  pretrained weights into any model or sub-module. New `PretrainedConfig` TypedDict
  and `pretrained` field on `ModelConfig`. Supports: whole-model warm-start,
  sub-module loading (e.g., BiasNet → Dalmatian's `bias_model`), extracting
  sub-module weights from full-model checkpoints (`source` prefix), and freezing
  loaded parameters. All training tools gain `--pretrained` CLI arg;
  `train_dalmatian.py` adds `--pretrained-bias` / `--freeze-bias`.
- **`tools/train_biasnet.py`**: Standalone BiasNet training tool matching the
  CLI conventions of `train_pomeranian.py` and `train_bpnet.py`. Defaults to
  `negative_peak` sampler (background-only training), `MSEMultinomialLoss`,
  and BiasNet architecture (f=12, 5 layers, linear head, ~9.3K params).
  Supports all BiasNet architecture parameters, multi-fold cross-validation,
  and cosine/warmup scheduling.
- **`examples/scatac_kidney_biasnet.sh`**: BiasNet training example for kidney
  scATAC-seq pseudobulk (negative peaks, reproduces exp19f).
- **`examples/atac_k562_biasnet.sh`**: BiasNet training example for K562
  ATAC-seq (negative peaks).
- **Dalmatian `signal_preset`**: ``"large"`` (default, f=256, ~3.9M params) or
  ``"standard"`` (f=64, ~150K params, matches Pomeranian K9). Individual
  `signal_*` args override the preset.
- **Dalmatian `zero_init` parameter**: Controls zero-initialization of signal
  output layers. `--no-zero-init` flag added to `tools/train_dalmatian.py`.

### Changed
- **DalmatianLoss simplified to 2-term loss**: Removed `signal_background_weight`
  and L_signal_bg (L1 penalty on signal outputs in background regions). Exp21
  confirmed this term had zero measurable effect on both kidney and K562 datasets
  — gradient detach already prevents SignalNet from activating on background.
  Loss is now `L_recon + bias_weight * L_bias`. The `signal_background_weight`
  parameter is accepted but ignored for backwards compatibility.
- **Dalmatian gradient separation**: Bias outputs are `.detach()`-ed before
  combining with signal outputs in `Dalmatian.forward()`. The combined
  reconstruction loss (L_recon) now trains only SignalNet; BiasNet receives
  gradients exclusively from L_bias (background reconstruction). This
  replicates ChromBPNet's freeze-bias design without two-stage training,
  preventing the bias model from learning TF footprints via L_recon.
- **Dalmatian BiasNet replaced**: The bias sub-network in `Dalmatian` is now a
  dedicated `BiasNet` (Conv1d + ReLU + residual, ~9.3K params, 105bp RF) instead
  of a Pomeranian (~72K params, 147bp RF). BiasNet is fully DeepLIFT/DeepSHAP
  compatible (no RMSNorm, GELU, or gating — only Conv1d, ReLU, and residual add).
  Default config: f=12, 5 residual tower layers, linear profile head. The
  `SimpleResidualBlock` layer is added to `cerberus.layers`. **Breaking change**:
  Dalmatian bias parameter names changed from `bias_expansion`/`bias_stem_expansion`
  to `bias_linear_head`/`bias_residual`. Default `bias_filters` changed from 64 to 12.
- **Dalmatian defaults updated** based on exp20 results: `zero_init` default
  changed from `True` to `False` (zero-init is harmful with gradient detach).
  `signal_preset` default changed from `"large"` to `"standard"` (24x fewer
  params with <0.01 Pearson difference).
- **PWMBiasModel** moved from `cerberus.models` to `debug/pwm_model/` as a
  self-contained debug package. No longer part of the cerberus public API.
  Includes `RegularizedProfileCountOutput` (output with optional reg_loss),
  `RegularizedMSEMultinomialLoss` (loss wrapper that consumes reg_loss), and
  cosine similarity decorrelation penalty (`decorrelation_weight` parameter)
  for encouraging distinct PWM filters.

### Added
- **BiasNet model** (`cerberus.models.BiasNet`): Lightweight bias model for
  Tn5 enzymatic sequence preference. Plain Conv1d + ReLU + residual stack with
  valid padding and linear profile head. 12 filters, 105bp RF, ~9.3K params.
  Fully DeepLIFT/DeepSHAP compatible via captum. Used as the bias sub-network
  in Dalmatian.
- **SimpleResidualBlock** (`cerberus.layers`): Conv1d + ReLU residual block
  with valid padding. All nn.Module-based (no F.relu) for captum compatibility.
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

### Changed
- `export_predictions.py`: Default evaluation now covers only test-fold chromosomes (`--eval-split test`). Previous behaviour (all chromosomes) is available via `--eval-split all`.

## [0.9.1] - 2026-03-01

### Added
- Stable training mode for BPNet using weight normalization and GELU activation.

### Fixed
- Removed unused imports from `bpnet.py`.

## [0.9.0] - 2026-03-01

### Added
- Initial tracked release.