# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Sequence logo helpers** (`cerberus.plots`): `plot_seqlogo`,
  `plot_attribution_heatmap`, and `plot_attribution_panel` render attribution
  maps as stacked letter logos (via `logomaker`), diverging-colormap heatmaps,
  and a combined logo+heatmap panel respectively. Accept numpy arrays or
  torch tensors, any alphabet (`"ACGT"` / `"ACGU"` / 20-letter protein), and
  three logo-height modes (`"attribution"` / `"probability"` / `"ic"`).
  All helpers take a caller-owned `ax` / `fig`; no side effects on matplotlib
  backends, no filesystem I/O. Optional-import pattern: function body raises
  a clear error pointing at `pip install 'cerberus[extras]'` if `logomaker`
  isn't installed. `logomaker` added to `extras`; `pandas` arrives as its
  transitive dep (used only inside `plot_seqlogo`'s body).
- **`--attribution-method taylor_ism`** in `tools/export_tfmodisco_inputs.py`:
  wires the new Taylor-ISM function into the TF-MoDISco export pipeline.
  Re-uses the existing `--ism-start` / `--ism-end` window; output `ohe.npz` /
  `shap.npz` shape and dtype match the `ism` path exactly so downstream
  `tools/run_tfmodisco.py` needs no flag changes. Typically ~100× faster
  than `ism` at peak widths. Documented in `docs/usage.md` alongside the
  existing ISM example.
- **Taylor-approximated ISM** (`compute_taylor_ism_attributions`):
  First-order Taylor approximation of exact ISM per Sasse et al. 2024
  (*iScience*). Replaces ``3 * L`` forward passes with one forward plus one
  backward; on linear models the output is bit-identical to
  `compute_ism_attributions` (verified by `torch.allclose`). Reference-base
  gradient computed via the dot product ``(g * x).sum(dim=1)``, matching
  the TISM reference implementation exactly and extending cleanly to soft /
  PWM inputs. Input and output contracts (shape, dtype, span zeroing, DNA
  channel validation, default TF-MoDISco ref-channel formatting) match exact
  ISM. The optional ``tf_modisco_format=False`` flag returns raw TISM deltas
  (reference channel == 0 in span) for the Majdandzic-bridge use case —
  `mean_center_attributions(raw_tism)` then equals `grad - grad.mean(dim=1)`
  (paper Eq. 8 / 11 / 15). Wrapped in `torch.enable_grad()` so it works from
  within `@torch.no_grad()` eval loops.

### Changed
- **Attribution module refactor** (`attribution.py`) — breaking, no deprecation shims:
  added `N_NUCLEOTIDES = 4` constant and `IsmSpan = tuple[int | None, int | None]`
  type alias; magic `4`s in `compute_ism_attributions` now use the named
  constant, and the DNA-alphabet assumption is greppable.
  `compute_ism_attributions(..., ism_start, ism_end)` →
  `compute_ism_attributions(..., span: IsmSpan)`; `resolve_ism_span(seq_len,
  start, end)` → `resolve_ism_span(seq_len, span)`. The TF-MoDISco reference
  override is now applied once after the ISM loop (vectorized `scatter_`)
  rather than per position, and lives in a private helper
  `_apply_tf_modisco_ref_override` that is ready for Taylor-ISM reuse.
  In-repo callers (tests, `tools/export_tfmodisco_inputs.py`) updated in the
  same commit; `--ism-start` / `--ism-end` CLI flags are unchanged (external
  contract; the tool packs into a tuple at the boundary).
- **Attribution API renames** (`attribution.py`) — breaking, no deprecation shims:
  `ATTRIBUTION_MODES` → `TARGET_REDUCTIONS` (the set names `AttributionTarget`
  *reductions*, not attribution *modes*, and would collide with the planned
  `AttributionMode` enum). `AttributionTarget` constructor arg `mode=` and
  attribute `.mode` → `reduction`. `apply_off_simplex_gradient_correction()`
  → `mean_center_attributions()` (shorter, and the operation is equivalent
  to Majdandzic off-simplex correction *and* paper ATISM *and* hypothetical
  uniform-baseline — "mean-center" names what it does without overloading
  one paper's framing). The `--target-mode` CLI flag in
  `tools/export_tfmodisco_inputs.py` is unchanged (external contract).
- **`tools/plot_training_results.py`** no longer depends on seaborn.
  The 9 repetitive per-metric plot blocks collapse into a single
  `_plot_curves` helper using `plt.plot` directly; the whitegrid look
  is preserved via matplotlib's bundled `seaborn-v0_8-whitegrid` style
  (no seaborn install required).  `seaborn` is removed from the
  `extras` dependency group in `pyproject.toml`.

### Added
- **`observed_log_counts` helper** (`predict_misc.py`):
  Mirror of `predict_log_counts` that extracts ground-truth total
  log-counts from a dataset's `target_signal_extractor` over each
  interval's `output_len`-centered window.  Auto-detects pseudocount and
  scaling from the model's `loss_cls` so predicted and observed values
  are guaranteed to be in the same log-space — designed to feed directly
  into evaluation scatter plots without manual config plumbing.
- **Pretrained-model comparison notebook**
  (`notebooks/chip_ar_mdapca2b_compare_models.py`):
  Loads both pretrained ensembles (BPNet and Pomeranian) for the
  MDA-PCA-2b AR ChIP-seq dataset, runs inference on the test intervals
  in `tests/data/fixtures/chip_ar_mdapca2b_intervals_test.bed.gz`, and
  produces a 3-panel scatter plot of predicted vs observed total
  log-counts plus model-vs-model agreement, with Pearson R annotated.
- **Batched variant effect scoring** (`predict_variants.py`):
  `score_variants()` generator composes `variant_to_ref_alt()` and
  `compute_variant_effects()` with batched model inference. Supports plain
  `nn.Module` or `ModelEnsemble` (with fold routing). Skips variants that
  fail sequence construction (boundary violations, ref mismatches) with
  logged warnings rather than crashing.
  `score_variants_from_ensemble()` convenience wrapper extracts `input_len`,
  `fasta_path`, and pseudocount parameters from the ensemble config.
  `VariantResult` frozen dataclass holds per-variant effects and provenance
  interval.
- **Saturation variant generation** (`variants.py`):
  `generate_variants(interval, fasta, max_indel_size=0)` yields all possible
  variants within a genomic interval. Default SNVs only (L x 3 alt bases);
  with `max_indel_size=k` also yields deletions up to k bases and insertions
  of all 4^k possible sequences. Composes directly with `score_variants`.
- **Variant scoring CLI tool** (`tools/score_variants.py`):
  Thin CLI wrapper around `score_variants_from_ensemble()`. Accepts VCF
  (`--vcf`) or TSV (`--variants`) input, writes per-variant effect metrics
  to TSV. Supports `--region` filtering, `--use_folds`, `--device`, and
  `--batch_size` options. Follows the `export_bigwig.py` pattern.
- **Variant scoring notebook** (`notebooks/chip_ar_mdapca2b_score_variants.py`):
  Demonstrates `score_variants_from_ensemble`, `score_variants`, and
  `generate_variants` with a pre-trained BPNet model. Includes named
  variants (peak summits, KLK3 missense, MSMB intergenic), saturation
  mutagenesis of a genomic region, and per-position max-effect tables.
- **Shared CLI utilities** (`utils.py`):
  `resolve_device()` auto-detects CUDA/MPS/CPU (replaces 6 inline copies
  across tools). `parse_use_folds()` parses `--use_folds` CLI arguments
  (replaces 3 inline copies). Both exported from `cerberus`.
- **Per-sample fold routing in ModelEnsemble** (`model_ensemble.py`):
  `_forward_models()` now routes each sample in a batch to the correct
  fold model(s) based on its interval, fixing a bug where heterogeneous
  batches (samples from different chromosomes/partitions) were all routed
  based on the first sample's interval.  New helpers
  `_get_partitions_for_interval()` and `_partitions_to_model_indices()`
  extracted from the routing logic.
- **Masked model aggregation** (`output.py`):
  `aggregate_models()` accepts an optional `masks` parameter — a list of
  `(B,)` bool tensors indicating which samples each model contributed to.
  Enables correct averaging when different models see different subsets
  of a batch.
- **Internal design document** (`docs/internal/variant_tool_design.md`):
  Records design process, tool audit, library gap analysis, and four
  design options (A-D) from minimal to full-pipeline.
- **Pretrained models** (`pretrained/`):
  Ships trained BPNet and Pomeranian models (AR ChIP-seq, MDA-PCA-2b,
  hg38) ready for inference via `ModelEnsemble("pretrained/chip_ar_mdapca2b_bpnet")` or
  `ModelEnsemble("pretrained/chip_ar_mdapca2b_pomeranian")`. Minimal footprint: `model.pt`,
  `hparams.yaml`, and `ensemble_metadata.yaml` only.

### Changed
- **Consolidated optional dependencies** (`pyproject.toml`):
  Replaced six optional-dependency groups (`variants`, `attribution`,
  `tfmodisco`, `interpret`, `plots`, `scatac`, `docs`) with two:
  `extras` (all non-core libraries) and `dev` (extras + tooling + mkdocs).
  Fixed modisco package name (`modisco` → `modisco-lite`). Removed unused
  `jupyter` and `ipywidgets`.
- **Simplified prediction notebook** (`notebooks/chip_ar_mdapca2b_predict_bpnet.py`):
  Now loads from `pretrained/chip_ar_mdapca2b_bpnet/` via `ModelEnsemble` instead of
  manually reconstructing all configs. Removed ~100 lines of redundant setup.

## [1.0.0a3] - 2026-04-08

### Added
- **`load_variants()` function** (`variants.py`):
  Lightweight alternative to `load_vcf()` for loading variants from
  tab-separated files with `chrom`, `pos`, `ref`, `alt` columns.
  Positions are 1-based by default (matching VCF/dbSNP/ClinVar convention);
  set `zero_based=True` for 0-based input. Supports optional `id` column,
  flexible column ordering, `#` comment lines, and extra columns.
- **Attribution module** (`attribution.py`):
  `AttributionTarget` wraps model output as a scalar for Captum attribution
  methods. `compute_ism_attributions()` computes single-position ISM deltas.
  `apply_off_simplex_gradient_correction()` subtracts per-position mean.
  Mode is validated at construction time via `ATTRIBUTION_MODES`.
- **TF-MoDISco interpretation tools** (`tools/export_tfmodisco_inputs.py`,
  `tools/run_tfmodisco.py`): Two-step workflow for exporting attribution
  arrays (Captum IG, DeepLiftShap, or ISM) and running TF-MoDISco motif
  discovery.
- **Internal analysis: attribution methods comparison**
  (`docs/internal/attribution_methods_tangermeme_vs_captum.md`): Technical
  reference comparing tangermeme and Captum attribution implementations.
- **Internal analysis: BPNet vs Pomeranian performance-interpretability tradeoff**
  (`docs/internal/interpretability_performance_tradeoff.md`): Documents why
  Pomeranian's higher predictive accuracy produces worse TF-MoDISco motifs,
  traces the cause to multiplicative gating, GRN, and non-linear profile head,
  and proposes concrete remediation strategies for both architectures.

### Changed
- **BPNet default residual architecture** (`bpnet.py`):
  Changed from `residual_post-activation_conv` to
  `residual_pre-activation_conv`.  Existing checkpoints trained with the old
  default are unaffected if `residual_architecture` is recorded in
  `hparams.yaml` (which it is for all cerberus-trained models).
- **BPNet final tower ReLU is now an `nn.Module`** (`bpnet.py`):
  Replaced `F.relu()` with `self.final_tower_relu = nn.ReLU()` so
  hook-based attribution methods (Captum DeepLiftShap) can register on the
  activation.  No effect on forward-pass numerics or checkpoint loading.
- **Refactored model-loading helpers** (`model_ensemble.py`):
  Extracted `find_latest_hparams()`, `select_best_checkpoint()`,
  `load_backbone_weights_from_checkpoint()`, and
  `load_backbone_weights_from_fold_dir()` as public module-level functions
  for reuse by interpretation tools.

### Fixed
- **Jitter mutation destroyed after first epoch** (`dataset.py`):
- **Case-insensitive narrowPeak detection** (`signal.py`, `samplers.py`):
  narrowPeak file detection is now case-insensitive and supports compound
  extensions: `.narrowPeak.bed.gz`, `.narrowPeak.bed`, `.narrowPeak.bb`,
  `.narrowPeak.bigbed`, `.narrowPeak.gz`, and plain `.narrowPeak`.
  Previously, files like `peaks.narrowPeak.bed.gz` silently lost summit
  information by falling through to the BED loader.
- **Extracted `_resolve_container_suffix()`** (`signal.py`):
  Compound extension resolution (stripping narrowPeak prefix, handling
  `.bed.gz`) is now a standalone helper, shared by `_resolve_extractor_cls`.

## [1.0.0a2] - 2026-04-08

### Changed
- **Simplified Dalmatian constructor** (`dalmatian.py`):
  Replaced 19 prefixed sub-model parameters (`bias_filters`, `signal_dropout`, etc.)
  with `bias_args`/`signal_args` forwarding dicts that use native sub-model parameter
  names (e.g. `bias_args={"filters": 12}`). Shared params (`input_len`, `output_len`,
  etc.) are injected automatically. Reduces constructor from 22 to 9 parameters.
- **Removed `zero_init` from Dalmatian** (`dalmatian.py`):
  The `zero_init` parameter and `_zero_init_signal_outputs()` method have been removed.
  Experiments showed zero-init is harmful with gradient detach.
- **Removed `_compute_shrinkage()` module-level function** (`dalmatian.py`):
  Replaced by `compute_shrinkage()` staticmethods on BiasNet, Pomeranian, and BPNet.
  Each model now owns its own geometry computation.

### Added
- **`Variant` dataclass** (`variants.py`):
  Frozen dataclass for genomic variants (SNPs, insertions, deletions) using
  0-based coordinates consistent with `Interval`.  Includes `ref_center` for
  symmetric window placement, `to_interval()` for fold routing, type
  classification properties (`is_snp`, `is_insertion`, `is_deletion`), and
  an `info` dict for VCF INFO fields.  First building block toward variant
  effect prediction support.
- **`load_vcf()` generator** (`variants.py`):
  Parses VCF/BCF files via cyvcf2 (optional dependency) and yields `Variant`
  objects with 0-based coordinates.  Supports region filtering (Interval or
  tabix string), PASS-only filtering, and selective INFO field capture.
  Requires biallelic, normalized input; multi-allelic records are skipped
  with a warning.
- **`variant_to_ref_alt()` function** (`variants.py`):
  Constructs one-hot ref and alt sequence tensors `(4, input_len)` from a
  `Variant` and a pyfaidx FASTA.  Window centered on `ref_center` (midpoint
  of ref allele footprint).  Indels handled via symmetric trimming from both
  flanks.  Validates ref allele against FASTA and rejects out-of-bounds
  windows.
- **`compute_signal()` function** (`output.py`):
  Converts any `ModelOutput` to linear-space predicted signal `(B, C, L)`.
  Handles `ProfileCountOutput` (softmax * counts), `ProfileLogRates`
  (exp), and `ProfileLogits` (raw fallback).  Supports batched and
  unbatched inputs with consistent pseudocount handling matching
  `compute_total_log_counts`.
- **`compute_profile_probs()` function** (`output.py`):
  Returns the normalized profile probability distribution `(B, C, L)` from
  any profile-producing model output.  Sums to 1 along the length axis.
- **`compute_channel_log_counts()` function** (`output.py`):
  Returns per-channel total counts in log space `(B, C)`.  Complements
  `compute_total_log_counts` which aggregates across channels to `(B,)`.
- **`compute_variant_effects()` function** (`variants.py`):
  Computes per-channel variant effect metrics (SAD, log fold change,
  Jensen-Shannon divergence, Pearson correlation, max absolute difference)
  between ref and alt model outputs.  Automatically adds `signal_*` metrics
  for `FactorizedProfileCountOutput` (Dalmatian) using the decomposed
  signal sub-model.
- **`shared_bias` parameter for Dalmatian** (`dalmatian.py`):
  When `shared_bias=True`, BiasNet has a single output channel (`["bias"]`) while
  SignalNet has the full N output channels. Enables multi-task training where Tn5
  insertion bias is shared across cell types (e.g. scATAC-seq pseudobulk).
- **`DalmatianLoss` shared bias support** (`loss.py`):
  When bias has fewer channels than targets, automatically sums targets across channels
  for L_bias computation (equivalent to ChromBPNet bulk training).
- **`compute_shrinkage()` staticmethod** on BiasNet, Pomeranian, and BPNet:
  Computes total valid-padding shrinkage in bp from architectural parameters.
- **Multi-task Dalmatian training tool** (`tools/train_dalmatian_multitask.py`):
  Accepts `--targets-json` for multi-task training on multiple BigWig targets.
  Adds `--shared-bias` flag for shared BiasNet across cell types.
- **Multi-task example files**: `examples/scatac_kidney_multitask_targets.json` and
  `examples/scatac_kidney_dalmatian_multitask.sh` for 14-cell-type kidney scATAC-seq.
- **Internal analysis: BPNet vs Pomeranian performance–interpretability tradeoff**
  (`docs/internal/interpretability_performance_tradeoff.md`): Documents why
  Pomeranian's higher predictive accuracy produces worse TF-MoDISco motifs,
  traces the cause to multiplicative gating, GRN, and non-linear profile head,
  and proposes concrete remediation strategies for both architectures.

### Fixed
- **Jitter mutation destroyed after first epoch** (`dataset.py`):
  `__getitem__` passed the sampler's stored `Interval` reference directly to
  transforms. Jitter mutated it in place (shrinking from `padded_size` to
  `input_len`), making jitter a no-op on all subsequent accesses to the same
  index. Now copies the interval before applying transforms.
- **Default bin pooling method changed to sum** (`transform.py`):
  `create_default_transforms` used the `Bin` class default of `"max"` pooling, but
  `predict_bigwig._process_island` divided by `output_bin_size` to recover per-bp
  signal, which is only correct for sum pooling. Explicitly set `method="sum"` in
  the default transform pipeline. Only affects models with `output_bin_size > 1`.
- **Off-by-one in peak exclusion zones** (`PeakSampler`, `NegativePeakSampler`):
  Peak intervals were added to the InterLap exclusion tree as closed `(start, end)`
  instead of `(start, end - 1)`, extending exclusion zones one base pair beyond
  each peak's half-open end. This caused background candidates starting exactly
  at a peak's exclusive end to be incorrectly rejected.
- **Shape mismatch with `count_per_channel=True` + `predict_total_count=True`**:
  The combination silently broadcast `(B, 1)` log_counts against `(B, C)` targets
  via PyTorch broadcasting, training the count head on the wrong objective. The
  affected losses (`MSEMultinomialLoss`, `PoissonMultinomialLoss`,
  `NegativeBinomialMultinomialLoss`) now raise `ValueError` on shape mismatch.
- **Misaligned windows in `predict_bigwig` near chromosome start**: When a region
  started within `offset` bp of position 0, the mid-loop `max(0, pos)` clamp
  broke stride alignment and produced an irregular first output gap. The clamp is
  now applied before the loop so the stride grid stays regular, and a warning is
  logged about the coverage gap.

## [1.0.0a1] - 2026-03-22

### Added
- **Ruff linter and formatter** configured in `pyproject.toml`: rules F (Pyflakes),
  I (isort), UP (pyupgrade), B (bugbear) enforced via `ruff check`; `ruff format`
  for deterministic whitespace normalization.
- **Pre-commit hooks** (`.pre-commit-config.yaml`): `ruff-format` and `ruff --fix`
  run automatically on every commit via the `pre-commit` framework.
- **`__all__` in `cerberus.models.__init__`**: All model re-exports are now explicit.
- `write_intervals_bed` and `load_intervals_bed` added to `cerberus.__all__`.

### Changed
- **Codebase-wide lint cleanup** (F, I, UP, B rules):
  - Removed ~150 unused imports (F401) and ~55 unused variable assignments (F841).
  - Sorted and formatted all import blocks (I001).
  - Modernized syntax: `collections.abc` imports (UP035), unquoted annotations
    (UP037), redundant open modes (UP015), `yield from` (UP028), `isinstance`
    union syntax (UP038).
  - Added exception chaining `raise ... from err/None` (B904).
  - Added `zip(strict=True)` for parallel iteration safety (B905); fixed a
    test mock in `test_complexity_matched_resample` that returned wrong-sized
    arrays (caught by `strict=True`).
  - Fixed mutable default argument in `MockSampler` (B006) and
    `ReverseComplement` (B008).
  - Prefixed unused loop variables with `_` (B007).
  - Removed duplicate `GenomeConfig` import in `dataset.py`.
- **`ruff format` applied to entire codebase**: 223 files reformatted for
  consistent whitespace, line wrapping, trailing commas, and quote style.

### Added
- **Eager file path validation** in `CerberusDataModule.setup()`: new
  `_validate_paths()` method checks that the genome FASTA and all input/target
  channel files exist before spawning DataLoader workers, surfacing missing-file
  errors immediately with a clear message naming the field and path.
- **Loss selection guide** in `docs/components.md`: added a "Choosing a Loss
  Function" table recommending a loss for common scenarios (ChIP-seq,
  low-coverage, coupled, bias-factorized, Poisson).
- **Expanded multi-GPU docs** (`docs/multi_gpu.md`): added minimal working
  examples for both CLI (`--multi` flag) and Python API, plus notes on
  `num_workers` interaction and `reload_dataloaders_every_n_epochs` in DDP.

### Breaking Changes
- **Pydantic V2 migration**: All config types (`GenomeConfig`, `DataConfig`,
  `SamplerConfig`, `TrainConfig`, `ModelConfig`, `CerberusConfig`) are now frozen
  Pydantic `BaseModel` classes instead of `TypedDict`. Bracket access
  (`config["key"]`) must change to attribute access (`config.key`). Mutations
  use `config.model_copy(update={...})`. See
  `docs/internal/pydantic_migration_breaking_changes.md` for full migration guide.
- **`count_pseudocount` moved from `DataConfig` to `ModelConfig`**: Specified in
  scaled units (raw × target_scale). `propagate_pseudocount()` deleted.
  `instantiate_metrics_and_loss()` now injects the value at construction time.
  Legacy `hparams.yaml` files are auto-migrated by `parse_hparams_config` with
  a deprecation warning.
- **`sampler_args` is `dict[str, Any]`**: Typed sampler args models
  (`PeakSamplerArgs`, `IntervalSamplerArgs`, etc.) have been removed.
  `SamplerConfig.sampler_args` is a plain dict; consumers use bracket access.
- **`fold_args` is `dict[str, Any]`**: The `FoldArgs` Pydantic model has been
  removed. `GenomeConfig.fold_args` is a plain dict; consumers use bracket
  access (e.g. `fold_args["k"]`).
- **Deleted functions**: `validate_genome_config`, `validate_data_config`,
  `validate_sampler_config`, `validate_train_config`, `validate_model_config`,
  `validate_data_and_sampler_compatibility`, `validate_data_and_model_compatibility`,
  `_sanitize_config`, `propagate_pseudocount`. Validation happens at Pydantic
  model construction time. Cross-validation runs in `CerberusConfig`'s
  `@model_validator`. Serialization uses `model.model_dump(mode="json")`.
- **`CerberusConfig.model_config_`**: The `ModelConfig` field is accessed as
  `model_config_` in Python (Pydantic reserves `model_config`). YAML key
  remains `"model_config"`.
- **Path validation removed from Pydantic models**: `@field_validator` on path
  fields has been deleted. Path resolution now happens only in
  `parse_hparams_config` via `_resolve_paths_in_config()`.
- **New dependency**: `pydantic>=2.0` added to `pyproject.toml`.

### Added
- **Expanded public API** (`__init__.py`): `TrainConfig`, `ModelConfig`,
  `PretrainedConfig`, `CerberusConfig`, `CerberusModule`, `instantiate`,
  `instantiate_model`, `train_single`, and `train_multi` are now importable
  directly from `cerberus`.
- **`pretrained.py` unit tests** (`tests/test_pretrained.py`): 19 tests
  covering prefix extraction, full/sub-module weight loading, freeze
  behavior, strict mode rejection, and multi-config application.
- **`predict_bigwig.py` unit tests** (`tests/test_predict_bigwig.py`): 18
  tests covering signal reconstruction (ProfileCountOutput, ProfileLogRates,
  multi-channel, pseudocount clamping), island processing (overlap averaging,
  target_scale undo), and end-to-end BigWig write (stride defaults, region
  mode, error cleanup).
- **`model_config_` alias warning** (`docs/configuration.md`): prominent
  admonition explaining the Pydantic V2 naming conflict and showing
  correct vs incorrect usage.
- **`model_copy()` examples** (`docs/configuration.md`): new section
  demonstrating how to derive modified copies of frozen configs.
- **Pydantic config regression tests** (`tests/test_pydantic_config.py`):
  76 tests covering construction, validation, typed sampler args, serialization
  round-trips, cross-validation, model_copy, backward compatibility, and the
  `model_config_` alias.
- **`complexity_center_size` parameter** for `PeakSampler`, `NegativePeakSampler`,
  and `ComplexityMatchedSampler`: crops intervals to their center N bp before
  computing complexity metrics. Default `None` preserves existing behavior.
- **Sampler and DataLoader context-length benchmarks**
  (`tests/benchmark/bench_sampler_context_length.py`,
  `bench_dataloader_context_length.py`, `bench_encode_dna.py`).

### Changed
- **All src/, tools/, tests/, notebooks/, docs/ migrated to Pydantic attribute
  access**: 158 files changed. `config["key"]` → `config.key`, `{**config}` →
  `model_copy(update=...)`, `_sanitize_config()` → `model_dump(mode="json")`,
  `cast(Config, dict)` → `Config(...)` or `Config.model_construct(...)`.
- **`encode_dna` rewritten with broadcast comparison**: ~11x faster at 128 kb.
- **`sampler_args` and `fold_args` simplified to plain dicts**: Deleted typed
  args classes (`FoldArgs`, `PeakSamplerArgs`, `NegativePeakSamplerArgs`,
  `IntervalSamplerArgs`, etc.). `SamplerConfig.sampler_args` is now
  `dict[str, Any]`; `GenomeConfig.fold_args` is now `dict[str, Any]`. All
  `tools/`, `notebooks/`, and `tests/` updated to use plain dict constructors
  and bracket access (`fold_args["test_fold"]` instead of `fold_args.test_fold`).

### Fixed
- **Post-migration audit of tools/, examples/, notebooks/, docs/**:
  - `tools/export_bigwig.py`: Fixed bracket access on frozen Pydantic config
    (`cerberus_config["data_config"]` -> `cerberus_config.data_config`) and
    replaced direct dict mutation with `model_copy(update=...)`.
  - `notebooks/model_ensemble_demo.py`: Removed unused `from typing import cast`
    and replaced `cast(ProfileCountOutput, ...)` calls with direct assignment.
  - `docs/prediction.md`: Replaced stale `search_paths` parameter documentation
    with the current config-override approach (`genome_config=`, `data_config=`).
- **`SamplerConfig.resolve_sampler_args` forwards validation context**: Passes
  `search_paths` via `model_validate()` instead of bare `__init__()`. Fixes
  `FileNotFoundError` when loading `hparams.yaml` with relative paths.

## [0.9.5] - 2026-03-20

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
- `count_pseudocount` validation relaxed from `> 0` to `>= 0` — allows `0.0`
  for Poisson/NB losses that do not use a pseudocount offset.
- `propagate_pseudocount` now warns when `count_pseudocount > 0` is paired
  with a loss that ignores it; also injects `log_counts_include_pseudocount`
  into `metrics_args` for correct multi-channel aggregation.
- `predict_to_bigwig` no longer takes a `count_pseudocount` parameter —
  auto-detects from the model config via `get_log_count_params`.
- `export_bigwig.py` CLI: removed `--count-pseudocount` argument (now automatic).
- `export_predictions.py`: replaced `isinstance` + `getattr` pseudocount
  detection with `get_log_count_params`; uses `compute_obs_log_counts`.

### Fixed
- **Multi-channel log-count metric aggregation with pseudocount**: `LogCountsMeanSquaredError`
  and `LogCountsPearsonCorrCoef` now correctly invert per-channel log-counts before summing
  when `log_counts_include_pseudocount=True`, avoiding the `log(total + C*pseudocount)` error
  from naive `logsumexp`. The new `log_counts_include_pseudocount` flag is propagated
  automatically via `propagate_pseudocount`.
- **Validation scatter plot refactored to use metric state**: Removed the separate
  `_accumulate_log_counts` method from `CerberusModule` which duplicated log-count
  accumulation and used the wrong dispatch mechanism (`hasattr` on instance attributes
  instead of the canonical `uses_count_pseudocount` class attribute). The scatter plot
  now reads directly from `LogCountsPearsonCorrCoef`'s accumulated `preds_list`/`targets_list`,
  ensuring correct pseudocount handling for all loss families including DalmatianLoss.
- **`PomeranianMetricCollection` and `BPNetMetricCollection` now accept `log_counts_include_pseudocount`**:
  Both model-specific MetricCollections were missing the `log_counts_include_pseudocount` parameter,
  causing `TypeError` when `instantiate_metrics_and_loss()` passed it from `propagate_pseudocount()`.
  The flag is now threaded through to the inner `LogCountsMeanSquaredError` and `LogCountsPearsonCorrCoef`
  sub-metrics, matching the existing behavior of `DefaultMetricCollection`.

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