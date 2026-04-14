# Variant Effect Prediction Tool: Design & Library Gap Analysis

**Date:** 2026-04-13
**Status:** Design proposal (no implementation)
**Predecessor:** `docs/internal/variant_support.md` (Phase 1 design from initial survey)

---

## 1. Goal

Design a CLI tool (`tools/score_variants.py`) that takes a trained cerberus
model (single-fold or multi-fold ensemble), a VCF or variant TSV file, and
produces per-variant effect scores.  The tool should be a thin CLI wrapper
around core library functionality -- the bulk of the logic should live in
`src/cerberus/` so it is reusable from notebooks, other tools, and
potentially training pipelines.

---

## 2. Existing Tool Patterns

An audit of all 17 files in `tools/` reveals several recurring patterns
relevant to the variant tool's design.

### 2.1 Model loading

Two distinct patterns exist, with no overlap:

| Pattern | Tools | Entry point |
|---------|-------|-------------|
| **ModelEnsemble** | export_bigwig, export_predictions | `ModelEnsemble(path, device=device)` -- discovers hparams, loads all folds, provides `cerberus_config` |
| **Single-fold manual** | export_tfmodisco_inputs | `_resolve_fold_dir()` + `find_latest_hparams()` + `parse_hparams_config()` + `instantiate_model()` + `load_backbone_weights_from_fold_dir()` (5 calls) |
| **BiasNet custom** | plot_biasnet_ism, plot_biasnet_pairwise_ism | `load_biasnet()` with manual hparams parsing, sub-model extraction (duplicated between both files) |

**ModelEnsemble** is the clear choice for the variant tool.  It handles
single-fold models (when `ensemble_metadata.yaml` lists one fold),
multi-fold ensembles, fold routing via intervals, and exposes
`cerberus_config` for `input_len`, `get_log_count_params`, and `fasta_path`.

The single-fold manual pattern exists because `export_tfmodisco_inputs.py`
predates `ModelEnsemble`'s current capabilities and needs to operate on
arbitrary fold directories that may lack `ensemble_metadata.yaml`.  The
variant tool should not inherit this limitation.

### 2.2 Device resolution

Every tool that uses a GPU has its own inline device detection:

```python
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

This appears in 12 of 17 tools.  `export_tfmodisco_inputs.py` is the only
one that extracts this into a `_resolve_device()` function, but it is
private to that file.  A shared utility would reduce boilerplate.

### 2.3 Dataset creation

Only `export_bigwig.py` and `export_predictions.py` create a
`CerberusDataset`.  Both construct it from `ensemble.cerberus_config`
with `is_train=False` and `sampler_config=None`.  `predict_misc.py`
provides `create_eval_dataset()` for exactly this purpose, but
**no tool uses it** -- each creates the dataset inline.

### 2.4 use_folds parsing

Only `export_bigwig.py` and `export_predictions.py` parse `--use_folds`.
Both use the same `re.split(r"[+,]", ...)` pattern with "all" expansion.
This is duplicated between them.

### 2.5 Duplicated code

The most significant duplication is between `plot_biasnet_ism.py` and
`plot_biasnet_pairwise_ism.py`: approximately 254 lines are
character-for-character identical, including `_resolve_fold_dir`,
`load_config`, `_extract_bias_state_dict`, `load_biasnet`,
`load_peak_intervals`, `extract_onehot`, `get_background_sequences`,
`get_real_sequences`, and `plot_logo`.

---

## 3. What the Library Already Provides

### 3.1 Variant primitives (variants.py)

| Function | What it does |
|----------|-------------|
| `Variant` dataclass | Frozen, 0-based coordinates, type properties (`is_snp`, `is_deletion`), `to_interval()`, `from_str()` |
| `load_vcf(path, region, pass_only, info_fields)` | Iterator of `Variant` from VCF/BCF via cyvcf2, region query via tabix |
| `load_variants(path, zero_based)` | Iterator of `Variant` from tab-delimited file |
| `variant_to_ref_alt(variant, fasta, input_len)` | Returns `(ref_tensor, alt_tensor, interval)` -- one-hot `(4, input_len)` tensors with symmetric indel trimming |
| `compute_variant_effects(ref_output, alt_output, ...)` | Returns `dict[str, Tensor]` of effect metrics (SAD, log_fc, JSD, pearson, max_abs_diff, plus Dalmatian signal-only metrics) |

### 3.2 Output utilities (output.py)

| Function | Relevance |
|----------|-----------|
| `get_log_count_params(model_config)` | Required to parameterize `compute_variant_effects` correctly |
| `compute_signal(output, ...)` | Used internally by `compute_variant_effects` |
| `compute_profile_probs(output)` | Used internally by `compute_variant_effects` |

### 3.3 Model loading (model_ensemble.py)

| Component | Relevance |
|-----------|-----------|
| `ModelEnsemble(checkpoint_path, device)` | Loads all folds, discovers config, exposes `cerberus_config` |
| `ModelEnsemble.forward(x, intervals)` | Accepts raw tensors + intervals for fold routing, no dataset needed |
| `find_latest_hparams()`, `parse_hparams_config()` | Available for single-fold fallback but not needed if using ModelEnsemble |

### 3.4 Prediction utilities (predict_misc.py)

| Function | Relevance |
|----------|-----------|
| `create_eval_dataset(config)` | Creates dataset for inference -- needed if model has input signal tracks |
| `load_bed_intervals(config, bed_path)` | Could provide region filtering for the variant tool |

---

## 4. What the Library Is Missing

### 4.1 Batched variant scoring function

**The primary gap.**  All per-variant primitives exist, but there is no
function that wires them together into a batched inference loop:

```
variants -> variant_to_ref_alt (per variant)
         -> batch ref/alt tensors
         -> model.forward (ref batch) / model.forward (alt batch)
         -> compute_variant_effects (per batch)
         -> unbatch and yield per-variant results
```

This ~50-line function handles: batching, device placement, paired
forward passes through the same folds, error handling per variant
(boundary violations, ref mismatches logged and skipped rather than
crashing), and progress reporting.

**Where it should live:** `src/cerberus/predict_variants.py` -- a peer
of `predict_bigwig.py` and `predict_misc.py`.  These modules compose
`ModelEnsemble` + output transforms into specific prediction workflows.
`variants.py` stays focused on data types and per-variant operations.

### 4.2 Variant tensor patching for multi-channel models

`variant_to_ref_alt()` constructs complete `(4, input_len)` tensors from
FASTA.  When a model was trained with additional input signal tracks
(e.g. accessibility BigWig via `data_config.inputs`), the model expects
more than 4 channels.  Currently there is no way to:

1. Get the full multi-channel input tensor for a genomic interval
   (via the dataset), and then
2. Replace only the sequence channels (0:4) with the alt allele.

A function like `apply_variant_to_tensor(input_tensor, variant, interval)`
would fill this gap.  It operates on an already-extracted tensor rather
than going back to FASTA, modifying only the sequence channels.

**Training relevance:** This same primitive would be needed for
training-time variant augmentation.  A VCF-based transform in the
dataset pipeline would call this function on each training sample's
input tensor.

**Where it should live:** `src/cerberus/variants.py` alongside
`variant_to_ref_alt`.

### 4.3 Shared device resolution utility

12 of 17 tools duplicate the same 6-line device detection block.
A `resolve_device(device_arg: str | None) -> torch.device` function
in `src/cerberus/utils.py` (or as a module-level function in
`__init__.py`) would eliminate this.

**Not critical** for the variant tool specifically, but a natural
cleanup given that the variant tool will need it too.

### 4.4 Shared use_folds parser

`export_bigwig.py` and `export_predictions.py` duplicate the same
`--use_folds` parsing logic.  The variant tool will also need it.
A `parse_use_folds(arg: str | None) -> list[str] | None` utility
would serve all three tools.

**Where it should live:** `src/cerberus/model_ensemble.py` alongside
the `ModelEnsemble` class that consumes the result.

### 4.5 Persistent VCF index for training use

`load_vcf()` currently opens and closes the VCF handle per call.
For training-time variant augmentation (where every training sample
queries the VCF for overlapping variants), a persistent object that
holds the tabix-indexed VCF open and supports fast region queries
would be needed.  Not required for the inference tool (which iterates
variants sequentially), but relevant for the broader variant support
story.

---

## 5. Design Options (Simple to Involved)

### 5.1 Option A: Minimal tool, no library additions

**Approach:** Put all logic in `tools/score_variants.py`.  The tool
does its own batching loop inline.

**Pros:**
- Zero library changes
- Self-contained, easy to review
- Ships immediately

**Cons:**
- Batching logic is not reusable (notebooks, other tools)
- Sequence-only models only (no multi-channel support)
- Follows the `export_tfmodisco_inputs.py` anti-pattern of complex
  tools with substantial logic that should be in the library

**Appropriate when:** You want a working tool today and plan to
refactor later.

### 5.2 Option B: Add `score_variants()` to library (recommended)

**Approach:** Add `src/cerberus/predict_variants.py` with a
`score_variants()` generator function.  The tool becomes a thin CLI
wrapper: parse args, load model, call `score_variants()`, write TSV.

**Library additions:**

```
src/cerberus/predict_variants.py  (new file, ~80-120 lines)
    score_variants(
        model: nn.Module | ModelEnsemble,
        variants: Iterable[Variant],
        fasta: pyfaidx.Fasta,
        input_len: int,
        log_count_params: tuple[bool, float],
        device: torch.device,
        batch_size: int = 64,
        use_folds: list[str] | None = None,
    ) -> Iterator[tuple[Variant, dict[str, torch.Tensor]]]
```

**Tool structure (mirrors export_bigwig.py):**
1. Parse CLI args
2. Load `ModelEnsemble`
3. Extract config: `input_len`, `fasta_path`, `get_log_count_params()`
4. Open FASTA, load variants (VCF or TSV)
5. Call `score_variants()`, write rows to TSV
6. Optionally write summary statistics

**Pros:**
- Core logic reusable from notebooks and other tools
- Tool is genuinely thin (~100 lines of argparse + I/O)
- Follows the established `predict_bigwig.py` / `predict_misc.py` pattern
- `score_variants()` handles errors per variant gracefully

**Cons:**
- Sequence-only models only (if model has input signal tracks, error)
- No dataset integration

**Appropriate when:** Variant scoring is primarily an inference-time
concern and all target models are sequence-only.

### 5.3 Option C: Dataset-integrated variant support

**Approach:** Extend Option B with `apply_variant_to_tensor()` so that
the scoring function can work with multi-channel models by going
through the dataset pipeline.

**Additional library additions:**

```
src/cerberus/variants.py  (additions to existing file)
    apply_variant_to_tensor(
        input_tensor: torch.Tensor,
        variant: Variant,
        interval: Interval,
        encoding: str = "ACGT",
    ) -> torch.Tensor
```

This takes a full multi-channel input tensor (as returned by
`dataset.get_interval()`), identifies the sequence channels,
and splices in the alt allele at the correct offset within the
interval.  Signal channels pass through unchanged.

**Changes to `score_variants()`:** Accept an optional `dataset`
parameter.  When provided and the model has input signals:
- Use `dataset.get_interval(interval)` to get the full ref input
- Call `apply_variant_to_tensor()` to produce the alt input
- Forward both through the model

When `dataset` is None (sequence-only model):
- Use `variant_to_ref_alt()` as before (no dataset needed)

**Pros:**
- Works with all model types (sequence-only and multi-channel)
- `apply_variant_to_tensor()` is reusable for training-time
  variant augmentation
- Clean separation: the tool doesn't need to know whether the
  model has input signals

**Cons:**
- More complex implementation
- Dataset creation requires valid file paths for input signals
  (BigWig files must be accessible at scoring time)
- Indel handling on tensor level is trickier than on string level
  (need to re-encode only the modified region)

**Appropriate when:** Multi-channel models (e.g. models with
accessibility input tracks) are a target for variant scoring.

### 5.4 Option D: Full pipeline with training support

**Approach:** Extend Option C with a variant-aware dataset transform
and a persistent VCF index, enabling both inference and training.

**Additional library additions:**

```
src/cerberus/variants.py  (additions)
    class VariantIndex:
        """Persistent tabix-indexed VCF handle for fast region queries."""
        def __init__(self, path: Path, pass_only: bool = True)
        def query(self, interval: Interval) -> list[Variant]

src/cerberus/transforms.py  (new transform)
    class ApplyVariants:
        """Dataset transform that applies VCF variants to sequence channels."""
        def __init__(self, variant_index: VariantIndex, ...)
        def __call__(self, inputs, targets, interval):
            overlapping = self.variant_index.query(interval)
            for v in overlapping:
                inputs = apply_variant_to_tensor(inputs, v, interval)
            return inputs, targets, interval
```

**Config changes:**
- `DataConfig` gains optional `variants_path: Path | None` field
- Dataset construction checks for this and adds the transform

**Pros:**
- Unified variant handling for inference and training
- VCF variants become a config option, not a separate tool concern
- Reuses the existing transform pipeline (jitter, RC, etc. compose
  naturally with variant application)

**Cons:**
- Significant design surface (config changes, new transform)
- Training with indel variants requires careful handling
  (sequence length changes, target signal realignment)
- VCF-based training is a research feature with unclear demand

**Appropriate when:** Training on personalized genomes is a
near-term goal and the team wants a unified variant pipeline.

---

## 6. Recommended Strategy

**Start with Option B.  Plan for Option C.**

Option B delivers a working variant scoring tool with reusable library
code and zero changes to existing modules.  The tool follows the
established pattern of `export_bigwig.py` (load ModelEnsemble, call
a library function, write output).

Concretely:

### New files

1. **`src/cerberus/predict_variants.py`** -- `score_variants()` generator
   function.  Handles batching, fold routing, error handling per variant.
   Yields `(Variant, dict[str, Tensor])` pairs.

2. **`tools/score_variants.py`** -- CLI wrapper.  Argparse, model loading,
   FASTA/VCF loading, calls `score_variants()`, writes TSV.

### Optional library improvements (not blocking, but reduce duplication)

3. **`resolve_device()` in `src/cerberus/utils.py`** -- shared device
   detection.  Eliminates 12 copies of the same inline block.

4. **`parse_use_folds()` in `src/cerberus/model_ensemble.py`** -- shared
   `--use_folds` argument parsing.  Eliminates duplication between
   export_bigwig, export_predictions, and the new variant tool.

### Future (Option C path)

5. **`apply_variant_to_tensor()` in `src/cerberus/variants.py`** -- when
   multi-channel model support is needed.  This is the bridge between
   the dataset pipeline and variant application, and the primitive
   that would also serve training-time variant augmentation.

---

## 7. CLI Design for `tools/score_variants.py`

Following the patterns established by `export_bigwig.py` and
`export_predictions.py`:

```
python tools/score_variants.py MODEL_PATH \
    --vcf variants.vcf.gz \          # or --variants variants.tsv
    --output effects.tsv \
    [--fasta /path/to/genome.fa] \   # default: from model config
    [--region chr1:1000000-2000000] \ # optional region filter
    [--regions-bed regions.bed] \     # optional BED filter
    [--batch-size 64] \
    [--device auto] \
    [--use-folds test+val] \
    [--channels 0,1] \               # which output channels to score
    [--output-format tsv|csv]
```

**Required arguments:**
- `MODEL_PATH` -- path to trained model directory (ModelEnsemble compatible)
- `--vcf` or `--variants` -- variant source (mutually exclusive)

**Output columns:**
`chrom`, `pos`, `ref`, `alt`, `id`, and per-channel effect metrics:
`sad_ch0`, `log_fc_ch0`, `jsd_ch0`, `pearson_ch0`, `max_abs_diff_ch0`,
plus `signal_sad_ch0`, `signal_log_fc_ch0`, `signal_jsd_ch0` for
Dalmatian models.

---

## 8. Comparison with Existing Tools

| Aspect | export_bigwig | export_predictions | score_variants |
|--------|--------------|-------------------|----------------|
| **Model loading** | ModelEnsemble | ModelEnsemble | ModelEnsemble |
| **Input source** | SlidingWindow/BED | Peak BED + BigWig | VCF/TSV |
| **Dataset needed** | Yes (for signals) | Yes (for targets) | No (sequence-only, Option B) |
| **Library function** | `predict_to_bigwig()` | inline | `score_variants()` |
| **Output** | BigWig | TSV + metrics JSON | TSV |
| **Forward passes** | 1 per interval | 1 per interval | 2 per variant (ref + alt) |
| **Key difference** | Spatial merge | Obs vs pred comparison | Ref vs alt comparison |

---

## 9. Library Gap Summary

| Gap | Blocks tool? | Blocks training? | Effort | Where |
|-----|-------------|-----------------|--------|-------|
| `score_variants()` batched generator | **Yes** (Option A works around) | No | Small | `predict_variants.py` (new) |
| `apply_variant_to_tensor()` | No (sequence-only OK) | **Yes** | Small | `variants.py` |
| `resolve_device()` shared utility | No (inline works) | No | Trivial | `utils.py` |
| `parse_use_folds()` shared utility | No (inline works) | No | Trivial | `model_ensemble.py` |
| `VariantIndex` persistent VCF handle | No | Yes (for perf) | Small | `variants.py` |
| `ApplyVariants` dataset transform | No | **Yes** | Medium | `transforms.py` |

---

## 10. Relationship to variant_support.md

The earlier `docs/internal/variant_support.md` document surveyed external
repositories and outlined a three-phase plan.  This document refines
Phase 1 (variant scoring tool) based on what was actually implemented
in `variants.py` and the current state of the library.

**What changed since the original design:**
- `variants.py` was implemented with `Variant`, `load_vcf`, `load_variants`,
  `variant_to_ref_alt`, and `compute_variant_effects`.
- `compute_variant_effects` lives in `variants.py` (the original plan
  placed signal reconstruction in variants.py; the actual implementation
  delegates to `output.py` functions).
- The `_reconstruct_linear_signal` duplication between `predict_bigwig.py`
  and `variants.py` was resolved -- `variants.py` uses `compute_signal()`
  from `output.py`.
- `AttributionTarget` and `compute_ism_attributions` were implemented in
  `attribution.py` (Phase 2 scope from the original plan).

**What remains from the original design:**
- The CLI tool itself (`tools/score_variants.py`)
- The batched scoring function (was implicit in the original tool design,
  now explicitly identified as the primary library gap)
- Multi-channel model support via tensor patching
- Training-time variant support (Phase 3, unchanged)
