# Prediction (Inference)

Prediction in Cerberus involves applying a trained model (or ensemble of models) to genomic intervals. This phase uses the `ModelEnsemble` class to manage model loading, fold selection, and output aggregation.

## Core Concepts

### ModelEnsemble
The `ModelEnsemble` class is the central entry point for prediction. It:
1.  **Loads Models**: Automatically detects single-fold or multi-fold model directories and loads weights. Prefers clean `model.pt` state dicts (faster, no prefix stripping) and falls back to Lightning `.ckpt` checkpoints for backward compatibility.
2.  **Manages Folds**: Identifies which model to run for a given genomic interval based on cross-validation fold definitions.
    *   **Rotation Logic**: Cerberus assumes a standard rotation where Model `i` uses Partition `i` as Test and Partition `(i+1)%k` as Validation.
    *   If you request `use_folds=["test"]`, the ensemble selects Model `i` for intervals falling in Partition `i`.
3.  **Aggregates**: Combines outputs from multiple models or overlapping intervals.
4.  **Resolves Paths**: Handles path adjustments when models trained on one machine are loaded on another (see *Path Resolution*).

### Aggregation Strategies
When multiple models predict on the same interval, or when tiling produces overlapping predictions, Cerberus needs to merge them.
*   **"model"**: Aggregates outputs across models (e.g., averaging predictions from all folds). Returns batched `ModelOutput` objects.
*   **"interval+model"**: Aggregates across models AND merges overlapping interval predictions into a single contiguous track. Returns unbatched `ModelOutput` objects.

    **Note on Alignment**: When merging overlapping predictions (e.g. from tiling), if `output_bin_size > 1`, Cerberus performs a **"snap-to-grid"** alignment. 
    Prediction starts are floored to the nearest bin relative to the merged interval start. This ensures consistent binning but means sub-bin shifts (e.g. from jitter) are quantized.

## Pretrained Models

Cerberus ships a pretrained BPNet model in the `pretrained/` directory, ready for inference without retraining:

| Model | Path | Dataset | Architecture |
|---|---|---|---|
| BPNet (AR ChIP-seq) | `pretrained/chip_ar_mdapca2b_bpnet/` | MDA-PCA-2b AR (hg38) | BPNet (kernel 21, 8 dilated layers) |
| Pomeranian (AR ChIP-seq) | `pretrained/chip_ar_mdapca2b_pomeranian/` | MDA-PCA-2b AR (hg38) | Pomeranian (ConvNeXtV2 stem, 8 PGC layers) |

```python
from cerberus.model_ensemble import ModelEnsemble

ensemble = ModelEnsemble("pretrained/chip_ar_mdapca2b_bpnet", device="cuda")
```

## Setup for Prediction

To perform prediction, you need to instantiate a `ModelEnsemble`.

```python
from cerberus.model_ensemble import ModelEnsemble

# Load from a training log directory (containing hparams.yaml and checkpoints)
ensemble = ModelEnsemble(
    checkpoint_path="logs/my_experiment",
    device="cuda"
)

# Access the configs used during training
genome_config = ensemble.cerberus_config.genome_config
data_config = ensemble.cerberus_config.data_config
```

### Path Resolution
If you move a trained model to a new environment (e.g., from a training cluster to a local machine), the absolute paths in `hparams.yaml` (pointing to FASTA or BigWig files) might be invalid.
You can override specific configs by passing `genome_config`, `data_config`, or `model_config` to `ModelEnsemble`:

```python
ensemble = ModelEnsemble(
    checkpoint_path="logs/my_experiment",
    genome_config=my_genome_config,  # Override genome paths
    data_config=my_data_config,      # Override data paths
)
```

## Prediction Methods

### 1. Batched Interval Prediction (Efficient)
Use `predict_intervals_batched` to process a large number of intervals (e.g. peaks) without loading everything into memory. It yields batches of results.

```python
# Create a sampler for your regions of interest
from cerberus.samplers import IntervalSampler
sampler = IntervalSampler(file_path="peaks.bed", ...)

# Iterate over batches
for batch_output, batch_intervals in ensemble.predict_intervals_batched(
    intervals=sampler,
    dataset=dataset,
    use_folds=["test", "val", "train"], # Use all folds to ensure coverage
    batch_size=256
):
    # Process batch_output (ModelOutput object)
    pass
```

### 2. Predict on Specific Intervals (In-Memory)
Use `predict_intervals` when you have a small list of intervals and want a single aggregated result tensor.

```python
output = ensemble.predict_intervals(
    intervals=my_intervals,
    dataset=dataset,
    use_folds=["test"],
    batch_size=64
)
```

### 3. Predict on Arbitrary Regions (Tiling)
Use `predict_output_intervals` when you have target regions of arbitrary length (e.g., entire peaks or genes). The system tiles the region with overlapping input windows and merges the results.

```python
outputs = ensemble.predict_output_intervals(
    intervals=gene_regions,
    dataset=dataset,
    stride=50,
    use_folds=["test"],
    batch_size=64
)
```

### 4. Genome-Wide Prediction (BigWig)
Use `predict_to_bigwig` to generate a genome-wide signal track. This is useful for visualization in genome browsers.

```python
from cerberus.predict_bigwig import predict_to_bigwig

predict_to_bigwig(
    output_path="predictions.bw",
    dataset=dataset,
    model_ensemble=ensemble,
    stride=50,
    use_folds=["test"],
    batch_size=256
)
```

## CLI Tools

Cerberus provides command-line tools for common prediction tasks.

### Export BigWig (`tools/export_bigwig.py`)
Exports genome-wide model predictions as a BigWig signal track for visualization in genome browsers (e.g., IGV, UCSC).

```bash
python tools/export_bigwig.py \
    path/to/model_dir \
    --output predictions.bw \
    --stride 50 \
    --batch_size 256 \
    --device cuda
```

This tool:
1. Loads the model ensemble and automatically resolves dataset paths.
2. Clears target extractors (no observed signal is needed — only sequence inputs are used).
3. Slides a window across all allowed chromosomes, grouping contiguous windows into islands.
4. Reconstructs linear signal per window (softmax × counts for BPNet, exp for log-rates) then spatially merges overlapping predictions using a streaming accumulator.
5. Automatically reads `count_pseudocount` from the model config and subtracts it when inverting `log_counts` (MSE-trained models).

The exported track contains the reconstructed linear signal (counts per bp). For multi-channel models, only channel 0 is written (BigWig is single-track).

#### Optional arguments

| Argument | Description |
|---|---|
| `--output` | Output BigWig path (default: `predictions.bw`). |
| `--stride` | Sliding window stride in bp. Defaults to `output_len // 2` (50% overlap). |
| `--use_folds` | Folds to use, e.g. `test`, `test+val`, `all`. Defaults to `test+val`. |
| `--batch_size` | Batch size for inference (default: 64). |
| `--device` | Override device selection (`cuda`, `mps`, `cpu`). |

### Export Predictions (`tools/export_predictions.py`)
Exports predicted vs observed log-counts for a set of peaks to a TSV file. This is useful for evaluating model performance on peak sets.

```bash
python tools/export_predictions.py \
    path/to/model_dir \
    path/to/peaks.bed \
    path/to/observed.bigwig \
    --output predictions.tsv.gz \
    --batch_size 128 \
    --device cuda
```

This tool:
1. Loads the model and automatically resolves dataset paths.
2. Overrides the target BigWig with the provided one (useful for cross-sample prediction).
3. Computes total log-counts for each peak.
4. Saves a compressed TSV with columns: `chrom`, `start`, `end`, `strand`, `pred_interval`, `predicted_log_count`, `observed_log_count`.
5. Saves a companion `predictions.metrics.json` with aggregated metrics and loss.

#### Log-space semantics

The `predicted_log_count` and `observed_log_count` columns are placed in the same log-space as the training objective, which depends on the loss:

| Loss family | Count head trains against | Column space |
|---|---|---|
| `MSEMultinomialLoss`, `CoupledMSEMultinomialLoss` | `log(total_counts + count_pseudocount)` | `log(x + p)` (defaults to `log1p` when `count_pseudocount=1.0`) |
| `PoissonMultinomialLoss`, `NegativeBinomialMultinomialLoss`, and their `Coupled` variants | `log(total_counts)` (via `PoissonNLLLoss(log_input=True)`) | `log` |

The tool reads `uses_count_pseudocount` from the loss class via `get_log_count_params` to determine this automatically — no manual configuration is needed.

The observed total is computed from raw BigWig signal (bypassing transforms) and then multiplied by `data_config.target_scale` before the log transform, matching what the loss was trained against.

#### Multi-channel models (`count_per_channel=True`)

When the model has multiple output channels and was trained with `count_per_channel=True` under an MSE loss, each channel's `log_counts` is in `log(count + count_pseudocount)` space. The tool correctly aggregates these to a single global log-count via:

```
log((exp(lc_ch0) - p) + (exp(lc_ch1) - p) + ... + p)
```

where `p = count_pseudocount`, rather than the naive `logsumexp`, which would give `log(n_channels * p + total_counts)`.

#### Optional arguments

| Argument | Description |
|---|---|
| `--eval-split` | Which chromosome split to evaluate on: `test` (default, held-out test chromosomes), `val`, `train`, or `all`. Use `all` only for exploratory analysis — it includes training chromosomes and inflates metrics. |
| `--include-background` | Include complexity-matched background (non-peak) intervals alongside peaks, replicating the training evaluation setup. Requires the model to have been trained with a `peak` sampler. Adds an `interval_source` column to the output (sampler class name). |
| `--background-ratio` | Ratio of background intervals to peaks (default: taken from model's sampler config). Only used with `--include-background`. |
| `--seed` | Random seed for background interval generation (default: 1234). Only used with `--include-background`. |
| `--use_folds` | Folds to use, e.g. `test`, `test+val`, `all`. Defaults to `test+val` for multi-fold ensembles, `all` for single-fold. |
| `--max_batches` | Limit the number of batches processed (useful for quick checks). |
| `--device` | Override device selection (`cuda`, `mps`, `cpu`). |

### Score Variants (`tools/score_variants.py`)
Scores the predicted effect of genomic variants (from VCF or TSV) by comparing model predictions on reference vs alternative sequences. See [Variant Effect Prediction](variants.md) for full documentation.

```bash
python tools/score_variants.py path/to/model_dir \
    --vcf variants.vcf.gz \
    --output effects.tsv \
    --batch_size 64 \
    --device cuda
```

| Argument | Description |
|---|---|
| `--vcf` / `--variants` | Variant source (mutually exclusive, required). |
| `--output` | Output TSV path (default: `variant_effects.tsv`). Supports `.gz`. |
| `--fasta` | Override FASTA path (default: from model config). |
| `--region` | Restrict to variants in a region (e.g. `chr1:1000000-2000000`). |
| `--batch_size` | Variants per inference batch (default: 64). |
| `--use_folds` | Folds for ensemble prediction (e.g. `test`, `test+val`, `all`). |
| `--device` | Device override (`cuda`, `mps`, `cpu`). |

## Model Outputs

Predictions return `ModelOutput` objects (defined in `cerberus.output`), which contain the raw tensors.

*   **ProfileLogits**: Unnormalized log-probabilities (shape `Batch, Channels, Length`).
*   **ProfileLogRates**: Log-scale counts/rates (shape `Batch, Channels, Length`). `exp(log_rates)` gives predicted counts per bin per channel.
*   **ProfileCountOutput**: Profile logits plus a total log-count head (BPNet style). `log_counts` has shape `Batch, Channels` when `count_per_channel=True`, otherwise `Batch, 1`.

You typically need to detach and post-process these (e.g., applying Softmax or Exponentials) for analysis.

### Extracting total log-counts

`cerberus.output.compute_total_log_counts` aggregates a `ModelOutput` into a scalar log-count per interval in the batch:

```python
from cerberus.output import compute_total_log_counts

# log_counts_include_pseudocount=True for MSEMultinomialLoss / CoupledMSEMultinomialLoss
# log_counts_include_pseudocount=False (default) for Poisson / NB losses
log_counts = compute_total_log_counts(batch_output, log_counts_include_pseudocount=True)
# → tensor of shape (batch_size,)
```

The `log_counts_include_pseudocount` parameter matters only for multi-channel `ProfileCountOutput` (i.e. `count_per_channel=True`):

- **`log_counts_include_pseudocount=False`** (default): assumes `log_counts` per channel is in natural-log space (Poisson/NB losses). Uses `logsumexp` to sum across channels: `log(Σ exp(log_counts_c))`.
- **`log_counts_include_pseudocount=True`**: assumes `log_counts` per channel is in `log(count + count_pseudocount)` space (MSE loss). Inverts per channel, sums, then reapplies: `log(Σ(exp(lc) - p) + p)`.

Using the wrong value with a multi-channel MSE model would give `log(n_channels * count_pseudocount + total_counts)` instead of `log(count_pseudocount + total_counts)`.

### Predicted vs observed log-counts

For evaluation (e.g. scatter plots of predicted vs ground-truth log-counts on a held-out fold), `cerberus.predict_misc` exposes mirrored helpers:

```python
from cerberus.predict_misc import (
    create_eval_dataset,
    observed_log_counts,
    predict_log_counts,
)

dataset = create_eval_dataset(ensemble.cerberus_config)

# Predicted total log-counts (one float per interval)
pred = predict_log_counts(ensemble, dataset, intervals)

# Observed total log-counts on the same intervals — extracted from the
# dataset's target signal over the model's output window.  Pseudocount and
# scaling are auto-detected from config.model_config_.loss_cls so the two
# arrays are guaranteed to be in the same log-space.
obs = observed_log_counts(dataset, intervals, ensemble.cerberus_config)
```

Both functions take the same input-length intervals; `observed_log_counts` internally crops to `output_len` before signal extraction, and `predict_log_counts` runs the ensemble's batched fold-routing inference.
