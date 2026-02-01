# Prediction (Inference)

Prediction in Cerberus involves applying a trained model (or ensemble of models) to genomic intervals. This phase uses the `ModelEnsemble` class to manage model loading, fold selection, and output aggregation.

## Core Concepts

### ModelEnsemble
The `ModelEnsemble` class is the central entry point for prediction. It:
1.  **Loads Models**: Automatically detects single-fold or multi-fold checkpoints and loads the appropriate model weights from a directory.
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
genome_config = ensemble.cerberus_config["genome_config"]
data_config = ensemble.cerberus_config["data_config"]
```

### Path Resolution
If you move a trained model to a new environment (e.g., from a training cluster to a local machine), the absolute paths in `hparams.yaml` (pointing to FASTA or BigWig files) might be invalid.
`ModelEnsemble` supports `search_paths` to resolve these files relative to new locations.

```python
from pathlib import Path

ensemble = ModelEnsemble(
    checkpoint_path="logs/my_experiment",
    # Look for files in current dir and tests/data if original paths fail
    search_paths=[Path.cwd(), Path("tests/data")] 
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
4. Saves a compressed TSV with columns: `chrom`, `start`, `end`, `strand`, `predicted_log_count`, `observed_log_count`.

## Model Outputs

Predictions return `ModelOutput` objects (defined in `cerberus.output`), which contain the raw tensors.

*   **ProfileLogits**: Unnormalized log-probabilities (shape `Batch, Channels, Length`).
*   **ProfileLogRates**: Log-scale counts/rates.
*   **ProfileCountOutput**: Tuple of profile logits and total log counts (BPNet style).

You typically need to detach and post-process these (e.g., applying Softmax or Exponentials) for analysis.
