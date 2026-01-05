# Prediction (Inference)

Prediction in Cerberus involves applying a trained model (or ensemble of models) to genomic intervals. This phase uses the `ModelEnsemble` class to manage model loading, fold selection, and output aggregation.

## Core Concepts

### ModelEnsemble
The `ModelEnsemble` class is the central entry point for prediction. It:
1.  **Loads Models**: Automatically detects single-fold or multi-fold checkpoints and loads the appropriate model weights.
2.  **Manages Folds**: Identifies which model to run for a given genomic interval based on cross-validation fold definitions.
    *   **Rotation Logic**: Cerberus assumes a standard rotation where Model `i` uses Partition `i` as Test and Partition `(i+1)%k` as Validation.
    *   If you request `use_folds=["test"]`, the ensemble selects Model `i` for intervals falling in Partition `i`.
3.  **Aggregates**: Combines outputs from multiple models or overlapping intervals.

### Aggregation Strategies
When multiple models predict on the same interval, or when tiling produces overlapping predictions, Cerberus needs to merge them.
*   **"model"**: Aggregates outputs across models (e.g., averaging predictions from all folds). Returns batched `ModelOutput` objects.
*   **"interval+model"**: Aggregates across models AND merges overlapping interval predictions into a single contiguous track. Returns unbatched `ModelOutput` objects.

    **Note on Alignment**: When merging overlapping predictions (e.g. from tiling), if `output_bin_size > 1`, Cerberus performs a **"snap-to-grid"** alignment. 
    Prediction starts are floored to the nearest bin relative to the merged interval start. This ensures consistent binning but means sub-bin shifts (e.g. from jitter) are quantized.

## Setup for Prediction

To perform prediction, you need to instantiate a `ModelEnsemble` and a `CerberusDataset`.
For inference-only tasks, you can omit `train_config` and `sampler_config` (if providing a sampler directly).

```python
# Instantiate ModelEnsemble (train_config is not required)
ensemble = ModelEnsemble(
    checkpoint_path="path/to/checkpoints",
    model_config=model_config,
    data_config=data_config,
    genome_config=genome_config,
    device=device
)

# Instantiate Dataset (sampler_config is not required if providing a custom sampler)
dataset = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config,
    sampler=my_custom_sampler, # Provide sampler directly
    is_train=False
)
```

## Prediction Methods

### 1. Predict on Specific Intervals
Use `predict_intervals` when you have a list of fixed-length intervals (matching `input_len`) and want predictions for each.

```python
output = model_ensemble.predict_intervals(
    intervals=my_intervals,
    dataset=dataset,
    use_folds=["test"],
    batch_size=64
)
```

### 2. Predict on Arbitrary Regions (Tiling)
Use `predict_output_intervals` when you have target regions of arbitrary length (e.g., entire peaks or genes). The system tiles the region with overlapping input windows and merges the results.

```python
outputs = model_ensemble.predict_output_intervals(
    intervals=gene_regions,
    dataset=dataset,
    stride=50,
    use_folds=["test"],
    batch_size=64
)
```

### 3. Genome-Wide Prediction (BigWig)
Use `predict_to_bigwig` to generate a genome-wide signal track. This is useful for visualization in genome browsers.

```python
from cerberus.predict_bigwig import predict_to_bigwig

predict_to_bigwig(
    output_path="predictions.bw",
    dataset=dataset,
    model_ensemble=model_ensemble,
    stride=50,
    use_folds=["test"],
    batch_size=256
)
```

## Model Outputs

Predictions return `ModelOutput` objects (defined in `cerberus.output`), which contain the raw tensors.

*   **ProfileLogits**: Unnormalized log-probabilities (shape `Batch, Channels, Length`).
*   **ProfileLogRates**: Log-scale counts/rates.
*   **ProfileCountOutput**: Tuple of profile logits and total log counts (BPNet style).

You typically need to detach and post-process these (e.g., applying Softmax or Exponentials) for analysis.
