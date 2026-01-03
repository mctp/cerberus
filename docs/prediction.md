# Prediction (Inference)

Prediction in Cerberus involves applying a trained model (or ensemble of models) to genomic intervals. This phase uses the `ModelEnsemble` class to manage model loading, fold selection, and output aggregation.

## Core Concepts

### ModelEnsemble
The `ModelEnsemble` class is the central entry point for prediction. It:
1.  **Loads Models**: Automatically detects single-fold or multi-fold checkpoints and loads the appropriate model weights.
2.  **Manages Folds**: Identifies which model to run for a given genomic interval based on cross-validation fold definitions (e.g., using the "test" model for intervals in the test fold).
3.  **Aggregates**: Combines outputs from multiple models or overlapping intervals.

### Aggregation Strategies
When multiple models predict on the same interval, or when tiling produces overlapping predictions, Cerberus needs to merge them.
*   **"model"**: Aggregates outputs across models (e.g., averaging predictions from all folds). Returns batched `ModelOutput` objects.
*   **"interval+model"**: Aggregates across models AND merges overlapping interval predictions into a single contiguous track. Returns unbatched `ModelOutput` objects.

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

## Configuration: PredictConfig

The prediction behavior is controlled by a `PredictConfig` dictionary.

```python
class PredictConfig(TypedDict):
    # List of folds to use for prediction (e.g., ["test"], ["train", "val", "test"])
    use_folds: list[str]
    
    # Stride for tiling input windows (used in predict_output_intervals and predict_to_bigwig)
    stride: int
    
    # Aggregation mode (optional, defaults to "model")
    aggregation: str
```

## Prediction Methods

### 1. Predict on Specific Intervals
Use `predict_intervals` when you have a list of fixed-length intervals (matching `input_len`) and want predictions for each.

```python
output = model_ensemble.predict_intervals(
    intervals=my_intervals,
    dataset=dataset,
    predict_config={"use_folds": ["test"]},
    batch_size=64
)
```

### 2. Predict on Arbitrary Regions (Tiling)
Use `predict_output_intervals` when you have target regions of arbitrary length (e.g., entire peaks or genes). The system tiles the region with overlapping input windows and merges the results.

```python
outputs = model_ensemble.predict_output_intervals(
    intervals=gene_regions,
    dataset=dataset,
    predict_config={"stride": 50, "use_folds": ["test"]},
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
    predict_config={"stride": 50, "use_folds": ["test"]},
    batch_size=256
)
```

## Model Outputs

Predictions return `ModelOutput` objects (defined in `cerberus.output`), which contain the raw tensors.

*   **ProfileLogits**: Unnormalized log-probabilities (shape `Batch, Channels, Length`).
*   **ProfileLogRates**: Log-scale counts/rates.
*   **ProfileCountOutput**: Tuple of profile logits and total log counts (BPNet style).

You typically need to detach and post-process these (e.g., applying Softmax or Exponentials) for analysis.
