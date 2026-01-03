# Configuration

Cerberus relies on three main configuration dictionaries to set up the data pipeline: `GenomeConfig`, `DataConfig`, and `SamplerConfig`.

## GenomeConfig

Defines the reference genome properties and validation strategy.

```python
class GenomeConfig(TypedDict):
    # Name of the genome assembly (e.g., "hg38")
    name: str
    
    # Path to the FASTA file
    fasta_path: str | Path
    
    # Dictionary mapping chromosome names to their lengths
    chrom_sizes: dict[str, int]
    
    # Dictionary mapping chromosome names to exclusion intervals (InterLap objects or paths)
    exclude_intervals: dict[str, InterLap] | dict[str, Path]
    
    # Cross-validation strategy
    fold_type: str  # e.g., "chrom_partition"
    
    # Arguments for the fold strategy
    fold_args: dict[str, Any]
    
    # Optional list of allowed chromosomes
    allowed_chroms: list[str] | None
```

## DataConfig

Defines the input and output data characteristics.

```python
class DataConfig(TypedDict):
    # Paths to input BigWig files (e.g., conservation scores)
    inputs: dict[str, Path]
    
    # Paths to target BigWig files (e.g., CAGE, ChIP-seq)
    targets: dict[str, Path]
    
    # Length of the input sequence
    input_len: int
    
    # Length of the output target signal
    output_len: int
    
    # Binning resolution for targets (default: 1)
    output_bin_size: int
    
    # DNA encoding ("ACGT" usually)
    encoding: str
    
    # Maximum jitter (shift) to apply during training
    max_jitter: int
    
    # Whether to apply log(1+x) transform to targets
    log_transform: bool
    
    # Whether to apply reverse complement augmentation
    reverse_complement: bool
    
    # Whether to use sequence input (default: True)
    use_sequence: bool
```

### Performance: In-Memory vs Disk-Based

The `in_memory` flag significantly impacts performance.

- **Disk-Based (`in_memory: False`)**:
  - **Pros**: Instant startup (< 1s). Low RAM usage.
  - **Cons**: Slower batch generation due to I/O. Throughput ~2,000 examples/sec.
  - **Use Case**: Huge datasets, quick debugging.

- **In-Memory (`in_memory: True`)**:
  - **Pros**: Extremely fast batch generation. Throughput ~16,000+ examples/sec.
  - **Cons**: Slow startup (reading all data). High RAM usage.
  - **Use Case**: Datasets that fit in RAM. Long training runs.

**Benchmark Results:**

| Mode       | Setup Time | Throughput (ex/s) |
|------------|------------|-------------------|
| Disk-Based | ~0.2s      | ~2,000            |
| In-Memory  | ~100s      | ~16,000           |

*Note: In-Memory loading pays a large upfront cost. For short training runs, Disk-Based may be faster overall.*

## SamplerConfig

Defines how data samples are selected from the genome.

```python
class SamplerConfig(TypedDict):
    # Type of sampler: "interval", "sliding_window", or "multi"
    sampler_type: str
    
    # Length of the window to sample (usually input_len)
    padded_size: int
    
    # Paths to exclusion regions (optional override)
    exclude_intervals: dict[str, Path] | None
    
    # Arguments specific to the chosen sampler type
    sampler_args: dict[str, Any]
```

## TrainConfig

Defines hyperparameters for training.

```python
class TrainConfig(TypedDict):
    # Batch size
    batch_size: int
    
    # Maximum number of epochs
    max_epochs: int
    
    # Learning rate
    learning_rate: float
    
    # Weight decay
    weight_decay: float
    
    # Optimizer name (e.g. "adamw")
    optimizer: str
    
    # Scheduler type (e.g. "default", "cosine")
    scheduler_type: str
    
    # Arguments for scheduler (passed to timm.scheduler.create_scheduler_v2)
    scheduler_args: dict[str, Any]
    
    # Whether to exclude bias and batch norm from weight decay
    filter_bias_and_bn: bool
    
    # Patience for EarlyStopping (used by entrypoints)
    patience: int
```

*Note: Runtime parameters like `num_workers`, `in_memory`, `precision`, `matmul_precision`, and `compile` are passed directly to the `train` or `instantiate` functions, not included in `TrainConfig`.*

## PredictConfig

Defines configuration for inference/prediction.

```python
class PredictConfig(TypedDict):
    # List of fold roles to use for prediction (e.g. ["test"], ["train", "val"])
    use_folds: list[str]
    
    # Stride for tiling input intervals (used in tiling predictions and bigwig generation)
    stride: int
    
    # Aggregation strategy: "model" (default) or "interval+model"
    aggregation: str
```

## ModelConfig

Defines the model architecture and its training components.

```python
class ModelConfig(TypedDict):
    # Name of the model
    name: str
    
    # Model class (nn.Module subclass)
    model_cls: type[nn.Module]
    
    # Loss class (nn.Module subclass)
    loss_cls: type[nn.Module]
    
    # Arguments for loss instantiation
    loss_args: dict[str, Any]
    
    # Metrics collection class
    metrics_cls: type[MetricCollection]
    
    # Arguments for metrics instantiation
    metrics_args: dict[str, Any]
    
    # Arguments for model instantiation.
    # Should include "input_channels", "output_channels" if required by the model.
    # Note: "input_len", "output_len", and "output_bin_size" are automatically passed
    # from DataConfig and do not need to be specified here.
    model_args: dict[str, Any]
```

### Sampler Arguments

*   **Interval Sampler**:
    ```python
    {
        "intervals_path": "/path/to/peaks.bed"
    }
    ```

*   **Sliding Window Sampler**:
    ```python
    {
        "stride": 50  # Step size for the sliding window
    }
    ```

*   **Multi Sampler**:
    ```python
    {
        "samplers": [
            {
                "type": "interval",
                "args": {"intervals_path": "peaks.bed"},
                "scaling": 0.5  # Subsample to 50%
            },
            {
                "type": "sliding_window",
                "args": {"stride": 1000},
                "scaling": 1.0
            }
        ]
    }
    ```

    *Note: Samplers can be nested recursively! A `MultiSampler` can contain other `MultiSampler` definitions.*
