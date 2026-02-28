# Configuration

Cerberus relies on five configuration dictionaries that together define the full training pipeline: `GenomeConfig`, `DataConfig`, `SamplerConfig`, `TrainConfig`, and `ModelConfig`.

## GenomeConfig

Defines the reference genome and cross-validation strategy.

```python
class GenomeConfig(TypedDict):
    # Name of the genome assembly (e.g., "hg38")
    name: str

    # Path to the FASTA file
    fasta_path: Path

    # Dictionary mapping chromosome names to their lengths
    chrom_sizes: dict[str, int]

    # Dictionary mapping names to BED file paths of regions to exclude
    # (e.g., blacklists, gaps). Paths are resolved at validation time.
    exclude_intervals: dict[str, Path]

    # Cross-validation strategy. Currently only "chrom_partition" is supported.
    fold_type: str  # e.g., "chrom_partition"

    # Arguments for the fold strategy.
    # For 'chrom_partition', required key: 'k' (int, number of folds).
    # Optional keys: 'test_fold' (int), 'val_fold' (int).
    fold_args: dict[str, Any]

    # List of chromosome names to include (e.g., ["chr1", ..., "chrX"])
    allowed_chroms: list[str]
```

## DataConfig

Defines the input and output data characteristics.

```python
class DataConfig(TypedDict):
    # Paths to input track files (BigWig, BigBed, BED).
    # Supported formats:
    # - BigWig (.bw, .bigwig): Continuous signal (coverage, fold-change).
    # - BigBed (.bb, .bigbed): Binary intervals (peaks, regions).
    # - BED (.bed, .bed.gz): Binary intervals (peaks, regions).
    inputs: dict[str, Path]

    # Paths to target track files (e.g., CAGE, ChIP-seq).
    # Supports same formats as inputs.
    targets: dict[str, Path]

    # Length of the input sequence window fed to the model
    input_len: int

    # Length of the output target signal
    output_len: int

    # Binning resolution for targets (1 = raw signal, >1 = pooled)
    output_bin_size: int

    # DNA one-hot encoding order (e.g., "ACGT")
    encoding: str

    # Maximum jitter (random shift) applied to the interval center during training
    max_jitter: int

    # Whether to apply log(1+x) transform to targets
    log_transform: bool

    # Whether to apply reverse complement augmentation during training
    reverse_complement: bool

    # Multiplicative scaling factor applied to targets before log transform.
    # Useful for rescaling normalized BigWig values to integer-like counts.
    # Set to 1.0 to disable.
    target_scale: float

    # Whether to include DNA sequence as an input channel (default: True)
    use_sequence: bool
```

### Performance: In-Memory vs Disk-Based

The `in_memory` flag (passed at runtime to `CerberusDataModule.setup()`) significantly impacts performance.

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
    # Type of sampler: "interval", "sliding_window", "random", "complexity_matched", "peak"
    sampler_type: str

    # Length of the window yielded by the sampler (after centering/padding).
    # Must be >= input_len + 2 * max_jitter to allow jitter headroom.
    padded_size: int

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

    # Scheduler type (e.g. "default", "cosine"). Optional, defaults to "default".
    scheduler_type: str

    # Arguments for scheduler (passed to timm.scheduler.create_scheduler_v2).
    # Optional, defaults to {}.
    scheduler_args: dict[str, Any]

    # Whether to exclude bias and batch norm from weight decay
    filter_bias_and_bn: bool

    # Patience for EarlyStopping
    patience: int

    # Reload dataloaders every n epochs (0 to disable). Optional, defaults to 0.
    reload_dataloaders_every_n_epochs: int
```

*Note: Runtime parameters like `num_workers`, `in_memory`, `precision`, `matmul_precision`, and `compile` are passed directly to the `train_single`/`train_multi` functions, not included in `TrainConfig`.*

## PredictConfig (Runtime)

Defines configuration arguments for inference/prediction functions (e.g., `predict_to_bigwig`). These are passed as runtime arguments rather than a static configuration object.

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
    # Name of the model (used for logging)
    name: str

    # Model class (fully qualified class name, e.g. "cerberus.models.BPNet")
    model_cls: str

    # Loss class (fully qualified class name, e.g. "cerberus.loss.MSEMultinomialLoss")
    loss_cls: str

    # Arguments for loss instantiation.
    # Any value may be set to the string "adaptive" to have cerberus compute it
    # automatically from the training data before the module is instantiated.
    # The adaptive value is: median_total_counts / 10, where median_total_counts
    # is the median summed signal across a sample of training intervals (after
    # applying target_scale). This is the recommended setting for BPNetLoss.alpha
    # and similar count-weight parameters in BPNet-style models.
    # Example: {"alpha": "adaptive"} → alpha is resolved at fit time.
    loss_args: dict[str, Any]

    # Metrics class (fully qualified class name, e.g. "torchmetrics.MetricCollection")
    metrics_cls: str

    # Arguments for metrics instantiation
    metrics_args: dict[str, Any]

    # Arguments for model instantiation.
    # Note: "input_len", "output_len", and "output_bin_size" are automatically passed
    # from DataConfig and do not need to be specified here.
    model_args: dict[str, Any]
```

## DataModule Runtime Arguments

Arguments passed to `CerberusDataModule.__init__` for hardware optimization:

- **pin_memory** (`bool`, default: `True`):
  - Whether to pin memory in DataLoaders (recommended for GPU training).
  - *Note: Automatically disabled on MPS devices.*

- **persistent_workers** (`bool`, default: `True`):
  - Whether to keep workers alive between epochs.
  - Improves performance when `num_workers > 0`.

- **multiprocessing_context** (`str | None`, default: `None`):
  - Explicitly set the multiprocessing start method (e.g., `'spawn'`, `'fork'`, `'forkserver'`).
  - Useful for MPS/macOS stability if default behaviors cause crashes.

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

*   **Random Sampler**:
    ```python
    {
        "num_intervals": 10000 # Number of random intervals to generate
    }
    ```
    *Note: Intervals are regenerated with a new random seed at the beginning of each epoch.*

*   **Complexity Matched Sampler**:
    ```python
    {
        "target_sampler": {
            "type": "interval",
            "args": {"intervals_path": "peaks.bed"}
        },
        "candidate_sampler": {
            "type": "random",
            "args": {"num_intervals": 100000}
        },
        "bins": 100,            # Number of complexity bins (default: 100)
        "candidate_ratio": 1.0, # Ratio of candidates to targets (default: 1.0)
        "metrics": ["gc", "dust", "cpg"] # Metrics to match (default: ["gc", "dust", "cpg"])
    }
    ```

*   **Peak Sampler**:
    ```python
    {
        "intervals_path": "peaks.bed",
        "background_ratio": 1.0,  # Ratio of complexity-matched negatives to peaks (default: 1.0)
        "min_candidates": 10000,  # Minimum number of candidates for background pool (default: 10000)
        "candidate_oversample_factor": 20.0 # Factor to oversample candidates relative to peaks (default: 20.0)
    }
    ```
    *Note: `PeakSampler` automatically creates a complexity-matched background set using a `RandomSampler` as the candidate pool, excluding the peaks themselves. Set `background_ratio=0.0` to disable background sampling.*
