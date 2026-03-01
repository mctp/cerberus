# Core Components

## Data Management

### CerberusDataset
The `CerberusDataset` class (`src/cerberus/dataset.py`) ties together configuration, sampling, and data extraction.

*   **Responsibility**:
    *   **Sampling**: Uses a `Sampler` to select genomic intervals.
    *   **Extraction**: Uses `SequenceExtractor` and `SignalExtractor` to retrieve data for those intervals.
    *   **Transformation**: Applies augmentation and normalization via `Transforms`.
*   **Modes**:
    *   **Disk-Based**: Reads data on-the-fly (low RAM, slower).
    *   **In-Memory**: Pre-loads data into RAM (high RAM, fastest).
*   **Splitting**: Supports `split_folds()` to create Train/Val/Test subsets that share underlying resources but sample from disjoint regions.

### CerberusDataModule
The `CerberusDataModule` class (`src/cerberus/datamodule.py`) wraps the dataset for PyTorch Lightning.

*   **Responsibility**:
    *   **Setup**: Initializes the dataset and performs fold splitting.
    *   **Loaders**: Creates `DataLoader` instances for training, validation, and testing.
    *   **Worker Seeding**: Implements `_worker_init_fn` to ensure correct random seeding across multi-process workers.
    *   **Epoch Management**: Calls `resample()` on the training dataset at the start of each epoch to refresh random samples.

## Samplers

Samplers determine *where* the model looks during training. They yield `Interval` objects (chromosome, start, end, strand).

### IntervalSampler
Used when you have a specific set of regions of interest, such as ChIP-seq peaks or TSS sites. Reads from BED or narrowPeak files.
*   **Use Case**: Training on peaks, fine-tuning on specific loci.
*   **Behavior**: Centers the window on the interval (or summit for narrowPeak) and pads to `padded_size`.

### SlidingWindowSampler
Systematically scans the genome with a fixed stride.
*   **Use Case**: Genome-wide training, baseline creation, or prediction.
*   **Behavior**: Generates intervals `[start, start + padded_size)` across all allowed chromosomes, skipping excluded regions.

### MultiSampler
A meta-sampler that combines multiple samplers.
*   **Use Case**: Balancing positive (peaks) and negative (background) examples.
*   **Behavior**: Mixes samples from sub-samplers. Resizing/balancing is handled by wrapping sub-samplers in `ScaledSampler`.

### ScaledSampler
Wraps a sampler to resize it (subsample or oversample) to a target number of samples.
*   **Use Case**: Controlling the epoch size or balancing datasets.
*   **Behavior**: Resamples the underlying sampler to exactly `num_samples`.
    *   **Subsampling**: Randomly selects intervals without replacement (if possible).
    *   **Oversampling**: Randomly selects intervals with replacement.

### RandomSampler
Samples random intervals from the genome, respecting exclusions.
*   **Use Case**: Generating unbiased background/negative sets.
*   **Behavior**: Generates `num_intervals` random intervals that do not overlap with `exclude_intervals`.
*   **Dynamic**: Supports `resample(seed)` to generate a fresh set of random intervals for each epoch.

### ComplexityMatchedSampler
Selects candidates from a `candidate_sampler` (e.g., RandomSampler) that match the distributional properties of a `target_sampler` (e.g., IntervalSampler of peaks).
*   **Use Case**: Creating balanced datasets where negatives match positives in complexity metrics (GC, DUST, CpG), removing bias.
*   **Behavior**: Pre-computes metrics for all intervals and bins targets. Samples candidates from corresponding bins to match the target distribution.
    *   **`metrics`**: List of metrics to match (e.g., `["gc"]` for GC-matching, `["gc", "dust"]` for multi-metric matching).
    *   **`candidate_ratio`**: Controls the ratio of candidates to targets in each bin (default: 1.0).
        *   `1.0`: Equal number of candidates as targets (1:1 balanced).
        *   `> 1.0`: More candidates (e.g., `2.0` for 2:1 negatives to positives).
        *   `< 1.0`: Fewer candidates.

### PeakSampler
A specialized sampler for training on peaks with a complexity-matched background.
*   **Use Case**: Standard ChIP-seq/ATAC-seq peak training where you want a balanced set of peaks (positives) and complexity-matched non-peak genomic regions (negatives).
*   **Behavior**:
    *   Loads peaks from a file.
    *   Automatically creates a background set that excludes the peaks.
    *   Selects background intervals that match the GC content of the peaks.
    *   Combines them into a single stream.
    *   **Arguments**: `intervals_path`, `background_ratio` (default 1.0).
    *   Set `background_ratio=0.0` to disable background sampling.

### Resampling

The `resample(seed)` method allows dynamic samplers to regenerate their internal list of intervals. This is called at the start of each training epoch.

*   **MultiSampler**: Propagates new seeds to sub-samplers and resamples the indices.
*   **ComplexityMatchedSampler**: Re-selects candidate intervals from the existing candidate pool.
*   **RandomSampler**: Regenerates its list of random intervals using the new seed.
*   **Base Samplers**: Static samplers like `IntervalSampler` and `SlidingWindowSampler` have a no-op `resample` method.

## Extractors

Extractors retrieve the actual data for a given `Interval`.

### SequenceExtractor
*   **Source**: FASTA file.
*   **Output**: One-hot encoded tensor (ACGT -> 4 channels).
*   **Modes**:
    *   `SequenceExtractor`: Reads from disk (using `pyfaidx`). Low memory, higher I/O.
    *   `InMemorySequenceExtractor`: Loads entire genome into RAM. High memory, fastest access.

    ```python
    extractor = SequenceExtractor("hg38.fa")
    interval = Interval("chr1", 0, 100, "+")
    seq_tensor = extractor.extract(interval)  # Returns (4, 100)
    ```

### UniversalExtractor
Intelligently routes input channels to the appropriate extractor based on file extension. This is the default extractor used by `CerberusDataset`.

*   **Source**: BigWig (.bw), BigBed (.bb), BED (.bed) files.
*   **Output**: Signal tensors or binary masks.
*   **Behavior**: Routes files to `SignalExtractor` (BigWig), `BigBedMaskExtractor` (BigBed), or `BedMaskExtractor` (BED).

### SignalExtractor
*   **Source**: BigWig files.
*   **Output**: Continuous values (e.g., read counts).
*   **Modes**:
    *   `SignalExtractor`: Reads from disk (using `pybigtools`).
    *   `InMemorySignalExtractor`: Loads tracks into RAM.

    ```python
    extractor = SignalExtractor({"H3K27ac": "H3K27ac.bw", "CTCF": "CTCF.bw"})
    interval = Interval("chr1", 50000000, 50000200, "+")
    signal = extractor.extract(interval)  # Returns (2, 200)
    ```

### MaskExtractor
*   **Source**: BigBed or BED files.
*   **Output**: Binary mask tensors indicating regions of interest (e.g., mappability, blacklists).
*   **Modes**:
    *   `BigBedMaskExtractor`: Reads BigBed from disk.
    *   `InMemoryBigBedMaskExtractor`: Loads BigBed into RAM.
    *   `BedMaskExtractor`: Reads BED files (always in-memory using InterLap).

    ```python
    extractor = BigBedMaskExtractor({"mappability": "mappability.bb"})
    interval = Interval("chr1", 150, 250, "+")
    mask = extractor.extract(interval)  # Returns tensor of shape (1, 100)
    ```

## Sequence Analysis

The `cerberus.complexity` module provides tools for analyzing DNA sequence properties. See [complexity.md](complexity.md) for the full API reference.

### GC Content
Calculates the fraction of G and C nucleotides.
*   **Function**: `calculate_gc_content(sequence)`
*   **Output**: Float (0.0 to 1.0).

### DUST Score
Calculates sequence complexity based on k-mer repetition. Higher scores indicate lower complexity (more repetitive).
*   **Function**: `calculate_dust_score(sequence, k=3)`
*   **Output**: Normalized float (0.0 to ~1.0).

### Normalized CpG Score
Calculates the normalized observed/expected CpG ratio.
*   **Function**: `calculate_log_cpg_ratio(sequence)`
*   **Output**: Normalized float (0.0 to 1.0).

## Transforms

Cerberus provides a composable transformation pipeline via `cerberus.transform.Compose`.

*   **Jitter**: Randomly shifts the interval start position to prevent overfitting to exact positioning. Updates interval coordinates to match the cropped region.
*   **ReverseComplement**: Randomly flips the sequence and signals (50% probability by default) to enforce strand invariance. Updates interval strand.
*   **TargetCrop**: Centers and crops the target signal to `output_len` (often smaller than `input_len` to avoid valid-padding edge effects).
*   **Scale**: Multiplies targets (or inputs) by a constant factor. Useful for rescaling normalized BigWig values to integer-like counts before log transform.
*   **Log1p**: Applies `log(x + 1)` scaling to targets.
*   **Sqrt**: Applies `sqrt(x)` scaling to targets.
*   **Arcsinh**: Applies `arcsinh(x)` scaling to targets.
*   **Bin**: Downsamples resolution via pooling (`max`, `avg`, or `sum`) for lower-resolution predictions.

    ```python
    transforms = [
        Jitter(input_len=50),
        TargetCrop(output_len=20)
    ]
    compose = Compose(transforms)
    inputs, targets, interval = compose(inputs, targets, interval)
    ```

### Default Pipeline Order
When `transforms` are not explicitly provided to `CerberusDataset`, they are automatically constructed from `DataConfig` by `create_default_transforms` in the following order:

1.  **Jitter**: Crops to `input_len`, with random offset up to `max_jitter` (or centered for inference).
2.  **ReverseComplement**: (If `reverse_complement=True` and not in deterministic/inference mode).
3.  **TargetCrop**: Crops targets to `output_len` (if `output_len < input_len`).
4.  **Scale**: Multiplies targets by `target_scale` (if `target_scale != 1.0`).
5.  **Bin**: Downsamples targets by `output_bin_size` (if `output_bin_size > 1`).
6.  **Log1p**: Applies `log(1+x)` to targets (if `log_transform=True`).

## Losses

Cerberus provides specialized loss functions for genomic signal prediction.

### ProfilePoissonNLLLoss
Standard Poisson NLL loss for models predicting log-rates (log-intensities).

*   **Behavior**: Computes `PoissonNLL(exp(log_rates), targets)`.
*   **Use Case**: Models predicting unnormalized log-counts directly (e.g., `ConvNeXtDCNN`).
*   **Inputs**: `ProfileLogRates`.

### MSEMultinomialLoss (BPNetLoss)
Standard BPNet loss combining Multinomial NLL (profile) and MSE (counts).

*   **Profile Loss**: Penalizes the shape of the predicted signal distribution (exact Multinomial NLL).
*   **Count Loss**: Penalizes the total count prediction (MSE on `log(1+x)`).
*   **Args**:
    *   `count_weight`: Weight for the count loss component (default: 1.0).
    *   `flatten_channels` (default: `False`): If `False`, computes Multinomial NLL independently per channel. If `True`, flattens all channels and length into a single dimension.
    *   `log1p_targets`: If targets are already log-transformed, reverses them before loss calculation.
*   **Inputs**: `ProfileCountOutput`.

### CoupledMSEMultinomialLoss
Coupled version of `MSEMultinomialLoss` for models predicting log-rates directly.

*   **Behavior**: Derives predicted counts by summing log-rates (LogSumExp) before computing MSE count loss.
*   **Inputs**: `ProfileLogRates`.

### PoissonMultinomialLoss
Combines Poisson NLL for total counts and Multinomial (Cross-Entropy) for profile shape.

*   **Count Loss**: Poisson Negative Log-Likelihood.
*   **Profile Loss**: Multinomial Negative Log-Likelihood (Cross-Entropy form).
*   **Inputs**: `ProfileCountOutput`.
*   **Args**: `count_weight` (default: 0.2), `flatten_channels` (default: `False`), `average_channels` (default: `True`), `log1p_targets`.

### CoupledPoissonMultinomialLoss
Coupled version of `PoissonMultinomialLoss` for models predicting log-rates.

*   **Behavior**: Derives predicted counts via LogSumExp before computing Poisson count loss.
*   **Inputs**: `ProfileLogRates`.

### NegativeBinomialMultinomialLoss
Combines Negative Binomial NLL for total counts and Multinomial (Cross-Entropy) for profile shape.

*   **Count Loss**: Negative Binomial NLL with fixed `total_count` dispersion parameter.
*   **Profile Loss**: Multinomial Negative Log-Likelihood.
*   **Inputs**: `ProfileCountOutput`.
*   **Args**: `total_count` (fixed dispersion parameter `r`).

### CoupledNegativeBinomialMultinomialLoss
Coupled version of `NegativeBinomialMultinomialLoss` for models predicting log-rates.

*   **Behavior**: Derives predicted counts via LogSumExp.
*   **Inputs**: `ProfileLogRates`.

## Model Outputs

Models return standardized dataclasses (defined in `cerberus.output`) to ensure compatibility with losses and metrics.

### ProfileLogits
*   **Content**: `logits` tensor (Batch, Channels, Length).
*   **Use Case**: Models predicting profile probability distributions (e.g., via Softmax).

### ProfileLogRates
*   **Content**: `log_rates` tensor (Batch, Channels, Length).
*   **Use Case**: Models predicting log-intensities (e.g., Poisson log-rates).

### ProfileCountOutput
*   **Content**: `logits` (Batch, Channels, Length) and `log_counts` (Batch, Channels).
*   **Use Case**: BPNet-style models with separate heads for profile shape and total count.

## Metrics

### ProfilePearsonCorrCoef
Pearson Correlation Coefficient for profile probabilities.

*   **Behavior**: Computes Pearson correlation per (example, channel) pair along the sequence length dimension, then averages across channels and examples. Numerically stable in float32. Operates on probabilities (Softmax of logits or log_rates).
*   **Inputs**: `ProfileLogRates` or `ProfileLogits`.

### CountProfilePearsonCorrCoef
Pearson Correlation for models with separate profile logits and counts (BPNet-style).

*   **Behavior**: Reconstructs predicted counts (`Softmax(logits) * (Exp(log_counts) - count_pseudocount)`) before computing per-example Pearson correlation along the sequence length. Numerically stable in float32.
*   **Inputs**: `ProfileCountOutput`.

### ProfileMeanSquaredError
MSE on profile probabilities.

*   **Behavior**: Computes MSE between predicted probabilities (Softmax of logits) and normalized target probabilities.
*   **Inputs**: `ProfileLogRates` or `ProfileLogits`.

### CountProfileMeanSquaredError
MSE on reconstructed profile counts (BPNet-style).

*   **Behavior**: Reconstructs predicted counts (`Softmax(logits) * Exp(log_counts)`) before computing MSE against targets.
*   **Inputs**: `ProfileCountOutput`.

### LogCountsMeanSquaredError
MSE on log counts.

*   **Behavior**: Computes MSE between predicted log counts and `log1p(target_counts)`.
*   **Inputs**: `ProfileCountOutput` (uses `log_counts`) or `ProfileLogRates` (uses `logsumexp(log_rates)`).

### LogCountsPearsonCorrCoef
Pearson Correlation on log counts.

*   **Behavior**: Collects per-example (pred, target) log-count scalar pairs and computes a single Pearson correlation at epoch end. Numerically stable.
*   **Inputs**: `ProfileCountOutput` or `ProfileLogRates`.

### DefaultMetricCollection
A pre-configured `MetricCollection` used when no custom metrics are specified.

*   **Includes**: `ProfilePearsonCorrCoef`, `ProfileMeanSquaredError`, `LogCountsMeanSquaredError`, `LogCountsPearsonCorrCoef`.

## Cross-Validation (Folds)

Cerberus has built-in support for K-fold cross-validation to ensure robust evaluation.
*   **Chromosomal Partitioning**: Splits data by chromosomes to avoid data leakage (e.g., neighboring correlated regions).
*   **Config**: Defined in `GenomeConfig` via `fold_type` and `fold_args`.
*   **Samplers**: The `split_folds` method ensures that training, validation, and test sets are disjoint based on these fold definitions.
