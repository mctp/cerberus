# Core Components

## Samplers

Samplers determine *where* the model looks during training. They yield `Interval` objects (Chromosome, Start, End, Strand).

### IntervalSampler
Used when you have a specific set of regions of interest, such as ChIP-seq peaks or TSS sites. It reads from BED or narrowPeak files.
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
    *   **Note**: Configuration via `create_sampler` still supports the `scaling` parameter for backward compatibility (including "min", "max", "count:N"), which automatically wraps the sampler in `ScaledSampler`. "min" scaling ignores empty samplers.

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

### GCMatchedSampler
Selects candidates from a `candidate_sampler` (e.g., RandomSampler) that match the GC content distribution of a `target_sampler` (e.g., IntervalSampler of peaks).
*   **Use Case**: Creating balanced datasets where negatives match positives in GC content, removing GC bias.
*   **Behavior**: Pre-computes GC content for all intervals and bins targets. Samples candidates from corresponding bins to match the target distribution.
    *   **`match_ratio`**: Controls the ratio of candidates to targets in each GC bin (default: 1.0).
        *   `1.0`: Selects an equal number of candidates as targets (1:1 balanced).
        *   `> 1.0`: Selects more candidates (e.g., `2.0` for 2:1 negatives to positives).
        *   `< 1.0`: Selects fewer candidates.

### PeakSampler
A specialized sampler for training on peaks with a GC-matched background.
*   **Use Case**: Standard ChIP-seq/ATAC-seq peak training where you want a balanced set of peaks (positives) and GC-matched non-peak genomic regions (negatives).
*   **Behavior**:
    *   Loads peaks from a file.
    *   Automatically creates a background set that excludes the peaks.
    *   Selects background intervals that match the GC content of the peaks.
    *   Combines them into a single stream.
    *   **Arguments**: `intervals_path`, `background_ratio` (default 1.0).
    *   **Defaults**: `background_ratio=1.0` ensures a 1:1 ratio between positives (peaks) and negatives (GC-matched background). `background_ratio=0.0` disables background sampling.

### Resampling

The `resample(seed)` method allows dynamic samplers to regenerate their internal list of intervals. This is typically called at the start of each training epoch.

*   **MultiSampler**: Resamples the indices from its sub-samplers based on scaling factors. This allows for:
    *   **Dynamic Subsampling**: Randomly selecting a new subset of negatives each epoch.
    *   **Oversampling**: Re-drawing samples with replacement.
    *   **Shuffling**: Ensuring a new random order of mixed samples.
*   **GCMatchedSampler**: Re-selects candidate intervals from the `candidate_sampler` to maintain the GC match with the target. This ensures that the model sees different negative examples that still match the GC profile of the positives.
*   **Base Samplers**: Static samplers like `IntervalSampler` and `SlidingWindowSampler` generally have a no-op `resample` method, as their interval set is fixed.

## Extractors

Extractors retrieve the actual data for a given `Interval`.

### SequenceExtractor
*   **Source**: FASTA file.
*   **Output**: One-hot encoded tensor (ACGT -> 4 channels).
*   **Modes**:
    *   `SequenceExtractor`: Reads from disk (using `pyfaidx`). Low memory, higher I/O.
    *   `InMemorySequenceExtractor`: Loads entire genome into RAM. High memory, fastest access.

    ```python
    # Example Usage (from tests/test_sequence_retrieval.py)
    extractor = SequenceExtractor("hg38.fa")
    interval = Interval("chr1", 0, 100, "+")
    seq_tensor = extractor.extract(interval) # Returns (4, 100)
    ```

### SignalExtractor
*   **Source**: BigWig files.
*   **Output**: Continuous values (e.g., read counts).
*   **Modes**:
    *   `SignalExtractor`: Reads from disk (using `pybigtools`).
    *   `InMemorySignalExtractor`: Loads tracks into RAM.

    ```python
    # Example Usage (from tests/test_signal.py)
    extractor = SignalExtractor({"H3K27ac": "H3K27ac.bw", "CTCF": "CTCF.bw"})
    interval = Interval("chr1", 50000000, 50000200, "+")
    signal = extractor.extract(interval) # Returns (2, 200)
    ```

### MaskExtractor
*   **Source**: BigBed files.
*   **Output**: Binary or categorical mask tensors indicating regions of interest (e.g., mappability, blacklists).
*   **Modes**:
    *   `MaskExtractor`: Reads from disk (using `pybigtools`).
    *   `InMemoryMaskExtractor`: Loads masks into RAM.

    ```python
    # Example Usage (from tests/test_mask.py)
    extractor = MaskExtractor({"mappability": "mappability.bb"})
    interval = Interval("chr1", 150, 250, "+")
    mask = extractor.extract(interval) # Returns tensor of shape (1, 100)
    ```

## Transforms

Cerberus provides a composable transformation pipeline via `cerberus.transform.Compose`.

*   **Jitter**: Randomly shifts the interval start position to prevent overfitting to exact positioning.
*   **ReverseComplement**: Randomly flips the sequence and signals (50% probability) to enforce strand invariance.
*   **TargetCrop**: Centers and crops the target signal to `output_len` (often smaller than `input_len` to avoid edge effects).
*   **Log1p**: Applies `log(x + 1)` scaling to targets.
*   **Sqrt**: Applies `sqrt(x)` scaling to targets.
*   **Arcsinh**: Applies `arcsinh(x)` scaling to targets.
*   **Bin**: Downsamples resolution (e.g., max pooling) for lower-resolution predictions.

    ```python
    # Example Usage (from tests/test_transforms.py)
    transforms = [
        Jitter(input_len=50),
        TargetCrop(output_len=20)
    ]
    compose = Compose(transforms)
    
    # Apply to tensors
    inputs, targets, interval = compose(inputs, targets, interval)
    ```

## Losses

Cerberus provides specialized loss functions for genomic signal prediction.

### BPNetLoss
Standard BPNet loss combining Multinomial NLL (profile) and MSE (counts).

*   **Profile Loss**: Penalizes the shape of the predicted signal distribution (Exact Multinomial NLL).
*   **Count Loss**: Penalizes the total count prediction (MSE on `log(1+x)`).
*   **Args**:
    *   `alpha`: Weight for the count loss component.
    *   `flatten_channels`: Whether to compute multinomial over all channels flattened (default: True, typical for BPNet).
    *   `implicit_log_targets`: If targets are already log-transformed, reverses them for profile loss calculation.

### PoissonMultinomialLoss (Unified)
A flexible loss function combining Poisson NLL for total counts and Multinomial (Cross-Entropy) for profile shape. It unifies the functionality of the deprecated `BPNetPoissonLoss` and supports both standard BPNet (Tuple) and generic (Tensor) model outputs.

*   **Ref**: Boshar et al. 2025 (Concept).
*   **Count Loss**: Poisson Negative Log-Likelihood.
*   **Profile Loss**: Multinomial Negative Log-Likelihood (Cross-Entropy form, ignoring constants).
*   **Supported Inputs**:
    *   **Tuple (BPNet)**: `(profile_logits, log_counts)`.
    *   **Tensor (Generic)**: `predicted_counts`.
*   **Args**:
    *   `count_weight`: Weight for the count loss component (aliases `alpha`).
    *   `flatten_channels`: Whether to flatten channels/length for profile loss (default: False).
    *   `implicit_log_targets`: Handling of log-transformed targets.

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
Computes Pearson Correlation Coefficient per channel on profile probabilities.

*   **Behavior**: Flattens batch and length dimensions to compute correlation between the predicted and target signal vectors for each channel independently, then averages across channels.
*   **Why**: Standard global Pearson correlation can be misleading if channels have vastly different dynamic ranges.

### CountProfilePearsonCorrCoef
Computes Pearson Correlation for models that output separate profile logits and counts (like BPNet).
*   **Behavior**: Reconstructs predicted counts (`Softmax(logits) * Exp(log_counts)`) before computing correlation.

### ProfileMeanSquaredError
Computes MSE on probability profiles (Softmax of logits) vs normalized target probabilities.

### CountProfileMeanSquaredError
Computes MSE on reconstructed counts (`Softmax(logits) * Exp(log_counts)`) vs target counts.

### Default Pipeline Order
When `transforms` are not explicitly provided to `CerberusDataset`, they are automatically constructed from `DataConfig` in the following order:
1.  **Jitter**: Enforce `input_len` and apply augmentation.
2.  **ReverseComplement**: (If enabled).
3.  **TargetCrop**: Enforce `output_len`.
4.  **Bin**: (If `output_bin_size > 1`).
5.  **Log1p**: (If enabled).

## Cross-Validation (Folds)

Cerberus has built-in support for K-fold cross-validation to ensure robust evaluation.
*   **Chromosomal Partitioning**: Splits data by chromosomes to avoid data leakage (e.g., neighboring correlated regions).
*   **Config**: Defined in `GenomeConfig` via `fold_type`.
*   **Samplers**: The `split_folds` method ensures that training, validation, and test sets are disjoint based on these fold definitions.
