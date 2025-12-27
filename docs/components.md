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
*   **Behavior**: Mixes samples from sub-samplers based on `scaling_factors`.
    *   `scaling < 1.0`: Subsamples (uses a random subset each epoch).
    *   `scaling > 1.0`: Oversamples (duplicates samples).

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

*   **Profile Loss**: Penalizes the shape of the predicted signal distribution.
*   **Count Loss**: Penalizes the total count prediction (MSE on `log(1+x)`).
*   **Args**:
    *   `alpha`: Weight for the count loss component.
    *   `flatten_channels`: Whether to compute multinomial over all channels flattened (default: True, typical for BPNet).
    *   `implicit_log_targets`: If targets are already log-transformed, reverses them for profile loss calculation.

### BPNetPoissonLoss
Variation of BPNet loss using Poisson NLL for the count component.

*   **Count Loss**: Poisson Negative Log-Likelihood.
*   **Args**: Same as `BPNetLoss`.

## Metrics

### FlattenedPearsonCorrCoef
Computes Pearson Correlation Coefficient per channel.

*   **Behavior**: Flattens batch and length dimensions to compute correlation between the predicted and target signal vectors for each channel independently, then averages across channels.
*   **Why**: Standard global Pearson correlation can be misleading if channels have vastly different dynamic ranges.

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
