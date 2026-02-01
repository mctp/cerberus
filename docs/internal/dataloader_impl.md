# Cerberus Dataloader Implementation Plan

This document outlines the classes and functions for the `Cerberus` dataloader implementation, describing their roles and connections.

## 1. Core Dataset
**Location**: `src/cerberus/dataset.py`

### `class CerberusDataset(torch.utils.data.Dataset)`
The primary entry point for PyTorch. Orchestrates data loading, sampling, and splitting.

*   **`__init__(self, genome_config: GenomeConfig, data_config: DataConfig, sampler_config: SamplerConfig | None = None, sequence_extractor: BaseSequenceExtractor | None = None, input_signal_extractor: BaseSignalExtractor | None = None, target_signal_extractor: BaseSignalExtractor | None = None, sampler: Sampler | None = None, exclude_intervals: dict[str, InterLap] | None = None, transforms: list[DataTransform] | None = None, deterministic_transforms: list[DataTransform] | None = None, in_memory: bool = False, is_train: bool = True, seed: int | None = None)`**
    *   **`genome_config`**: Configuration for genome FASTA and chromosome sizes.
    *   **`data_config`**: Configuration for input/output dimensions and tracks.
    *   **`sampler_config`**: Configuration for the sampling strategy.
    *   **`sequence_extractor`**: Optional pre-initialized extractor (for sharing across subsets).
    *   **`input_signal_extractor`**: Optional pre-initialized extractor for input signals (e.g. conservation).
    *   **`target_signal_extractor`**: Optional pre-initialized extractor for target signals (e.g. ChIP-seq counts).
    *   **`sampler`**: Optional pre-initialized sampler (for subsets).
    *   **`exclude_intervals`**: Optional pre-loaded exclusion intervals (for sharing across subsets).
    *   **`transforms`**: Optional list of transforms for training.
    *   **`deterministic_transforms`**: Optional list of deterministic transforms for val/test.
    *   **`in_memory`**: Whether to load data into memory.
    *   **`is_train`**: Whether this dataset is used for training.
    *   **`seed`**: Optional random seed for sampler initialization.
    *   Initializes `DataSource`s (Genome/Signals) if not provided.
    *   Initializes `Sampler` if not provided.
    *   Supports `InMemory` versions of extractors based on `in_memory`.

*   **`__len__(self) -> int`**
    *   Delegates to the active `Sampler`.

    *   **`__getitem__(self, idx: int) -> Tuple[Tensor, Tensor]`**
    *   Retrieves interval from `Sampler`.
    *   Fetches sequence data via `SequenceExtractor` -> `(4, L)`.
    *   Fetches input signal data via `InputSignalExtractor` -> `(C_in, L)`.
    *   Concatenates sequence and inputs along channel dimension -> `inputs`.
    *   Fetches target signal data via `TargetSignalExtractor` -> `targets`.
    *   Applies configured transforms (Jitter, ReverseComplement, Binning, etc.) via `self.transforms`, passing and updating the `interval` metadata.
    *   Returns a dictionary `{"inputs": inputs, "targets": targets, "intervals": str(interval)}`.

*   **`resample(self)`**
    *   Triggers resampling of the underlying sampler (e.g. `MultiSampler`). This is a no-op for samplers that do not support dynamic resampling.

*   **`split_folds(self, test_fold: int, val_fold: int) -> Tuple['CerberusDataset', 'CerberusDataset', 'CerberusDataset']`**
    *   Delegates to `self.sampler.split_folds` to get train/val/test samplers.
    *   Creates new `CerberusDataset` instances for each split using `_subset`.
    *   Returns `(train_dataset, val_dataset, test_dataset)`.

*   **`_subset(self, sampler: Sampler) -> 'CerberusDataset'`**
    *   Internal method to create a lightweight copy of the dataset.
    *   Shares `sequence_extractor`, `input_signal_extractor`, `target_signal_extractor`, and `exclude_intervals` with the new instance to avoid memory duplication and re-computation.
    *   Uses the provided `sampler` (e.g., a `BaseSampler`).

## 2. Sampling Components
**Location**: `src/cerberus/samplers.py`

### `class Sampler(Protocol)`
Interface for defining iteration strategies.

### `class BaseSampler`
Base class providing common splitting functionality.
*   **`_subset(self, indices: List[int]) -> BaseSampler`**: Returns a new sampler containing only the specified indices.
*   **`split_folds(self, test_fold: int, val_fold: int) -> Tuple[BaseSampler, BaseSampler, BaseSampler]`**:
    *   Uses pre-computed chromosome folds to generate train/val/test splits.
    *   Returns `BaseSampler`s for each split.
*   **`create_folds(chrom_sizes: dict, num_folds: int)`**: Static method to greedily distribute chromosomes into balanced folds.

### `class ListSampler(BaseSampler)`
Base class for samplers that store a concrete list of intervals.
*   **`__init__(self, intervals: List[Interval] | None = None, ...)`**
    *   A lightweight wrapper around a list of intervals.

### `class IntervalSampler(BaseSampler)`
*   **`__init__(self, file_path: Path, ..., num_folds: int = 5)`**
    *   **`num_folds`**: Number of folds to pre-compute for cross-validation.
*   **`__iter__(self) -> Iterator[Interval]`**: Yields regions from a file.
    *   **Supported Formats**: BED3-BED12, MACS narrowPeak.
    *   **Centering/Resizing**: Adjusts intervals based on `padded_size`.
    *   **Validation**: Checks genome bounds.
    *   **Exclusion**: Filters out intervals overlapping `exclude_intervals`.
    *   **Folds**: Initializes `self.folds` using a greedy chromosome splitting strategy (`create_folds`) to ensure balanced folds.

### `class SlidingWindowSampler(BaseSampler)`
*   **`__init__(self, chrom_sizes: dict, ..., num_folds: int = 5)`**
    *   Generates sliding windows across the genome.
    *   Initializes `self.folds` for cross-validation.

### `class MultiSampler(BaseSampler)`
*   **`__init__(self, samplers: List[Sampler], ...)`**
    *   Combines multiple samplers (e.g., Peaks and Background).
    *   **Scaling**:
        *   Scaling/balancing is handled by wrapping sub-samplers in `ScaledSampler`.
    *   **Resampling**: Supports epoch-based resampling of subsets.

## 3. PyTorch Lightning Integration
**Location**: `src/cerberus/datamodule.py`

### `class CerberusDataModule(pl.LightningDataModule)`
Standard interface for PyTorch Lightning integration.

*   **`__init__(self, genome_config, data_config, sampler_config, test_fold=0, val_fold=1, pin_memory=True)`**
    *   Accepts configuration dictionaries and dataloader parameters.
    *   **`pin_memory`**: Whether to copy tensors into CUDA pinned memory.
    *   **`test_fold`, `val_fold`**: Indices of folds to use for testing and validation.
    *   **Note**: Runtime parameters (`batch_size`, `num_workers`) are NOT passed to `__init__`.

*   **`setup(self, stage: str | None = None, batch_size: int | None = None, num_workers: int | None = None)`**
    *   **`batch_size`, `num_workers`**: Optional arguments to configure runtime settings.
    *   Initializes the full `CerberusDataset`.
    *   Splits dataset into `train_dataset`, `val_dataset`, and `test_dataset` using `split_folds`.

*   **`worker_init_fn(worker_id)`** (Static)
    *   Ensures reproducibility by seeding `numpy` random state based on `torch.initial_seed()`.
    *   Crucial for preventing identical augmentations across workers.

*   **`train_dataloader()`, `val_dataloader()`, `test_dataloader()`**
    *   Returns configured `DataLoader` instances.
    *   **Resampling**: `train_dataloader` detects if it's running within a `pl.Trainer`. If so, it uses the current epoch and global rank to seed the `resample()` method of the dataset. This ensures proper randomization and mixing (e.g., for `StratifiedSampler`) at the start of each epoch, provided `reload_dataloaders_every_n_epochs=1` is set in the Trainer.

## 4. Data Extraction Components
**Location**: `src/cerberus/dataset.py`

### `class SequenceExtractor`
Handles retrieval and encoding of DNA sequences using `pyfaidx`.

*   **`__init__(self, fasta_path: str, encoding: str = 'ACGT')`**
    *   Initializes `pyfaidx.Fasta`.
*   **`extract(self, interval: Interval) -> Tensor`**
    *   Fetches sequence string from `pyfaidx`.
    *   Calls `encode_dna`.

### `class SignalExtractor`
Handles retrieval of raw continuous signals using `pybigtools`.

*   **`__init__(self, bigwig_paths: Dict[str, str])`**
    *   Initializes `pybigtools.open` handles for each BigWig.
    *   Implements lazy loading (handles opened on first access) to support `multiprocessing`.
    *   Implements `__getstate__` to exclude handles from pickling.
*   **`extract(self, interval: Interval) -> Tensor`**
    *   Iterates over BigWig handles.
    *   Fetches values array using `bw.values(chrom, start, end)`.
    *   Handles `NaN` by replacing with 0.
    *   Handles truncation (padding with 0 if end > chrom end).
    *   Returns tensor of shape `(Channels, Length)`.
    *   *Note*: Does NOT currently perform binning or log-transformation. Returns raw values.

### `class InMemorySignalExtractor`
In-memory version of `SignalExtractor` for faster access when data fits in RAM.
*   **`__init__(self, bigwig_paths: Dict[str, str])`**
    *   Loads entire chromosomes from BigWigs into memory as `torch.Tensor`s during initialization.
*   **`extract(self, interval: Interval) -> Tensor`**
    *   Slices from cached tensors.

### `class MaskExtractor`
Handles extraction of binary masks from BigBed files using `pybigtools`. Useful for mappability tracks or other binary genomic features.

*   **`__init__(self, bigbed_paths: dict[str, Path])`**
    *   Initializes with a dictionary mapping channel names to BigBed file paths.
    *   Implements lazy loading and pickling support similar to `SignalExtractor`.
*   **`extract(self, interval: Interval) -> Tensor`**
    *   Queries BigBed files for entries overlapping the interval.
    *   Constructs a binary mask where 1.0 indicates overlap and 0.0 indicates no overlap.
    *   Returns tensor of shape `(Channels, Length)`.

### `class InMemoryMaskExtractor`
In-memory version of `MaskExtractor`.
*   **`__init__(self, bigbed_paths: dict[str, Path])`**
    *   Loads entire BigBed files into memory as binary mask tensors during initialization.
    *   Uses `torch.Tensor.share_memory_()` for efficiency across worker processes.
*   **`extract(self, interval: Interval) -> Tensor`**
    *   Slices from cached tensors.

## 4. Data Sources (IO Backend)

We use lightweight, specialized libraries for direct file access, avoiding heavy dependencies.

*   **FASTA**: `pyfaidx` provides efficient random access to indexed FASTA files.
*   **BigWig**: `pybigtools` provides fast access to BigWig signal tracks.

### Alternative Backend: `GenVarLoader`
*   **Description**: `GenVarLoader` is a powerful library for handling various genomic data types with support for genetic variants (VCFs).
*   **Status**: Currently considered too heavy for the core requirements of `Cerberus`.
*   **Future Use**: May be integrated as an alternative backend if variant support or complex haplotype construction becomes necessary.

## 5. Utility Functions
**Location**: `src/cerberus/genome.py`

*   **`encode_dna(sequences: str, encoding: str) -> Tensor`**
    *   One-Hot Encoding.
    *   Supports 'ACGT' and 'AGCT' encodings.
    *   Optimized using numpy indexing.

## 6. Region Pruning / N-Content Filtering

To ensure training data quality, intervals containing excessive unknown bases (Ns) (e.g., telomeres, centromeres, assembly gaps) must be pruned.

*   **Strategy**:
    *   **Pre-filtering (Recommended)**: Use standard "blacklist" regions (e.g., ENCODE blacklist, assembly gaps BED) to filter intervals during Sampler initialization.
    *   **Sampler Logic**:
        *   `IntervalSampler` accepts `exclude_intervals`.
        *   Intervals overlapping `exclude_intervals` are discarded.
    *   **Lazy Filtering (Alternative)**: The dataset can check N-content during `__getitem__`. If Ns > threshold, the sample is marked invalid. However, this complicates batch construction (variable batch size) and requires custom collate functions.
    *   **Implementation**: Phase 2 focuses on implementing `exclude_intervals` support in Samplers.

## 7. Interval Exclusion & Overlap Handling

Reliable exclusion of specific genomic regions (blacklists, gaps, unmappable regions) is critical.

### Implementation Strategy: Interval Trees (InterLap)
Cerberus adopts `InterLap` for precise and efficient interval exclusion.

*   **Concepts**:
    *   **InterLap**: A fast interval tree structure that allows for exact overlap queries.
    *   **Coordinate System**:
        *   **Standard (BED)**: Half-open intervals `[start, end)`.
        *   **InterLap**: Closed intervals `[start, end]`.
        *   **Handling**: The system automatically converts BED intervals to InterLap format by subtracting 1 from the end coordinate (`end - 1`) during both storage and querying. This ensures correct overlap detection for standard genomic data.
*   **Timing**: Exclusion is checked against the interval tree during sampling (e.g., `SlidingWindowSampler` generation or `IntervalSampler` filtering).
*   **Interface**:
    *   Samplers accept `exclude_intervals` as a dictionary mapping chromosome names to `InterLap` objects.

## 8. Exclude Intervals Loading Lifecycle

To ensure efficiency and consistency, exclude intervals (blacklists) are managed centrally by the Dataset.

1.  **Configuration**:
    *   `GenomeConfig` includes an `exclude_intervals` section mapping names (e.g., "blacklist") to file paths (BED).
2.  **Instantiation (`CerberusDataset.__init__`)**:
    *   **Step 1**: Load exclusion intervals from files defined in `genome_config['exclude_intervals']` using `exclude.py` into `InterLap` objects.
    *   **Step 2**: Store these as a dictionary `self.exclude_intervals: Dict[str, InterLap]`.
    *   **Step 3**: Instantiate `Sampler`, passing `self.exclude_intervals` as the argument.
3.  **Benefit**:
    *   Exclusions are loaded once.
    *   Samplers perform fast intersection queries to check for exclusion.

## 9. Transforms Architecture
**Location**: `src/cerberus/transform.py`

Transforms implement the `DataTransform` protocol:
`__call__(self, inputs: Tensor, targets: Tensor, interval: Interval) -> Tuple[Tensor, Tensor, Interval]`

*   **Interval Tracking**: Transforms receive the genomic interval and can modify it.
    *   **Jitter**: Updates interval start/end coordinates to reflect the cropped region.
    *   **ReverseComplement**: Flips the interval strand (e.g. `+` -> `-`).
*   **Benefits**: Ensures that metadata returned by the dataset accurately reflects the data augmentations applied.
