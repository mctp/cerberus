# Cerberus Dataloader Design Document

## 1. Overview
This document outlines the architecture for the `Cerberus` dataloader, a unified data pipeline designed to support training, validation, and testing of genomic deep learning models. It specifically targets the requirements of **ASAP** (Atac Seq Analysis Pipeline), **BPNet-lite**, and **ChromBPNet-pytorch**.

The goal is to provide a flexible, efficient, and reproducible way to load genomic sequences, signal tracks (BigWig), and interval data (BED), while supporting advanced data augmentation and sampling strategies.

## 2. Design Goals
1.  **Universality**: Support both binned (ASAP) and base-resolution (BPNet) signal tracks.
2.  **Multi-modal Input**: Support DNA sequence as primary input, but allow additional input channels (e.g., conservation scores, bias tracks, ATAC "input" control).
3.  **Flexible Targets**: Support profile prediction (1D tensors) and scalar prediction (total counts).
4.  **Advanced Sampling**: Support peak-centric training, whole-genome sliding windows, and "negative peak" sampling (mixing positives and negatives).
5.  **Augmentation**: Integrated support for jitter, reverse complement, and signal smoothing/binning.
6.  **Efficiency**: Leverage lightweight, GIL-free libraries (`pyfaidx`, `pybigtools`) for efficient I/O, following `Tangermeme` and `BPNet-lite` strategies.

## 3. Architecture

### 3.1. Core Concepts

The dataloader is built around the concept of a **Sample**, which is derived from a genomic **Interval**.

-   **Interval**: A specific genomic region `(chrom, start, end)`.
-   **Input (`X`)**:
    -   **Sequence**: One-hot encoded DNA (4 channels).
    -   **Signals**: Optional extra channels (e.g., conservation, control bias).
-   **Tensor Dimensions**:
    -   **Ordering**: `(Channels, Length)` for a single sample.
        -   PyTorch `Conv1d` expects `(Batch, Channels, Length)`.
        -   The `DataLoader` will stack samples to produce this batch dimension.
    -   **Sequence Channel Order**:
        -   **Canonical**: `ACGT` (Index 0=A, 1=C, 2=G, 3=T). Standard for **BPNet/ChromBPNet**.
        -   **ASAP-Compatibility**: `AGCT` (Index 0=A, 1=G, 2=C, 3=T). Required when using pre-trained ASAP weights.
-   **Target (`Y`)**:
    -   **Profile**: Binned or unbinned signal tracks (e.g., ChIP/ATAC read counts).
    -   **Scalar**: Aggregated values (e.g., total counts) for tasks like count prediction or mappability classification.

### 3.2. Component Structure

The system will consist of the following main components:

#### A. `DataSource` (The "What")
Wrappers around raw data files on disk, handling random access and extraction.
-   **`SequenceSource`**: Wrapper for Fasta files (via `pyfaidx`). Handles fetching sequence strings/bytes.
-   **`SignalSource`**: Wrapper for BigWig files (via `pybigtools`). Handles fetching continuous tracks.
-   **`IntervalSource`**: Wrapper for BED files. Manages lists of intervals (Peaks, Negatives, Blacklists).

#### B. `DataExtractor` (The "How")
Responsible for taking an **Interval** and extracting raw tensors from `DataSources`.
-   **`SequenceExtractor`**:
    -   Extracts sequence `S` of length `L_in`.
    -   Applies One-Hot Encoding (ACGT or AGCT ordering).
    -   Supports **Lazy Loading** (disk-based) or **In-Memory** operation.
-   **`SignalExtractor`**:
    -   Extracts signal `T` of length `L_out` (or `L_in`).
    -   Returns **raw values** (no binning/logging applied at extraction time).
    -   Supports **Lazy Loading** (disk-based, GIL-free via `pybigtools`) or **In-Memory** operation.
-   **`MaskExtractor`**:
    -   Extracts binary mask `M` of length `L_in` from BigBed files.
    -   Returns **binary values** (1.0 for overlap, 0.0 otherwise).
    -   Supports **Lazy Loading** (disk-based) or **In-Memory** operation.

#### C. `Sampler` (The "Where")
Determines *which* intervals are fed to the model.
-   **`IntervalSampler`**: Iterates through a given BED file. Supports sequential, random, or **weighted** sampling (e.g., based on signal strength).
-   **`SlidingWindowSampler`**: Iterates across the genome with a stride (ASAP WGDataset style).
-   **`MultiSampler`**: Mixes samples from multiple sources (e.g., 50% Peaks, 50% Background) per epoch (BPNet style). Supports dynamic scaling strategies (e.g., matching background count to peaks) to handle class imbalance.

#### D. `Transform` (The "Changes")
Applied to the extracted data on-the-fly.
-   **`Jitter`**: Randomly shifts the center of the interval within a max shift range.
-   **`ReverseComplement`**: Randomly flips sequence and signal (50% probability).
-   **`Smooth/Bin`**: Downsamples high-res signal to model output resolution (e.g., 4bp bins for ASAP).

#### E. `CerberusDataset` (The Interface)
The PyTorch `Dataset` implementation that ties everything together.
-   **`__getitem__(index)`**:
    1.  Get `interval` from `Sampler`.
    2.  Apply `Jitter` (modifies interval).
    3.  Extract `inputs` via `SequenceExtractor` + `SignalExtractor`.
    4.  Extract `targets` via `SignalExtractor`.
    5.  Apply `ReverseComplement`.
    6.  Return `(inputs, targets, metadata)`.

## 4. Detailed Functionality

### 4.1. Inputs
The dataloader must accept a flexible dictionary of inputs to construct the `DataSources`:
```python
input_config = {
    "genome": "path/to/hg38.fa",
    "inputs": {
        "conservation": "path/to/cons.bw",
        "bias": "path/to/bias.bw"
    },
    "targets": {
        "counts": "path/to/counts.bw"
    },
    "intervals": {
        "peaks": "path/to/peaks.bed",
        "negatives": "path/to/negatives.bed",
        "blacklist": "path/to/blacklist.bed"
    }
}
```

When implementing this consider using names such that it is easy tell model inputs apart from targets. All of the above would be inputs not targets.

### 4.2. Signal Binning and Scalar Reduction
To support **ASAP** (4bp bins) and **BPNet** (1bp resolution), `SignalExtractor` will support a `resolution` or `bin_size` parameter.
-   **Unbinned**: `bin_size=1`. Returns `[L]`.
-   **Binned**: `bin_size=k`. Returns `[L/k]`. Aggregation mode: `mean`, `max`, or `sum`.
-   **Scalar**: `bin_size=L`. Returns `[1]` (equivalent to aggregating the entire window).
-   **Simultaneous Output**:
    -   For **BPNet/ChromBPNet** architectures (which have separate profile and count heads), the dataloader can yield **both** the profile tensor `[C, L]` and the scalar total counts `[C, 1]`.
    -   **Computation**: The scalar is computed **on-the-fly** from the extracted profile window during `__getitem__` (e.g. via optimized binning logic or simple sum). This avoids storing redundant scalar files on disk while ensuring the counts exactly match the profile provided to the model.

### 4.3. Mappability and Blacklists
-   **Blacklist**: Regions in the `blacklist` BED will be excluded from valid sampling regions during `Sampler` initialization.
    -   **Overlap Threshold**: To avoid discarding large windows due to trivial overlaps (e.g., 1bp), the sampler supports a `max_blacklist_fraction` (e.g., 0.01). Windows are only discarded if the blacklisted overlap exceeds this fraction of the window length.
    -   **Efficient Implementation**: Intersection queries are computationally expensive if done per-sample. The dataloader performs these operations during initialization using vectorized interval logic (e.g., via `ruranges`) to pre-filter the allowable regions list.
-   **Mappability**:
    -   Can be treated as an **Input Signal** (masking input).
    -   Can be treated as a **Target** (for auxiliary tasks like in ASAP).
    -   Can be used as a **Filter** (discard windows with < 65% mappability).

### 4.4. Negative Sampling (BPNet/ChromBPNet)
BPNet requires training on a mix of Peaks and Non-Peaks (Negatives).
-   The `MultiSampler` accepts a list of sub-samplers (e.g., one for peaks, one for negatives).
-   Configurable `scaling` per sub-sampler (e.g., `"min"`, `"max"`, or float ratio) allows easy balancing of classes.
-   In each epoch, it yields a mix of positive and negative samples.
    -   **GC-Matching**: If "negatives" are not provided, an optional pre-processing step can generate GC-matched negative regions from the genome background.

### 4.5. Data Augmentation Details
1.  **Jitter (reference: `seqpro.jitter`)**:
    -   Randomly shifts the center of the interval within a `max_jitter` range (e.g., +/- 128bp).
    -   The implementation must handle the synchronized jittering of multiple arrays (Sequence, Signal, Control) to ensure alignment is maintained.
    -   Requires reading a larger window (`input_window + 2 * max_jitter`) from disk.
2.  **Reverse Complement (reference: `seqpro.reverse_complement`)**:
    -   Applied to Sequence (reverse order, complement base).
    -   Applied to Signal (reverse order).
    -   Applied to *stranded* signal if present (swaps plus/minus strands and reverses).

### 4.6. Signal Normalization and Transformation
To ensure consistent dynamic ranges and model stability, the dataloader will support configurable signal transformations:

1.  **Log Transformation**:
    -   Apply `log1p(x) = log(x + 1)` to signal tracks.
    -   Standard practice in **BPNet/ChromBPNet** (for counts/bias) and **ASAP** (via `logspace=True` parameter) to compress dynamic range and stabilize training.
2.  **Total Count Normalization**:
    -   Option to normalize profiles to sum to 1 (probability distribution) for use with Multinomial NLL loss.
    -   Usually handled by the loss function, but can be pre-computed here.
3.  **Max Clipping**:
    -   Clip signal values at a specified percentile (e.g., 99.9th) to remove extreme outliers that can destabilize training.
4.  **Read Depth Normalization**:
    -   Normalize by library size (CPM - Counts Per Million) if comparing across samples with vastly different sequencing depths.
    -   *Note*: For standard peak-calling/profile prediction within a single cell type, raw counts (or log-counts) are often preferred.

### 4.7. Windowing and Stride
The dataloader defines explicit parameters for windowing logic, particularly for the `SlidingWindowSampler`:

-   **Binning (e.g. `seqpro.bin_coverage`)**:
    -   Used for ASAP mode (e.g., 4bp bins).
    -   Efficiently sums or means coverage over non-overlapping windows using optimized reduction.
-   **`input_window`**: Total length of the input sequence (e.g., 2048 bp).
-   **`output_window`**: Length of the target signal (e.g., 1024 bp). Usually centered within the input.
-   **`step_size` (Stride)**: The distance between the centers of consecutive windows.
    -   **Non-overlapping**: `step_size >= input_window`.
    -   **Overlapping**: `step_size < input_window`. The dataloader fully supports this case, which is common for "tiling" the genome during training or robust inference.
-   **Redundant Computation**:
    -   When windows overlap, the same genomic regions are read and processed (binned/normalized) multiple times.
    -   **Design Decision**: `Cerberus` accepts this redundancy. The random access speed of `pyfaidx`/`pybigtools` makes re-reading small overlapping regions negligible compared to the model forward pass.
    -   *Optimization*: Complex caching of binned regions is avoided to maintain architectural simplicity and shuffle-independence during training.

### 4.8. Handling Unknown Bases (N)
Genomic reference sequences often contain ambiguous bases ('N').
-   **Reference Strategy**: Both **ASAP** and **ChromBPNet** map 'N' to a zero-vector `[0, 0, 0, 0]` in the one-hot encoding.
-   **Design Decision**: `Cerberus` adopts this standard. `N` bases will result in zero-padding in the sequence tensor.
-   **Implementation**: This is natively supported by robust OHE functions (e.g. in `SeqPro`) by treating 'N' as an unknown character outside the core ACGT alphabet.

## 5. Implementation Strategy

### 5.1. Dependencies
-   **`pyfaidx`**: Efficient, random access Fasta reader.
-   **`pybigtools`**: GIL-free, thread-safe BigWig reader (Rust-based).
-   **`SeqPro` (Reference)**: Provides patterns for fast OHE, RC, and padding operations.
-   **`PyTorch`**: Standard `Dataset`/`DataLoader`.

### 5.2. Class Interface (Current Implementation)

```python
class CerberusDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        genome_config: dict | GenomeConfig, 
        data_config: dict | DataConfig, 
        sampler_config: dict | SamplerConfig,
        ...
    ):
        # 1. Initialize Sampler (filters regions using exclude_mask)
        self.sampler = ...
        
        # 2. Initialize Extractors (lazy loading or in-memory)
        self.sequence_extractor = ...
        self.input_signal_extractor = ...
        self.target_signal_extractor = ...
        
    def __getitem__(self, idx):
        interval = self.sampler[idx]
        
        # 3. Extract raw data
        seq = self.sequence_extractor.extract(interval) # (4, L)
        
        if self.input_signal_extractor:
            input_signals = self.input_signal_extractor.extract(interval) # (C_in, L)
            inputs = torch.cat([seq, input_signals], dim=0)
        else:
            inputs = seq
            
        if self.target_signal_extractor:
            targets = self.target_signal_extractor.extract(interval) # (C_out, L)
        
        # 4. Transform
        inputs, targets, interval = self.transforms(inputs, targets, interval)
        
        return {
            "inputs": inputs, 
            "targets": targets, 
            "intervals": str(interval)
        }
```

## 6. Usecase Configurations

### A. ASAP Mode
-   **Input**: 2048 bp
-   **Output**: 1024 bp (center), Binned 4bp (Length 256)
-   **Encoding**: AGCT
-   **Sampling**: WGDataset (Sliding Window) or PeakDataset (Peaks only)
-   **Target**: Max-pooled signal.
-   **Normalization**: `log1p` (logspace=True).

### B. BPNet / ChromBPNet Mode
-   **Input**: 2114 bp
-   **Output**: 1000 bp (center), Unbinned (Length 1000)
-   **Encoding**: ACGT
-   **Sampling**: Stratified (Peaks + Negatives)
-   **Target**: Raw counts (Stranded or Unstranded).

## 7. Performance & IO Strategy

### 7.1. Lazy vs In-Memory Loading
The dataloader supports two distinct modes of operation to balance memory usage and speed:

1.  **Lazy Loading (Default)**:
    -   **Mechanism**: Reads data from disk on-the-fly for each batch.
    -   **Pros**: Zero startup time; Supports datasets much larger than RAM (e.g., Whole Genome); Efficient for large-scale pre-training.
    -   **Cons**: Slower per-epoch than RAM; Disk I/O bottleneck.
    -   **Implementation**: Relies on `pyfaidx` and `pybigtools` which are efficient and thread-safe, allowing multiple PyTorch workers to read concurrently without GIL contention.

2.  **In-Memory Caching (BPNet-lite Style)**:
    -   **Mechanism**: Pre-loads all intervals (sequences and signals) into RAM tensors at initialization.
    -   **Pros**: Extremely fast training loop (no Disk I/O); ideal for smaller peak-centric datasets (< 100k regions).
    -   **Cons**: High memory footprint; Long startup time; Cannot handle whole-genome sliding windows easily.
    -   **Implementation**: A `cache_dataset=True` flag in `CerberusDataset` triggers a one-time load (using `tangermeme`-like `extract_loci` logic) into CPU/GPU tensors.
    -   **Multiprocessing Optimization**: To prevent memory explosion when using PyTorch DataLoader workers (especially with the `spawn` start method used on macOS/Windows and DDP), in-memory tensors are explicitly moved to **Shared Memory** using `torch.Tensor.share_memory_()`. This allows multiple worker processes to read from the same physical memory block without duplication.

### 7.2. Backend: pyBigWig vs pybigtools
-   **pyBigWig**: The standard Python wrapper for `libBigWig` (C).
    -   *Limitation*: **GIL-bound**. It holds the Global Interpreter Lock during read operations. This essentially serializes reads even when using multiple threads, forcing the use of process-based workers (`num_workers > 0` in PyTorch) which significantly increases memory usage and startup time.
-   **pybigtools** (Primary):
    -   *Advantage*: **GIL-Free**. Implemented in Rust, it releases the GIL during file I/O.
    -   *Benefit*: Enables true multi-threaded data loading within a single process or efficient multi-process loading. This drastically reduces memory overhead.
    -   *Reference*: See `repos/tangermeme/tangermeme/io.py` for usage examples.

### 7.3. Alternative: GenVarLoader
`GenVarLoader` is a powerful library for handling various genomic data types with support for genetic variants (VCFs). While `Cerberus` defaults to lightweight libraries, `GenVarLoader` remains an alternative backend if variant support or complex haplotype construction becomes necessary.

### 7.4. Genome Partitioning and Folds
To support robust evaluation and ensemble training, the dataloader includes a flexible mechanism for data partitioning (Folds). It supports both coarse (chromosome-level) and granular (region-level) splitting.

1.  **Chromosome Splitting**:
    -   **Concept**: Traditional method where whole chromosomes are assigned to Train/Val/Test (e.g., `train=['chr1'], val=['chr2']`).
    -   **Usage**: Simple config lists passed to the dataloader.
2.  **Granular BED-based Partitioning**:
    -   **Concept**: Folds are defined by a master BED file where each region is annotated with a `fold_id`.
    -   **Format**: A BED file with columns `(chrom, start, end, fold_id)`.
    -   **Application**:
        -   Allows blocking the genome into smaller chunks (e.g., 1MB blocks) to maximize data usage while preventing leakage.
        -   The `IntervalSampler` filters sampling regions based on intersection with the active fold's regions from the partition file.
3.  **Implementation**:
    -   `CerberusDataModule` (PyTorch Lightning style wrapper) orchestrates the splits.
    -   For **BED-based**, it loads the partition file and initializes datasets by passing the subset of allowed regions (or a filter mask) corresponding to the requested fold.

### 7.5. Performance Analysis (Reference: SeqPro)
High-performance sequence operations (exemplified by `SeqPro`) are critical for the dataloader:
-   **Numba Optimization**: Critical string operations (One-Hot Encoding, Padding, Tokenization) should be implemented as **Numba Generalized Ufuncs (gufuncs)**. This compiles Python logic to machine code, bypassing the slow Python interpreter loops typically associated with string processing.
-   **Alternative: NumPy Lookup Table**: A highly efficient, dependency-free alternative for OHE is using a **Lookup Table (LUT)**. By casting byte arrays to `uint8` and indexing into a pre-computed `(256, 4)` array, we can achieve speeds comparable to compiled code using pure NumPy.
-   **Byte-Level Operations**: Efficient handling of DNA sequences involves casting strings/bytes to `uint8` numpy views. This avoids overhead from Python string objects and allows direct memory manipulation.
-   **Vectorization**: Augmentations like `jitter` and `reverse_complement` must be fully vectorized and support operating on multiple arrays (e.g., input sequence + target signal) simultaneously to ensure data consistency.
-   **Conclusion**: Adopting optimized primitives (like those in `SeqPro`) is significantly more performant and robust than re-implementing these operations in pure Python/Pandas.

### 7.6. PyTorch Optimization & Best Practices
The dataloader design adheres to PyTorch performance best practices while respecting the unique threaded nature of `pybigtools`:

1.  **Parallel Loading & Workers**:
    -   **pybigtools Specifics**: `pybigtools` releases the GIL, allowing concurrent reads.
    -   **Recommendation**: Use `num_workers=2` to `4` to parallelize the Python-side **Transform** operations (OHE, Log, etc.) while allowing efficient I/O.
    -   **Persistent Workers**: Use `persistent_workers=True` to keep worker processes alive between epochs, reducing startup overhead.
    -   **Multiprocessing Context**: On macOS/MPS, explicitly set `multiprocessing_context='spawn'` if stability issues (crashes/hangs) arise.
2.  **Memory Management**:
    -   **`pin_memory=True`**: Ensures data is allocated in pinned (page-locked) memory, allowing for non-blocking high-speed transfer to the GPU.
    -   *Note*: Automatically disabled on MPS devices where pinned memory is currently unsupported.
3.  **Reproducibility**:
    -   **`worker_init_fn`**: Global seeding must be managed carefully for random augmentations and NumPy.

## 9. Code References
The following files in the `repos/` submodules were consulted and are relevant for the implementation:

### ASAP
-   `repos/ASAP/src/asap/dataset.py`: High-level dataset orchestration.
-   `repos/ASAP/src/asap/dataloader/base.py`: Base dataset logic.
-   `repos/ASAP/src/asap/dataloader/wg.py`: Sliding window dataset logic.
-   `repos/ASAP/src/asap/dataloader/peak.py`: Peak-centered dataset logic.
-   `repos/ASAP/src/asap/dataloader/utils/data_bw.py`: BigWig reading and binning utilities.

### BPNet-lite
-   `repos/bpnet-lite/bpnetlite/io.py`: `PeakNegativeSampler` and `DataGenerator` logic.

### ChromBPNet-pytorch
-   `repos/chrombpnet-pytorch/chrombpnet/dataset.py`: PyTorch Lightning DataModule and Dataloader.
-   `repos/chrombpnet-pytorch/chrombpnet/data_utils.py`: Low-level sequence/signal extraction.

### GenVarLoader (Alternative)
-   `repos/GenVarLoader/python/genvarloader/_bigwig.py`: Efficient BigWig reader.
-   `repos/GenVarLoader/python/genvarloader/_fasta.py`: Fasta reader.

### SeqPro
-   `repos/SeqPro/python/seqpro/_encoders.py`: One-hot encoding utilities.
-   `repos/SeqPro/python/seqpro/_modifiers.py`: Sequence augmentation (jitter, RC).

### Tangermeme
-   `repos/tangermeme/tangermeme/io.py`: `extract_loci` function and `pybigtools` usage.

## 10. Extensibility & Future Directions
The architecture is designed to be extensible beyond 1D signal tracks and BigWig files. Future implementations can leverage the abstract `DataSource` and `DataExtractor` patterns to support:

### 10.1. Additional Signal Formats and Outputs
-   **BigWig Writing**:
    -   A utility to write prediction tensors back to BigWig format (using `pyBigWig`), handling the reverse mapping of `bin_size` (e.g. broadcasting 4bp bins back to genomic coordinates).
-   **Raw Alignments (BAM/CRAM/Fragment)**:
    -   Instead of pre-computed BigWigs, coverage can be computed on-the-fly from alignment files.
    -   **Benefit**: Allows dynamic filtering (e.g., by fragment size, mapping quality) and variable resolution without re-processing files.
    -   **Implementation**: A `BamSource` class wrapping `pysam` or `rust-bio` to stream reads and compute pileups.
-   **HDF5 / Zarr / N5**:
    -   For massive, chunked datasets where random access in BigWig is too slow or limiting.
    -   **Zarr** is particularly promising for cloud-native storage and parallel reading.

### 10.2. High-Dimensional Data (2D/3D)
The current `(C, L)` tensor structure can be extended to support multi-dimensional genomic data:
-   **Hi-C / Micro-C (Contact Maps)**:
    -   **Format**: `.cool` (Cooler) or `.mcool`.
    -   **Tensor Shape**: `(C, L, L)` representing contact frequencies between loci in the window.
    -   **Augmentation**: 2D Jitter (shifting the window along the diagonal) and 2D augmentation.
-   **Single-Cell Matrices**:
    -   **Format**: `AnnData` (`.h5ad`) or TileDB.
    -   **Integration**: Returning cell x gene matrices corresponding to the genomic interval (if applicable, e.g. for scATAC).

## 11. Downstream Compatibility
The dataloader is specifically engineered to feed into downstream analysis tools like **Tangermeme**:

-   **Tensor Shapes**:
    -   Sequence: `(Batch, 4, Length)` matches `tangermeme.predict` and `tangermeme.deep_lift_shap` expectations.
    -   Signal: `(Batch, Channels, Length)` aligns with Tangermeme's extraction logic.
-   **Interpretation & Attribution**:
    -   **N-Handling**: The zero-vector encoding for 'N' is compatible with DeepLIFT/SHAP difference-from-reference calculations.
    -   **Jitter**:
        -   For **Lazy Loading**, jitter is applied on-the-fly, so the model receives a fixed-size window.
        -   For **In-Memory** (BPNet style), users can request `input_window + 2*max_jitter` to allow `tangermeme`'s samplers to handle randomization during training.
-   **Model Inputs**: The separate `(input, target, control)` tuple structure is flexible enough to support the various signatures required by `tangermeme` wrappers.
