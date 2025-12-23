# Cerberus Dataloader: Fork & Pickle Safety

## Overview
When using PyTorch `DataLoader` with `num_workers > 0`, the dataset instance is replicated across multiple worker processes. This replication involves either `fork` (default on Unix) or `spawn` (default on Windows/macOS, or if configured).

Handling file descriptors (such as open file handles for FASTA or BigWig files) during this process requires careful design to avoid:
1.  **Race Conditions**: Multiple processes attempting to use the same file descriptor state simultaneously (if inherited via `fork`).
2.  **Pickling Errors**: Open file handles cannot be pickled, causing failures when using `spawn` or `forkserver`.
3.  **Silent Corruption**: Some C/Rust libraries might appear to work but suffer from internal state corruption when forked.

## Strategy: Lazy Initialization with PID Check

To ensure robustness, `Cerberus` adopts a **Lazy Initialization** strategy for all data sources (`SequenceExtractor`, `SignalExtractor`).

### 1. No Open Handles in `__init__`
The `__init__` method of extractors **must not** open any files. It should only store the configuration (paths, encoding, etc.) and initialize the handle storage to `None`.

### 2. Initialization on First Use
File handles are opened only when `extract()` (or a similar data access method) is called for the first time.

### 3. Pickling Support & Process Safety
To support `spawn` (which pickles the dataset), we implement `__getstate__` to explicitly exclude file handle attributes from the pickled state. When unpickled in the worker, these attributes default to `None`, triggering initialization on the first use.

For `fork` (default on Unix), we rely on the standard PyTorch `DataLoader` pattern where the dataset is initialized in the main process (without accessing data) and then forked. Workers will see uninitialized handles (`None`) and open their own.

## Implementation Details

### `SequenceExtractor`
-   **State**: `self.fasta` (pyfaidx.Fasta).
-   **Logic**: Checks `None` on `extract`. Uses `pyfaidx.Fasta` which handles random access.

### `SignalExtractor`
-   **State**: `self._bigwig_files` (dict of pybigtools handles).
-   **Logic**: Checks `None` on `extract`. Uses `pybigtools.open`.

This design ensures that `Cerberus` works seamlessly with any `num_workers` setting and any multiprocessing start method.
