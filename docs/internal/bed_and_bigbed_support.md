# BED and BigBed Input Track Support Plan

## Overview
This document outlines the plan to add support for BED and BigBed files as input tracks (features) in Cerberus. Currently, Cerberus only supports BigWig files (`.bw`) for input/target signals via configuration.

## Missing Features & Tests
1.  **BED Support:** No support for text-based BED files as input tracks.
2.  **BigBed Config Support:** BigBed files are supported via `MaskExtractor` but cannot be used in `DataConfig` because `CerberusDataset` defaults to `SignalExtractor`.
3.  **Tests:** Missing tests for:
    -   Using BED files as inputs.
    -   Using BigBed files as inputs via config.
    -   Graceful failure when unsupported files are passed to `SignalExtractor`.

## Implementation Plan

### 1. `BedMaskExtractor`
Implement a new class `BedMaskExtractor` in `src/cerberus/mask.py` (or similar).
-   **Input:** Dictionary of channel names to BED file paths.
-   **Storage:** Load BED intervals into `interlap.InterLap` structures in memory.
-   **Extraction:** `extract(interval)` queries the InterLap structure and returns a binary tensor (1.0 for overlap, 0.0 otherwise).
-   **Interface:** Must conform to `BaseMaskExtractor` / `BaseSignalExtractor` protocol.

### 2. Extractor Factory
Refactor `CerberusDataset` in `src/cerberus/dataset.py` to dynamically select the appropriate extractor based on file extension.

**Logic:**
-   Gather all input paths.
-   Group by extension type? Or enforce same type per group?
    -   *Constraint:* `CerberusDataset` currently takes a single `input_signal_extractor`. If inputs are mixed (e.g., some BigWig, some BED), we might need a `CompositeSignalExtractor` or enforce one type per dictionary.
    -   *Decision:* For now, we will assume all inputs in `data_config['inputs']` are of the same type OR we will implement a factory that creates a composite or handles mixed types if needed.
    -   *Better approach:* Check extensions for each channel. If they differ, create separate extractors and wrap them, or (simpler) create a `UniversalSignalExtractor` that holds a dict of specific extractors per channel.
    -   *Simpler MVP:* Inspect the first file. If `.bw` -> `SignalExtractor`. If `.bb` -> `BigBedMaskExtractor`. If `.bed` -> `BedMaskExtractor`. Warn if mixed?
    -   *Robust approach:* A `UniversalExtractor` that inspects each path and assigns it to an internal handler (BW/BB/BED). This allows mixing continuous signals (BW) with sparse peaks (BED).

### 3. Testing (`tests/test_track_support_gaps.py`)
-   **`test_bed_mask_extractor`**: Verify `BedMaskExtractor` works with dummy BED files.
-   **`test_dataset_routing`**: Verify `CerberusDataset` correctly loads BED and BigBed files when specified in `inputs`.
-   **`test_mixed_inputs`** (Optional): Verify mixing BW and BED works (if implemented).

## Class Design

```python
class BedMaskExtractor:
    def __init__(self, bed_paths: dict[str, Path]):
        self.intervals = {} # channel -> InterLap
        # load...

    def extract(self, interval: Interval) -> torch.Tensor:
        # query InterLap...
```

```python
# Universal Extractor Idea
class UniversalExtractor:
    def __init__(self, paths: dict[str, Path]):
        self.extractors = {}
        # for name, path in paths.items():
        #   if .bw: self.extractors[name] = SignalExtractor({name: path})
        #   elif .bed: self.extractors[name] = BedMaskExtractor({name: path})
        #   ...
    
    def extract(self, interval):
        # iterate channels, extract, stack...
```
