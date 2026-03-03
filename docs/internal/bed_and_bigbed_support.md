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

### 2. Extractor Factory — IMPLEMENTED

`UniversalExtractor` now uses a module-level `_EXTRACTOR_REGISTRY` to resolve file extensions to extractor classes. New formats are added via `register_extractor(extension, cls, in_memory_cls)` — no modification to `UniversalExtractor` needed.

Channels are grouped by resolved class so same-type channels share one extractor instance (preserving the batching optimization). Mixed inputs (e.g., BigWig + BED) are fully supported.

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
# UniversalExtractor now uses registry-based dispatch (implemented in signal.py):
# register_extractor('.bw', SignalExtractor, InMemorySignalExtractor)
# register_extractor('.bed', BedMaskExtractor)
# ... new formats added via register_extractor() calls
```
