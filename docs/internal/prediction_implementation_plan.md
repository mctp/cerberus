# Prediction and BigWig Export Implementation Plan

## Overview
This document outlines the plan to implement flexible output prediction and BigWig export in Cerberus, utilizing `pybigtools`. The goal is to support:
- Whole genome or specific interval predictions (e.g., `chr1:1000-2000`).
- Overlapping window smoothing (stride < output_len).
- Multi-fold model support (automatically selecting the correct model for a region).
- Ensemble averaging (if multiple models apply).

## Analysis of ASAP Approach
The `asap` codebase uses `pyBigWig` and implements prediction by iterating over chromosomes. It handles robustness by shifting input windows and averaging the results. It has specific logic for SNVs and whole-genome tracks, manually managing buffers and writing to BigWig files with headers.

## Cerberus Design

### Key Components
1.  **`predict_to_bigwig` Entrypoint**: The high-level wrapper that orchestrates the process and writes to BigWig.
2.  **`_predict_to` Generator**: The core generator function that handles intervals, model loading, inference, and yields `(chrom, start, end, value)` tuples. This decouples prediction logic from output format.
3.  **Interval Parsing & Merging**: Flexible input handling (`parse_intervals`, `merge_intervals`).
4.  **Model Manager**: Maps genomic regions to model checkpoints based on fold configuration (`ModelManager`).
5.  **Sequence Extractor**: Extracts DNA sequences efficiently (`SequenceExtractor`).
6.  **BigWig Writer**: Uses `pybigtools` to write the output.

### 1. Interval Handling
We support inputs as:
- List of strings: `["chr1", "chr2:1000-5000"]`
- BED file path.
- None (implies whole genome from `GenomeConfig`).

Intervals are parsed into `(chrom, start, end)` tuples and merged if overlapping to ensure efficient processing and correct averaging at boundaries.

### 2. Model Selection
`ModelManager` handles loading models. `get_models(chrom, start, end, use_folds)`:
- Checks `GenomeConfig` to see which fold(s) contain this interval in their test (or validation/train) set.
- Returns a list of initialized `CerberusModule` instances.
- Caches models to avoid reloading.

### 3. Prediction Loop with Stride
Inside `_predict_to`:
- **Buffer**: Allocate arrays `accumulated_values` and `counts` of shape `(E - S) // bin_size`.
- **Input Generation**:
    - Iterate with `stride`.
    - Input window start $I_s$ such that the output window $O_s$ overlaps `[S, E)`.
    - $O_s = I_s + \text{offset}$, where $\text{offset} = (\text{input\_len} - \text{output\_len}) // 2$.
    - We need inputs such that $[O_s, O_s + \text{output\_len})$ covers the target.
- **Inference**: Run model(s). Average outputs if multiple models are active.
- **Accumulation**: Add prediction to the buffer.
    - Calculate intersection of prediction window and target interval.
    - Update corresponding indices in buffer.
- **Yielding**: Normalize (`accum / counts`) and yield chunks as `(chrom, start, end, val)`.

### 4. BigWig Writing
- Initialize `bw = pybigtools.open(output_path, "w")`.
- `pybigtools` requires a dictionary of chromosome lengths (`chroms`).
- Write intervals using the generator: `bw.write(chrom_sizes, generator)`.
- Note: `pybigtools.BigWigWrite` does not support context manager protocol (`with ...`).

## Implementation Checklist

- [x] **Create `src/cerberus/predict.py`**
    - [x] `parse_intervals(intervals, genome_config)`
    - [x] `merge_intervals(intervals)`
    - [x] `_predict_to(...)` generator function.
    - [x] `predict_to_bigwig(...)` main function.

- [x] **Interval Logic**
    - [x] Support `chr:start-end` format.
    - [x] Support whole chromosome names.
    - [x] Merge overlapping intervals.

- [x] **Prediction Logic**
    - [x] Implement sliding window generator.
    - [x] Implement accumulation buffer.
    - [x] Handle `output_bin_size`.
    - [x] Handle edge cases (start of chrom, end of chrom).
    - [x] Support multi-fold models via `ModelManager`.

- [x] **BigWig Integration**
    - [x] Use `pybigtools.open`.
    - [x] Construct `vals` iterator from processed buffers.
    - [x] Handle lack of context manager support.

- [x] **Testing**
    - [x] `tests/test_predict.py`: End-to-end test with dummy model and mocked data.
    - [x] `tests/test_predict_config.py`: Configuration validation tests.

## Refactoring for Generic Output
To support future output formats (e.g., Zarr, BedGraph), the core prediction logic was extracted into `_predict_to`.
```python
def _predict_to(...) -> Iterator[tuple[str, int, int, float]]:
    # ... yields (chrom, start, end, value) ...
```
`predict_to_bigwig` simply wraps this generator:
```python
def predict_to_bigwig(...):
    generator = _predict_to(...)
    bw = pybigtools.open(output_path, "w")
    bw.write(genome_config["chrom_sizes"], generator)
```
