# Cerberus Dataloader Progress Tracker

This document tracks the iterative implementation of the Cerberus dataloader.

## Phase 1: Configuration & Stub
- [x] Create `src/cerberus/dataset.py` with `CerberusDataset` stub.
- [x] Define configuration schemas (Input/Run configs).
- [x] Implement config validation logic.
- [x] Test config validation with valid/invalid inputs.
- [x] Separate inputs/targets in configuration.
- [x] Validate file paths and existence in config.
- [x] Refactor config to `src/cerberus/config.py`.
- [x] Update dependencies (numpy, torch).
- [x] Simplify typing using Python 3.12+ features (including simplified `dict` usage).
- [x] Stub `__getitems__` for vectorized data access.
- [x] Test configuration with .narrowPeak files.
- [x] Refactor Input/Run configs into DataConfig.
- [x] Update DataConfig with missing parameters (input_len, max_jitter, etc.).

## Phase 2: Samplers
- [x] Implement `IntervalSampler` (BED file reading).
    - [x] Support generic BED (BED3-BED12).
    - [x] Support MACS narrowPeak format.
    - [x] Implement summit centering/resizing.
    - [x] Implement genome bounds validation (check against `chrom_sizes`).
    - [x] Implement exclusion logic (filtering using `ruranges`).
- [x] Implement `exclude.py` for efficient interval exclusion loading.
- [x] Test `IntervalSampler` (including exclusions).
- [x] Refactor SamplerConfig and IntervalSampler (padded_size, sampler_args).
- [x] Implement `SlidingWindowSampler`.
- [x] Test `SlidingWindowSampler`.
- [x] Implement `StratifiedSampler` (implemented as `MultiSampler` for multi-source mixing).
- [x] Test `StratifiedSampler` (via `tests/test_multi_sampler.py`).
- [x] Update documentation to include strategy for N-content filtering/pruning.
- [x] Implement `k_fold_split` and `SubsetSampler` in `BaseSampler`.
- [x] Implement `create_folds` logic for balanced chromosome splitting.

## Phase 3: Data Sources & Extractors
- [x] Implement `SequenceExtractor` (using pyfaidx).
- [x] Test `SequenceExtractor`.
- [x] Implement `SignalExtractor` (using pybigtools).
    - [x] Lazy loading (GIL-free).
    - [x] In-Memory implementation.
    - [x] Raw signal extraction (no binning/logging).
- [x] Test `SignalExtractor`.
- [x] Implement `InMemorySequenceExtractor` and `InMemorySignalExtractor`.
- [x] Implement `MaskExtractor` and `InMemoryMaskExtractor`.
- [x] Test `MaskExtractor`.

## Phase 4: Core Dataset
- [x] Implement `CerberusDataset` initialization.
- [x] Implement `CerberusDataset` resource sharing (subsetting).
- [x] Implement `split_folds` for cross-validation.
- [x] Implement `CerberusDataset.__getitem__` (Raw data extraction).
- [x] Implement `CerberusDataset._transform` (vectorized operations: jitter, RC, binning, Tanh).
- [x] Test dataset configuration and initialization.
- [x] Test full dataset pipeline (with transforms) with mock data.

## Phase 5: Integration & Verification
- [x] Verify compatibility with ASAP requirements.
- [x] Verify compatibility with BPNet/ChromBPNet requirements (Analysis in `docs/compatibility_report.md`).
    - [x] Implement `MixedSampler` (as `MultiSampler`) for peak/negative mixing.
    - [ ] Implement Flexible Output Formatting.
    - [ ] Implement Dynamic Sampling Hooks.
- [ ] Performance benchmarking (optional).

## Phase 6: Quality Assurance & Robustness
- [x] Implement robust unit tests for `Interval` core logic.
- [x] Implement mocked tests for `SignalExtractor` (independent of file system).
- [x] Implement error handling tests for `SequenceExtractor`.
- [x] Verify `MultiSampler.split_folds` logic with unit tests.
- [x] Verify `Jitter` randomness and bounds with statistical distribution tests.
- [x] Verify `Log1p` safety check for negative values.
- [x] Verify `SignalExtractor` missing data (NaN) and Infinity handling.
- [x] Ensure high coverage for core components.

## Phase 7: Advanced Features & Compatibility
- [ ] Implement Flexible Output Formatting (Dict/Tuple support beyond `(inputs, targets)`).
- [ ] Implement Dynamic Resampling Hooks (e.g. `on_epoch_start` callback for `MultiSampler`).
- [ ] Implement BigWig writing utility for predictions.
- [ ] Implement BAM/CRAM DataSource for on-the-fly pileup.
- [ ] Implement HDF5/Zarr backend for massive datasets.
- [ ] Implement support for High-Dimensional data (Hi-C matrices).
