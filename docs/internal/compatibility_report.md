# BPNet / ChromBPNet Compatibility Analysis

## 1. Data Loading Strategy

### BPNet (`bpnet-lite`)
- **Strategy**: Eagerly loads all sequences and signals into memory tensors.
- **Sampling**:
  - Training batches mix peaks and negatives.
  - Peaks are shuffled.
  - Negatives are randomly sampled (with replacement/independent prob) to achieve a target `negative_ratio` (e.g., 0.1).
  - Effectively, the dataset size is `n_peaks * (1 + negative_ratio)`.
- **Augmentation**:
  - **Jitter**: Input windows are wider than model input. Random cropping is applied.
  - **Reverse Complement**: Applied randomly (p=0.5).

### ChromBPNet (`chrombpnet-pytorch`)
- **Strategy**: Eager loading (or cached H5).
- **Sampling**:
  - Explicitly constructs a training set for each epoch by subsampling negatives to a target ratio.
  - Resamples negatives at the start of each epoch (`shuffle_at_epoch_start`).
- **Augmentation**:
  - **Jitter**: Random cropping from a wider window.
  - **Reverse Complement**: Applied.

### Cerberus Current State
- **Strategy**: Lazy loading (default) or In-Memory (supported via `InMemory*Extractor`).
- **Sampling**: `IntervalSampler` provides a static list of intervals from a BED file. `SlidingWindowSampler` generates static windows.
- **Augmentation**: `Jitter` and `ReverseComplement` transforms are implemented and compatible.

### Gaps & Recommendations
1.  **Mixed/Ratio Sampling**: Cerberus lacks a native way to mix "peaks" and "negatives" with a specific ratio, especially with "resampling negatives" behavior.
    - **Recommendation**: Implement `MixedSampler` or update `StratifiedSampler` to accept multiple source samplers (e.g., Peaks, Negatives) and a mixing strategy (e.g., "1.0 of A, 0.1 of B, resample B every epoch").

2.  **Resampling Hook**: ChromBPNet resamples negatives every epoch. Standard PyTorch Datasets are static.
    - **Recommendation**: The `Sampler` or `CerberusDataset` needs a `on_epoch_start()` or similar hook, or the `Sampler` needs to be dynamic (re-shuffling internal indices on `__iter__`).

## 2. Input/Output Format

### BPNet
- **Input**: `(sequence, [control_signal])`. Sequence is One-Hot `(4, L)`.
- **Output**: `(predicted_signal)`. Shape `(Tasks, L)`.
- **Format**: Tuple of tensors.

### ChromBPNet
- **Input/Output**: Dictionary `{'onehot_seq': ..., 'profile': ...}`.

### Cerberus Current State
- **Format**: `__getitem__` returns a tuple `(inputs, targets)`.
  - `inputs`: `torch.cat([seq, input_signals])`. Shape `(4+C_in, L)`.
  - `targets`: `target_signals`. Shape `(C_out, L)`.

### Gaps & Recommendations
1.  **Flexible Output**: Models may expect Dicts or specific Tuple structures (e.g., separated Sequence and Control). Cerberus concatenates Sequence and Control.
    - **Recommendation**: Add an `output_format` option to `CerberusDataset` or a `FormattingTransform`.
    - BPNet models might require separating Sequence (4 channels) from Control tracks. Cerberus merging them into `inputs` might require a model adapter layer (Splitting the input tensor) or a change in Dataloader output.

## 3. Configuration

- BPNet/ChromBPNet use `in_window` (model input), `out_window` (prediction), and `max_jitter`.
- Cerberus `DataConfig` supports these exact parameters.
- **Compatibility**: High.

## 4. Summary of Tasks

To fully support BPNet/ChromBPNet workflows, Cerberus needs:
1.  **`MixedSampler`**: To handle Peak + Negative mixing with ratios.
2.  **Output Formatting**: To support Dict outputs or split Tuple outputs (Seq, Control, Target).
3.  **Dynamic Sampling**: To support "new negatives every epoch".
