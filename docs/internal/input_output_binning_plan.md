# Input/Output Binning and 3D Tensor Transformation Plan

## 1. Overview
This document outlines the plan to refactor Cerberus to support decoupled `input_bin_size` and `output_bin_size`, moving away from the single global `bin_size` parameter. A key component of this refactor is the transformation of input sequences into a 3D tensor format (`Channels`, `Length // Bin`, `Bin_Size`) and the alignment of input signals (control tracks) to this structure.

This change aims to enable more efficient modeling of long-range interactions by reducing the resolution of the "length" dimension while preserving high-resolution sequence information within the "bin" dimension.

## 2. Motivation

### Scientific Motivation
*   **Efficiency**: Genomic models often operate on very long sequences (e.g., 100kb+). processing these at single-base resolution is computationally expensive. Operating on binned representations (e.g., 128bp bins) significantly reduces the sequence length dimension.
*   **Information Preservation**: While binning output signals (like ChIP-seq coverage) via averaging is standard, binning DNA sequence is non-trivial. Averaging one-hot encoded DNA loses the specific motif information. Reshaping the sequence into a `(L/B, B)` structure preserves the exact local sequence within each bin, allowing the model to learn local motifs (inside the bin) and long-range interactions (across bins) simultaneously.
*   **Input Signal Handling**: Control tracks (e.g., bias, input DNA) are often smoother or used for normalization. Pooling these signals to match the bin resolution is often scientifically valid and reduces noise, provided they are aligned correctly with the sequence tensor.

### Technical Motivation
*   **Decoupled Resolution**: Current implementation forces inputs and outputs to have the same "binning" concept (though currently inputs are unbinned and outputs are binned in the model head). Explicit `input_bin_size` allows the model body to operate entirely at a lower resolution.
*   **Model Flexibility**: Supporting 3D inputs allows for the use of `Conv2d` architectures or specialized `Conv1d` approaches on flattened channels, opening up design space for hierarchical models.

## 3. Configuration Changes

We will modify `src/cerberus/config.py` to support the new binning parameters.

### `DataConfig` Updates
*   **Deprecate**: `bin_size` (int).
*   **Add**:
    *   `input_bin_size` (int, default=1): The bin size applied to inputs.
    *   `output_bin_size` (int, default=1): The bin size applied to targets.
*   **Validation**:
    *   `input_len` must be divisible by `input_bin_size`.
    *   `output_len` must be divisible by `output_bin_size`.

## 4. Data Transformation Strategy

### 4.1. Sequence Transformation (Lossless)
Instead of a 2D tensor `(4, Length)`, the sequence will be reshaped into a 3D tensor.

*   **Original Shape**: `(4, Length)`
*   **New Shape**: `(4, Length // input_bin_size, input_bin_size)`
    *   **Dimension 0**: Channels (A, C, G, T)
    *   **Dimension 1**: Binned Length (Time steps)
    *   **Dimension 2**: Bin Size (Local resolution)

This transformation is lossless. It changes the tensor rank but retains all boolean flags of the one-hot encoding.

### 4.2. Input Signal Transformation (Lossy + Expansion)
Input signals (e.g., control tracks) usually have shape `(Channels, Length)`. To combine them with the 3D sequence tensor, we must address the shape mismatch.

**Transformation Steps:**
1.  **Binning (Pooling)**: Apply `Mean` or `Max` pooling to reduce resolution.
    *   `Original`: `(C_sig, Length)`
    *   `Pooled`: `(C_sig, Length // input_bin_size)`
2.  **Expansion (Broadcasting)**: To concatenate with the sequence tensor, we treat the pooled value as constant across the bin.
    *   `Unsqueeze`: `(C_sig, Length // input_bin_size, 1)`
    *   `Repeat`: `(C_sig, Length // input_bin_size, input_bin_size)`

### 4.3. Concatenation
With both sequence and signals in `(C, L_new, B)` format, they can be concatenated along the channel dimension.

*   **Resulting Input Tensor**: `(4 + C_sig, Length // input_bin_size, input_bin_size)`

## 5. Convolutional Architectures: 2D vs 3D Inputs

Adapting models to consume this 3D input tensor requires changing the first layer strategy.

### Option A: 2D Convolutions (`nn.Conv2d`)
Treat the input as an image with `Height = Length // Bin` and `Width = Bin`.

*   **Input**: `(Batch, Channels, L_new, Bin)`
*   **Kernel**: `(Kernel_Size, Bin)`
    *   A kernel height of `K` and width of `Bin` spans `K` bins and the *entire* bin width.
    *   This effectively learns features based on the full content of `K` bins.
*   **Pros**: Explicitly models the bin structure. Can use standard 2D pooling/stride logic.
*   **Cons**: `Conv2d` is generally slower than `Conv1d` for the same number of effective operations if not optimized carefully.

### Option B: Flattened 1D Convolutions (`nn.Conv1d`)
Flatten the "Bin" dimension into the "Channel" dimension.

*   **Transformation**:
    *   `Input`: `(Batch, Channels, L_new, Bin)`
    *   `Flatten`: `(Batch, Channels * Bin, L_new)`
*   **Layer**: `nn.Conv1d(in_channels = (4 + C_sig) * input_bin_size, ...)`
*   **Pros**: Uses highly optimized 1D convolution implementations. Conceptually simpler for "sequence" modeling.
*   **Cons**: Greatly increases the number of input channels, which increases parameter count in the first layer `(Channels * Bin * Kernel_Size * Out_Channels)`.

### Recommendation
For the initial implementation, **Option B (Flattened 1D)** is likely easier to integrate with existing `VanillaCNN` architectures, as it only changes the `input_channels` definition. However, **Option A** aligns better with the "3D tensor" concept and allows for more structured constraints (e.g., sharing weights across positions within a bin if we use `Kernel_Width < Bin`).

## 6. Implementation Plan

### Step 1: Config Updates
*   Modify `src/cerberus/config.py` to add `input_bin_size`, `output_bin_size` and remove `bin_size`.
*   Update validation logic.

### Step 2: Transform Implementation
*   Modify `src/cerberus/transform.py`.
*   Create `ReshapeBin`: Handles the sequence reshaping.
*   Create `BroadcastBin`: Handles signal pooling and expansion.
*   Update `create_default_transforms` to wire these up based on config.

### Step 3: Dataset Integration
*   Verify `src/cerberus/dataset.py` handles the concatenation correctly. Since transforms return modified tensors, if we apply transforms *after* concatenation, we need a transform that handles the concatenated `(4+C, L)` tensor, splits it, processes parts, and recombines.
*   **Alternative**: Apply transforms *before* concatenation?
    *   Current architecture applies transforms to `inputs` (concatenated) and `targets`.
    *   We may need a specialized `InputTransform` that knows which channels are sequence vs signal, OR rely on the transform to operate on the whole block.
    *   **Proposed**: A transform that takes the `(4+C, L)` input, reshapes the first 4 channels (Sequence), pools and expands the rest (Signals), and returns the `(4+C, L/B, B)` tensor.

### Step 4: Model Updates
*   Update `VanillaCNN` to calculate `input_channels` dynamically or accept a flattened dimension size if using Option B.
*   Ensure the regression head aligns with `output_bin_size`.
