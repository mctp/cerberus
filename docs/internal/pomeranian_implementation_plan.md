# Implementation Plan - Pomeranian Model

## Overview
We will implement `Pomeranian`, a new model in `src/cerberus/models/pomeranian.py`. This model combines the architectural advancements of `GemiNet` (ConvNeXtV2 stem, PGC tower) with the `padding='valid'` paradigm of `BPNet`. It is designed to be a lightweight model (~150k parameters) suitable for input sequences of ~2kb yielding output predictions of ~1kb.

## Key Changes
- **New Model File**: `src/cerberus/models/pomeranian.py`
- **New Classes**:
    - `Pomeranian`: Main model class (BPNet-compatible defaults: 2114/1000).
    - `Pomeranian1k`: Aligned subclass (defaults: 2112/1024).
    - `_ConvNeXtV2BlockValid`: Adapted `ConvNeXtV2Block` with valid padding and residual cropping.
    - `_PGCBlockValid`: Adapted `PGCBlock` with valid padding and residual cropping.
- **Configuration**: Defaults set to `filters=64` and `n_dilated_layers=8` to meet the ~150k parameter target.

## Implementation Steps
1.  **Define Adapted Blocks**:
    - Implement `_ConvNeXtV2BlockValid` in `pomeranian.py` (or `layers.py` if preferred, but local keeps `layers.py` clean).
    - It will use `padding='valid'` for the depthwise convolution.
    - It will crop the residual path (`x_`) to match the output size before addition.
    - Implement `_PGCBlockValid` similarly: `padding='valid'` for the dilated convolution, with center cropping for the residual connection.

2.  **Implement Pomeranian Class**:
    - **Stem**: `_ConvNeXtV2BlockValid` with kernel size 21 (default).
    - **Body**: Stack of 8 `_PGCBlockValid` layers with exponentially increasing dilation ($2^1$ to $2^8$).
    - **Profile Head**: Decoupled Head (Strategy A).
        - `Conv1d(filters, filters, 1) -> GELU -> Conv1d(filters, out, 75, valid)`.
    - **Count Head**: MLP Head (Strategy C).
        - `GlobalAvgPool -> Linear(64->32) -> GELU -> Linear(32->1)`.
    - **Forward Pass**:
        - Compute output size based on input size and total shrinkage (1114 bp for standard config).
        - Verify input length is sufficient.
        - Return `ProfileCountOutput`.

3.  **Parameter Verification**:
    - Verify that `filters=64`, `layers=8` results in ~150k parameters.
    - (Estimate: ~142k params, as calculated).

4.  **Tests**:
    - Create `tests/test_pomeranian.py` mirroring `tests/test_geminet2.py` and `tests/test_bpnet_implementation.py`.
    - Verify output shapes and parameter counts.

## Technical Considerations
- **Valid Padding & Residuals**: The core challenge is handling residual connections when the spatial dimension shrinks. We will implement center-cropping on the residual branch, matching the logic in `BPNet`'s `DilatedResidualBlock`.
- **Receptive Field**: With 8 layers (dilations 2-256), stem kernel 21, and profile kernel 75, the total receptive field shrinkage is 1114 bp.
    - Input 2048 -> Output 934 (approx 1kb).
    - Input 2114 -> Output 1000.
    - This aligns with standard BPNet dimensions.
    - **Heads Strategy**:
    - **Profile Head (Decoupled)**: Standard BPNet uses a single large linear convolution (k=75). We enhance this by inserting a pointwise convolution (1x1) followed by GELU activation before the final spatial convolution. This structure (`Conv1d(1x1) -> GELU -> Conv1d(Valid)`) decouples channel mixing from spatial aggregation. The pointwise layer mixes features per-position, and the activation adds non-linearity, allowing the model to refine the representation before the final shape prediction. This improves expressivity with minimal parameter cost compared to increasing the kernel size or depth of the main tower.
    - **Count Head (MLP)**: Standard BPNet uses a single linear layer. We replace this with a 2-layer MLP (`64->32->1`) with GELU activation. This captures non-linear relationships between the pooled features and total counts, which simple linear regression might miss, for a negligible cost (~2k extra params).

## Success Criteria
- [x] `Pomeranian` model is importable and runnable.
- [x] Parameter count is approx 150k.
- [x] Input/Output delta is correct (e.g., input length - output length = 1114).
- [x] `padding='valid'` logic works correctly without shape mismatch errors.
