# Implementation Plan: `conv_profile_all_base` in Cerberus

## 1. Overview
This document outlines the plan to implement the `conv_profile_all_base` architecture from Gopher into Cerberus. This model serves as a **Baseline CNN** for profile prediction, featuring a standard convolutional body, a dense bottleneck/rescaling section, and a shared convolutional head.

## 2. Architecture Specification (TF to PyTorch Mapping)

### Input
- **TF**: `(Batch, Length, 4)`
- **Cerberus**: `(Batch, 4, Length)`
- **Action**: Implement PyTorch model to accept `(N, 4, L)`.

### Layers

| Layer | TF Specification | PyTorch Implementation Plan | Notes |
| :--- | :--- | :--- | :--- |
| **Conv 1** | `Conv1D(192, 19, padding='same')` | `nn.Conv1d(4, 192, 19, padding='same')` | `padding='same'` requires PyTorch 1.10+. |
| **Block 1** | `BN` -> `Act` -> `MaxPool(8)` -> `Drop(0.1)` | `nn.BatchNorm1d(192)` -> `Act` -> `nn.MaxPool1d(8)` -> `Dropout(0.1)` | `MaxPool` in TF with padding='valid' drops last if < 8. PyTorch default is similar (`ceil_mode=False`). |
| **Conv 2** | `Conv1D(256, 7, padding='same')` | `nn.Conv1d(192, 256, 7, padding='same')` | |
| **Block 2** | `BN` -> `Act` -> `MaxPool(4)` -> `Drop(0.1)` | `nn.BatchNorm1d` -> `Act` -> `nn.MaxPool1d(4)` -> `Dropout` | |
| **Conv 3** | `Conv1D(512, 7, padding='same')` | `nn.Conv1d(256, 512, 7, padding='same')` | |
| **Block 3** | `BN` -> `Act` -> `MaxPool(4)` -> `Drop(0.2)` | `nn.BatchNorm1d` -> `Act` -> `nn.MaxPool1d(4)` -> `Dropout` | |
| **Flatten** | `Flatten()` | `nn.Flatten()` | **Warning**: TF flattens `(N, L, C)`. PT flattens `(N, C, L)`. Weights are not compatible without transpose, but architecture logic holds. |
| **Dense 1** | `Dense(256)` -> `BN` -> `Act` -> `Drop` | `nn.Linear(In_Feats, 256)` -> `nn.BatchNorm1d(256)` -> ... | `In_Feats` must be calculated dynamically (dummy pass). |
| **Dense 2** | `Dense(Out_Len * Bottleneck)` | `nn.Linear(256, Out_Len * Bottleneck)` | `Out_Len = output_shape[0]`. `Bottleneck` default 8. |
| **Reshape** | `Reshape([Out_Len, Bottleneck])` | `View(Batch, Bottleneck, Out_Len)` | **Critical Change**: TF reshapes to `(N, L, C)`. We reshape to `(N, C, L)` to use `Conv1d` next. |
| **Conv 4** | `Conv1D(256, 7, padding='same')` | `nn.Conv1d(Bottleneck, 256, 7, padding='same')` | Operating on the reshaped "Bottleneck" channels. |
| **Head** | `Dense(num_tasks, activation='softplus')` | `nn.Conv1d(256, num_tasks, 1)` + `Softplus` | In TF, `Dense` on 3D input applies to last dim. Equivalent to `Conv1d(k=1)` in PT. |

## 3. Loss Functions and Outputs

### Output Shape
- **TF Output**: `(Batch, Out_Len, Num_Tasks)` (Channels Last).
- **Cerberus Output**: `(Batch, Num_Tasks, Out_Len)` (Channels First).
- **Plan**: The proposed PyTorch architecture naturally produces `(N, Num_Tasks, Out_Len)` because we reshape to `(N, Bottleneck, Out_Len)` and then use `Conv1d` which preserves the length dimension and changes channels to `Num_Tasks`.

### Loss Compatibility
- **Model Output**: `Softplus` activation implies positive values (Rates/Counts).
- **Cerberus Default**: `PoissonNLLLoss`.
    - `nn.PoissonNLLLoss(log_input=True)` (Default PyTorch) expects **Log-Rates**.
    - `nn.PoissonNLLLoss(log_input=False)` expects **Rates**.
- **Strategy**:
    1.  **Exact Port**: Keep `Softplus` at end. Use `PoissonNLLLoss(log_input=False)`.
    2.  **Optimization Friendly**: Remove `Softplus`. Output raw logits. Use `PoissonNLLLoss(log_input=True)`.
    - **Recommendation**: Use **Strategy 2** (Logits + LogInput=True) for better numerical stability, unless exact reproduction of Gopher weights is required. If implementing from scratch, Strategy 2 is preferred.
    - **Policy**: **We explicitly favor the more numerically stable solution (Strategy 2) for the Cerberus implementation.**

## 4. Technical Incompatibilities & Challenges

### 4.1. Flattening and Spatial Information
- **Challenge**: The model uses `Flatten()` followed by `Dense()`. This destroys spatial structure, but the subsequent `Reshape` attempts to "reconstruct" a spatial grid of size `Out_Len`.
- **Implication**: The network learns a global mapping from the input features to the output grid.
- **PyTorch Difference**: Because PyTorch is `(N, C, L)`, flattening produces a vector where all timepoints for Channel 0 come first, then Channel 1, etc. In TF `(N, L, C)`, it interleaves channels.
- **Resolution**: Since the Dense layer is fully connected, it can learn any permutation. As long as we train from scratch, this difference doesn't matter for correctness.

### 4.2. Padding 'Same'
- **Challenge**: `padding='same'` in TF is dynamic. In PyTorch `Conv1d`, `padding='same'` requires stride=1 and odd kernel size.
- **Status**: All Conv layers in this model use `stride=1` and odd kernels (19, 7).
- **Resolution**: `nn.Conv1d(..., padding='same')` will work perfectly in modern PyTorch.

### 4.3. Pooling Arithmetic & The "128 Constraint"
- **Origin**: The architecture applies three successive MaxPooling operations with sizes 8, 4, and 4.
    - `Input Length` -> `MaxPool(8)` -> `L/8`
    - `L/8` -> `MaxPool(4)` -> `L/32`
    - `L/32` -> `MaxPool(4)` -> `L/128`
- **Total Reduction**: The sequence length is reduced by a factor of $8 \times 4 \times 4 = 128$.
- **The Constraint**: For this model to function correctly without dropping data or creating misalignment at the Flatten layer:
    1.  The `input_len` must be **divisible by 128**.
    2.  If `input_len` is *not* divisible by 128 (e.g., 1000), the final pooled length will vary depending on rounding behavior (floor vs ceil), and pixels at the edge of the sequence will be effectively ignored or padded.
    3.  More importantly, the `Flatten` layer produces a fixed-size vector. If `input_len` changes (even by 128), the Flatten size changes, and the subsequent `Dense` layer (with fixed weight matrix size) will fail.
- **Implication**: This model is strictly bound to a fixed input length (unlike Fully Convolutional Networks). The input length chosen at initialization (e.g. 2048) must be strictly adhered to during inference.

### 4.4. Batch Norm
- **TF**: `momentum=0.99`, `epsilon=0.001`.
- **PyTorch**: `momentum=0.1` (equivalent to TF 0.9), `epsilon=1e-5`.
- **Resolution**: Adjust PyTorch BN parameters to match if strictly reproducing. `momentum=0.1` in PyTorch corresponds to `0.9` in TF notation (decay vs smooth).

## 5. Alternative Naming

The name `ConvProfileAllBase` ("All" tasks, "Base" resolution) is somewhat opaque. Better alternatives for the Cerberus codebase:

1.  **`GlobalProfileCNN`**: Highlights the key architectural feature—using a global Dense bottleneck to project the entire input sequence to the output profile, rather than using local convolutions throughout.
2.  **`StandardProfileCNN`**: Indicates this is a standard/baseline CNN architecture for profile prediction.
3.  **`BaselineProfileCNN`**: Explicitly marks it as the baseline model.
4.  **`FixedInputProfileCNN`**: Highlights the constraint that input size is fixed due to the dense layers.

**Recommendation**: **`GlobalProfileCNN`** accurately describes the mechanism (Global projection) which distinguishes it from BPNet/Basenji (Local/Fully Convolutional).

## 6. Implementation Steps

1.  Create `src/cerberus/models/global_profile_cnn.py` (using recommended name).
2.  Define class `GlobalProfileCNN(nn.Module)`.
3.  Implement `__init__`:
    - Calculate `nr_bins` from `output_len`.
    - Define Conv blocks.
    - Implement "dummy forward pass" to calculate Flatten size.
    - Define Dense bottleneck and rescaling layers.
    - Define Final Conv and Head.
4.  Implement `forward`:
    - Apply Convs/Pools.
    - Flatten.
    - Apply Dense layers.
    - `View` (Reshape) to `(N, Bottleneck, Out_Len)`.
    - Apply Final Conv.
    - Apply Head (Conv1x1).
    - Return Logits (**Do NOT apply Softplus**, to adhere to the numerically stable strategy).
