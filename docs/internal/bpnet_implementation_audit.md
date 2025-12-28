# BPNet Implementation Audit

## Overview

This document presents an audit of the `cerberus` implementation of `BPNet` and `BPNetLoss` against the reference `chrombpnet-pytorch` implementation. The goal is to ensure 100% architectural and numerical equivalence.

## Findings

### 1. Architecture (`BPNet` Class)

The core backbone (Dilated Residual Networks) is largely consistent, but there are significant deviations regarding "Control Tracks" integration and initialization.

| Feature | `cerberus` (`src/cerberus/models/bpnet.py`) | `chrombpnet-pytorch` (Reference) | Match? |
| :--- | :--- | :--- | :--- |
| **Convolution Padding** | `valid` | `valid` | ✅ Yes |
| **Residual Block** | Post-Activation (`x + ReLU(Conv(x))`) | Post-Activation (`x + ReLU(Conv(x))`) | ✅ Yes |
| **Cropping Logic** | Explicit cropping in residual blocks | Explicit cropping in residual blocks | ✅ Yes |
| **Initial Conv** | `Conv1d` -> `ReLU` | `Conv1d` -> `ReLU` | ✅ Yes |
| **Control Tracks (Input)** | Not supported in `forward` or `__init__` | Supported (`x_ctl` arg) | ❌ **No** |
| **Profile Head** | `Conv1d(filters, out_channels)` | `Conv1d(filters + n_controls, out_channels)` | ❌ **No** |
| **Counts Head** | `GlobalAvgPool` -> `Linear` | `GlobalAvgPool` + `log1p(sum(controls))` -> `Linear` | ❌ **No** |
| **Weight Init** | PyTorch Default (Kaiming Uniform) | TF-Style (Xavier Uniform w/ Zero Bias) | ❌ **No** |
| **ChromBPNet Composition**| Not implemented (only base `BPNet`) | Implements `ChromBPNet` (Bias + Acc combination) | ❌ **No** |

**Critical Gaps:**
1.  **Control Tracks**: The reference implementation concatenates control tracks (e.g., bias tracks) at the Profile Head (channel concatenation) and the Counts Head (concatenation of log-total-counts). The current `cerberus` implementation lacks this mechanism entirely.
2.  **Counts Head Dimensionality**:
    *   **Reference (`chrombpnet-pytorch`)**: Hardcodes the Counts Head to output a **single scalar** (total counts) regardless of the number of profile output channels.
    *   **`bpnet-lite`**: Also hardcodes `self.linear` to output `1` scalar, reshaped to `(Batch, 1)`.
    *   **`cerberus`**: Outputs `n_output_channels` values (per-channel counts).
    *   **Impact**: This is a semantic deviation if `n_output_channels > 1`. Cerberus is more general (per-track counts), whereas the standard BPNet ecosystem assumes "Total Counts" prediction distributed to strands via the profile softmax.
3.  **Weight Initialization**: For strict numerical parity (especially if comparing untrained models or starting training), `chrombpnet-pytorch` explicitly re-initializes layers to match TensorFlow defaults (Xavier Uniform). `cerberus` uses PyTorch defaults.
    *   **Xavier (Glorot) Uniform** (Used by `chrombpnet`): Designed for symmetric activations (tanh/sigmoid). TensorFlow's default for `Conv1D`/`Dense`. Formula scales by $\frac{1}{n_{in} + n_{out}}$.
    *   **Kaiming (He) Uniform** (Used by `cerberus`/PyTorch): Designed for ReLU activations. PyTorch's default for `Conv1d`. Formula scales by $\frac{1}{n_{in}}$.
    *   **Impact**: While Kaiming is theoretically better for ReLU networks, achieving a 100% match with the reference implementation requires strictly following its initialization scheme (Xavier), as this affects initial training dynamics and is essential for reproducing untrained baselines.

### 2. Loss Function (`BPNetLoss`)

| Feature | `cerberus` (`BPNetLoss`) | `chrombpnet-pytorch` (`multinomial_nll`) | Match? |
| :--- | :--- | :--- | :--- |
| **Profile Loss Formula** | Exact Multinomial NLL (incl. factorial terms) | Exact Multinomial NLL (incl. factorial terms) | ✅ Yes |
| **Count Loss Formula** | MSE on `log1p(total_counts)` | MSE on `log1p(total_counts)` | ✅ Yes |
| **Channel Handling** | Supports multi-channel (Stranded) natively | Typically single-channel in wrapper | ✅ Yes |

The loss function implementation in `cerberus` appears mathematically equivalent to the reference, including the constant combinatorial terms often omitted in other implementations.

## Recommendations for 100% Match

To achieve 100% parity with `chrombpnet-pytorch`, the following changes are required in `cerberus`:

### 1. Update `BPNet` Class
*   **Add `n_control_tracks` parameter** to `__init__`.
*   **Update `forward` signature** to accept optional `control_tracks` (or `x_ctl`).
*   **Modify Profile Head**: If control tracks are present, concatenate them to the residual output before the final convolution.
*   **Modify Counts Head**: If control tracks are present, calculate their log-sum (`log1p(sum)`) and concatenate to the global average pooled features before the final linear layer.
*   **Add TF-Style Initialization**: Add a method or flag to re-initialize weights using `nn.init.xavier_uniform_` for weights and `nn.init.zeros_` for biases to match the reference.

### 2. Implement `ChromBPNet` Class
*   Create a container class that manages two `BPNet` instances (Bias and Accessibility).
*   Implement the combination logic:
    *   `Profile = Profile_Acc + Profile_Bias` (Logits addition)
    *   `Counts = log(exp(Counts_Acc) + exp(Counts_Bias))` (Log-Sum-Exp)

### 3. Verify Numerical Implementation
*   Create a unit test that loads exact weights from a `chrombpnet` checkpoint into the modified `cerberus` model and asserts outputs match on random input to within float precision ($10^{-6}$).

## Proposed Code Changes

### `src/cerberus/models/bpnet.py`

```python
# Add to __init__
self.n_control_tracks = n_control_tracks
# ...
# Update Profile Head definition
self.profile_conv = nn.Conv1d(
    filters + n_control_tracks, # Adjusted input channels
    self.n_output_channels,
    ...
)
# Update Counts Head definition
self.count_dense = nn.Linear(
    filters + (1 if n_control_tracks > 0 else 0), # Adjusted input features
    self.n_output_channels
)

# Add to forward
def forward(self, x, control_tracks=None):
    # ... (backbone) ...
    
    # Profile Head
    if control_tracks is not None:
        # Crop control tracks to match x spatial dim if needed
        # Concat
        x_profile_in = torch.cat([x, control_tracks], dim=1)
    else:
        x_profile_in = x
    profile_logits = self.profile_conv(x_profile_in)
    
    # Counts Head
    x_pooled = x.mean(dim=-1)
    if control_tracks is not None:
        ctl_sum = control_tracks.sum(dim=(1, 2)).unsqueeze(1).log1p() # check dims
        x_count_in = torch.cat([x_pooled, ctl_sum], dim=1)
    else:
        x_count_in = x_pooled
    log_counts = self.count_dense(x_count_in)
```
