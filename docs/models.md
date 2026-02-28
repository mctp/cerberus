# Models

Cerberus provides implementations of standard deep learning architectures for genomic sequence modeling.

## Pomeranian

**Implementation**: `cerberus.models.Pomeranian` (Default) / `PomeranianK5`  
**Source**: `src/cerberus/models/pomeranian.py`

A lightweight, efficient model (~150k params) mirroring BPNet's valid-padding paradigm but utilizing modern components (ConvNeXtV2 Stem, PGC Body).

### Key Features
*   **Valid Padding**: Ensures no zero-padding artifacts; strictly maps input to output geometrically (2112bp -> 1024bp).
*   **Factorized Stem**: Uses a 2-layer stem (`[11, 11]`) with dense and depthwise convolutions for efficient initial feature extraction.
*   **PGC Body**: Stack of 8 Pointwise-Gated Convolution (PGC) blocks.
*   **Variants**:
    *   **Pomeranian** (Default): Uses Kernel=9 and Dilation=64 (max). Best for hardware efficiency and modeling large motifs.
    *   **PomeranianK5**: Uses Kernel=5 and Dilation=128 (max). Traditional small-kernel approach.

## BPNet

**Implementation**: `cerberus.models.BPNet` (Standard) / `BPNet1024`
**Source**: `src/cerberus/models/bpnet.py`

An implementation of the BPNet architecture (Avsec et al., 2021) following the "Consensus" specification (Post-Activation Residual Blocks). It is designed for base-resolution profile prediction. Weights are initialized with Xavier uniform (Glorot) to match the TensorFlow/Keras defaults used by the original BPNet and chrombpnet-pytorch.

### Key Features
*   **Valid Padding**: Uses `'valid'` padding throughout; excess length is center-cropped at the profile head.
*   **Dual Heads**:
    *   **Profile Head**: Conv1D over the tower output → logits for the profile distribution.
    *   **Count Head**: Global average pool → Linear → log(total counts).
*   **Dilated Residual Tower**: 8 blocks with exponentially increasing dilations (2¹..2⁸).
*   **Binning**: Optional output binning via average pooling (`output_bin_size > 1`).
*   **Variants**:
    *   **BPNet** (Default): Canonical dimensions (2114bp → 1000bp), 64 filters, `profile_kernel_size=75`.
    *   **BPNet1024**: Tuned for 2112bp → 1024bp with no center-cropping; 77 filters, `profile_kernel_size=49`. Receptive-field shrinkage is exactly 1088bp (20 + 1020 + 48).

### Recommended Training Settings
BPNet has no normalization layers, so weight decay directly shrinks convolutional activations. Use plain Adam with no weight decay and a constant learning rate (matching chrombpnet-pytorch):

| Parameter | Value |
|---|---|
| Optimizer | `adam` |
| Learning rate | `1e-3` |
| Weight decay | `0` |
| Adam ε | `1e-7` |
| Scheduler | `default` (constant) |

### Usage
```python
from cerberus.models import BPNet, BPNet1024

# Standard BPNet: 2114bp -> 1000bp
model = BPNet(
    input_len=2114,
    output_len=1000,
    filters=64,
    n_dilated_layers=8,
    input_channels=["A", "C", "G", "T"],
    output_channels=["signal"],
)

# BPNet1024: 2112bp -> 1024bp (no center-cropping, comparable to Pomeranian)
model = BPNet1024(
    input_channels=["A", "C", "G", "T"],
    output_channels=["signal"],
)
# Both return ProfileCountOutput(logits=..., log_counts=...)
```

## GemiNet

**Implementation**: `cerberus.models.GemiNet`
**Source**: `src/cerberus/models/geminet.py`

A modern profile prediction architecture using Pointwise-Gated Convolutions (PGC) for efficient long-range context modeling.

### Key Features
*   **PGC Blocks**: Uses depthwise-separable convolutions with gating mechanisms.
*   **Efficiency**: Higher throughput than BPNet with comparable or better performance.
*   **Output**: Dual-head (Profile + Counts) similar to BPNet.

## LyraNet

**Implementation**: `cerberus.models.LyraNet`
**Source**: `src/cerberus/models/lyra.py`

A hybrid architecture combining PGC blocks for local context and S4D layers (State Space Models) for efficient global context modeling.

### Key Features
*   **Hybrid Stem/Body**: Uses Convolutional Stem, PGC layers for local interactions, and S4D layers for global sequence modeling.
*   **State Space Models (S4D)**: Efficiently models very long-range dependencies.
*   **Variants**: `LyraNet`, `LyraNetMedium`, `LyraNetLarge`, `LyraNetExtraLarge` scaling from ~140k to ~5.4M parameters.
*   **Output**: Dual-head (Profile + Counts).

## ConvNeXtDCNN (ASAP)

**Implementation**: `cerberus.models.ConvNeXtDCNN`
**Source**: `src/cerberus/models/asap.py`

An architecture leveraging ConvNeXtV2 blocks for hierarchical feature extraction.

### Key Features
*   **ConvNeXtV2 Blocks**: Modern CNN building blocks with LayerScale and GRN (Global Response Normalization).
*   **Output**: Predicts log-rates (ProfileLogRates).

## GlobalProfileCNN (Baseline)

**Implementation**: `cerberus.models.GlobalProfileCNN`  
**Source**: `src/cerberus/models/gopher.py`

A baseline architecture based on the "Gopher" model (ResNet-style CNN with global pooling).

### Key Features
*   **Structure**: 3 Convolutional Blocks -> Global Dense Bottleneck -> Global Projection -> Final Conv.
*   **Output**: Predicts signal profiles directly (Single Head). Returns a tuple containing the profile tensor.
