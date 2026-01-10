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

**Implementation**: `cerberus.models.BPNet`  
**Source**: `src/cerberus/models/bpnet.py`

An implementation of the BPNet architecture (Avsec et al., 2021) following the "Consensus" specification (Post-Activation Residual Blocks). It is designed for base-resolution profile prediction.

### Key Features
*   **Dual Heads**:
    *   **Profile Head**: Predicts the shape of the signal distribution (logits).
    *   **Count Head**: Predicts the total read count (log-transformed).
*   **Dilated Convolutions**: Uses exponentially increasing dilation rates to capture long-range interactions.
*   **Padding**: Supports `'same'` padding to maintain input resolution.
*   **Binning**: Optional output binning via average pooling.

### Usage
```python
from cerberus.models import BPNet

model = BPNet(
    input_len=2114,
    output_len=1000,
    filters=64,
    n_dilated_layers=9
)
# Returns (profile_logits, log_counts)
```

## GemiNet

**Implementation**: `cerberus.models.GemiNet`
**Source**: `src/cerberus/models/geminet.py`

A modern profile prediction architecture using Pointwise-Gated Convolutions (PGC) for efficient long-range context modeling.

### Key Features
*   **PGC Blocks**: Uses depthwise-separable convolutions with gating mechanisms.
*   **Efficiency**: Higher throughput than BPNet with comparable or better performance.
*   **Output**: Dual-head (Profile + Counts) similar to BPNet.

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
