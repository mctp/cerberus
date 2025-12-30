# Models

Cerberus provides implementations of standard deep learning architectures for genomic sequence modeling.

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

## GlobalProfileCNN (Baseline)

**Implementation**: `cerberus.models.GlobalProfileCNN`  
**Source**: `src/cerberus/models/baseline_gopher.py`

A baseline architecture based on the "Gopher" model (ResNet-style CNN with global pooling).

### Key Features
*   **Structure**: 3 Convolutional Blocks -> Global Dense Bottleneck -> Global Projection -> Final Conv.
*   **Output**: Predicts signal profiles directly (Single Head). Returns a tuple containing the profile tensor.
