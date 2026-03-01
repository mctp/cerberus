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

#### Baseline (chrombpnet-compatible)
BPNet has no normalization layers in its default configuration, so weight decay directly shrinks convolutional activations. Use plain Adam with no weight decay and a constant learning rate (matching chrombpnet-pytorch):

| Parameter | Value |
|---|---|
| Optimizer | `adam` |
| Learning rate | `1e-3` |
| Weight decay | `0` |
| Adam ε | `1e-7` |
| Scheduler | `default` (constant) |

#### Stable Training Mode
For improved training stability with AdamW and cosine LR scheduling, enable `weight_norm=True` and `activation="gelu"` via the `--stable` flag or model arguments. These changes are **fully compatible with DeepLIFT/DeepSHAP via captum**:

- **Weight normalization** (`weight_norm=True`): Reparameterizes Conv1d weights as `weight = weight_g / ‖weight_v‖ × weight_v`. This is applied at the parameter level, not the activation level — captum's DeepLIFT still treats Conv1d as a linear passthrough. The decoupling of magnitude and direction stabilises gradient norms across the deep dilated tower and enables effective AdamW weight decay.
- **GELU activation** (`activation="gelu"`): Smooth gradients at zero prevent dying neurons in deep dilated stacks and work better with the low-LR phase of cosine schedules. captum has a registered rule for GELU.

| Parameter | Stable Mode |
|---|---|
| `weight_norm` | `True` |
| `activation` | `"gelu"` |
| Optimizer | `adamw` |
| Learning rate | `1e-3` |
| Weight decay | `0.01` |
| Adam ε | `1e-7` |
| Scheduler | `cosine` |
| Warmup epochs | `5` |
| Min LR | `1e-5` |

From the training script:
```bash
python tools/train_bpnet.py --stable --bigwig signal.bw --peaks peaks.bed --output-dir models/my_model
```

From Python:
```python
model = BPNet(
    input_len=2114, output_len=1000, filters=64, n_dilated_layers=8,
    input_channels=["A", "C", "G", "T"], output_channels=["signal"],
    activation="gelu",
    weight_norm=True,
)
```

### Loss: BPNetLoss

`BPNetLoss` computes the sum of a multinomial profile loss and a counts MSE loss:

```
L = L_profile + alpha * L_counts
```

The `alpha` parameter balances the two terms. Because the multinomial NLL scales linearly with total signal depth N while the counts MSE scales as `(log N)²`, a fixed alpha cannot stay balanced as dataset depth varies.

**Recommended**: set `alpha="adaptive"` in `ModelConfig.loss_args`. Cerberus will compute `alpha = median_total_counts / 10` from the training set before the module is instantiated, automatically matching the dataset depth:

```python
model_config = {
    "name": "BPNet_AR",
    "model_cls": "cerberus.models.bpnet.BPNet",
    "loss_cls": "cerberus.models.bpnet.BPNetLoss",
    "loss_args": {"alpha": "adaptive"},  # resolved from training data at fit time
    "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
    "metrics_args": {},
    "model_args": {"n_dilated_layers": 8, "output_channels": ["signal"]},
}
```

See `docs/internal/adaptive_counts_loss_weight.md` for the full mathematical derivation.

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

## ConvNeXtDCNN (ASAP)

**Implementation**: `cerberus.models.ConvNeXtDCNN`
**Source**: `src/cerberus/models/asap.py`

A large-capacity architecture combining a ConvNeXtV2 stem with a Basenji-style dilated residual tower (DCNN). Designed for high-resolution profile prediction over a wide genomic context.

### Key Features
*   **ConvNeXtV2 Stem**: Modern CNN building block with LayerScale and Global Response Normalization (GRN) for initial feature extraction.
*   **MaxPool Downsampling**: 2× stride after the stem to halve the spatial resolution before the residual tower.
*   **Basenji Dilated Residual Tower**: Stack of residual blocks with exponentially increasing dilations (`rate_mult=1.5`), each consisting of a dilated convolution followed by a pointwise convolution with a zero-initialized BN gamma (enabling smooth residual learning from initialization).
*   **Configurable Depth and Width**: `residual_blocks` (default 11), `filters0` (default 256), `filters1` (default 128).
*   **Output**: Returns `ProfileLogRates(log_rates=...)` consumed by `ProfilePoissonNLLLoss`.
*   **I/O**: Default 2048bp → 512 bins at 4bp resolution (2048bp coverage). Output bins = `input_len / output_bin_size`.

### Recommended Training Settings

| Parameter | Value |
|---|---|
| Optimizer | `adamw` |
| Learning rate | `1e-3` |
| Weight decay | `0.01` |
| Scheduler | `cosine` with 10 warmup epochs |
| Min LR | `1e-5` |
| Loss | `ProfilePoissonNLLLoss(log_input=True, log1p_targets=True)` |

### Loss: ProfilePoissonNLLLoss

ASAP outputs log-rates and is trained with Poisson NLL directly on the profile (no separate count head). The data config sets `log_transform=True` so targets are log1p-transformed before computing the loss:

```python
model_config = {
    "name": "ConvNeXtDCNN",
    "model_cls": "cerberus.models.asap.ConvNeXtDCNN",
    "loss_cls": "cerberus.loss.ProfilePoissonNLLLoss",
    "loss_args": {"log_input": True, "log1p_targets": True},
    "metrics_cls": "cerberus.metrics.DefaultMetricCollection",
    "metrics_args": {},
    "model_args": {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "residual_blocks": 11,
        "filters0": 256,
        "filters1": 128,
        "dropout": 0.3,
    },
}
```

### Usage
```python
from cerberus.models.asap import ConvNeXtDCNN

# Default: 2048bp -> 512 bins (2048bp at 4bp resolution)
model = ConvNeXtDCNN(
    input_len=2048,
    output_len=2048,
    output_bin_size=4,
    input_channels=["A", "C", "G", "T"],
    output_channels=["signal"],
)
# Returns ProfileLogRates(log_rates=...)  shape: (batch, output_channels, 512)
```

## GlobalProfileCNN (Gopher)

**Implementation**: `cerberus.models.GlobalProfileCNN`
**Source**: `src/cerberus/models/gopher.py`

A baseline architecture corresponding to the "Baseline CNN" from the Gopher manuscript (`conv_profile_all_base`). It uses a standard convolutional body followed by a global dense bottleneck that compresses the full sequence into a fixed-size representation, then projects back to a spatial grid.

### Key Features
*   **3 Convolutional Blocks**: Conv1D → BatchNorm → ReLU → MaxPool (pooling factors: 8×, 4×, 4× = 128× total), with increasing filter counts (192, 256, 512).
*   **Global Dense Bottleneck**: Flattens the pooled feature map into a single vector (Dense → BN → ReLU → Dropout).
*   **Global Projection**: Projects the bottleneck to `output_bins × bottleneck_channels`, then reshapes back to a spatial feature map.
*   **Final Conv Head**: Conv1D(256) → Conv1D(output_channels) producing log-rate logits.
*   **Output**: Returns `ProfileLogRates(log_rates=...)` — log rates consumed by `ProfilePoissonNLLLoss`.
*   **I/O**: 2048bp → 1024bp at 4bp resolution (256 prediction bins). Input length must be divisible by 128.

### Recommended Training Settings

| Parameter | Value |
|---|---|
| Optimizer | `adamw` |
| Learning rate | `1e-3` |
| Weight decay | `0.01` |
| Scheduler | `cosine` with 10 warmup epochs |
| Min LR | `1e-5` |
| Loss | `ProfilePoissonNLLLoss(log_input=True, log1p_targets=True)` |

### Loss: ProfilePoissonNLLLoss

Gopher outputs log-rates and is trained with Poisson NLL directly on the profile (no separate count head). The data config should set `log_transform=True` so targets are log1p-transformed before computing the loss:

```python
model_config = {
    "name": "GlobalProfileCNN",
    "model_cls": "cerberus.models.gopher.GlobalProfileCNN",
    "loss_cls": "cerberus.loss.ProfilePoissonNLLLoss",
    "loss_args": {"log_input": True, "log1p_targets": True},
    "metrics_cls": "cerberus.metrics.DefaultMetricCollection",
    "metrics_args": {},
    "model_args": {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "bottleneck_channels": 8,
    },
}
```

### Usage
```python
from cerberus.models import GlobalProfileCNN

# Default: 2048bp -> 256 bins (1024bp at 4bp resolution)
model = GlobalProfileCNN(
    input_len=2048,
    output_len=1024,
    output_bin_size=4,
    input_channels=["A", "C", "G", "T"],
    output_channels=["signal"],
    bottleneck_channels=8,
)
# Returns ProfileLogRates(log_rates=...)  shape: (batch, output_channels, 256)
```
