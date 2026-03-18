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
*   **Depthwise-Only Mode** (`expansion=0`): When `expansion=0`, the PGC tower uses depthwise-only blocks — no pointwise projections or gating. Each block becomes `RMSNorm → Depthwise Conv → Dropout → Residual`. Geometry (input/output lengths) is unchanged. Useful for lightweight baselines and ablation studies testing whether inter-channel mixing matters.
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

## BiasNet

**Implementation**: `cerberus.models.BiasNet`
**Source**: `src/cerberus/models/biasnet.py`

A lightweight bias model for Tn5 enzymatic sequence preference. Uses a plain Conv1d + ReLU stack with residual connections and valid padding. Designed for ChromBPNet-style bias factorization in the Dalmatian model.

All operations are `nn.Module`-based (no `F.relu`) for full DeepLIFT/DeepSHAP compatibility via captum.

### Architecture

- **Stem**: 2-layer `[11, 11]` Conv1d + ReLU (valid padding)
- **Body**: 5 × SimpleResidualBlock (Conv1d + ReLU + Dropout + Residual, k=9, dilation=1)
- **Head**: Single linear Conv1d(f, 1, 45) spatial conv (valid padding)
- **Count**: GlobalAvgPool → Linear → ReLU → Linear

Default configuration: 12 filters, 105bp RF, ~9.3K params.

### Key Features
*   **DeepLIFT/DeepSHAP compatible**: Only Conv1d, ReLU, and residual add — all have well-defined captum propagation rules.
*   **Linear profile head**: A single spatial convolution (no pointwise + ReLU) maximizes interpretability for attribution methods.
*   **Valid padding**: Ensures no zero-padding artifacts; strictly maps input to output geometrically.
*   **Residual connections**: Enable gradient flow through the 5-layer tower without normalization layers.

### Recommended Training Settings

| Parameter | Value |
|---|---|
| Optimizer | `adamw` |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Adam ε | `1e-8` |
| Scheduler | `default` (constant) |
| Loss | `MSEMultinomialLoss` (`count_weight="adaptive"`) |
| Sampler | `negative_peak` (background regions only) |

### Training Tool

Use `tools/train_biasnet.py` for command-line training:

```bash
# Train on negative peaks (default — bias-only training)
python tools/train_biasnet.py \
    --bigwig path/to/signal.bw \
    --peaks path/to/peaks.narrowPeak \
    --output-dir models/my_bias_model

# Train on peaks + background (like Pomeranian)
python tools/train_biasnet.py \
    --bigwig path/to/signal.bw \
    --peaks path/to/peaks.narrowPeak \
    --output-dir models/my_bias_model \
    --sampler-type peak

# Cosine schedule with warmup
python tools/train_biasnet.py \
    --bigwig path/to/signal.bw \
    --peaks path/to/peaks.narrowPeak \
    --output-dir models/my_bias_model \
    --scheduler-type cosine --warmup-epochs 10
```

### Usage
```python
from cerberus.models import BiasNet

model = BiasNet(
    input_len=1128,
    output_len=1024,
    filters=12,
    n_layers=5,
)
# Returns ProfileCountOutput(logits=..., log_counts=...)
```

## Dalmatian

**Implementation**: `cerberus.models.Dalmatian`
**Source**: `src/cerberus/models/dalmatian.py`

An end-to-end bias-factorized sequence-to-function model for ATAC-seq data. Replaces ChromBPNet's two-stage training (freeze bias, then train signal) with joint training of a BiasNet and a Pomeranian SignalNet, using architectural constraints, gradient separation, and a peak-conditioned loss to separate Tn5 enzyme bias from regulatory signal.

The name "Dalmatian" follows the dog-breed naming convention (BPNet, Pomeranian) and alludes to the two-component spotted pattern.

### Architecture

Dalmatian composes two sub-networks:

- **BiasNet** (~9.3K params): Lightweight Conv1d+ReLU stack with short receptive field (~105bp) to capture local Tn5 sequence bias. Uses 12 filters, 5 residual tower layers (all dilation=1), kernel size 9, two-layer stem `[11,11]`, linear profile head (k=45). Fully DeepLIFT/DeepSHAP compatible.
- **SignalNet** (~2-3M params): Full Pomeranian with long receptive field (~1089bp) to capture regulatory grammar (TF footprints, nucleosome positioning). Uses 256 filters, 8 dilated layers `[1,1,2,4,8,16,32,64]`, kernel size 9, two-layer stem `[11,11]`, profile kernel 45.

Their outputs are combined:
- **Profile**: logit addition (`combined_logits = bias_logits + signal_logits`)
- **Counts**: log-space addition via logsumexp (`combined_log_counts = logsumexp(bias_log_counts, signal_log_counts)`)

### Key Features
*   **Gradient separation**: Bias outputs are `.detach()`-ed before combining with signal outputs, so the combined reconstruction loss (L_recon) trains only SignalNet. BiasNet receives gradients exclusively from L_bias (background reconstruction). This replicates ChromBPNet's freeze-bias design without requiring a two-stage training procedure.
*   **Zero-initialized signal outputs** (`zero_init`, default `False`): When enabled, signal outputs are identity elements (logits=0, log_counts=-10) at initialization, so the combined output equals the bias-only output. **Recommended: leave disabled.** Experiments (exp20) show zero-init is harmful with gradient detach — SignalNet wastes epochs "turning on" from zero output. Without detach it served a purpose (soft two-stage training); with detach it's just a bad initialization.
*   **Signal presets**: `signal_preset="large"` (f=256, expansion=2, ~3.9M params) or `signal_preset="standard"` (default, f=64, expansion=1, ~150K params — matches standalone Pomeranian K9). Individual `signal_*` args override the preset. Experiments (exp20) show the 24x parameter difference buys <0.01 profile Pearson.
*   **Per-channel counts**: Both sub-models use `predict_total_count=False` because ATAC-seq channels represent independent samples (not forward/reverse strands).
*   **Clean geometry**: SignalNet shrinkage exactly maps `input_len` to `output_len` with no excess cropping. BiasNet receives a center-cropped input sized to its own receptive field needs.
*   **Input validation**: Rejects configurations where SignalNet shrinkage doesn't produce the exact `output_len`.
*   **DeepLIFT-compatible bias model**: BiasNet uses only Conv1d + ReLU + residual add, all with well-defined captum propagation rules.

### Output: FactorizedProfileCountOutput

Returns a `FactorizedProfileCountOutput` (extends `ProfileCountOutput`) containing both the combined predictions and the decomposed bias/signal components:

```python
@dataclass
class FactorizedProfileCountOutput(ProfileCountOutput):
    bias_logits: torch.Tensor       # (B, C, L)
    bias_log_counts: torch.Tensor   # (B, C)
    signal_logits: torch.Tensor     # (B, C, L)
    signal_log_counts: torch.Tensor # (B, C)
```

This is fully compatible with existing losses and metrics that expect `ProfileCountOutput` (they see `logits` and `log_counts`), while the decomposed fields enable the peak-conditioned `DalmatianLoss`.

### Loss: DalmatianLoss

`DalmatianLoss` wraps any profile+count base loss (e.g., `MSEMultinomialLoss`) and adds a peak-conditioned bias term:

```
L = L_recon(combined, target)                          # all examples
  + bias_weight * L_bias(bias_only, target)            # background only
```

- **L_recon**: Standard reconstruction loss on the combined output (all examples). Gradients flow only to SignalNet (bias outputs are detached before combining).
- **L_bias**: Reconstruction loss using only the bias component (background/non-peak examples only). This is the sole gradient source for BiasNet, forcing it to learn Tn5 bias rather than TF footprints.
- **`peak_status`**: A per-example binary tensor passed as batch context via `**kwargs`. The `CerberusModule._shared_step` automatically forwards all non-input/target batch fields as keyword arguments.

No explicit signal suppression term is needed — gradient detach already prevents SignalNet from activating on background (verified in exp21: removing the L1 penalty had zero measurable effect).

```python
model_config = {
    "name": "Dalmatian",
    "model_cls": "cerberus.models.dalmatian.Dalmatian",
    "loss_cls": "cerberus.loss.DalmatianLoss",
    "loss_args": {
        "base_loss_cls": "cerberus.loss.MSEMultinomialLoss",
        "base_loss_args": {"count_per_channel": True},
        "bias_weight": 1.0,
    },
    "metrics_cls": "cerberus.models.pomeranian.PomeranianMetricCollection",
    "metrics_args": {},
    "model_args": {
        "input_len": 2112,
        "output_len": 1024,
        "output_channels": ["sample1"],
    },
}
```

### Usage
```python
from cerberus.models import Dalmatian
from cerberus.loss import DalmatianLoss

# Default: 2112bp -> 1024bp
model = Dalmatian(
    input_len=2112,
    output_len=1024,
    input_channels=["A", "C", "G", "T"],
    output_channels=["sample1"],
)

loss = DalmatianLoss(
    base_loss_cls="cerberus.loss.MSEMultinomialLoss",
    bias_weight=1.0,
)

# Returns FactorizedProfileCountOutput with combined + decomposed fields
out = model(torch.randn(2, 4, 2112))
# out.logits, out.log_counts       -- combined
# out.bias_logits, out.bias_log_counts    -- bias component
# out.signal_logits, out.signal_log_counts -- signal component
```

### Training Tool

Use `tools/train_dalmatian.py` for command-line training:

```bash
# Bulk pseudobulk (all cell types merged)
python tools/train_dalmatian.py \
    --bigwig "tests/data/scatac_kidney_pseudobulk/bulk.bw" \
    --peaks "tests/data/scatac_kidney_pseudobulk/bulk_merge.narrowPeak.bed.gz" \
    --output-dir models/kidney_dalmatian

# Or use the example script
bash examples/scatac_kidney_dalmatian.sh
```

## PWMBiasModel (experimental)

**Location**: `debug/pwm_model/pwm_bias.py` (not part of cerberus public API)

A structured bias model that learns `n_motifs` position-weight matrix (PWM) filters of fixed `motif_width` and combines them via linear mixing. Designed for capturing enzymatic sequence bias (e.g., Tn5 transposase preference in ATAC-seq), where the bias is a short motif (~21bp) with low information content (~2 bits total).

Self-contained in `debug/pwm_model/` with its own output type (`RegularizedProfileCountOutput`), loss wrapper (`RegularizedMSEMultinomialLoss`), and tests. Compatible with cerberus training infrastructure via `import_class`.

### Key Features
*   **Learnable PWM filters**: `Conv1d(4, n_motifs, motif_width, valid)` — each filter is a learnable PWM that can be extracted and visualized as a sequence logo.
*   **Linear mixing**: `Conv1d(n_motifs, n_output_channels, 1)` — pointwise linear combination of filter activations. With linear mixing, multiple filters are mathematically equivalent to a single effective filter (provably via einsum folding).
*   **Non-negative mixing** (`nonneg_mixing=True`): Applies softplus to mixing weights, preventing filter cancellation where filters learn opposing patterns.
*   **ReLU activation** (`relu_activation=True`): Adds ReLU between PWM conv and profile mixing, breaking linear equivalence so each filter fires independently.
*   **Decorrelation penalty** (`decorrelation_weight`): Cosine similarity penalty between flattened PWM filters, encouraging distinct motif patterns. Used with `RegularizedMSEMultinomialLoss`.
*   **Exact geometry**: `input_len = output_len + motif_width - 1` (no cropping). Validates at construction time.
*   **Count head**: `GlobalAvgPool → Linear → GELU → Linear` for log count prediction.
*   **Output**: Returns `RegularizedProfileCountOutput(logits=..., log_counts=..., reg_loss=...)`, compatible with `MSEMultinomialLoss` and `RegularizedMSEMultinomialLoss`.

### Architecture Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_motifs` | 2 | Number of learnable PWM filters |
| `motif_width` | 21 | Width of each filter in bp |
| `nonneg_mixing` | False | Softplus on mixing weights (prevents cancellation) |
| `relu_activation` | False | ReLU between PWM conv and mixing (breaks linear equivalence) |
| `decorrelation_weight` | 0.0 | Cosine similarity penalty weight (0=disabled) |
| `predict_total_count` | True | Single total count vs per-channel counts |

### Usage
```python
# Add debug/ to sys.path first
from pwm_model.pwm_bias import PWMBiasModel

model = PWMBiasModel(
    input_len=1044, output_len=1024,
    n_motifs=2, motif_width=21,
    nonneg_mixing=True, relu_activation=True,
    decorrelation_weight=0.1,
)

# After training, extract filters for visualization:
pwm_weights = model.get_pwm_weights()   # (n_motifs, 4, motif_width)
mixing = model.get_mixing_weights()      # (n_output_channels, n_motifs, 1)
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
