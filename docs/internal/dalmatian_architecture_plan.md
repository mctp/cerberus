# Dalmatian: End-to-End Bias-Factorized Sequence-to-Function Model

## Context

ChromBPNet separates ATAC-seq signal into two components: (1) Tn5 transposase enzyme sequence bias, which is ubiquitous and local (~19bp motif preference), and (2) regulatory signal from TF binding footprints and nucleosome positioning, which is peak-specific and long-range. The original ChromBPNet uses a two-stage training procedure: the bias model is trained first in isolation on non-peak (background) regions, then frozen while the signal (accessibility) model trains on peak regions with the bias subtracted. We want to replace this with end-to-end joint training, using architectural constraints and a peak-conditioned loss to force the two sub-networks to learn different things without manual staging.

The name "Dalmatian" follows the dog-breed naming convention (BPNet, Pomeranian) and alludes to the two-component spotted pattern.

## Reference: ChromBPNet Architecture & Parameter Counts

Source: `s2f-models/repos/chrombpnet-pytorch/`

### ChromBPNet = Main BPNet (accessibility) + Bias BPNet

**Main BPNet** (`bpnet.py`): n_filters=512, n_layers=8, conv1_kernel_size=21, rconvs_kernel_size=3, profile_kernel_size=75

| Component | Params |
|---|---|
| Initial Conv: `Conv1d(4->512, k=21)` | 4x512x21 + 512 = **43,520** |
| 8x Dilated Conv: `Conv1d(512->512, k=3)`, dilations 2,4,8,16,32,64,128,256 | 8 x (512x512x3 + 512) = **6,295,552** |
| Profile Head: `Conv1d(512->1, k=75)` | 512x75 + 1 = **38,401** |
| Count Head: `Linear(512->1)` | 512 + 1 = **513** |
| **Main BPNet Total** | **~6.38M** |

**Bias BPNet** (`chrombpnet.py`): n_filters=128, n_layers=4 (hardcoded)

| Component | Params |
|---|---|
| Initial Conv: `Conv1d(4->128, k=21)` | **10,880** |
| 4x Dilated Conv: `Conv1d(128->128, k=3)` | **197,120** |
| Profile Head: `Conv1d(128->1, k=75)` | **9,601** |
| Count Head: `Linear(128->1)` | **129** |
| **Bias BPNet Total** | **~218K** |

**ChromBPNet Total: ~6.6M parameters** (97% in main model)

The dominant cost is the 8 **dense** dilated conv layers at 512 filters: each is `Conv1d(512->512, k=3)` = 787K params.

### ChromBPNet Forward Pass

```python
# chrombpnet.py:106-136
acc_profile, acc_counts = self.model(x)       # Main BPNet
bias_profile, bias_counts = self.bias(x)      # Bias BPNet
y_profile = acc_profile + bias_profile        # Logit addition
y_counts = log(exp(acc_counts) + exp(bias_counts))  # LogSumExp
return y_profile.squeeze(1), y_counts
```

ChromBPNet returns **combined only** from forward(). Decomposed outputs (acc vs bias) are internal. There is no public API to access them; you'd call `self.model(x)` and `self.bias(x)` separately.

### ChromBPNet Training (Two-Stage)

1. Train bias model on non-peak (background) regions only
2. Freeze bias model (`requires_grad=False`)
3. Train main model on peak regions with bias model predictions subtracted
4. Loss: `alpha * MSE(log_counts) + beta * MultinomialNLL(profile)`
5. Alpha scaling: `median_count / 10`
6. Default LR: 0.001 with Adam (eps=1e-7)

### ChromBPNet Config Defaults (`model_config.py`)

```python
out_dim=1000, n_filters=512, n_layers=8, conv1_kernel_size=21,
profile_kernel_size=75, n_outputs=1, n_control_tracks=0
```

### Key Architectural Differences: BPNet vs Pomeranian

| Aspect | BPNet (ChromBPNet) | Pomeranian |
|--------|-------------------|------------|
| Dilated conv type | **Dense** `Conv1d(F->F, k=3)` | **Depthwise separable** PGCBlock |
| Params per dilated layer (512 filters) | ~787K | ~790K (comparable but structured) |
| Params per dilated layer (64 filters) | ~12.3K | ~13.4K (PGCBlock) |
| Stem | Single Conv1d | ConvNeXtV2Block (with inv. bottleneck) |
| Profile head | Single large Conv1d(k=75) | Decoupled: Pointwise -> GELU -> Spatial |
| Count head | Linear(F->1) | MLP: Linear(F->F/2) -> GELU -> Linear(F/2->1) |
| Activation | ReLU | GELU + gating (PGC) |
| Normalization | None | RMSNorm + GRN |
| Input/Output | 2114 bp → 1000 bp | 2112 bp → 1024 bp |

### Pomeranian Parameter Count (default, filters=64)

| Component | Params |
|---|---|
| Stem (2x ConvNeXtV2Block, expansion=2) | ~37K |
| Body (8x PGCBlock, expansion=1) | ~108K |
| Profile Head (pointwise + spatial) | ~7K |
| Count Head (MLP) | ~2K |
| **Pomeranian Total** | **~154K** |

**Size gap: Pomeranian (154K) is ~43x smaller than ChromBPNet (6.6M).**

To match ChromBPNet's size: `filters=512, expansion=1` gives ~6.5M; `filters=384, expansion=2` gives ~6.8M.

## 1. Architecture Overview

```
                       DNA Sequence (B, 4, input_len)
                              |
                   +----------+----------+
                   |                     |
              BiasNet                SignalNet
          (Pomeranian)            (Pomeranian)
         small, local RF         large, full RF
          ~35K params             ~2-3M params
                   |                     |
          bias_profile,          signal_profile,
          bias_counts            signal_counts
                   |                     |
                   +----------+----------+
                              |
                         Combination
                    (see Section 2 below)
                              |
                      DalmatianOutput
          (combined logits/counts + decomposed fields)
```

Both sub-networks are `Pomeranian` instances with different hyperparameters. They share the same input (one-hot DNA sequence) and produce the same output type (`ProfileCountOutput`). Their outputs are then combined into a single `DalmatianOutput`.

## 2. Combination Rules

The two sub-model outputs are combined differently for profile shape vs total counts, following ChromBPNet's design (`chrombpnet-pytorch/chrombpnet/chrombpnet.py:116-121`):

### 2.1 Profile (shape): Addition in Logit Space

```python
combined_logits = bias_logits + signal_logits    # (B, C, output_len)
```

The model predicts **logits** (unnormalized log-probabilities). The actual profile shape is obtained via `softmax(logits)`, and per-position predicted counts via `softmax(logits) * exp(log_counts)`. Adding logits before softmax means the two models contribute **multiplicatively** to the final per-position probability:

```
P(cut at position i) ∝ exp(bias_logit_i) * exp(signal_logit_i)
```

This is biologically appropriate: Tn5 has a baseline cutting preference at each position (bias), and regulatory elements modulate that preference up or down (signal). Multiplication in probability space = addition in log-probability space.

### 2.2 Counts (total): Log-Sum-Exp

```python
combined_log_counts = torch.logsumexp(
    torch.stack([bias_log_counts, signal_log_counts], dim=-1), dim=-1
)
```

This computes `log(exp(bias_log_counts) + exp(signal_log_counts))` = `log(bias_counts + signal_counts)`. The total number of reads is the **sum** of bias-driven cuts and signal-driven cuts. This is additive in linear count space, computed in log space for numerical stability.

### 2.3 Identity Elements

| Component | Identity value | Effect |
|-----------|---------------|--------|
| **Profile logits** | `signal_logits = 0` | `combined_logits = bias_logits + 0 = bias_logits` (exact identity) |
| **Log counts** | `signal_log_counts = −∞` | `logsumexp(bias, −∞) = bias` (exact identity) |

In practice, initializing signal_log_counts ≈ −10 gives exp(−10) ≈ 0.000045 phantom reads, which is negligible. Initializing to 0 would give exp(0) = 1, adding 1 phantom read — not a true identity.

### 2.4 Summary Table

| Component | Operation | Math Space | Biological Meaning |
|-----------|-----------|-----------|-------------------|
| **Profile** (shape) | `bias_logits + signal_logits` | Log-probability | Multiplicative modulation of per-position cutting probability |
| **Counts** (total) | `logsumexp(bias_log_counts, signal_log_counts)` | Log-count | Additive: `total_reads = bias_reads + signal_reads` |

## 3. Three Mechanisms for Forcing Separation

The central challenge of end-to-end training is preventing mode collapse (both models learning the same thing, or one model learning everything while the other learns nothing). We use three complementary mechanisms:

### 3.1 Receptive Field Constraint (Architectural)

Tn5 insertion bias is a **local** sequence preference (~19bp motif). By limiting the bias model's receptive field to ~80bp, it physically **cannot** learn long-range regulatory grammar (TF binding site spacing, nucleosome positioning patterns spanning hundreds of bp). The signal model gets a full ~1000bp+ receptive field and can capture these patterns.

This is the strongest inductive bias -- it's not a soft penalty but a hard architectural constraint.

### 3.2 Capacity Asymmetry (Architectural)

The bias model is intentionally small (~35K parameters) while the signal model is large (~2-3M parameters). Even within its limited receptive field, the bias model can only represent simple local sequence patterns. The signal model has the capacity for complex combinatorial regulatory grammar.

### 3.3 Peak-Conditioned Loss (Training)

The dataset provides `peak_status` per example (1 = peak-centered window, 0 = complexity-matched background). This metadata enables three loss terms with different biological motivations:

```
L_total = L_recon + lambda_bias * L_bias_only + lambda_sparse * L_signal_sparse
```

| Term | Applied To | Gradients Flow To | Biological Motivation |
|------|-----------|-------------------|----------------------|
| **L_recon** | ALL examples | Both models | Combined output should match the observed ATAC-seq signal everywhere |
| **L_bias_only** | NON-PEAK examples only | Bias model only | In non-peak regions, the observed signal is almost entirely Tn5 cutting bias. The bias model alone should explain it. |
| **L_signal_sparse** | NON-PEAK examples only | Signal model only | The signal model should output ~0 outside peaks. There's no regulatory signal to learn in background regions. |

**Why gradient routing is natural (no stop-gradient needed):**

- `L_recon` is computed from `combined_logits` and `combined_log_counts`, which depend on both models -> both get gradients.
- `L_bias_only` is computed from `bias_logits` and `bias_log_counts` only -> only the bias model's parameters are in the computation graph.
- `L_signal_sparse` is computed from `signal_logits` and `signal_log_counts` only -> only the signal model's parameters are in the computation graph.

**Together these create complementary pressures:**

1. The bias model is pulled toward explaining the ubiquitous Tn5 bias (L_bias_only)
2. The signal model is pushed away from learning Tn5 bias (L_signal_sparse makes it silent outside peaks)
3. In peaks, the residual signal (TF footprints, regulatory grammar) can only be captured by the signal model -- the bias model's tiny receptive field can't represent it
4. The combined model is accurate everywhere (L_recon)

## 4. Sub-Model Configurations

### 4.1 BiasNet (Pomeranian variant -- small, local)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `filters` | 64 | Small capacity (~35K total params) |
| `n_dilated_layers` | 4 | Fewer layers, limited depth |
| `dilations` | `[1, 1, 2, 4]` | Limited RF; max dilation 4 keeps RF ~80bp |
| `dil_kernel_size` | 5 | Small spatial kernel |
| `conv_kernel_size` | 11 | Single stem layer (not factorized) |
| `profile_kernel_size` | 21 | Small smoothing window |
| `expansion` | 1 | Minimal PGC expansion |
| `stem_expansion` | 1 | Minimal stem expansion |

**Receptive field estimate:** stem(11) + 4 layers x (dil x (k-1)) = 11 + (1x4 + 1x4 + 2x4 + 4x4) = 11 + 32 + profile(21) = 64bp effective, ~80bp total.

### 4.2 SignalNet (Pomeranian variant -- large, full RF)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `filters` | 256 | Large capacity (~2-3M total params) |
| `n_dilated_layers` | 8 | Full depth |
| `dilations` | `[1, 1, 2, 4, 8, 16, 32, 64]` | Full RF ~1000bp+ |
| `dil_kernel_size` | 9 | Standard Pomeranian kernel |
| `conv_kernel_size` | `[11, 11]` | Factorized 2-layer stem |
| `profile_kernel_size` | 45 | Standard profile head |
| `expansion` | 2 | Full PGC expansion |
| `stem_expansion` | 2 | Standard stem expansion |

### 4.3 Zero-Initialization of Signal Output Layers

The signal model's final output layers are initialized so that it produces the **identity element** for both combination rules at initialization:

**Profile head — zero weights and zero biases:**

- `signal_model.profile_pointwise` (Conv1d): `weight=0, bias=0`
- `signal_model.profile_spatial` (Conv1d): `weight=0, bias=0`

**Effect:** `signal_logits = 0` everywhere. Combined profile = bias-only profile (exact identity for logit addition).

**Count head — zero weights and bias = −10:**

- `signal_model.count_mlp[-1]` (Linear, final layer of count MLP): `weight=0, bias=-10.0`

**Effect:** `signal_log_counts ≈ −10` regardless of input. Since `exp(−10) ≈ 0.000045`, the signal model contributes negligible counts via logsumexp. This is the approximate identity for the count combination rule.

**Why bias = −10 and not bias = 0 for counts:** The identity element for `logsumexp(a, b)` is `b = −∞`, not `b = 0`. If we used `bias = 0`, then `exp(0) = 1`, and the signal model would contribute exactly 1 phantom read to every prediction: `combined = log(exp(bias_counts) + 1)`. With `bias = −10`, the contribution is truly negligible (0.000045 reads).

**Implementation:**

```python
def _zero_init_signal_outputs(self):
    """Zero-initialize signal model output layers for identity-element start."""
    # Profile head: zero -> signal_logits = 0 (identity for addition)
    nn.init.zeros_(self.signal_model.profile_pointwise.weight)
    nn.init.zeros_(self.signal_model.profile_pointwise.bias)
    nn.init.zeros_(self.signal_model.profile_spatial.weight)
    nn.init.zeros_(self.signal_model.profile_spatial.bias)

    # Count head: zero weights + large negative bias -> signal_log_counts ≈ −10
    # (identity for logsumexp: exp(−10) ≈ 0.000045 phantom reads)
    final_linear = self.signal_model.count_mlp[-1]
    nn.init.zeros_(final_linear.weight)
    nn.init.constant_(final_linear.bias, -10.0)
```

**Training dynamics at epoch 0:**

1. Combined output = bias-only output. L_recon trains only the bias model effectively.
2. L_bias trains the bias model directly on background.
3. L_sparse is already satisfied (signal outputs are zero/negligible).
4. As training progresses, the bias model converges on Tn5 bias. The signal model's intermediate representations develop through L_recon gradients, and its output layers gradually activate.

## 5. Output Design

### 5.1 DalmatianOutput Dataclass

**File:** `src/cerberus/output.py` (alongside existing `ProfileCountOutput`)

```python
@dataclass
class DalmatianOutput(ProfileCountOutput):
    """Combined output with decomposed sub-model outputs for the Dalmatian model."""
    # Inherited from ProfileCountOutput:
    #   logits: torch.Tensor       # (B, C, L) -- combined profile logits
    #   log_counts: torch.Tensor   # (B, C)   -- combined log counts
    # Inherited from ModelOutput (kw_only):
    #   out_interval: Interval | None = None

    # Decomposed fields:
    bias_logits: torch.Tensor       # (B, C, L) -- bias model profile logits
    bias_log_counts: torch.Tensor   # (B, C)   -- bias model log counts
    signal_logits: torch.Tensor     # (B, C, L) -- signal model profile logits
    signal_log_counts: torch.Tensor # (B, C)   -- signal model log counts

    def detach(self):
        return DalmatianOutput(
            logits=self.logits.detach(),
            log_counts=self.log_counts.detach(),
            bias_logits=self.bias_logits.detach(),
            bias_log_counts=self.bias_log_counts.detach(),
            signal_logits=self.signal_logits.detach(),
            signal_log_counts=self.signal_log_counts.detach(),
            out_interval=self.out_interval,
        )
```

### 5.2 Compatibility with Existing Cerberus Infrastructure

Because `DalmatianOutput` extends `ProfileCountOutput`, all existing infrastructure works transparently:

| Component | File | How it works |
|-----------|------|-------------|
| **Metrics** (`PomeranianMetricCollection`) | `metrics.py` | `isinstance(preds, ProfileCountOutput)` → True. Metrics see combined `logits` and `log_counts` only. |
| **`compute_total_log_counts()`** | `output.py:265` | `isinstance(model_output, ProfileCountOutput)` → True. Uses combined `log_counts`. |
| **`unbatch_modeloutput()`** | `output.py:54` | Uses `dataclasses.asdict()` which includes all fields (combined + decomposed). Extra tensor fields are correctly unbatched along batch dimension. |
| **`aggregate_intervals()`** | `output.py:174` | Aggregates all tensor fields. Decomposed fields are spatially merged alongside combined fields. |
| **`aggregate_models()`** | `output.py:224` | Stacks and averages all tensor fields across ensemble members. |
| **`_accumulate_log_counts()`** | `module.py:182` | Calls `compute_total_log_counts()` which uses combined outputs. Correct. |
| **Count scatter plot** | `module.py:215` | Uses accumulated log counts from combined outputs. Correct. |

The decomposed fields (`bias_logits`, `signal_logits`, etc.) are consumed only by:
1. `DalmatianLoss` during training
2. Interpretation tools when analyzing bias vs. regulatory attribution (future work)

### 5.3 Dalmatian.forward()

```python
def forward(self, x) -> DalmatianOutput:
    bias_out = self.bias_model(x)       # ProfileCountOutput
    signal_out = self.signal_model(x)   # ProfileCountOutput

    combined_logits = bias_out.logits + signal_out.logits
    combined_log_counts = torch.logsumexp(
        torch.stack([bias_out.log_counts, signal_out.log_counts], dim=-1),
        dim=-1,
    )

    return DalmatianOutput(
        logits=combined_logits,
        log_counts=combined_log_counts,
        bias_logits=bias_out.logits,
        bias_log_counts=bias_out.log_counts,
        signal_logits=signal_out.logits,
        signal_log_counts=signal_out.log_counts,
    )
```

## 6. Loss Function

### 6.1 DalmatianLoss

**File:** `src/cerberus/loss.py`

```python
class DalmatianLoss(nn.Module):
    """
    Peak-conditioned loss for end-to-end bias-signal separation.

    Wraps any ProfileCountOutput-compatible base loss (e.g. MSEMultinomialLoss,
    PoissonMultinomialLoss) with auxiliary losses that force the bias and signal
    sub-networks to specialize.

    Three terms:
      L_total = L_recon + bias_weight * L_bias_only + sparse_weight * L_signal_sparse

    Args:
        base_loss_cls: Fully qualified class name of the base loss
            (e.g. "cerberus.loss.MSEMultinomialLoss").
        base_loss_args: Keyword arguments for base loss constructor.
        bias_weight: Weight for bias-only reconstruction on non-peak examples.
        sparse_weight: Weight for signal sparsity penalty on non-peak examples.
    """
    def __init__(
        self,
        base_loss_cls: str,
        base_loss_args: dict | None = None,
        bias_weight: float = 1.0,
        sparse_weight: float = 0.1,
        count_pseudocount: float = 1.0,
    ):
        super().__init__()
        from cerberus.config import import_class
        loss_cls = import_class(base_loss_cls)
        self.base_loss = loss_cls(**(base_loss_args or {}))
        self.bias_weight = bias_weight
        self.sparse_weight = sparse_weight
        # count_pseudocount accepted for compatibility with propagate_pseudocount.
        # The actual pseudocount is in base_loss_args, not used directly here.
        self.count_pseudocount = count_pseudocount

    def forward(self, output: DalmatianOutput, target: torch.Tensor,
                peak_status: torch.Tensor) -> torch.Tensor:
        # 1. Combined reconstruction loss (all examples)
        combined = ProfileCountOutput(
            logits=output.logits, log_counts=output.log_counts)
        L_recon = self.base_loss(combined, target)

        # 2. Bias-only reconstruction + signal sparsity (non-peak examples)
        non_peak = peak_status == 0
        L_bias = torch.tensor(0.0, device=target.device)
        L_sparse = torch.tensor(0.0, device=target.device)

        if non_peak.any():
            bias_out = ProfileCountOutput(
                logits=output.bias_logits[non_peak],
                log_counts=output.bias_log_counts[non_peak])
            L_bias = self.base_loss(bias_out, target[non_peak])

            L_sparse = (
                output.signal_logits[non_peak].abs().mean()
                + output.signal_log_counts[non_peak].abs().mean()
            )

        return L_recon + self.bias_weight * L_bias + self.sparse_weight * L_sparse
```

### 6.2 Config Integration: Nested Base Loss Instantiation

The standard cerberus loss API (`instantiate_metrics_and_loss` in `module.py:382`) does:

```python
loss_cls = import_class(model_config["loss_cls"])
criterion = loss_cls(**model_config["loss_args"])
```

DalmatianLoss takes `base_loss_cls` (a string) and `base_loss_args` (a dict) as constructor arguments, and instantiates the base loss internally using `import_class`. This means it works with the existing factory without any changes to `module.py`'s instantiation logic.

**Example YAML config:**

```yaml
model_config:
  name: "dalmatian"
  model_cls: "cerberus.models.dalmatian.Dalmatian"
  loss_cls: "cerberus.loss.DalmatianLoss"
  loss_args:
    base_loss_cls: "cerberus.loss.MSEMultinomialLoss"
    base_loss_args:
      count_weight: 1.0
      profile_weight: 1.0
    bias_weight: 1.0
    sparse_weight: 0.1
  metrics_cls: "cerberus.models.pomeranian.PomeranianMetricCollection"
  metrics_args: {}
  model_args:
    bias_filters: 64
    bias_n_layers: 4
    bias_dilations: [1, 1, 2, 4]
    bias_dil_kernel_size: 5
    bias_conv_kernel_size: 11
    bias_profile_kernel_size: 21
    bias_expansion: 1
    bias_stem_expansion: 1
    signal_filters: 256
    signal_n_layers: 8
    signal_dilations: [1, 1, 2, 4, 8, 16, 32, 64]
    signal_dil_kernel_size: 9
    signal_conv_kernel_size: [11, 11]
    signal_profile_kernel_size: 45
    signal_expansion: 2
    signal_stem_expansion: 2
```

### 6.3 `propagate_pseudocount` Compatibility

The `propagate_pseudocount` function in `config.py:775` does:

```python
loss_args.setdefault("count_pseudocount", scaled_pseudocount)
```

This sets `count_pseudocount` on DalmatianLoss's top-level `loss_args`. DalmatianLoss accepts it for compatibility but does **not** use it directly — the actual pseudocount is inside `base_loss_args`. To propagate the pseudocount into the nested base loss, DalmatianLoss's constructor should forward it:

```python
def __init__(self, base_loss_cls, base_loss_args=None, bias_weight=1.0,
             sparse_weight=0.1, count_pseudocount=1.0):
    ...
    args = dict(base_loss_args or {})
    args.setdefault("count_pseudocount", count_pseudocount)
    self.base_loss = loss_cls(**args)
```

This ensures that when `propagate_pseudocount` sets `count_pseudocount` on the outer loss_args, it flows through to the base loss. The user can also set it explicitly in `base_loss_args` to override.

### 6.4 Loss Signature and Module Integration

`DalmatianLoss.forward()` takes three arguments: `(output, target, peak_status)`. The standard cerberus loss API is `(output, target)`. This requires a small change to `CerberusModule._shared_step()` in `module.py`.

**Current code** (`module.py:146`):

```python
loss = self.criterion(outputs, targets)
```

**New code:**

```python
if isinstance(self.criterion, DalmatianLoss):
    loss = self.criterion(outputs, targets, batch["peak_status"])
else:
    loss = self.criterion(outputs, targets)
```

This is the **only change** to `module.py`. The `peak_status` key is already present in every batch dict (from `CerberusDataset.__getitem__()` at `dataset.py:354-358`), defaulting to 1 when the sampler doesn't support `get_peak_status()`.

### 6.5 Edge Cases

- **All-peak batch** (`non_peak.any()` is False): L_bias = 0, L_sparse = 0. Only L_recon contributes. This is fine — the bias model still gets gradients through L_recon.
- **All-non-peak batch**: All three terms are active. Unlikely in practice with balanced PeakSampler but handled correctly.
- **Single-example batch**: Works correctly since all operations are batch-indexed.

## 7. Data Pipeline Integration

### 7.1 Existing Infrastructure (no changes needed)

The cerberus data pipeline already provides all the required components:

1. **`PeakSampler`** (`src/cerberus/samplers.py:1186`): Combines `IntervalSampler` for peaks with `ComplexityMatchedSampler` for GC-matched backgrounds. Returns a `MultiSampler`.

2. **`MultiSampler.get_peak_status()`** (`src/cerberus/samplers.py:375-394`): Returns `1` for intervals from the first sub-sampler (peaks) and `0` for all others (backgrounds). Convention: `samplers[0]` = peaks, subsequent = backgrounds. Preserved after `split_folds`.

3. **`CerberusDataset.__getitem__()`** (`src/cerberus/dataset.py:354-358`): Includes `peak_status` in the returned batch dict:
   ```python
   result["peak_status"] = (
       self.sampler.get_peak_status(idx)
       if hasattr(self.sampler, "get_peak_status")
       else 1
   )
   ```

4. **BigBed support** (`src/cerberus/mask.py`): `.bb`/`.bigbed` files are registered as extractor types with both on-the-fly (`BigBedMaskExtractor`) and in-memory (`InMemoryBigBedMaskExtractor`) variants.

### 7.2 Typical Training Configuration

A Dalmatian training run would use `sampler_type: "peak"` in the sampler config:

```yaml
sampler_config:
  sampler_type: "peak"
  padded_size: 2112      # input_len
  sampler_args:
    intervals_path: "peaks.narrowPeak"  # MACS3 output
    background_ratio: 1.0               # 1 background per peak
```

## 8. Model Constructor

```python
class Dalmatian(nn.Module):
    """
    Dalmatian: End-to-end bias-factorized sequence-to-function model.

    Composes two Pomeranian sub-networks:
    - BiasNet: small, limited receptive field -- learns Tn5 sequence bias
    - SignalNet: large, full receptive field -- learns regulatory grammar

    Their outputs are combined (profile addition in logit space, count
    addition in log space via logsumexp) and returned as DalmatianOutput.
    """
    def __init__(
        self,
        input_len: int = 2112,
        output_len: int = 1024,
        output_bin_size: int = 1,
        input_channels: list[str] | None = None,
        output_channels: list[str] | None = None,
        # --- BiasNet configuration ---
        bias_filters: int = 64,
        bias_n_layers: int = 4,
        bias_dilations: list[int] | None = None,     # default: [1, 1, 2, 4]
        bias_dil_kernel_size: int = 5,
        bias_conv_kernel_size: int | list[int] = 11,
        bias_profile_kernel_size: int = 21,
        bias_expansion: int = 1,
        bias_stem_expansion: int = 1,
        bias_dropout: float = 0.1,
        # --- SignalNet configuration ---
        signal_filters: int = 256,
        signal_n_layers: int = 8,
        signal_dilations: list[int] | None = None,   # default: [1,1,2,4,8,16,32,64]
        signal_dil_kernel_size: int = 9,
        signal_conv_kernel_size: int | list[int] | None = None,  # default: [11, 11]
        signal_profile_kernel_size: int = 45,
        signal_expansion: int = 2,
        signal_stem_expansion: int = 2,
        signal_dropout: float = 0.1,
    ):
```

**Key constructor implementation notes:**

1. Both `self.bias_model` and `self.signal_model` are `Pomeranian` instances constructed with their respective hyperparameters. Both receive the same `input_len`, `output_len`, `output_bin_size`, `input_channels`, `output_channels`.

2. Default dilations are set in the constructor body:
   - `bias_dilations` defaults to `[1, 1, 2, 4]`
   - `signal_dilations` defaults to `[1, 1, 2, 4, 8, 16, 32, 64]`
   - `signal_conv_kernel_size` defaults to `[11, 11]`

3. After constructing both Pomeranians, call `self._zero_init_signal_outputs()` (Section 4.3).

4. The constructor matches the cerberus convention: `instantiate_model` in `module.py:285-318` calls `model_cls(input_len=..., output_len=..., output_bin_size=..., **model_args)`, so all Dalmatian-specific args must be in `model_args` in the config.

## 9. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/cerberus/output.py` | **Modify** | Add `DalmatianOutput` dataclass extending `ProfileCountOutput` with decomposed fields and `detach()` |
| `src/cerberus/loss.py` | **Modify** | Add `DalmatianLoss` with nested base loss instantiation, three-term loss |
| `src/cerberus/models/dalmatian.py` | **Create** | Dalmatian model composing two Pomeranians, zero-init, forward returning DalmatianOutput |
| `src/cerberus/module.py` | **Modify** | `isinstance` check in `_shared_step` to pass `peak_status` to `DalmatianLoss` (~3 lines) |
| `src/cerberus/models/__init__.py` | **Modify** | Add `from .dalmatian import Dalmatian` |
| `tests/test_dalmatian.py` | **Create** | Unit tests (see Section 11 for full list) |
| `CHANGELOG.md` | **Modify** | Add entry |
| `docs/models.md` | **Modify** | Document new model |

## 10. Implementation Plan: Step-by-Step with Test Cycles

Each step is a self-contained implement-test cycle. Complete all tests for a step before proceeding to the next. This ensures each layer is verified before building on it.

### Step 1: DalmatianOutput (output.py)

**Implement:**
- Add `DalmatianOutput` dataclass in `src/cerberus/output.py` (after `ProfileCountOutput`)
- Fields: `bias_logits`, `bias_log_counts`, `signal_logits`, `signal_log_counts`
- Implement `detach()` method

**Test (add to `tests/test_dalmatian.py`):**
```python
def test_dalmatian_output_is_profile_count_output():
    """DalmatianOutput must be isinstance of ProfileCountOutput."""
    out = DalmatianOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.randn(2, 1),
        bias_logits=torch.randn(2, 1, 100),
        bias_log_counts=torch.randn(2, 1),
        signal_logits=torch.randn(2, 1, 100),
        signal_log_counts=torch.randn(2, 1),
    )
    assert isinstance(out, ProfileCountOutput)

def test_dalmatian_output_detach():
    """detach() returns new instance with all tensors detached."""
    out = DalmatianOutput(
        logits=torch.randn(2, 1, 100, requires_grad=True),
        log_counts=torch.randn(2, 1, requires_grad=True),
        bias_logits=torch.randn(2, 1, 100, requires_grad=True),
        bias_log_counts=torch.randn(2, 1, requires_grad=True),
        signal_logits=torch.randn(2, 1, 100, requires_grad=True),
        signal_log_counts=torch.randn(2, 1, requires_grad=True),
    )
    det = out.detach()
    assert isinstance(det, DalmatianOutput)
    assert not det.logits.requires_grad
    assert not det.bias_logits.requires_grad
    assert not det.signal_logits.requires_grad

def test_dalmatian_output_unbatch():
    """unbatch_modeloutput works with DalmatianOutput (all tensor fields split)."""
    out = DalmatianOutput(
        logits=torch.randn(4, 1, 100),
        log_counts=torch.randn(4, 1),
        bias_logits=torch.randn(4, 1, 100),
        bias_log_counts=torch.randn(4, 1),
        signal_logits=torch.randn(4, 1, 100),
        signal_log_counts=torch.randn(4, 1),
    )
    items = unbatch_modeloutput(out, 4)
    assert len(items) == 4
    assert "bias_logits" in items[0]
    assert items[0]["bias_logits"].shape == (1, 100)

def test_dalmatian_output_compute_total_log_counts():
    """compute_total_log_counts sees combined log_counts from DalmatianOutput."""
    out = DalmatianOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.tensor([[3.0], [4.0]]),
        bias_logits=torch.randn(2, 1, 100),
        bias_log_counts=torch.tensor([[2.0], [3.0]]),
        signal_logits=torch.randn(2, 1, 100),
        signal_log_counts=torch.tensor([[1.0], [2.0]]),
    )
    lc = compute_total_log_counts(out)
    # Should use combined log_counts, not decomposed
    assert torch.allclose(lc, torch.tensor([3.0, 4.0]))
```

**Verify:** `pytest -v tests/test_dalmatian.py -k "output"` + `npx pyright src/cerberus/output.py`

---

### Step 2: Dalmatian Model (models/dalmatian.py)

**Implement:**
- Create `src/cerberus/models/dalmatian.py`
- `Dalmatian(nn.Module)` with constructor (Section 8)
- Two Pomeranian instances (bias_model, signal_model)
- `_zero_init_signal_outputs()` method (Section 4.3)
- `forward(x) -> DalmatianOutput` (Section 5.3)

**Test:**
```python
def test_dalmatian_forward_shape():
    """Forward produces DalmatianOutput with correct shapes."""
    model = Dalmatian(input_len=2112, output_len=1024)
    x = torch.randn(2, 4, 2112)
    out = model(x)
    assert isinstance(out, DalmatianOutput)
    assert out.logits.shape == (2, 1, 1024)
    assert out.log_counts.shape == (2, 1)
    assert out.bias_logits.shape == (2, 1, 1024)
    assert out.signal_logits.shape == (2, 1, 1024)

def test_dalmatian_zero_init():
    """Signal model outputs are zero/negligible at initialization."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    with torch.no_grad():
        out = model(x)
    # Signal logits should be exactly 0
    assert torch.allclose(out.signal_logits, torch.zeros_like(out.signal_logits), atol=1e-6)
    # Signal log_counts should be ≈ -10
    assert (out.signal_log_counts < -9.0).all()
    # Combined logits should equal bias logits
    assert torch.allclose(out.logits, out.bias_logits, atol=1e-6)

def test_dalmatian_combined_equals_bias_at_init():
    """At initialization, combined output ≈ bias-only output."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    with torch.no_grad():
        out = model(x)
    # Profile: combined = bias + 0 = bias
    assert torch.allclose(out.logits, out.bias_logits, atol=1e-6)
    # Counts: logsumexp(bias, -10) ≈ bias (when bias >> -10)
    # Not exact, but very close when bias model outputs > 0
    diff = (out.log_counts - out.bias_log_counts).abs()
    assert diff.max() < 0.01  # Negligible difference

def test_dalmatian_backward():
    """Gradients flow through both sub-models."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    out = model(x)
    loss = out.logits.sum() + out.log_counts.sum()
    loss.backward()
    # Both models should have gradients
    for name, p in model.bias_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"bias_model.{name} has no gradient"
    for name, p in model.signal_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"signal_model.{name} has no gradient"

def test_dalmatian_param_count():
    """Parameter count is in expected range."""
    model = Dalmatian()
    total = sum(p.numel() for p in model.parameters())
    bias_params = sum(p.numel() for p in model.bias_model.parameters())
    signal_params = sum(p.numel() for p in model.signal_model.parameters())
    assert 20_000 < bias_params < 80_000, f"Bias params {bias_params} out of range"
    assert 1_000_000 < signal_params < 5_000_000, f"Signal params {signal_params} out of range"
    assert total == bias_params + signal_params
```

**Verify:** `pytest -v tests/test_dalmatian.py -k "model or forward or init or backward or param"` + `npx pyright src/cerberus/models/dalmatian.py`

---

### Step 3: DalmatianLoss (loss.py)

**Implement:**
- Add `DalmatianLoss` class in `src/cerberus/loss.py`
- Nested base loss instantiation via `import_class(base_loss_cls)`
- Three-term loss with `peak_status` routing
- `count_pseudocount` forwarding to base loss

**Test:**
```python
def test_dalmatian_loss_gradient_routing():
    """Verify gradient routing: L_recon→both, L_bias→bias only, L_sparse→signal only."""
    model = Dalmatian(input_len=2112, output_len=1024)
    x = torch.randn(2, 4, 2112)
    targets = torch.rand(2, 1, 1024) * 100
    peak_status = torch.tensor([1, 0])  # First=peak, second=background

    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
        sparse_weight=0.1,
    )

    out = model(x)
    loss = loss_fn(out, targets, peak_status)
    loss.backward()

    # Both models should have gradients (L_recon flows to both)
    bias_grads = {n: p.grad for n, p in model.bias_model.named_parameters() if p.grad is not None}
    signal_grads = {n: p.grad for n, p in model.signal_model.named_parameters() if p.grad is not None}
    assert len(bias_grads) > 0, "Bias model should have gradients"
    assert len(signal_grads) > 0, "Signal model should have gradients"

def test_dalmatian_loss_all_peak_batch():
    """All-peak batch: only L_recon contributes."""
    model = Dalmatian()
    x = torch.randn(4, 4, 2112)
    targets = torch.rand(4, 1, 1024) * 100
    peak_status = torch.ones(4, dtype=torch.long)

    loss_fn = DalmatianLoss(base_loss_cls="cerberus.loss.MSEMultinomialLoss")
    out = model(x)
    loss = loss_fn(out, targets, peak_status)
    assert loss.isfinite()

def test_dalmatian_loss_all_nonpeak_batch():
    """All-non-peak batch: all three terms active."""
    model = Dalmatian()
    x = torch.randn(4, 4, 2112)
    targets = torch.rand(4, 1, 1024) * 100
    peak_status = torch.zeros(4, dtype=torch.long)

    loss_fn = DalmatianLoss(base_loss_cls="cerberus.loss.MSEMultinomialLoss")
    out = model(x)
    loss = loss_fn(out, targets, peak_status)
    assert loss.isfinite()

def test_dalmatian_loss_bias_only_gradient_isolation():
    """L_bias sends gradients only to bias model, not signal model."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    out = model(x)

    # Compute L_bias manually (bias-only reconstruction on all examples)
    from cerberus.loss import MSEMultinomialLoss
    base = MSEMultinomialLoss()
    targets = torch.rand(2, 1, 1024) * 100
    bias_out = ProfileCountOutput(logits=out.bias_logits, log_counts=out.bias_log_counts)
    l_bias = base(bias_out, targets)
    l_bias.backward()

    # Signal model should have NO gradients
    for name, p in model.signal_model.named_parameters():
        assert p.grad is None or p.grad.abs().max() == 0, \
            f"signal_model.{name} got gradient from L_bias"
    # Bias model should have gradients
    has_bias_grad = any(p.grad is not None and p.grad.abs().max() > 0
                        for p in model.bias_model.parameters())
    assert has_bias_grad, "Bias model should have gradients from L_bias"
```

**Verify:** `pytest -v tests/test_dalmatian.py -k "loss"` + `npx pyright src/cerberus/loss.py`

---

### Step 4: Module Integration (module.py)

**Implement:**
- Add `isinstance` check in `CerberusModule._shared_step()` at line 146
- Import `DalmatianLoss` at top of `module.py`

**Test:**
```python
def test_module_shared_step_with_dalmatian():
    """CerberusModule._shared_step passes peak_status to DalmatianLoss."""
    from cerberus.module import CerberusModule
    from cerberus.models.pomeranian import PomeranianMetricCollection

    model = Dalmatian()
    loss_fn = DalmatianLoss(base_loss_cls="cerberus.loss.MSEMultinomialLoss")
    metrics = PomeranianMetricCollection()

    module = CerberusModule(model=model, criterion=loss_fn, metrics=metrics)

    batch = {
        "inputs": torch.randn(4, 4, 2112),
        "targets": torch.rand(4, 1, 1024) * 100,
        "peak_status": torch.tensor([1, 0, 1, 0]),
    }
    loss = module._shared_step(batch, 0, "train_")
    assert loss.isfinite()
```

**Verify:** `pytest -v tests/test_dalmatian.py -k "module"` + `npx pyright src/cerberus/module.py`

---

### Step 5: Exports and Config Validation

**Implement:**
- Add `from .dalmatian import Dalmatian` to `src/cerberus/models/__init__.py`
- Add `DalmatianLoss` to the imports at top of `loss.py` (if not already importable)

**Test:**
```python
def test_dalmatian_import():
    """Dalmatian is importable from cerberus.models."""
    from cerberus.models import Dalmatian
    assert Dalmatian is not None

def test_dalmatian_loss_import():
    """DalmatianLoss is importable from cerberus.loss."""
    from cerberus.loss import DalmatianLoss
    assert DalmatianLoss is not None

def test_dalmatian_config_instantiation():
    """Dalmatian can be instantiated through the config system."""
    from cerberus.config import import_class
    cls = import_class("cerberus.models.dalmatian.Dalmatian")
    model = cls(input_len=2112, output_len=1024)
    assert isinstance(model, Dalmatian)

    loss_cls = import_class("cerberus.loss.DalmatianLoss")
    loss = loss_cls(base_loss_cls="cerberus.loss.MSEMultinomialLoss")
    assert isinstance(loss, DalmatianLoss)
```

**Verify:** `pytest -v tests/test_dalmatian.py -k "import or config"` + `npx pyright tests/ src/`

---

### Step 6: Full Integration Test

**Test:**
```python
def test_dalmatian_end_to_end_training_step():
    """End-to-end: model + loss + metrics + module for one training step."""
    from cerberus.module import CerberusModule
    from cerberus.models.pomeranian import PomeranianMetricCollection

    model = Dalmatian()
    loss_fn = DalmatianLoss(base_loss_cls="cerberus.loss.MSEMultinomialLoss")
    metrics = PomeranianMetricCollection()

    module = CerberusModule(model=model, criterion=loss_fn, metrics=metrics)

    # Simulate mixed peak/non-peak batch
    batch = {
        "inputs": torch.randn(8, 4, 2112),
        "targets": torch.rand(8, 1, 1024) * 100,
        "peak_status": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0]),
    }

    # Training step
    loss = module.training_step(batch, 0)
    assert loss.isfinite()

    # Check metrics update without error
    metrics_dict = module.train_metrics.compute()
    assert "train_pearson" in metrics_dict or "train_mse_log_counts" in metrics_dict
```

**Verify:** `pytest -v tests/test_dalmatian.py` (all tests) + `npx pyright tests/ src/`

---

### Step 7: Documentation and Changelog

**Implement:**
- Update `CHANGELOG.md` with Dalmatian entry
- Update `docs/models.md` with Dalmatian documentation

**Verify:** All tests pass: `pytest -v tests/` + `npx pyright tests/ src/`

## 11. Verification Summary

### Unit Tests (`tests/test_dalmatian.py`)

| Test | Verifies |
|------|----------|
| `test_dalmatian_output_is_profile_count_output` | Inheritance from ProfileCountOutput |
| `test_dalmatian_output_detach` | detach() works for all fields |
| `test_dalmatian_output_unbatch` | unbatch_modeloutput compatibility |
| `test_dalmatian_output_compute_total_log_counts` | compute_total_log_counts sees combined |
| `test_dalmatian_forward_shape` | Correct output shapes |
| `test_dalmatian_zero_init` | Signal outputs are zero/−10 at init |
| `test_dalmatian_combined_equals_bias_at_init` | Combined = bias at init |
| `test_dalmatian_backward` | Gradients flow to both models |
| `test_dalmatian_param_count` | Parameter counts in expected range |
| `test_dalmatian_loss_gradient_routing` | L_recon→both, L_bias→bias, L_sparse→signal |
| `test_dalmatian_loss_all_peak_batch` | Edge case: no non-peak examples |
| `test_dalmatian_loss_all_nonpeak_batch` | Edge case: all non-peak examples |
| `test_dalmatian_loss_bias_only_gradient_isolation` | L_bias does not touch signal model |
| `test_module_shared_step_with_dalmatian` | peak_status flows from batch to loss |
| `test_dalmatian_import` | Import from cerberus.models |
| `test_dalmatian_loss_import` | Import from cerberus.loss |
| `test_dalmatian_config_instantiation` | Config system instantiation |
| `test_dalmatian_end_to_end_training_step` | Full training step |

### Integration

- Configure with `PeakSampler`, verify `peak_status` flows from dataset through module to `DalmatianLoss`

### Post-training Separation Check (manual)

- Bias model alone should explain non-peak signal well
- Signal model output should be near-zero on non-peak regions
- Combined model should outperform bias-only on peak regions

### Full Suite

- `pytest -v tests/` — all existing + new tests pass
- `npx pyright tests/ src/` — no type errors

## Appendix A: Design Decisions Log

### A.1 Output API: DalmatianOutput Dataclass

**Decision:** `forward()` returns `DalmatianOutput` (extends `ProfileCountOutput`) with decomposed bias/signal fields always present.

**Alternatives considered:**
1. **Loss takes model + inputs** -- DalmatianLoss.forward(model, x, target, peak_status) calls model.forward_decomposed() internally. Rejected: unusual pattern, loss drives forward pass, harder to test.
2. **Two forward modes** -- forward() for inference (returns ProfileCountOutput), forward_decomposed() for training (returns dict). Module calls different methods for train vs eval. Rejected: requires model-specific branching in module.py.
3. **Combined only from forward(), separate forward_decomposed()** -- Mirrors ChromBPNet exactly. Rejected: requires either the loss to call the model (unusual) or the module to branch.

**Rationale:** DalmatianOutput is the simplest approach. It extends ProfileCountOutput so existing metrics/export/unbatching work transparently. The extra decomposed fields ride along with zero overhead for inference code that doesn't use them.

### A.2 Peak Mask Granularity

**Decision:** Per-example boolean (`peak_status` = 1 for peak, 0 for background).

**Alternative:** Per-position mask within each window (B, L). More granular but adds loss computation complexity and doesn't match the natural sampling paradigm (peak-centered windows vs background windows).

**Rationale:** Matches ChromBPNet's training data structure. The existing `PeakSampler` + `MultiSampler.get_peak_status()` already provides this. BigBed is also supported as an extractor type for future per-position use if needed.

### A.3 Signal Zero-Initialization

**Decision:** Zero-init signal model's profile output layers (weight=0, bias=0) and count output layer (weight=0, bias=−10) so signal contribution starts at identity for both combination rules.

**Rationale:** At initialization, the combined output equals the bias model's output. This lets the bias model establish its Tn5 baseline during early training without interference from random signal model outputs. The signal model gradually activates as its internal representations develop through L_recon gradients on peak examples.

### A.4 Separation Approach: Why Not Adversarial/Gradient Reversal?

**Considered:** Adding a discriminator that tries to predict peak/non-peak from the bias model's representation, with gradient reversal to prevent the bias model from encoding peak-specific information.

**Rejected:** Adds significant complexity (extra network, adversarial training instability). The receptive field constraint + peak-conditioned loss should be sufficient. The hard architectural RF limit is more robust than a soft adversarial penalty. Can revisit if separation is insufficient in practice.

### A.5 Why End-to-End Instead of Two-Stage?

**Motivation:**
1. Simpler training pipeline (single training run vs two sequential runs)
2. Bias model can adapt jointly with signal model -- in two-stage training, if the bias model is imperfect, the signal model must compensate for bias model errors
3. End-to-end allows the two models to find a better joint optimum
4. The loss-based separation is differentiable and principled

**Risk:** Mode collapse where one model dominates. Mitigated by the three separation mechanisms (RF constraint, capacity asymmetry, peak-conditioned loss).

### A.6 DalmatianLoss Config Integration: Nested Base Loss

**Decision:** `DalmatianLoss.__init__` takes `base_loss_cls: str` and `base_loss_args: dict`, and instantiates the base loss internally using `import_class`.

**Alternative:** `DalmatianLoss.__init__` takes `base_loss: nn.Module` (already instantiated). This would require the config system to support nested object instantiation (construct base loss first, then pass to DalmatianLoss). The existing `instantiate_metrics_and_loss` does `loss_cls(**loss_args)` with flat kwargs, so nested module args would require changes.

**Rationale:** The string-based approach works with the existing factory without any changes. The config YAML naturally nests the base loss config inside `loss_args`. `propagate_pseudocount` sets `count_pseudocount` on the outer `loss_args`, and DalmatianLoss forwards it to the base loss via `setdefault`.

## Appendix B: Key Reference Files

### chrombpnet-pytorch (reference implementation)
- `chrombpnet/bpnet.py` -- BPNet model definition
- `chrombpnet/chrombpnet.py` -- ChromBPNet wrapper (main + bias)
- `chrombpnet/model_config.py` -- ChromBPNetConfig defaults
- `chrombpnet/model_wrappers.py` -- BPNetWrapper training logic
- `chrombpnet/interpret.py` -- Interpretation (DeepSHAP via ProfileWrapper/CountWrapper)

### cerberus (target codebase)
- `src/cerberus/models/pomeranian.py` -- Pomeranian model (sub-network base)
- `src/cerberus/layers.py` -- ConvNeXtV2Block, PGCBlock, DilatedResidualBlock
- `src/cerberus/output.py` -- ProfileCountOutput, DalmatianOutput (to add)
- `src/cerberus/loss.py` -- MSEMultinomialLoss, PoissonMultinomialLoss, DalmatianLoss (to add)
- `src/cerberus/module.py` -- CerberusModule._shared_step (line 138-168)
- `src/cerberus/dataset.py` -- CerberusDataset.__getitem__ (peak_status at line 354)
- `src/cerberus/samplers.py` -- PeakSampler (line 1186), MultiSampler.get_peak_status (line 375)
- `src/cerberus/mask.py` -- BigBedMaskExtractor, InMemoryBigBedMaskExtractor
- `src/cerberus/signal.py` -- SignalExtractor, InMemorySignalExtractor
- `src/cerberus/config.py` -- GenomeConfig, DataConfig, SamplerConfig, ModelConfig, import_class, propagate_pseudocount
- `src/cerberus/metrics.py` -- PomeranianMetricCollection, CountProfilePearsonCorrCoef, LogCountsMeanSquaredError
