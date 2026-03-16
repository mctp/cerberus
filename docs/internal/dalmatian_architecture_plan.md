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

### 3.2.1 Signal Strength Hierarchy Within Accessible Regions

Understanding why architectural constraints and gradient routing are both necessary requires appreciating the hierarchy of forces that shape base-resolution ATAC-seq profiles. Within an accessible region, four factors create structure at different scales and magnitudes:

| Factor | Mechanism | Scale | Magnitude | Sequence-dependent? |
|--------|-----------|-------|-----------|---------------------|
| **TF footprints** | Bound proteins physically block Tn5 | 10–30bp | 10–50× | Yes (TF motifs) |
| **Nucleosome phasing** | Histone octamers wrap ~147bp, exposing minor groove at ~10bp periodicity | ~147bp / 10bp | 3–10× | Partially (positioning signals) |
| **DNA shape** | Minor groove width, roll, propeller twist affect Tn5 accessibility | ~5–10bp | 2–5× | Yes (dinucleotide/trinucleotide) |
| **Tn5 sequence preference** | Enzymatic recognition of ~21bp target motif | ~21bp | 1.2–1.4× | Yes (the PWM we model) |

Several consequences follow:

1. **Tn5 bias is the weakest factor by far.** At each position, Tn5 preference shifts cutting probability from 0.25 (uniform) to ~0.30–0.35 — a ~1.3× enrichment. TF footprints create 10–50× contrast. Without gradient isolation, the bias model will absorb TF footprints (which are much stronger and also local/sequence-dependent) rather than learning the subtle Tn5 motif.

2. **Receptive field alone is insufficient.** TF footprints (10–30bp) and DNA shape effects (5–10bp) both fall within the bias model's ~80bp receptive field. The RF constraint prevents learning nucleosome-scale patterns but NOT TF-scale patterns. Without gradient detach, the bias model's strongest gradient comes from these local regulatory features, not Tn5.

3. **Background-only training is insufficient.** At background depth (~75 counts per 1024bp), the Tn5 signal produces ~0.02 count difference per position — drowned by Poisson noise (σ ≈ 0.26). The multinomial NLL provides essentially zero gradient for the bias model on background regions. The bias model needs peak-region signal where total counts are high enough (hundreds to thousands) for the ~30% Tn5 modulation to produce detectable gradients. But it must receive this gradient only through L_bias, not L_recon.

4. **Standard metrics are blind to bias learning.** A perfect Tn5 predictor achieves only ~0.15 per-window Pearson on background regions (Tn5 bias accounts for 0.6–2% of multinomial NLL at N≈75). Monitoring val_loss or Pearson correlation will show flat curves even when the bias model is learning correctly. Evaluation requires motif analysis of learned filters or aggregated correlation across thousands of windows.

### 3.3 Hard Gradient Routing (Training)

The initial assumption that the bias model could be "softly steered" using only `L_bias_only` proved insufficient, as the gradient signal from `L_recon` on peak regions overwhelmingly dominates the small baseline signal of Tn5 sequence bias (just 1-2% of total loss).

To enforce absolute separation, Dalmatian uses **hard gradient routing**: the bias model's outputs are `.detach()`ed before being combined with the signal model's outputs. 

```
# Detach bias outputs before computing combined logic
combined_logits = bias_logits.detach() + signal_logits
combined_log_counts = logsumexp(bias_log_counts.detach(), signal_log_counts)
```

The dataset provides `peak_status` per example (1 = peak-centered window, 0 = complexity-matched background). This metadata enables three loss terms:

```
L_total = L_recon + lambda_bias * L_bias_only + lambda_sparse * L_signal_sparse
```

| Term | Applied To | Gradients Flow To | Biological Motivation |
|------|-----------|-------------------|----------------------|
| **L_recon** | ALL examples | **Signal model only** | Combined output should match the observed ATAC-seq signal everywhere. The signal model must adapt to the frozen bias baseline. |
| **L_bias_only** | NON-PEAK examples only | **Bias model only** | In non-peak regions, the observed signal is almost entirely Tn5 cutting bias. (May optionally be applied to ALL examples with `.detach()`). |
| **L_signal_sparse** | NON-PEAK examples only | **Signal model only** | The signal model should output ~0 outside peaks. There's no regulatory signal to learn in background regions. |

**Together these create an absolute separation:**
1. The bias model learns exclusively from `L_bias` and is completely blind to peak-specific profiles driven by `L_recon`.
2. The signal model is pushed away from learning Tn5 bias (L_signal_sparse makes it silent outside peaks) and learns the residual signal in peaks relative to the fixed bias representation.

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

*Note: It is strictly critical that the receptive field remains ≤80bp to avoid capturing TF motifs. Previously, higher dilations (`[1, 2, 4, 8]`) with `k=9` inflated RF to 147bp.*

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

        # Hard gradient routing: detach bias outputs before combination.
        # BiasNet receives gradient only from L_bias, never from L_recon.
        if getattr(self, "detach_bias_in_recon", True):
            combined_logits = bias_out.logits.detach() + signal_out.logits
            combined_log_counts = torch.logsumexp(
                torch.stack([bias_out.log_counts.detach(), signal_out.log_counts], dim=-1),
                dim=-1,
            )
        else:
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

---

## Appendix C: Post-Implementation Revision — Gradient-Routed Dalmatian

After implementing and testing the Dalmatian model on real data, a systematic
debug study (5 experiments across 2 datasets, `debug/dalmatian/research_log.md`)
revealed that the soft loss-weighted separation mechanism in Sections 3.3 and
6.1 is fundamentally inadequate. This appendix summarizes the evidence and
prescribes concrete changes to the architecture and loss.

### C.1 Experimental Evidence

Six standalone Pomeranian experiments isolated the bias model's learning capacity:

| Exp | Model | Dataset | Sampler | Val Loss | Pearson Profile | Pearson Counts | Epochs |
|-----|-------|---------|---------|----------|-----------------|----------------|--------|
| 1a | 72K, RF=147bp | kidney (N≈75) | peak+bg | 565 | 0.549 | 0.570 | 12 |
| 1b | 72K, RF=147bp | kidney (N≈75) | bg-only | 266 | 0.443 | 0.466 | 18 |
| 1d | 72K, RF=147bp | kidney (N≈75) | peak, cw=adaptive | 576 | 0.551 | 0.639 | 17 |
| 2a | 220K, RF=155bp | kidney (N≈75) | bg-only | 266 | 0.443 | 0.422 | 17 |
| 2b | 30K, RF=25bp | kidney (N≈75) | bg-only | 274 | 0.431 | 0.449 | 25 |
| 3a | 30K, RF=25bp | K562 (N≈206) | bg-only | 299 | 0.512 | 0.597 | 20 |

Key findings:

1. **Background-only training fails uniformly.** Profile Pearson plateaus at
   ~0.43 (kidney) and ~0.51 (K562) regardless of model capacity. A 7× parameter
   range (30K → 220K) produces identical results. The bottleneck is signal
   strength, not model capacity.

2. **Tn5 bias is an inherently tiny signal.** Published measurements of Tn5
   insertion preference converge on **total IC = 1.0–2.0 bits** across the
   ~21bp motif (Adey et al. 2010: ~1.0 bits from libraries; Li et al. 2021:
   ~1.9 bits from naked DNA; Wolpe et al. 2023: 2.00 bits from naked DNA).
   For comparison, CTCF has 12–15 bits. The strongest single-position effect
   is G at position -4 at 45% (vs 25% uniform) — only 1.8× enrichment
   (Wolpe et al. 2023, Fig 1A).

3. **Tn5 bias is 0.6–2.0% of multinomial NLL on background.** The multinomial
   NLL profile loss at count depth N is dominated by shot noise:
   `L ≈ N × log(L_bins)`. The KL divergence between Tn5-biased and uniform
   insertion is ~0.02–0.07 nats/position, giving a maximum learnable signal of
   1.5–5.3 nats on background (N=75) vs total loss of ~266 nats. The observed
   exp1b loss drop (267.6 → 266.0 = 1.6 nats) matches this prediction exactly.

4. **Pearson correlation cannot detect bias learning.** With N=75 counts in
   1024 bins, a perfect Tn5 predictor achieves per-window Pearson ≈ 0.15 at
   best. The observed Pearson of ~0.44 at epoch 0 reflects the count head's
   regional coverage prediction (GC content, mappability), not profile shape.

5. **Higher coverage helps counts but not profiles.** K562 (N≈206) shows higher
   starting Pearson (~0.51 vs ~0.43) but equally flat profile learning, confirming
   the limitation is fundamental to bp-resolution multinomial NLL at any
   realistic ATAC-seq depth.

6. **Tn5 has the highest sequence bias among common enzymes** but it is still
   weak in absolute terms (Wolpe et al. 2023, Fig 1A):

   | Enzyme | Total IC (bits) |
   |--------|----------------|
   | Tn5 | 2.00 |
   | DNase I | 1.06 |
   | MNase | 0.77 |
   | Cyanase | 0.53 |
   | Benzonase | 0.50 |

7. **Strand-merging attenuates Tn5 IC by ~30–50%.** Mao et al. 2024 (PRINT)
   showed that merging + and − strand cut sites into unstranded signal reduces
   per-position IC from ~0.25–0.3 bits (per-strand) to ~0.15–0.2 bits (unstranded
   +4/−4), a ~30–50% loss. With the asymmetric +4/−5 shift convention, the
   unstranded motif is nearly destroyed (logos are flat). The symmetric +4/−4
   convention (default in `tools/scatac_pseudobulk.py`) is essential for preserving
   any Tn5 motif in unstranded training data. The effective total IC in our
   unstranded BigWig training data is therefore ~0.7–1.2 bits, pushing the
   learnable signal toward the lower end of the 0.6–2% loss range.

7. **Tn5 bias is sample-specific.** TraceBIND (Avsec et al. 2025) showed that
   Tn5 bias correlates highly within a sample but poorly across labs/studies,
   and sample-specific correction reduces footprinting false positives by >60×.

### C.2 Why Soft Separation Fails

The current DalmatianLoss (Section 6.1) relies on three loss terms:

```
L_total = L_recon(all) + bias_weight × L_bias(bg) + sparse_weight × L_signal_bg(bg)
```

The multinomial NLL scales linearly with counts N. In a 50/50 peak/background
batch with the kidney data:

| Term | Regions | Effective N | Loss magnitude | % of total |
|------|---------|-------------|----------------|------------|
| L_recon | all (peak-dominated) | ~380 | ~560 | ~96% |
| L_bias | background only | ~75 | ~266 × 0.5 = ~20 | ~3.4% |
| L_signal_bg | background only | — | ~0.1–1 | ~0.2% |

**The bias model receives >95% of its gradient from L_recon on peak examples.**
It learns peak-specific signal (GC content, dinucleotide patterns) rather than
Tn5 bias. The auxiliary terms L_bias and L_signal_bg are invisible in the
gradient landscape.

This is not fixable by reweighting. Even with `bias_weight=50`, the bias model
still sees L_recon gradients that are structurally different from Tn5 bias.
The problem is that L_recon provides a gradient path from peaks to BiasNet —
any gradient flowing through this path teaches BiasNet the wrong thing.

### C.3 Specific Corrections to This Document

1. **Section 3.1 (RF Constraint):** The plan proposed RF≈80bp for BiasNet
   (dilations=[1,1,2,4], k=5). The implementation used dilations=[1,2,4,8],
   k=9 → RF=147bp. This is too large — 147bp allows BiasNet to capture short
   TF motifs (6–20bp) and partial nucleosome features. The Tn5 motif extends
   only ±11bp from the insertion site (Wolpe et al. 2023). **Revert to RF≤80bp.**

2. **Section 3.3 (Peak-Conditioned Loss):** The claim that L_bias "effectively
   steers the bias model toward Tn5 bias" is incorrect. L_bias contributes
   ~3% of total gradient. The bias model is steered primarily by L_recon on
   peaks, which teaches it peak-specific patterns. **Soft loss weighting is
   insufficient; hard gradient routing is required.**

3. **Section 4.3 (Zero-Init Training Dynamics):** The plan states "L_recon
   trains only the bias model effectively" at epoch 0 and that "the bias model
   converges on Tn5 bias." The first part is correct (SignalNet outputs zero),
   but the second is wrong — L_recon on peaks teaches BiasNet to explain peak
   profiles, not Tn5 bias. **BiasNet converges on the wrong target before
   SignalNet activates.**

4. **Section 6.1 (DalmatianLoss):** The base loss config should include
   `count_weight="adaptive"` (ChromBPNet-style `median_count / 10`). Without
   this, count loss is ~1.2 vs ~560 profile loss — the count head barely
   trains. Experiment 1d showed adaptive count_weight improved count Pearson
   from 0.570 to 0.639.

5. **Section 3.2 (Capacity Asymmetry):** The claim that "even within its
   limited receptive field, the bias model can only represent simple local
   sequence patterns" is misleading. The debug study showed a 7× capacity
   range (30K–220K params) produces identical background Pearson. The bottleneck
   is signal strength (Tn5 bias = 0.6–2% of loss), not model capacity. Capacity
   asymmetry is a secondary defense, not a primary separation mechanism.

### C.4 Recommended Architecture: Gradient-Routed Dalmatian

Replace soft loss-weighted separation with hard gradient routing in the forward
pass. The bias model's outputs are **detached** before combining with SignalNet,
severing the gradient path from L_recon to BiasNet.

#### C.4.1 Forward Pass Change

```python
# In Dalmatian.forward():
def forward(self, x):
    bias_out = self.bias_model(x)
    signal_out = self.signal_model(x)

    # Hard gradient routing: detach bias outputs before combination.
    # BiasNet receives gradient only from L_bias, never from L_recon.
    if self.detach_bias_in_recon:
        combined_logits = bias_out.logits.detach() + signal_out.logits
        combined_log_counts = torch.logsumexp(
            torch.stack([bias_out.log_counts.detach(),
                         signal_out.log_counts], dim=-1),
            dim=-1,
        )
    else:
        combined_logits = bias_out.logits + signal_out.logits
        combined_log_counts = torch.logsumexp(
            torch.stack([bias_out.log_counts,
                         signal_out.log_counts], dim=-1),
            dim=-1,
        )

    return FactorizedProfileCountOutput(
        logits=combined_logits,
        log_counts=combined_log_counts,
        bias_logits=bias_out.logits,
        bias_log_counts=bias_out.log_counts,
        signal_logits=signal_out.logits,
        signal_log_counts=signal_out.log_counts,
    )
```

Add `detach_bias_in_recon: bool = True` to the constructor. Default `True`
enables hard separation; `False` reverts to the original soft separation for
comparison experiments.

#### C.4.2 Gradient Flow After Detach

| Term | Gradients to BiasNet | Gradients to SignalNet |
|------|---------------------|-----------------------|
| L_recon (combined, all) | **None** (detached) | Yes |
| L_bias (bias-only, bg) | Yes | None |
| L_signal_bg (signal, bg) | None | Yes |

BiasNet learns exclusively from L_bias on background regions. SignalNet learns
from L_recon on all regions, seeing the full combined target minus a frozen
bias contribution. This is analogous to ChromBPNet's freeze-then-train but
end-to-end: the bias contribution is treated as a fixed offset in L_recon
while being optimized separately through L_bias.

#### C.4.3 Reduced BiasNet RF Defaults

Revert BiasNet to the originally planned receptive field:

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| `bias_dilations` | [1, 2, 4, 8] | [1, 1, 2, 4] | Limit RF to ~80bp |
| `bias_dil_kernel_size` | 9 | 5 | Smaller spatial kernel |
| `bias_profile_kernel_size` | 17 | 21 | Matches Section 4.1 |
| **Resulting RF** | **147bp** | **~80bp** | Tn5 motif is ±11bp |

The Tn5 insertion motif extends ±11bp from the cut site (Wolpe et al. 2023),
with broader compositional effects out to ±15bp. An 80bp RF is generous for
capturing this while excluding TF motifs and nucleosome-scale features.

#### C.4.4 Adaptive Count Weight

Add `count_weight: "adaptive"` to the default `base_loss_args`:

```yaml
loss_args:
  base_loss_cls: "cerberus.loss.MSEMultinomialLoss"
  base_loss_args:
    count_weight: "adaptive"   # median_count / 10
    profile_weight: 1.0
  bias_weight: 1.0
  signal_background_weight: 0.1
```

#### C.4.5 Consider Training Bias on All Regions

With `.detach()` in place, it becomes safe to compute L_bias on **all**
examples (peak + background), not just background. The gradient path from
L_recon to BiasNet is severed, so BiasNet cannot learn peak-specific patterns
through the combined output. Training on peaks gives the bias model 5× more
signal per example (N≈343 vs N≈75 for kidney data), dramatically improving
Tn5 bias learning.

This is a departure from ChromBPNet's background-only protocol, but with hard
gradient routing the risk of contamination is eliminated. The bias model's
limited RF (≤80bp) provides a second layer of protection — it architecturally
cannot represent long-range regulatory patterns even if exposed to peak regions.

Whether L_bias should cover all regions or just background is an empirical
question. Recommend testing both configurations:

```python
# Option A: Background-only (current, conservative)
if non_peak.any():
    l_bias = self.base_loss(bias_out[non_peak], target[non_peak])

# Option B: All regions (proposed, with detach providing safety)
l_bias = self.base_loss(bias_out, target)
```

### C.5 Summary: From Soft to Hard Separation

| Mechanism | Original (Sections 3, 6) | Revised |
|-----------|-------------------------|---------|
| **RF constraint** | RF=147bp (too large) | RF≤80bp (matches Tn5 motif) |
| **Capacity asymmetry** | Primary mechanism | Secondary (capacity is not the bottleneck) |
| **Loss weighting** | Soft: bias_weight × L_bias | Hard: `.detach()` severs gradient path |
| **Bias training data** | Background only | Background or all (safe with detach) |
| **Count weight** | Default (negligible) | Adaptive (median_count / 10) |

The original design assumed that loss weighting could steer two jointly-trained
models toward different targets. The debug study proved this fails when one
signal (Tn5 bias, 1–2% of loss) is overwhelmed by another (peak-specific
profile, 96% of loss). Hard gradient routing via `.detach()` replaces the soft
pressure with an absolute constraint: BiasNet's parameters are invisible to
L_recon, period. Combined with the architectural RF constraint, this gives
Dalmatian the same separation guarantees as ChromBPNet's two-stage protocol
while retaining the simplicity of end-to-end training.

### C.6 Additional Refinements & Future Work

Based on experimental results, several minor but important refinements are recommended:

1. **Investigate Batch-Level Bias Normalization:**
   The adaptive count weight (`median_count / 10`) handles global sequence depth, but local variations can cause large batch-to-batch loss variance. Consider calculating adaptive weights dynamically per-batch or introducing a global scale normalization factor.

2. **Explicit Bias-Only Evaluation Metrics:**
   Extend `PomeranianMetricCollection` to explicitly output metrics for the `bias_model` evaluated on background regions in isolation. Currently, it evaluates the combined output only. An isolated evaluation provides real-time signal regarding whether the bias model collapses.

3. **Continuous Signal Intensity Weighting for `L_bias`:**
   Instead of a strict binary threshold (`peak_status` = 1 or 0), investigate applying a continuous weight mapping (via sigmoid on background total counts) for `L_bias` to naturally taper the penalty for ambiguous regulatory elements near the calling threshold.

4. **Testing Signal Net with a Learnable Gamma:**
   ChromBPNet employs a fixed scalar ($\gamma$) to match count distributions. While Dalmatian uses `logsumexp` to bypass fixed scaling organically, introducing a single learnable scalar parameter before the `logsumexp` combination step may allow for natural adaptation if the bias model systematically under-predicts the background rate when exposed to peak contexts.

### C.7 Literature References for Tn5 Bias Strength

| Study | Method | Total IC (bits) | Key finding |
|-------|--------|----------------|-------------|
| Goryshin & Reznikoff 1998 | In vitro/in vivo | — | Consensus: A-GNTYWRANCT-T (9bp core) |
| Adey et al. 2010 | Sequencing libraries | ~1.0 | Max IC=0.16 bits/pos; "little impact at level of coverage" |
| Green et al. 2012 | Comparative | — | Tn5 biased toward G/C |
| Li et al. 2021 | Naked DNA, 8 species | ~1.9 | Max IC≈0.35 bits/pos; only 16–29% of insertions in motif |
| Wolpe et al. 2023 | Naked DNA, rule ensemble | **2.00** | Highest of 5 enzymes; G at pos -4 = 45% |
| Mao et al. 2024 (PRINT) | PWM from ATAC-seq | — | +4/−4 shift preserves Tn5 motif in unstranded signal; +4/−5 destroys it. Per-strand IC ~0.25–0.3 bits/pos; unstranded +4/−4 ~0.15–0.2; unstranded +4/−5 ~flat. Strand-merging attenuates IC by ~30–50%. |
| Pampari et al. 2025 | ChromBPNet | — | Background-only bias, frozen, then subtracted |
| TraceBIND 2025 | Sample-specific mtDNA | — | Bias varies across labs; >60× FP reduction |

## Appendix D: Designing a Better Bias Model for Weak Signals

The debug study (Appendix C) established that Tn5 bias is a tiny signal: 1–2
bits total IC, 0.6–2% of multinomial NLL on background. Five experiments across
two datasets showed the bottleneck is signal strength, not model capacity (7×
parameter range had no effect). This appendix outlines design principles for
better capturing weak enzymatic bias signals.

The core insight: **when the signal is weak and simple, match the model to the
signal's structure rather than making the model bigger.** The problem is not the
model — it is the loss function, the training signal, and the evaluation metric.

### D.1 Count-Invariant Profile Loss

**Problem.** Multinomial NLL scales linearly with total counts N:

```
L_multinomial = -sum_i(target_i * log_softmax(logits)_i) ≈ N * log(L_bins) + bias_signal
```

At N=75 counts in 1024 bins, `N * log(1024) ≈ 520 nats` of irreducible shot
noise dwarfs the ~1.5–5.3 nat Tn5 signal. The gradient is ~98% noise.

**Fix.** Use a loss that compares normalized distributions, removing count-depth
dependence:

```python
# KL divergence on normalized distributions:
pred_dist = F.log_softmax(logits, dim=-1)                    # predicted shape
obs_dist = target / target.sum(dim=-1, keepdim=True)         # observed shape
loss = F.kl_div(pred_dist, obs_dist, reduction='batchmean')
```

Now a 2-bit motif contributes ~0.02–0.07 nats/position regardless of count
depth. The Tn5 signal fraction increases from ~1% to a much larger share of
the loss.

**Caveat.** At N=75, the observed normalized distribution is extremely noisy —
93% of bins are zero, and any single window's profile is a poor estimate of
the true shape. A count-invariant loss should be combined with smoothing or
aggregation (D.3) to be effective.

### D.2 Structured PWM Output

**Problem.** A general Pomeranian (conv→dilated→profile head) has tens of
thousands of parameters for learning what is fundamentally a ~21bp
position-weight matrix with 84 free parameters (4 nucleotides × 21 positions).
The excess capacity lets the model learn GC content, dinucleotide frequencies,
and regional coverage patterns instead of the Tn5 motif.

**Fix.** Replace the free-form profile head with a structured output that
encodes what Tn5 bias actually is — a short convolutional motif:

```python
class StructuredBiasModel(nn.Module):
    """Bias model that directly learns a position-weight matrix.

    Instead of predicting a free 1024bp profile, learns a short PWM
    and convolves it with the input sequence. This has exactly the right
    inductive bias for enzymatic sequence preference.
    """

    def __init__(self, motif_width: int = 21):
        super().__init__()
        # Learnable PWM: (1, 4, motif_width) log-odds over uniform
        self.pwm = nn.Parameter(torch.zeros(1, 4, motif_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L) one-hot DNA sequence
        # Convolve PWM with sequence → per-position bias score
        bias_logits = F.conv1d(x, self.pwm, padding='same')  # (B, 1, L)
        return bias_logits
```

This model has **84 parameters** and is exactly the right inductive bias for
Tn5. It physically cannot learn anything except a local sequence preference.

**Extensions.** If 1st-order PWM is insufficient (Tn5 has weak dinucleotide
dependencies), add a small number of wider or higher-order filters:

```python
# 1st-order PWM (21bp):                   84 params
# + dinucleotide filter (16 × 20bp):     320 params
# + DNA shape (4 features × 21bp):        84 params
# Total:                                  488 params
```

This remains orders of magnitude smaller than even the tiniest Pomeranian
(30K params) and encodes the known structure of enzymatic sequence bias.

**Integration with Dalmatian.** The structured bias model replaces
`self.bias_model` in the Dalmatian class. Its output (per-position logits)
feeds into the same combination rule (logit addition with SignalNet). The count
head can remain a small MLP since regional count variation (GC, mappability) is
a legitimate bias-related signal.

### D.3 Aggregate-Then-Evaluate (and Train)

**Problem.** Per-window Pearson correlation is noise-limited. With N=75 counts
in 1024 bins, a perfect Tn5 predictor achieves Pearson ≈ 0.15. The metric
cannot distinguish a model that learned Tn5 bias from one that learned nothing.

**Fix: aggregate evaluation.** Average predicted and observed profiles across
thousands of windows before computing metrics:

```python
# Aggregate across K windows (e.g., K=10,000):
mean_pred = pred_profiles.mean(dim=0)    # (1, L) — noise cancels
mean_obs = obs_profiles.mean(dim=0)      # (1, L) — noise cancels
# SNR improves by sqrt(K): with K=10,000, noise drops 100×
# Now Pearson can detect the 1.8× enrichment at position -4
agg_pearson = pearsonr(mean_pred, mean_obs)
```

This is analogous to how sequence logos are computed: no single insertion site
is informative, but millions of sites averaged together reveal the motif.

**Aggregate training loss.** The same principle can be applied during training.
Instead of computing loss per window and averaging, accumulate predicted
profiles across a mini-batch (or across several mini-batches using a running
average) and compute the loss on the aggregate:

```python
# Conceptual: aggregate-then-loss (not standard PyTorch per-example loss)
batch_pred = F.softmax(logits, dim=-1).mean(dim=0)   # avg predicted shape
batch_obs = (target / target.sum(-1, keepdim=True)).mean(dim=0)  # avg observed shape
loss = kl_div(batch_pred.log(), batch_obs)
```

This increases the effective N per gradient step by the batch size, improving
the signal-to-noise ratio of the gradient proportionally. With batch_size=64,
effective N goes from 75 to ~4,800, making Tn5 bias ~8% of the loss instead
of ~1%.

**Caveat.** Aggregation assumes the Tn5 motif is the same across all windows
(translation-invariant). This is true by definition for a sequence-only bias
model — the Tn5 preference depends only on local sequence, not genomic
position. However, aggregation over heterogeneous sequences dilutes the
position-specific signal; alignment to cut sites or grouping by local k-mer
composition would improve it.

### D.4 Alternative Evaluation: Motif Recovery

Instead of Pearson correlation, evaluate whether the bias model has recovered
the known Tn5 motif:

1. **Filter inspection.** Extract the learned conv filters from the first layer
   and compare to the Adey/Li/Wolpe Tn5 PWM using TOMTOM or Pearson on the
   information content vectors.

2. **Aggregated insertion profile.** Predict profiles for 100K+ background
   windows, align by observed insertion sites, average the predicted bias
   around each insertion. A model that learned Tn5 bias should show the
   characteristic G enrichment at position -4.

3. **Comparison to known PWM.** Compute Pearson between the model's predicted
   per-position log-odds and the Tn5 PWM from Wolpe et al. (total IC = 2.00
   bits). This directly measures whether the model captured the motif,
   independent of shot noise.

### D.5 Summary: Design Priorities

| Design choice | Impact | Why |
|---|---|---|
| Structured PWM output | **High** | Encodes exactly what Tn5 bias is (21bp motif, 84 params) |
| Count-invariant loss | **High** | Removes shot-noise dominance from gradient |
| Aggregate evaluation | **High** | Makes weak signal visible in metrics |
| Train on all regions (with detach) | **Medium** | 5× more signal per example |
| Aggregate training loss | **Medium** | Increases effective N per gradient step |
| Motif-based evaluation | **Medium** | Directly tests whether the right thing was learned |
| More parameters | **None** | Debug study proved capacity is not the bottleneck |
| Larger receptive field | **Negative** | Lets model learn non-bias patterns |

The overarching principle: when the signal is weak and simple, **shrink the
model to match the signal's structure** (D.2), **remove the noise from the loss**
(D.1), and **aggregate to beat down shot noise** (D.3). A 84-parameter PWM with
a count-invariant loss and aggregated evaluation would likely outperform any
general-purpose conv net on this task.
