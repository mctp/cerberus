# BPNet vs Pomeranian: The Performance–Interpretability Tradeoff

*Why a better-predicting model produces worse TF-MoDISco motifs, and what to do about it*

---

# Part I: The Observation

Side-by-side comparison of BPNet and Pomeranian on the same data shows:

| Property | BPNet | Pomeranian |
|---|---|---|
| Train/Val/Test metrics | Lower | Higher |
| Overfitting | More | Less |
| Hyperparameter sensitivity | Higher | Lower |
| Training speed (epochs) | Faster | Slower |
| Training speed (wall time/epoch) | Faster | Slower |
| TF-MoDISco motif quality | Clean, biologically convincing | Noisy, fragmented |

Pomeranian is the better *predictor*. BPNet is the better *explainee*. This document explains why, identifies the specific architectural mechanisms responsible, and proposes concrete remediation strategies for both models.

---

# Part II: Architectural Comparison

## 2.1 BPNet: A Piecewise-Linear Additive Stack

BPNet's forward path is:

```
one-hot → Conv1d(4→64, k=21) → ReLU → [DilatedResidualBlock × 8] → Conv1d(64→C, k=75) → profile
                                                                   ↘ GAP → Linear → log_counts
```

Each `DilatedResidualBlock` (default: `residual_post-activation_conv`) computes:

```python
# layers.py:350-354 — DilatedResidualBlock.forward
out = self.act(self.conv(x))           # act = ReLU
residual = self._center_crop_to_length(x, out.shape[-1])
return residual + out                  # x + ReLU(Conv(x))
```

The profile head is a single linear convolution:

```python
# bpnet.py:139-145
self.profile_conv = nn.Conv1d(filters, self.n_output_channels,
                              kernel_size=profile_kernel_size, padding="valid")
```

**Key mathematical properties:**

1. **Piecewise-linear.** ReLU is the only non-linearity. Within each activation region, the entire network from input to profile output is a linear function of the input.
2. **Additive residuals.** Each block adds its contribution: $h^{(\ell+1)} = \text{crop}(h^{(\ell)}) + \text{ReLU}(W_\ell * h^{(\ell)})$.
3. **Linear profile head.** The final profile logit at position $t$ is a direct linear combination of tower features: $z_t = \sum_{c,\tau} U_{c,\tau} \, h^{(L)}_c(t+\tau)$.
4. **Standard dense convolutions.** Each `Conv1d` mixes all channels at every position — no depthwise/grouped structure.

## 2.2 Pomeranian: A Gated, Normalized, Non-Linear Stack

Pomeranian's forward path is:

```
one-hot → [ConvNeXtV2Block × 2] → [PGCBlock × 8] → Conv1d(1×1) → GELU → Conv1d(k=45) → profile
                                                   ↘ GAP → Linear → GELU → Linear → log_counts
```

### The PGC Block: Multiplicative Gating

Each `PGCBlock` computes:

```python
# layers.py:208-244 — PGCBlock.forward (full mode)
x = self.in_proj(x)                      # pointwise: dim → 2*hidden_dim

x = x.transpose(1, 2)
x = self.norm1(x.float()).type_as(x)      # RMSNorm
x = x.transpose(1, 2)

x, v = torch.chunk(x, 2, dim=1)          # split into X and V
x = self.conv(x)                          # depthwise dilated conv on X

# ... center-crop v to match x ...

x = x * v                                # ← MULTIPLICATIVE GATING

x = self.out_proj(x)                      # pointwise: hidden_dim → dim

x = x.transpose(1, 2)
x = self.norm2(x.float()).type_as(x)      # RMSNorm
x = x.transpose(1, 2)

x = self.dropout(x)
return residual + x
```

The gating operation `x * v` is the critical difference. For any element $z = x \cdot v$:

$$\frac{\partial z}{\partial x} = v, \qquad \frac{\partial z}{\partial v} = x$$

The attribution of $x$ depends on $v$, and vice versa. The same input motif produces different attributions depending on what the gate branch $v$ encodes from surrounding context.

### The ConvNeXtV2 Stem: Global Response Normalization

The stem uses GRN, which introduces a global dependency across all positions:

```python
# layers.py:21-24 — GRN1d.forward
Gx = torch.norm(x, p=2, dim=(1,), keepdim=True)           # L2 norm over positions → (B, 1, C)
Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)          # normalize across channels → (B, 1, 1)
return self.gamma * (x * Nx) + self.beta + x               # scale + residual
```

At this point, $x$ has shape `(B, L, C)` (after the permute in `ConvNeXtV2Block`). The norm $G_c = \|x_{\cdot,c}\|_2$ aggregates over *all positions* for each channel, making every position's output depend on the global activation pattern. This is another multiplicative interaction: $x_{t,c}$ is scaled by a factor derived from $\{x_{s,c}\}_{s=1}^L$.

Note: GRN appears only in the ConvNeXtV2 stem, not in the PGC tower. The tower uses RMSNorm, which normalizes across channels at each position independently — a weaker but still non-trivial inter-channel dependency.

### The Non-Linear Profile Head

```python
# pomeranian.py:163-170
self.profile_pointwise = nn.Conv1d(filters, filters, kernel_size=1)
self.profile_act = nn.GELU()
self.profile_spatial = nn.Conv1d(filters, self.n_output_channels,
                                  kernel_size=profile_kernel_size, padding="valid")
```

Unlike BPNet's single linear convolution, Pomeranian interposes a GELU between two convolutions. This means the final profile logit is a *non-linear* function of the tower features, breaking the direct linear readout that makes BPNet's attributions clean.

### Dropout: No Direct Effect, But Shapes the Representation

BPNet has **no dropout anywhere** in its architecture. Pomeranian applies `Dropout(0.1)` in every PGCBlock, after the second normalization and before the residual addition:

```python
# layers.py:195 — PGCBlock.__init__
self.dropout = nn.Dropout(dropout)

# layers.py:244 — PGCBlock.forward (after norm2, before residual add)
x = self.dropout(x)
return residual + x
```

Dropout is deterministic at inference time (it becomes an identity), so it does not directly interfere with attribution methods — DeepLIFT, ISM, and integrated gradients all see a clean, dropout-free forward pass. However, dropout has an important **indirect** effect on interpretability through the representation it encourages during training.

During training, dropout randomly zeros 10% of the features before each residual addition. This means no single feature channel can be relied upon — the model must spread predictive information across multiple redundant channels. The result is a more **distributed, redundant encoding** where the same motif's signal is carried by many features rather than concentrated in a few strong detectors. This is precisely why Pomeranian overfits less (redundancy = robustness to noise), but it also means that attribution scores for a motif are spread thinly across many channels rather than concentrated in a sharp, localized pattern.

This is a subtle but important contributor to the interpretability gap. BPNet, with no dropout, is free to develop sharp, specialized motif detectors — individual channels that fire strongly for specific patterns. These produce high-magnitude, spatially localized attribution scores that TF-MoDISco clusters cleanly. Pomeranian's dropout-trained features are more diffuse by construction, producing lower-magnitude, more distributed attributions that fragment during clustering.

## 2.3 Summary Table

| Mechanism | BPNet | Pomeranian | Effect on Attribution |
|---|---|---|---|
| Tower non-linearity | ReLU (piecewise-linear) | Multiplicative gating ($x \cdot v$) | Gating makes attribution context-dependent |
| Normalization | None | RMSNorm (tower) + GRN (stem) | Creates inter-position/inter-channel dependencies |
| Convolution structure | Dense Conv1d | Depthwise + pointwise (factored) | Distributes features across more parameters |
| Profile head | Linear Conv1d | Conv1d → GELU → Conv1d | Non-linear head breaks linear readout |
| Residual blocks | $x + \text{ReLU}(\text{Conv}(x))$ | $x + \text{Dropout}(\text{Norm}(\text{Proj}((\text{Conv}(X)) \cdot V)))$ | More complex path = noisier backprop |
| Regularization | None | Dropout (0.1) in every PGCBlock | No direct effect on attribution (deterministic at inference), but shapes representation toward distributed encoding |

---

# Part III: Why TF-MoDISco Favors BPNet

## 3.1 How TF-MoDISco Works

TF-MoDISco does not evaluate predictive performance. Its pipeline is:

1. **Input:** Per-base hypothetical contribution scores (all 4 bases × each position) from DeepLIFT/DeepSHAP across many sequences.
2. **Seqlet extraction:** Identify short windows with high total attribution.
3. **Alignment and clustering:** Align seqlets by similarity, cluster into metaclusters.
4. **Output:** Contribution Weight Matrices (CWMs) — motif-like patterns.

This pipeline works best when the same biological motif produces a **stable, reproducible attribution signature** across instances. That assumption is approximately satisfied by BPNet and systematically violated by Pomeranian.

## 3.2 The Attribution Stability Problem

### BPNet: Stable Attributions

Because BPNet is piecewise-linear with a linear profile head, the attribution of a GATA motif at position $t$ is approximately:

$$\text{attribution}(t) \approx \sum_\ell W_\ell^{\text{eff}} \cdot \text{one\_hot}(t)$$

where $W_\ell^{\text{eff}}$ is the effective weight matrix along the active ReLU path. Within a linear region, this is constant — the same GATA motif in different sequences (if the ReLU activation pattern is similar) gets approximately the same attribution. Seqlets cluster tightly. TF-MoDISco recovers clean PWMs.

### Pomeranian: Context-Dependent Attributions

For Pomeranian, the attribution of the same GATA motif depends on:

1. **The gate values $v$** at every PGC layer — which encode surrounding sequence context.
2. **The GRN scaling factors** in the stem — which depend on the global activation pattern.
3. **The GELU in the profile head** — which applies a non-linear transform conditioned on the channel mix.

The same motif in different contexts produces different attribution magnitudes, different attribution shapes, and different relative importance compared to flanking positions. When TF-MoDISco extracts seqlets for this motif, they vary across instances. The clustering algorithm sees noisy, inconsistent patterns and either:

- Fragments the motif into multiple weak clusters, or
- Merges it with other motifs that happen to co-occur, or
- Misses it entirely if the attribution is too diffuse.

## 3.3 DeepLIFT's Specific Problem with Multiplicative Gates

DeepLIFT propagates contribution scores using rules that decompose the output change $\Delta z = z - z_0$ (relative to a reference) into input contributions. For the gating operation $z = x \cdot v$:

$$\Delta z = x \cdot v - x_0 \cdot v_0 = \underbrace{(x - x_0) \cdot v_0}_{\text{attr. to } x} + \underbrace{x_0 \cdot (v - v_0)}_{\text{attr. to } v} + \underbrace{(x - x_0)(v - v_0)}_{\text{cross-term}}$$

The cross-term $(x - x_0)(v - v_0)$ has no principled decomposition. The Rescale rule distributes it proportionally; the RevealCancel rule handles it differently. Both introduce approximation error. With 8 stacked PGC blocks, each containing one multiplicative gate, this error compounds through the network.

This is distinct from gradient noise — it is an inherent limitation of the DeepLIFT decomposition for non-additive operations.

## 3.4 What Pomeranian Has Actually Learned

Pomeranian has not learned "wrong" biology. It has almost certainly learned the same motifs BPNet has, **plus** context-dependent interactions between them: motif-motif cooperativity, spacing effects, flanking-sequence modulation, nucleosome-scale compositional signals. These are real biological phenomena, and modeling them is why Pomeranian predicts better.

The problem is not that Pomeranian's representation is wrong — it is that the representation is *distributed* and *context-conditional*, which is exactly what TF-MoDISco's clustering step assumes biology is not.

---

# Part IV: Other Attribution Methods

## 4.1 ISM (In-Silico Mutagenesis)

ISM computes $f(x) - f(x_{\text{mut}})$ via forward passes. It does not backpropagate through the network and is therefore immune to the DeepLIFT cross-term problem.

**Per-sequence ISM** on Pomeranian should reveal motifs clearly — the forward pass correctly captures the model's response to mutations regardless of internal architecture.

**However**, if ISM scores are fed into TF-MoDISco, the same clustering problem remains: Pomeranian genuinely responds differently to the same motif in different contexts (because that is what it learned), so ISM effect sizes will vary across loci, and seqlet clustering will still fragment.

**Additional caveat:** TF-MoDISco expects hypothetical contribution scores (all 4 bases at each position). Standard single-mutation ISM produces actual-base scores only (zeros where the input is zero). A full saturation mutagenesis matrix can approximate the hypothetical format, but this is 3× more forward passes per position and is not the standard ISM workflow.

## 4.2 Integrated Gradients

Integrated gradients (IG) integrates along a path from reference to input:

$$\text{IG}_i = (x_i - x_i^0) \int_0^1 \frac{\partial f(x^0 + \alpha(x - x^0))}{\partial x_i} d\alpha$$

This handles non-linearities more faithfully than DeepLIFT (no rule-based decomposition), but the resulting attributions still reflect the model's context-dependence. IG on Pomeranian will produce more accurate per-instance attributions than DeepLIFT, but the downstream TF-MoDISco clustering will still struggle with cross-instance variability.

## 4.3 Gradient × Input

The noisiest option for Pomeranian. Raw gradients through multiplicative gates and normalization layers produce high-variance attribution maps. Not recommended for TF-MoDISco with either architecture.

## 4.4 TPCAV: Concept-Based Attribution as an Alternative to TF-MoDISco

A recent method — **TPCAV** (Testing with PCA-projected Concept Activation Vectors; Yang & Mahony, 2025) — sidesteps the entire TF-MoDISco pipeline and its associated bottlenecks. Rather than decomposing the input-output function into per-base scores and clustering them, TPCAV asks a different question: **is a known biological concept linearly separable in the model's internal embedding space?**

### How TPCAV Works

1. **Concept construction:** For a concept of interest (e.g., the GATA motif), generate positive examples by inserting motif instances (sampled from the PWM) into random genomic sequences. Negative examples are unmodified random sequences.
2. **PCA decorrelation:** Extract hidden-layer activations for both sets. Apply PCA to decorrelate the embedding features — this is critical because models with dropout or redundant encoding (like Pomeranian) spread concept information across many correlated channels.
3. **Linear classifier:** Train a linear SVM in the decorrelated space to distinguish concept-present from concept-absent. The normal vector to the decision hyperplane is the **Concept Activation Vector (CAV)** — the direction in embedding space that encodes the concept.
4. **Scoring:** The TPCAV score for each concept is derived from the projection of DeepLIFT layer attributions onto the CAV direction, with sign determined by the classifier boundary.

### Why TPCAV Bypasses the Pomeranian Bottlenecks

The two problems we identified for Pomeranian + TF-MoDISco were:

1. **DeepLIFT cross-term error at multiplicative gates** — TPCAV uses Layer DeepLIFT (attribution computed at a chosen hidden layer, not backpropagated to the input). The attribution only needs to pass through the layers *above* the probed layer (typically just the head). If the probed layer is the final tower output, the multiplicative PGC blocks are upstream and don't affect the attribution computation.

2. **Context-dependent attributions fragmenting during clustering** — TPCAV does not cluster seqlets. It operates globally: one linear classifier per concept, evaluated across many sequences. If the same GATA motif produces different activation patterns in different contexts, the PCA decorrelation + linear classifier can still find the shared concept direction as long as it is linearly separable — a much weaker requirement than "seqlets must look identical."

The PCA step is particularly well-suited to Pomeranian's dropout-induced distributed encoding. If motif signal is spread across 20 correlated channels (because dropout prevented concentration), PCA recovers the principal direction that captures the shared variance — exactly the concept direction.

### What TPCAV Can and Cannot Do

**Strengths relative to TF-MoDISco:**

- **Architecture-agnostic.** Works on CNNs, transformers, foundation models (DNABERT-2, HyenaDNA), multimodal models (sequence + chromatin). The paper demonstrates successful concept recovery on an Enformer-inspired architecture with attention, BatchNorm, and skip connections — all components we flagged as hostile to TF-MoDISco.
- **Input-format agnostic.** Works with tokenized DNA (k-mer encodings) where TF-MoDISco cannot operate at all (it requires one-hot input).
- **Handles redundant/correlated features.** The PCA decorrelation step is designed for exactly the scenario that Pomeranian creates.
- **Global scoring.** Produces a single aggregated score per concept, avoiding the ambiguity where TF-MoDISco can assign the same motif to both positive and negative pattern clusters.
- **Concept-specific attribution maps.** By projecting activations onto the CAV column space (via QR decomposition) and zeroing the orthogonal component during DeepLIFT, TPCAV can produce per-base attribution maps for a specific concept. This recovers spatial resolution for known motifs even in distributed representations.

**Limitations relative to TF-MoDISco:**

- **Concept-driven, not discovery-driven.** You must define what concepts to test. TPCAV can confirm "does Pomeranian use GATA?" but it cannot discover novel, unanticipated motifs the way TF-MoDISco can. For exploratory analysis on a new system, TF-MoDISco remains essential.
- **Requires a concept database.** The paper uses the Vierstra non-redundant human motif clustering database, plus repeat and chromatin accessibility concepts. Coverage depends on the completeness of the database.
- **Linear separability assumption.** If a concept is encoded in a genuinely non-linear manifold in embedding space (not just correlated linear features), the linear SVM will miss it. This is unlikely for well-defined motifs but possible for complex combinatorial concepts.

### Implications for the BPNet/Pomeranian Comparison

TPCAV suggests a **third path** beyond the two remediation strategies in Part V:

- Path 1: Improve BPNet's performance while keeping it interpretable via TF-MoDISco (Section 5.1).
- Path 2: Make Pomeranian more TF-MoDISco-compatible by linearizing the head and removing GRN (Section 5.2).
- **Path 3: Keep Pomeranian's full architecture and use TPCAV for concept-level interpretation.**

Path 3 preserves Pomeranian's performance and training robustness entirely — no architectural compromises. The tradeoff is shifting from discovery-driven motif finding (TF-MoDISco) to hypothesis-driven concept testing (TPCAV). In practice, a combined workflow is likely optimal:

1. Use **BPNet + TF-MoDISco** for initial motif discovery (what motifs matter?).
2. Use **Pomeranian + TPCAV** for concept validation and quantification (how much does each known motif contribute? which peaks use which motifs? are there task-specific differences?).
3. Use **Pomeranian + concept-specific attribution maps** for spatial visualization of known motifs in individual sequences.

This workflow leverages each model's strength: BPNet's additive structure for discovery, Pomeranian's superior prediction for quantitative concept analysis.

## 4.5 Summary

| Method | BPNet quality | Pomeranian quality | Bottleneck for Pomeranian |
|---|---|---|---|
| DeepLIFT/DeepSHAP → TF-MoDISco | Excellent | Poor | Cross-term error + clustering |
| ISM (per-sequence) | Good | Good | None (forward-pass exact) |
| ISM → TF-MoDISco | Good | Moderate | Clustering (context-variability) |
| Integrated Gradients → TF-MoDISco | Good | Moderate | Clustering (context-variability) |
| Gradient × Input | Fair | Poor | Gradient variance + clustering |
| TPCAV (concept-level) | Good | Good (expected) | Requires predefined concepts; not discovery-driven |

The pattern is clear: **per-instance** attribution methods work fine on Pomeranian; the problem is **aggregation across instances** via clustering. TPCAV avoids clustering entirely by operating at the concept level in embedding space, making it the most promising existing method for interpreting Pomeranian-class architectures with known biological concepts.

---

# Part V: Remediation

## 5.1 Improving BPNet Without Destroying Interpretability

The goal is to close the performance gap with Pomeranian while preserving BPNet's piecewise-linear, additive structure. The key constraint: **no multiplicative gating, no normalization layers in the tower, and keep the linear profile head.**

### 5.1.1 Weight Normalization (Already Supported)

BPNet already supports `weight_norm=True`, which decouples weight magnitude from direction without introducing any non-linear operations into the forward pass:

```python
# bpnet.py:159-163
if weight_norm:
    _apply_weight_norm(self.iconv)
    for block in self.res_layers:
        if isinstance(block, DilatedResidualBlock):
            _apply_weight_norm(block.conv)
```

Weight normalization is a *reparameterization* ($W = g \cdot \hat{v}/\|\hat{v}\|$), not an activation normalization. During inference, it folds into a standard linear weight matrix. DeepLIFT's linear passthrough rule applies unchanged. This is the safest first step for improving BPNet training dynamics.

### 5.1.2 GELU Activation (Already Supported)

Switching from ReLU to GELU (`activation="gelu"`) provides smoother gradients and reduces dying-neuron risk. GELU is a smooth monotonic-ish non-linearity — while not strictly piecewise-linear, DeepLIFT/DeepSHAP handle it reasonably well via the rescale rule, and the attributions remain local and stable because there is no multiplicative interaction between different feature dimensions.

The model remains additive: $x + \text{GELU}(\text{Conv}(x))$. No context-dependence is introduced.

### 5.1.3 Wider Initial Convolution or Factored Stem

BPNet's initial conv (k=21) is the narrowest bottleneck for capturing motif features. Options:

- **Wider kernel** (k=25 or k=29): Captures longer motifs directly. Increases parameters linearly with kernel size, remains fully linear.
- **Factored stem** (two convolutions, e.g., k=11 + k=11 with ReLU between): Similar effective receptive field to k=21 but with a non-linearity between layers, allowing the stem to learn more complex initial features. The additive structure is preserved because there is no cross-feature multiplication.

```python
# Hypothetical factored stem for BPNet — additive, no gating
self.iconv1 = nn.Conv1d(4, filters, kernel_size=11, padding="valid")
self.iconv2 = nn.Conv1d(filters, filters, kernel_size=11, padding="valid")
# forward:
x = F.relu(self.iconv1(x))
x = self.iconv2(x)                 # linear into tower (or with activation)
```

### 5.1.4 Deeper Tower with Smaller Kernels

Pomeranian uses k=9 with dilations `[1,1,2,4,8,16,32,64]`; BPNet uses k=3 with dilations `[2,4,...,256]`. The effective receptive field per layer is `dilation × (kernel - 1)`:

- BPNet top layer: $256 \times 2 = 512$
- Pomeranian top layer: $64 \times 8 = 512$

These are comparable, but Pomeranian's larger kernel at lower dilation captures *denser* local patterns. BPNet could benefit from k=5 with adjusted dilations to increase local feature density while preserving the same total receptive field. The structure remains $x + \text{ReLU}(\text{Conv}(x))$ — fully piecewise-linear and additive.

### 5.1.5 Pre-Activation Residual Architecture

The `residual_pre-activation_conv` variant (`x + Conv(ReLU(x))`) is already implemented and can improve gradient flow through deep stacks (He et al. 2016, "Identity Mappings in Deep Residual Networks"). This changes nothing about the attribution properties — the same operations in a different order, still piecewise-linear and additive.

### 5.1.6 What NOT to Do

Do not add to BPNet:

- **Normalization layers** (BatchNorm, LayerNorm, RMSNorm) — introduces inter-feature dependencies in attribution.
- **Gating / GLU / SwiGLU** — introduces multiplicative non-linearity.
- **Squeeze-and-excitation / channel attention** — makes attributions channel-context-dependent.
- **Non-linear profile head** — breaks the linear readout that produces clean motif attributions.
- **Dropout in the tower** — deterministic at inference, so no direct effect on attribution methods. But dropout during training pushes the model toward distributed, redundant representations where no single channel is a clean motif detector (see Section 2.2, "Dropout: No Direct Effect, But Shapes the Representation"). This is a significant contributor to Pomeranian's interpretability gap that operates entirely through representation geometry, not attribution mechanics.

## 5.2 Improving Pomeranian's Interpretability Without Sacrificing Performance

The goal is to make Pomeranian's attributions more stable across instances while retaining the gated architecture's capacity and training robustness.

### 5.2.1 Linear Profile Head Variant

The single highest-impact change. Replace the non-linear decoupled head:

```python
# Current (pomeranian.py:163-170) — non-linear
self.profile_pointwise = nn.Conv1d(filters, filters, kernel_size=1)
self.profile_act = nn.GELU()
self.profile_spatial = nn.Conv1d(filters, self.n_output_channels,
                                  kernel_size=profile_kernel_size, padding="valid")
```

with a single linear convolution:

```python
# Proposed — linear, BPNet-style
self.profile_conv = nn.Conv1d(filters, self.n_output_channels,
                              kernel_size=profile_kernel_size, padding="valid")
```

This restores the linear readout from tower features to profile logits. The tower still uses gating (preserving capacity), but the final projection is transparent to attribution methods. This is the cleanest test of whether the interpretability gap comes from the head vs. the body.

**Expected effect:** Moderate improvement in TF-MoDISco motif quality. The tower's gating still creates context-dependent features, but the linear head ensures those features map to profile logits in a stable, position-by-position manner.

### 5.2.2 Remove GRN from Stem

GRN's global position-aggregation is the most aggressive source of attribution non-locality. Replace it with a simpler mechanism or remove it:

```python
# Current ConvNeXtV2Block (layers.py:69-72)
if grn:
    self.grn = GRN1d(self.inv_bottleneckwidth)
else:
    self.grn = nn.Identity()
```

Pomeranian's stem can be instantiated with `grn=False`. This removes the global L2 norm dependency while retaining the ConvNeXtV2 block's depthwise conv → expand → compress structure. The tower's RMSNorm (per-position, across channels) is a much weaker dependency and likely acceptable.

### 5.2.3 Reduce Tower Depth / Gating Stages

Each PGC block adds one multiplicative gate and its associated cross-term error. Fewer layers = fewer compounding stages. A shallower tower (e.g., 6 layers instead of 8) with wider kernels could maintain receptive field coverage with fewer gating operations:

```python
# Fewer gates, wider kernels — same receptive field
dilations = [1, 2, 4, 16, 64, 128]      # 6 layers
dil_kernel_size = 13                      # wider kernel compensates
```

This is a direct tradeoff: each removed layer eliminates one source of cross-term error but reduces the model's sequential depth. Worth testing empirically.

### 5.2.4 SiLU Gate Activation (From PomeranianV2 Plan)

The PomeranianV2 architecture plan proposes replacing the linear gate `x * v` with a SiLU-activated gate `x * SiLU(v)`. While this doesn't directly improve interpretability (it's still multiplicative), the SiLU activation makes the gate more decisive — values are pushed toward 0 or pass-through, which can produce *sharper* gating patterns. Sharper gates mean more consistent attribution patterns across instances for the same motif:

```python
# Current: x = x_conv * v                    (linear gate)
# SiLU:    x = x_conv * F.silu(v)            (activated gate)
```

If the gate learns to be approximately binary (on/off) for specific motif detectors, the cross-term error shrinks because either $\Delta x \approx 0$ or $\Delta v \approx 0$ for most positions.

### 5.2.5 Interpretation-Aware Training (Soft Constraint)

An advanced approach: add an auxiliary loss that encourages attribution stability. For example, a penalty on the variance of attribution magnitudes for high-confidence motif regions across the training batch. This is speculative and complex to implement, but it directly targets the root cause.

### 5.2.6 Hybrid Architecture: Gated Body + Linear Head + No GRN

Combining 5.2.1, 5.2.2, and optionally 5.2.3:

```python
# Pomeranian-Interpretable: gated body, linear head, no GRN
Pomeranian(
    conv_kernel_size=[11, 11],     # ConvNeXtV2 stem with grn=False
    n_dilated_layers=8,            # PGC tower (gating preserved)
    profile_kernel_size=75,        # wider linear head (BPNet-style)
    # ... with linear profile head and no GRN ...
)
```

This preserves: gated feature interactions, dropout regularization, depthwise/pointwise factorization, RMSNorm stability.

This removes: GRN's global coupling, non-linear head, (optionally) depth.

**Expected effect:** The model retains most of Pomeranian's training robustness (from gating + normalization in the tower) while producing more interpretable attributions (from the linear head + removed GRN). The gating-induced context-dependence remains, but it is "read out" linearly, reducing attribution variance.

---

# Part VI: Experimental Validation Plan

To separate *attribution-method incompatibility* from *genuinely distributed representations*, run the following experiments on the same data:

## 6.1 Diagnostic Experiments

| Experiment | Question Answered | Method |
|---|---|---|
| ISM on both models → TF-MoDISco | Is the gap attribution-method bias or representation? | If ISM-MoDISco gap is small, blame DeepLIFT. If large, blame the representation. |
| Per-sequence ISM heatmaps | Can Pomeranian's motifs be seen per-instance? | Visual inspection of top-scoring loci. |
| Pomeranian with linear head → DeepSHAP → TF-MoDISco | How much does the non-linear head contribute? | Retrain, then run standard interpretation pipeline. |
| Pomeranian without GRN → DeepSHAP → TF-MoDISco | How much does GRN contribute? | Retrain with `grn=False` in stem blocks. |
| TPCAV on both models (same concept set) | Does Pomeranian encode known motifs as well as BPNet? | If TPCAV concept scores are comparable, the gap is in the TF-MoDISco pipeline, not the representation. |
| TPCAV concept-specific attribution maps on Pomeranian | Can known motifs be spatially resolved in Pomeranian? | Extract per-base maps for top concepts; compare sharpness to BPNet TF-MoDISco CWMs. |

## 6.2 Remediation Experiments

| Experiment | Hypothesis |
|---|---|
| BPNet + weight_norm + GELU | Closes part of the performance gap, motifs unchanged. |
| BPNet + factored stem (k=11,11) | Better low-level features, motifs unchanged. |
| BPNet + k=5, adjusted dilations | Denser local features, motifs unchanged. |
| Pomeranian with linear head | Moderate motif improvement, small performance cost. |
| Pomeranian with linear head + no GRN | Larger motif improvement, moderate performance cost. |
| Pomeranian with fewer layers (6) + wider kernels | Fewer gating stages, test receptive field tradeoff. |

## 6.3 Motif Insertion / Marginal Footprinting

Instead of relying solely on TF-MoDISco clustering, test interpretability via **motif insertion experiments**: insert known motifs (e.g., CTCF, GATA) into random background sequences and measure the predicted response. This bypasses attribution entirely and directly tests whether the model has learned the motif. If both BPNet and Pomeranian respond correctly to inserted motifs, the issue is confirmed to be in the attribution/clustering pipeline, not in the learned biology.

---

# Part VII: Key Takeaways

1. **The gap is real and architecturally determined.** Pomeranian's multiplicative gating, GRN, and non-linear head create context-dependent attributions that TF-MoDISco's clustering step cannot cleanly aggregate. BPNet's piecewise-linear additive structure produces stable attributions by design.

2. **Better prediction does not imply better interpretability.** Pomeranian predicts better *because* it models context-dependent regulatory grammar. TF-MoDISco assumes motifs have context-independent attribution signatures. These assumptions are in direct tension.

3. **The problem is partly the method, partly the model.** DeepLIFT's cross-term approximation error at multiplicative gates makes things worse, but even exact (ISM-based) attributions will show context-variability because the model genuinely learned context-dependent features. The clustering step is the true bottleneck.

4. **Both models can be improved.** BPNet can adopt training improvements (weight norm, GELU, wider stems) without touching its interpretability. Pomeranian can swap to a linear head and remove GRN to reduce attribution instability while retaining its gated tower.

5. **The cleanest experiment is the linear-head ablation.** A Pomeranian with a linear profile head directly tests how much of the interpretability gap comes from the head (fixable) vs. the tower (fundamental to the architecture's advantage).

6. **Concept-based methods bypass the bottleneck entirely.** TPCAV (Yang & Mahony, 2025) operates in embedding space rather than input space, using PCA decorrelation to handle the distributed representations that Pomeranian creates. It is architecture-agnostic and does not require seqlet clustering. The tradeoff is that it is concept-driven (you must specify what to look for) rather than discovery-driven (TF-MoDISco finds novel motifs). A practical workflow combines BPNet + TF-MoDISco for discovery with Pomeranian + TPCAV for quantitative concept validation.

---

# Part VIII: Design Principles for Interpretability-Friendly Architectures

The preceding analysis focuses on BPNet and Pomeranian specifically. This section generalizes: what architectural properties make a model maximally compatible with TF-MoDISco and gradient-based attribution methods?

## 8.1 What TF-MoDISco Requires from the Model

TF-MoDISco's pipeline — extract seqlets, align, cluster into CWMs — implicitly assumes the model's input-output relationship is approximately decomposable as:

$$f(x) \approx \sum_{i} g_i(\text{motif}_i, \text{position}_i) + \text{baseline}$$

an **additive function of local motif contributions**. When this holds, the attribution for motif $M$ at position $t$ is approximately $g_M$ regardless of what else is in the sequence. Seqlets are stable across instances, clustering works.

This is satisfied when four properties hold simultaneously:

1. **Local features** — each intermediate representation depends on a bounded input window.
2. **Additive combination** — features combine by summation, not multiplication.
3. **Linear readout** — the output is a linear function of the final feature map.
4. **Concentrated encoding** — predictive signal for a motif lives in few channels, not distributed across many.

## 8.2 The Maximally Interpretable Architecture Template

Working backward from these requirements:

```
one-hot → [Wide Linear Conv] → ReLU → [Additive Residual Tower] → [Linear Conv] → output
                                        (no norm, no gating,
                                         no dropout, no attention)
```

Each component choice is driven by the four properties above.

### Activations: Piecewise-Linear Only

ReLU and LeakyReLU are ideal. Within each linear region, the entire network is a linear function of the input, and DeepLIFT's propagation rules are exact (no approximation error at all). GELU is acceptable — it is smooth and monotonic, so the rescale rule works reasonably well. The critical constraint is: avoid any activation that creates **interactions between different features** (gating, softmax, sigmoid applied to cross-channel combinations).

### Residual Connections: Strictly Additive

$x + f(x)$ is clean. The contribution of layer $\ell$ adds independently to all subsequent layers. DeepLIFT decomposes this trivially. The pre-activation variant $x + \text{Conv}(\text{ReLU}(x))$ is equally clean — same operations, different order.

### No Normalization in the Body

BatchNorm, LayerNorm, RMSNorm, and GRN all create dependencies between features or positions. The Jacobian $\partial y_i / \partial x_j$ becomes non-zero for $i \neq j$, meaning the attribution of position $i$ depends on what is happening at position $j$. This is the mechanism that makes the same motif produce different attributions in different contexts.

If normalization is needed for training stability, **weight normalization** is the safe choice — it is a reparameterization of the weight matrix ($W = g \cdot \hat{v} / \|\hat{v}\|$), not a runtime operation on activations. At inference it folds into a standard linear layer. BPNet already supports this:

```python
# bpnet.py:159-163
if weight_norm:
    _apply_weight_norm(self.iconv)
    for block in self.res_layers:
        if isinstance(block, DilatedResidualBlock):
            _apply_weight_norm(block.conv)
```

### Linear Profile Head

The single most impactful property for TF-MoDISco quality. If the output is:

$$z_t = \sum_{c,\tau} W_{c,\tau} \, h_c(t+\tau)$$

then the attribution at each position is a direct linear projection of the tower features. There is a clean, interpretable mapping: "this channel detects this pattern, and contributes this much to the output at this position." A non-linear head (Conv → GELU → Conv) breaks this — the same tower feature can contribute differently to the output depending on the activation of other features at the same position.

### No Dropout

As discussed in Section 2.2, dropout does not break attribution methods at inference, but it shapes the representation during training toward distributed, redundant encodings. Without dropout, the model is free to develop sharp, specialized motif detectors — individual channels that fire strongly for specific patterns. Weight decay alone provides sufficient regularization for this architecture class.

### Dense (Not Depthwise) Convolutions

Dense `Conv1d` layers where each filter sees all input channels can learn complete motif detectors in a single filter. Depthwise-separable convolutions force the model to factor pattern detection into per-channel spatial operations + cross-channel mixing, distributing the motif representation across the factorization. Dense convolutions are more parameter-expensive per layer but produce more concentrated features.

## 8.3 Extensions Beyond BPNet

BPNet already satisfies most of the above. To push interpretability *further* while staying within the additive/linear framework:

### 8.3.1 Sparse Activations

If only a few channels are active at each position (most are zero), attributions become even more concentrated. Options:

- **Top-k activation**: After ReLU, keep only the top-k channels per position, zero the rest. Forces the model to encode each motif in a small number of specialist channels.
- **Grouped sparsity**: Partition channels into groups, enforce that only one group is active per position (like a mixture-of-experts over local patterns). Each group becomes an interpretable "motif slot."

Both stay within the piecewise-linear, additive framework. They don't introduce multiplicative interactions — they just increase the sparsity of the ReLU activation pattern.

### 8.3.2 Explicit Motif-Like First Layer

Instead of learning the initial convolution freely, use a wide first-layer convolution (k=19 to k=25) with **no subsequent non-linearity before the residual tower**. This forces the first layer to act as a bank of linear motif scanners — essentially learned PWMs. The rest of the network combines their outputs additively. Attributions through this layer are trivially interpretable: each first-layer filter's contribution is a linear function of the input one-hot encoding.

BPNet already approximates this (initial Conv → ReLU → tower), but making the first layer wider and keeping it strictly linear (moving the ReLU to after the first residual block) sharpens the motif-detector interpretation:

```python
# Hypothetical: explicit linear motif bank
self.motif_bank = nn.Conv1d(4, filters, kernel_size=25, padding="valid")
# No activation here — first layer is a pure linear scanner
# ReLU applied inside the first residual block instead
```

### 8.3.3 Linear Bottleneck Before the Head

A 1×1 convolution that reduces channels (e.g., 64 → 16) before the linear profile head forces the model to compress its representation into fewer dimensions. Fewer active features = sparser attributions = cleaner clustering. The compression must be linear (no activation between the bottleneck and the spatial head) to preserve the linear readout property:

```python
# Bottleneck before profile head — both layers linear
self.bottleneck = nn.Conv1d(filters, 16, kernel_size=1)     # compress
self.profile_conv = nn.Conv1d(16, n_output_channels,
                              kernel_size=profile_kernel_size, padding="valid")
# forward:
profile_logits = self.profile_conv(self.bottleneck(x))       # linear chain
```

This is equivalent to a single lower-rank convolution, but by making the bottleneck dimension explicit, it provides a tunable knob for the sparsity/capacity tradeoff.

### 8.3.4 Dense Skip Connections

Force the model to be explicitly additive over layer contributions. Instead of only the last layer feeding the head, sum all layers' outputs with learned scalar weights:

$$z = W_{\text{head}} * \left( \sum_{\ell=0}^{L} \alpha_\ell \, h^{(\ell)} \right)$$

This is DenseNet-style aggregation where all layers contribute directly to the output. Each layer can be interpreted independently, and attributions decompose cleanly across layers. The scalar weights $\alpha_\ell$ are interpretable too — they indicate which resolution scale matters most for the task.

```python
# Dense skip connections — all layers contribute to head
alphas = nn.Parameter(torch.ones(n_dilated_layers + 1))    # learnable weights

# forward:
features = [x]                   # x after initial conv
for layer in self.res_layers:
    x = layer(x)
    features.append(center_crop_to_match(x, features))

# Weighted sum (all cropped to same length)
combined = sum(a * f for a, f in zip(F.softmax(alphas, dim=0), features))
profile_logits = self.profile_conv(combined)
```

## 8.4 Architectures That Resist Interpretability

For contrast, these architectural patterns are maximally hostile to TF-MoDISco:

### Self-Attention (Enformer, Borzoi)

- **Multiplicative**: Attention weights are $\text{softmax}(QK^T/\sqrt{d})$, creating the same cross-term problem as gating.
- **Global**: Every position attends to every other position, making attributions maximally context-dependent.
- **Dynamic**: The effective "kernel" changes per input, so the same motif gets different attributions in different sequences.

In practice, attention-based genomics models (Enformer, Borzoi) are interpreted via ISM or contribution scores computed on the full attention graph, and TF-MoDISco results are noticeably noisier than for BPNet-class models.

Note: **additive attention** (Bahdanau-style: $a_{ij} = v^T \tanh(W_q q_i + W_k k_j)$) is somewhat better — the query and key contributions are additive before the tanh, reducing (but not eliminating) cross-term issues.

### Gated Architectures (Mamba, Hyena, Pomeranian)

The $x \cdot v$ gating pattern, while powerful for learning, creates the cross-term decomposition problem at every layer. Stacking 8+ gated layers compounds the error.

### Heavy Normalization (any architecture with BatchNorm/LayerNorm in every block)

Each normalization layer redistributes attribution across the features it normalizes. In a tower with 8 blocks, each containing one or more normalization layers, the cumulative effect can substantially blur motif attributions.

## 8.5 Summary: The Interpretability-Friendly Design Space

| Property | Most interpretable | Acceptable | Avoid |
|---|---|---|---|
| Activation | ReLU, LeakyReLU | GELU | Gating, Softmax, SiLU on cross-channel |
| Residual | Additive ($x + f(x)$) | Dense/skip connections | Multiplicative ($x \cdot f(x)$) |
| Normalization | None, or weight norm | — | BatchNorm, LayerNorm, RMSNorm, GRN |
| Profile head | Single linear Conv1d | Linear bottleneck → linear Conv1d | Non-linear (Conv → Act → Conv) |
| Regularization | Weight decay only | Early stopping | Dropout (encourages distributed encoding) |
| Convolution | Dense Conv1d | — | Depthwise-separable |
| Feature sparsity | Top-k, grouped | ReLU (natural sparsity) | Dense activations (GELU, SiLU) |
| Context | Local (dilated conv) | — | Global (attention, global pooling) |

The fundamental insight is that **interpretability through TF-MoDISco is not a free property of any good model — it requires the architecture to be approximately an additive function of local features**. This is a strong inductive bias that limits model capacity. BPNet works well for TF-MoDISco precisely because it enforces this bias. Models that relax it (Pomeranian, Enformer) gain capacity at the cost of clean motif decomposition.
