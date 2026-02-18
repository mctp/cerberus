# Pomeranian V2 Architecture Improvements Plan

## Context

The Pomeranian model (~150k params) is a lightweight CNN for genomic sequence-to-function prediction: it takes 2112bp one-hot DNA and predicts ChIP-seq signal profile shape (1024bp, via multinomial NLL) and total counts (scalar, via MSE). It uses a ConvNeXtV2 stem, 8 PGC (Projected Gated Convolution) body layers with exponential dilations, and decoupled profile + count heads. All layers use valid padding with precise geometric alignment (shrinkage: 20 stem + 1024 body + 44 head = 1088).

These improvements target accuracy gains while maintaining the lightweight parameter budget (~100k-300k) and improving robustness to training hyperparameters and initialization. Each proposal is grounded in published evidence and architectural principles from CNN design, sequence-to-function modeling, and modern neural network engineering.

---

## TIER 1: High-confidence improvements (implement together as "PomeranianV2")

All Tier 1 changes are implemented in a new `PGCBlockV2` class in `layers.py`. The original `PGCBlock` is left untouched. `ConvNeXtV2Block` is NOT modified (it is shared with ASAP/other models). `PomeranianV2` in `pomeranian.py` uses `PGCBlockV2` while the original `Pomeranian` class remains unchanged.

**Target PGCBlockV2 structure:**
```
input -> RMSNorm -> in_proj -> split(X,V) -> depthwise_conv(X) -> X * SiLU(V) -> out_proj -> layer_scale -> drop_path -> residual_add
```

### 1.1 Normalization: Single Pre-Norm

**Change**: Replace the double-norm pattern (norm1 after in_proj + norm2 after out_proj) with a single RMSNorm before in_proj.

**Current V1**: `in_proj -> norm1(2*H) -> split -> conv -> gate -> out_proj -> norm2(D) -> dropout -> residual`
**Proposed V2**: `norm(D) -> in_proj -> split -> conv -> gate -> out_proj -> layer_scale -> drop_path -> residual`

**Reasoning**: Pre-norm is the dominant pattern in modern residual architectures (GPT-2/3, LLaMA, ConvNeXt). Xiong et al. 2020 ("On Layer Normalization in the Transformer Architecture") demonstrated pre-norm provides better gradient flow and training stability. The current double-norm constrains representational capacity -- norm2 re-normalizes the output, partially undoing the learned transformation before it enters the residual stream.

**On removing all internal normalization**: The PGCBlock is structurally a **gated MLP with depthwise conv** (analogous to Mamba's block), NOT a traditional CNN like ConvNeXt. Mamba's block uses exactly this pattern: single pre-norm, no internal normalization, with the depthwise conv operating on unnormalized expanded features. LLaMA's SwiGLU FFN is identical: `RMSNorm -> gate_proj + up_proj -> SiLU(gate) * up -> down_proj`, zero internal norms. The stability concern is addressed by zero-init (1.3) + layer scale (1.4): at initialization the block output is ~0, so even if internal features are poorly normalized, the contribution to the residual stream is negligible. As training progresses and layer_scale grows from 1e-4, the internal weights have adapted.

**Fallback if unstable**: If training shows instability (loss spikes, NaN), add a single RMSNorm after the depthwise conv (between conv and gating), creating a ConvNeXt-inspired "mid-norm" pattern. This should be tried only if pure pre-norm fails.

### 1.2 SiLU (Swish) Gate Activation

**Change**: Apply SiLU activation to the gate branch V before the multiplicative gate. Currently `x = x_conv * v` (linear gate / vanilla GLU). Change to `x = x_conv * F.silu(v)` (SwiGLU-style).

**Reasoning**: Shazeer 2020 ("GLU Variants Improve Transformer") demonstrated SwiGLU consistently outperforms vanilla GLU across tasks. Validated at scale by LLaMA, PaLM, Mistral, Gemma. In convolution architectures, Mamba and Hyena both use SiLU gating. The smooth non-saturating nonlinearity allows richer modulation than linear gating. Zero parameter overhead, negligible compute cost.

### 1.3 Zero-Initialize Output Projection

**Change**: Initialize `out_proj.weight` to zeros in `PGCBlockV2` only. Do NOT modify `ConvNeXtV2Block` (shared with ASAP and other models -- changing its initialization would be an uncontrolled side-effect).

**Reasoning**: Zero-init of the residual path is used by GPT-2/3/4, ConvNeXt (via Layer Scale init to 1e-6), and ReZero (Bachlechner et al. 2020). Each residual block starts as identity, providing well-conditioned gradients from step 1.

### 1.4 Learnable Layer Scale

**Change**: Add a learnable per-channel scale parameter to each PGCBlockV2 output, initialized to 1e-4. Applied after out_proj: `x = x * self.layer_scale[None, :, None]`.

**Reasoning**: Layer Scale (Touvron et al. 2021, CaiT; adopted by ConvNeXt, MetaFormer). Provides smooth ramp-up of each layer's contribution. Works synergistically with zero-init: zero-init sets the starting point, layer scale controls the ramp-up rate. The ConvNeXtV2Block has implicit scaling via GRN (zero-initialized gamma/beta), but the current PGCBlock has no equivalent. Tiny overhead: 8 x 64 = 512 params.

### 1.5 Stochastic Depth (DropPath)

**Change**: Replace `nn.Dropout` on activations with DropPath on the entire residual branch. Use linearly increasing drop rates: 0 at layer 0, up to `drop_path_rate` at the last layer. DropPath is applied to the branch output `x` BEFORE addition with `residual` (i.e., `return residual + drop_path(x)`), not to the sum.

**Reasoning**: Stochastic depth (Huang et al. 2016) is standard in all modern residual architectures (ConvNeXt, DeiT, Swin, BEiT). Key advantages over standard dropout: (1) preserves internal block structure (dropout can break the gating mechanism by zeroing random gate elements), (2) linear schedule means early motif-detection layers are always trained while later context layers get regularized, (3) creates implicit ensemble of sub-networks of different depths.

**New parameter for PomeranianV2.__init__**: `drop_path_rate: float = 0.1`

---

## TIER 2: Medium-confidence improvements (implement selectively after Tier 1 evaluation)

### 2.1 Profile Head: Pre-Norm Residual Block

**Change**: Replace the current `Conv1x1 -> GELU -> Conv_spatial` with a proper pre-norm residual block before the spatial conv:

```python
# __init__:
self.profile_norm = nn.RMSNorm(filters)
self.profile_pointwise1 = nn.Conv1d(filters, filters, kernel_size=1)
self.profile_act = nn.GELU()
self.profile_pointwise2 = nn.Conv1d(filters, filters, kernel_size=1)
self.profile_spatial = nn.Conv1d(filters, n_output_channels, kernel_size=profile_kernel_size, padding='valid')

# forward:
profile_x = self.profile_norm(x.transpose(1,2)).transpose(1,2)
profile_x = self.profile_pointwise1(profile_x)
profile_x = self.profile_act(profile_x)
profile_x = self.profile_pointwise2(profile_x)
profile_x = profile_x + x  # proper 2-layer residual block
profile_logits = self.profile_spatial(profile_x)
```

**Reasoning** (revised from earlier naive skip): The earlier proposal of `x + GELU(Conv1x1(x))` was an unconventional single-layer residual. A standard pre-norm residual block with two pointwise convolutions (expand -> GELU -> compress -> skip) is better grounded. The block learns a nonlinear refinement of the body features, and the skip ensures no information is lost. This follows the standard `x + Block(Norm(x))` pattern. The second pointwise can be zero-initialized so the head starts as pure spatial conv on unrefined features.

**Parameter cost**: One extra Conv1d(64, 64, K=1) = ~4k params. Negligible.

### 2.2 Counts Head: Sigmoid-Gated Pooling

**Change**: Replace GAP (`x.mean(dim=-1)`) with sigmoid-gated pooling (NOT softmax attention).

```python
# __init__:
self.count_gate = nn.Conv1d(filters, 1, kernel_size=1)

# forward:
gate_weights = torch.sigmoid(self.count_gate(x_for_counts))  # (B, 1, L), values in (0,1)
x_pooled = (x_for_counts * gate_weights).mean(dim=-1)  # (B, filters)
```

**Reasoning** (revised from softmax): Softmax over 1024 positions has a winner-take-all tendency -- it can become very peaked on a single position, producing sparse gradients and failing to capture multi-peak signals. Sigmoid gating avoids this by allowing multiple positions to independently have high (or low) weight. Each position gets a gate value in (0, 1) based on its features, and the final pooling is a weighted average. This degrades gracefully to uniform weighting (GAP) if the gate learns to output ~0 everywhere (since sigmoid(0) = 0.5, uniform). 65 new parameters.

### 2.3 Stem: Larger First Kernel (Keep ConvNeXtV2 Structure)

**Change**: Increase the first ConvNeXtV2Block kernel from K=11 to K=15, keeping the full block structure. Do NOT simplify to a plain Conv1d -- the ConvNeXtV2Block's inverted bottleneck and GRN provide nonlinear motif processing that a simple linear conv cannot replicate. The concern about over-parameterization with 4 input channels is addressed by the fact that the first block is dense (groups=1, not depthwise), so it genuinely performs cross-channel mixing of the one-hot input.

**Geometric adjustment**: Stem shrinkage changes from 20 to (15-1)+(11-1)=24. Adjust profile head kernel from K=45 to K=41 (shrinkage 40). Total: 24+1024+40=1088. Output: 2112-1088=1024.

**Reasoning**: BPNet first-layer filters learn TF binding motifs of 6-20bp (Avsec et al. 2021). K=15 captures these more directly than K=11. The ConvNeXtV2Block structure is retained because (a) it preserves nonlinear processing capacity, and (b) changing to a plain Conv1d would be a capacity regression, not simplification.

### 2.4 Expansion Factor = 1.5

**Change**: Increase PGC expansion from 1 to 1.5.

**Reasoning**: With expansion=1, hidden_dim=dim=64. The gate and conv operate at the same width as the residual stream. LyraNet uses 1.5, Mamba uses 2, LLaMA ~2.7. At expansion=1.5 with 8 layers, body params increase from ~107k to ~160k -- still lightweight.

---

## TIER 3: Worth investigating (require empirical validation)

### 3.1 Dilation Schedule: Repeating Cycles

**Change**: Replace `[1,1,2,4,8,16,32,64]` with a repeating cycle like `[1,2,4,8,1,2,4,8]`.

**Geometric constraint**: Shrinkage changes drastically (8*30=240 vs 8*128=1024 with K=9). Requires geometric redesign. Best explored as a separate model variant.

### 3.2 EMA (Exponential Moving Average) of Weights

**Change**: Training pipeline callback, not architecture. Shadow copy with decay=0.999.

### 3.3 Gradient Checkpointing

**Change**: Optional `torch.utils.checkpoint` in PGC tower loop. Trades compute for memory.

---

## Implementation Plan

### Scope and safety

- `PGCBlock` in `layers.py` is **NOT modified**. A new `PGCBlockV2` class is created alongside it.
- `ConvNeXtV2Block` in `layers.py` is **NOT modified**. It is shared with ASAP and other models.
- `Pomeranian` and `PomeranianK5` in `pomeranian.py` are **NOT modified**. A new `PomeranianV2` class is created.
- All changes are additive, ensuring zero regression risk for existing models.

### Files to modify

| File | Changes |
|------|---------|
| `src/cerberus/layers.py` | Add `DropPath` class. Add `PGCBlockV2` class (pre-norm, SiLU gating, zero-init, layer scale, drop_path). |
| `src/cerberus/models/pomeranian.py` | Add `PomeranianV2` using `PGCBlockV2`. Tier 2: profile head residual block, sigmoid-gated count pooling. |
| `tests/test_pomeranian.py` | Add tests for `PomeranianV2`: shapes, geometric alignment, parameter counts, init properties (zero-init verified), DropPath behavior (train vs eval mode). |

### Step-by-step

**Step 1**: Add `DropPath` module to `layers.py`.

**Step 2**: Create `PGCBlockV2` in `layers.py` with all Tier 1 changes. New constructor params: `layer_scale_init: float = 1e-4`, `drop_path: float = 0.0`.

**Step 3**: Create `PomeranianV2` in `pomeranian.py` using `PGCBlockV2`. New constructor param: `drop_path_rate: float = 0.1`. Compute per-layer drop rates with `torch.linspace(0, drop_path_rate, n_layers)`.

**Step 4**: Add Tier 2 improvements to `PomeranianV2`: profile head pre-norm residual block (2.1), sigmoid-gated count pooling (2.2).

**Step 5**: Add tests for PomeranianV2.

### Verification

1. **Unit tests**: `pytest tests/test_pomeranian.py` -- shapes, geometric alignment, parameter counts
2. **Existing model tests**: `pytest tests/` -- verify Pomeranian, ASAP, and all other models are unaffected
3. **Init verification**: Assert `PGCBlockV2.out_proj.weight` is all zeros at init
4. **Train/eval mode**: Assert DropPath is active during training, disabled during eval
5. **Smoke test**: Train PomeranianV2 for 5 epochs, verify loss decreases and no NaN
6. **Stability monitoring**: If pure pre-norm shows instability (loss spikes, NaN), add post-conv RMSNorm as fallback
