# Cerberus-Tangermeme Integration: Compatibility Report & Strategy

## 1. Executive Summary

This document analyzes the compatibility between cerberus (genomic sequence-to-function models) and tangermeme (post-hoc sequence analysis toolkit), identifies blocking incompatibilities, and proposes a wrapper-based integration strategy.

**Bottom line:** The two packages are compatible at the input level (encoding, tensor format) but incompatible at the output level. Cerberus models return structured dataclass outputs (`ProfileCountOutput`) that tangermeme cannot process. A thin `TangermemeWrapper(nn.Module)` resolves all incompatibilities while keeping the packages decoupled.

---

## 2. Tangermeme Capability Overview

Tangermeme provides post-hoc analysis tools for any PyTorch sequence model:

| Tool | Function | What it does |
|------|----------|-------------|
| `predict` | `predict(model, X)` | Batched GPU inference with memory management |
| `marginalize` | `marginalize(model, X, motif)` | Measure effect of inserting a motif |
| `ablate` | `ablate(model, X, start, end)` | Measure effect of shuffling out a region |
| `saturation_mutagenesis` | `saturation_mutagenesis(model, X)` | All single-nucleotide substitution effects (ISM) |
| `deep_lift_shap` | `deep_lift_shap(model, X)` | DeepLIFT/SHAP attribution scores |
| `variant_effect` | `substitution_effect(model, X, subs)` | Predict effect of specific variants |
| `space` | `space(model, X, motifs, spacing)` | Test cooperative effects at different motif spacings |
| `design` | `greedy_substitution(model, X, ...)` | Optimize sequences toward target predictions |

**Model interface tangermeme expects:**
```python
# Minimum contract:
y = model(X)  # X: (B, 4, L) → y: torch.Tensor or tuple of torch.Tensors

# For deep_lift_shap specifically:
y = model(X)  # Must return (B, n_targets) — 2D tensor only
```

Source: `tangermeme/predict.py:119-124` — raises `ValueError` for any output that is not `torch.Tensor` or `tuple/list` of tensors.

---

## 3. Cerberus Model Interface Summary

Cerberus BPNet and Pomeranian models follow a dual-head architecture:

```python
# Forward signature (both BPNet and Pomeranian):
def forward(self, x: torch.Tensor) -> ProfileCountOutput:
    # x: (B, 4, input_len)
    # Returns dataclass, NOT a raw tensor
    ...
```

**ProfileCountOutput** (defined in `src/cerberus/output.py:40-49`):
```python
@dataclass
class ProfileCountOutput(ProfileLogits):
    log_counts: torch.Tensor  # (B, 1) or (B, C)

# Inherited from ProfileLogits:
    logits: torch.Tensor      # (B, C, L) — unnormalized multinomial parameters
```

**Model specifications:**

| Model | Input shape | Profile logits | Log counts | Parameters |
|-------|-----------|----------------|------------|------------|
| BPNet | `(B, 4, 2114)` | `(B, 1, 1000)` | `(B, 1)` | ~65k |
| BPNet1024 | `(B, 4, 2112)` | `(B, 1, 1024)` | `(B, 1)` | ~80k |
| Pomeranian | `(B, 4, 2112)` | `(B, 1, 1024)` | `(B, 1)` | ~150k |

**Output semantics:**
- `logits`: Unnormalized log-probabilities over spatial positions. `softmax(logits, dim=-1)` gives the predicted shape (probability of reads at each position).
- `log_counts`: `log1p(total_counts)` — the predicted magnitude in log scale. `expm1(log_counts)` recovers total read count.
- **Reconstructed counts** (the biologically meaningful signal): `softmax(logits, dim=-1) * expm1(log_counts).unsqueeze(-1)` → `(B, C, L)`.

---

## 4. Detailed Compatibility Matrix

### 4.1 Compatible aspects

| Aspect | Cerberus | Tangermeme | Status |
|--------|----------|------------|--------|
| Alphabet order | ACGT (A=0, C=1, G=2, T=3) via `sequence.py:19` | ACGT (A=0, C=1, G=2, T=3) via `utils.py:352` | Compatible |
| Input tensor shape | `(B, 4, L)` float32 | `(B, len(alphabet), L)` | Compatible |
| Base class | `torch.nn.Module` | Expects `torch.nn.Module` | Compatible |
| Eval mode | Standard PyTorch | Sets `.eval()` + `torch.no_grad()` | Compatible |
| Device handling | Standard PyTorch | Auto CPU↔GPU transfer | Compatible |
| N/unknown bases | All-zero columns | All-zero columns | Compatible |

### 4.2 Incompatible aspects

#### Issue 1: Output type mismatch (BLOCKING)

**Cerberus** returns `ProfileCountOutput` (a Python dataclass).
**Tangermeme** checks output type at `predict.py:119-124`:

```python
if isinstance(y_, torch.Tensor):
    y_ = y_.cpu()
elif isinstance(y_, (list, tuple)):
    y_ = tuple(yi.cpu() for yi in y_)
else:
    raise ValueError("Cannot interpret output from model.")
```

A `ProfileCountOutput` is none of these types → `ValueError` is raised.

**Impact:** All tangermeme operations fail immediately.

#### Issue 2: Output dimensionality for DeepLIFT/SHAP (BLOCKING)

`deep_lift_shap` requires `(B, n_targets)` output (stated at `deep_lift_shap.py:228-232`):

> "NOTE: predictions MUST yield a (batch_size, n_targets) tensor, even if n_targets is 1."

Cerberus profile logits are `(B, C, L)` — 3D. Even after fixing Issue 1, the spatial dimension must be reduced to a scalar per target for DeepLIFT/SHAP.

**Impact:** `deep_lift_shap` fails even with a naive tuple wrapper.

#### Issue 3: Output semantic decomposition (DESIGN CONCERN)

Cerberus separates shape (logits) from magnitude (log_counts). Neither component alone represents the full predicted signal. This creates a design choice for integration:

- **Reconstructed counts** `softmax(logits) * expm1(log_counts)`: Full biological signal, comparable to BigWig tracks. Appropriate for marginalization, variant effect analysis.
- **Profile logits**: Shape-only signal without magnitude. Appropriate for shape-focused attribution (e.g., "where does the model predict reads?") and for ISM on the multinomial distribution.
- **Log counts**: Global magnitude only. Appropriate for count-level analysis (e.g., "does this motif increase total binding?").

No single output mode is universally correct — the choice depends on the analysis question.

#### Issue 4: Fixed input length requirement (USAGE CONCERN)

Cerberus models use valid padding throughout, requiring exact input lengths:
- BPNet: exactly 2114 bp
- BPNet1024 / Pomeranian: exactly 2112 bp

Tangermeme does not enforce input lengths. Operations that modify sequence length (deletion_effect, insertion_effect) will break. Operations that preserve length (predict, marginalize, ISM, deep_lift_shap, ablate, substitution_effect) work correctly if the user provides correctly-sized input.

**Impact:** User must ensure input sequences match model requirements. Some tangermeme operations (deletion/insertion variant effects) are architecturally incompatible without padding/trimming logic.

---

## 5. Tangermeme Tuple Output Handling

Tangermeme natively supports models that return tuple/list outputs in most (but not all) operations. Understanding this is important for the wrapper design because cerberus models naturally produce two outputs (logits and log_counts).

### 5.1 How tangermeme processes tuples — step by step

In `predict.py`, tuple outputs go through three stages:

**Stage 1: Per-batch type detection** (`predict.py:119-124`)
```python
if isinstance(y_, torch.Tensor):
    y_ = y_.cpu()
elif isinstance(y_, (list, tuple)):
    y_ = tuple(yi.cpu() for yi in y_)  # Each element moved to CPU independently
else:
    raise ValueError("Cannot interpret output from model.")
```

**Stage 2: Accumulation** (`predict.py:126`)
Each batch result is appended to a list:
```python
y.append(y_)
# After all batches, y = [(logits_b0, counts_b0), (logits_b1, counts_b1), ...]
```

**Stage 3: Batch concatenation** (`predict.py:128-132`)
```python
if isinstance(y[0], torch.Tensor):
    y = torch.cat(y)          # Single tensor: concatenate across batches
else:
    y = [torch.cat(y_) for y_ in list(zip(*y))]  # Tuple: transpose then cat
```

The `zip(*y)` transposes the nested structure:
```
Before zip: [(logits_b0, counts_b0), (logits_b1, counts_b1), ...]
After zip:  [(logits_b0, logits_b1, ...), (counts_b0, counts_b1, ...)]
```
Then `torch.cat` concatenates each group along the batch dimension.

**Final result for a cerberus model in `"both"` mode:**
```python
y = [
    logits_all_batches,    # y[0]: (N, C, L) — all logits concatenated
    log_counts_all_batches # y[1]: (N, 1)    — all log_counts concatenated
]
```

**Key point:** The elements are kept completely separate. No element is picked, averaged, or combined with any other element. The return type is `list[Tensor]` where each position corresponds to one tuple element from the model's `forward()`.

### 5.2 What happens to tuple elements in each downstream operation

After `predict` returns a `list[Tensor]`, each tangermeme operation that calls `predict` receives this list and must handle it. Here is a precise trace of what each operation does:

#### `marginalize` — transparent pass-through
`marginalize.py:105-108`: Calls `func(model, X)` twice (before/after substitution) and returns both results unchanged. No type checking. Whatever `predict` returns, `marginalize` returns.

```python
y_before = func(model, X, ...)       # list[Tensor] if model returns tuple
y_after  = func(model, X_perturb, ...) # list[Tensor] if model returns tuple
return y_before, y_after
```

Concrete result for `"both"` mode:
```python
y_before = [logits_before, counts_before]  # list of 2 tensors
y_after  = [logits_after, counts_after]    # list of 2 tensors
# User must index: y_before[0] vs y_after[0] for logit changes,
#                  y_before[1] vs y_after[1] for count changes
```

#### `ablate` — each element reshaped independently
`ablate.py:135-139`: After `func()` returns the shuffled-region predictions, `ablate` reshapes the result to add the shuffle-repeat dimension. For tuples, it reshapes each element independently:

```python
if isinstance(y_after, torch.Tensor):
    y_after = y_after.reshape(*X_perturb.shape[:2], *y_after.shape[1:])
else:
    y_after = [y.reshape(*X_perturb.shape[:2], *y.shape[1:]) for y in y_after]
```

Concrete result for `"both"` mode with `n=20` shuffles:
```python
y_before = [logits_before, counts_before]  # unchanged from predict
y_after  = [
    logits_shuffled,   # (B, 20, C, L) — logits for each shuffle
    counts_shuffled    # (B, 20, 1)    — counts for each shuffle
]
```

#### `variant_effect` — pure pass-through
`variant_effect.py:104-106`: Like marginalize, calls `func()` twice and returns results unchanged. No tuple-specific code.

#### `space` — each element stacked/transposed independently
`space.py:113-117`: Collects predictions for multiple spacings, then stacks. Uses `zip(*y_afters)` correctly to transpose the list structure:

```python
if isinstance(y_afters[0], torch.Tensor):
    y_afters = torch.stack(y_afters).transpose(0, 1)
else:
    y_afters = [torch.stack(y_).transpose(0, 1) for y_ in list(zip(*y_afters))]
```

Concrete result for `"both"` mode with 4 spacings:
```python
y_before = [logits_before, counts_before]  # single prediction
y_afters = [
    logits_stacked,   # (B, 4, C, L) — logits for each spacing
    counts_stacked    # (B, 4, 1)    — counts for each spacing
]
```

#### `saturation_mutagenesis` — **CRASHES in default mode with tuples**

Two separate problems:

**Problem 1: Raw output reshaping works** (`saturation_mutagenesis.py:197-204`). Uses `zip(*y_hat)` correctly:
```python
else:
    y_hat = [
        torch.cat(y_).reshape(X.shape[0], X.shape[2], X.shape[1],
            *y_[0].shape[1:]).transpose(2, 1) for y_ in zip(*y_hat)
    ]
```

**Problem 2: Attribution computation crashes** (`saturation_mutagenesis.py:206-207`). When `raw_outputs=False` (the default), the code calls:
```python
attr = _attribution_score(y0, y_hat, target)
```
But `_attribution_score` (`saturation_mutagenesis.py:37`) does:
```python
attr = y_hat[:, :, :, target] - y0[:, None, None, target]
```
When `y0` and `y_hat` are lists (from tuple output), this indexing fails — you cannot use `[:, :, :, target]` on a Python list.

**Conclusion:** With tuple outputs, `saturation_mutagenesis` works ONLY with `raw_outputs=True`. The default mode (`raw_outputs=False`) crashes.

#### `deep_lift_shap` — completely incompatible with tuples
`deep_lift_shap.py:453`: `y = model(X_)[:, target]` — indexing a tuple with `[:, target]` raises `TypeError`.

#### `design` — assumes single tensor
`design.py` calls `predict()` and passes the result directly to a loss function. No tuple branching. Will crash if model returns tuple.

### 5.3 Per-operation tuple support summary

| Operation | Tuple support | After batch concat | Final return with tuple model | Failure mode |
|-----------|--------------|-------------------|------------------------------|-------------|
| `predict` | Full | `list[Tensor]` — elements separate | `[tensor_0, tensor_1, ...]` | N/A |
| `marginalize` | Full | Pass-through | `([t0, t1], [t0, t1])` | N/A |
| `ablate` | Full | Each element reshaped | `([t0, t1], [t0_reshaped, t1_reshaped])` | N/A |
| `variant_effect` | Full | Pass-through | `([t0, t1], [t0, t1])` | N/A |
| `space` | Full | Each element stacked | `([t0, t1], [t0_stacked, t1_stacked])` | N/A |
| `saturation_mutagenesis` | **`raw_outputs=True` only** | Each element reshaped | `([t0, t1], [t0_reshaped, t1_reshaped])` | Default mode crashes: `_attribution_score` indexes a list |
| `deep_lift_shap` | **No** | N/A | N/A | `model(X_)[:, target]` fails on tuple |
| `design` | **No** | N/A | N/A | Loss function receives list |

### 5.4 The `target` parameter — never indexes into tuple elements

The `target` parameter has different semantics across operations, but in no case does it select a tuple element:

- **`deep_lift_shap`**: Indexes **dimension 1** of the output tensor: `y = model(X_)[:, target]`. The model MUST return a single `(B, n_targets)` tensor.
- **`saturation_mutagenesis`**: Indexes the **last dimension** of the output tensor via `y_hat[:, :, :, target]`. For a `(B, C, L)` output, this selects spatial positions, not channels. Only reachable when model returns a single tensor (otherwise crashes before this point).
- **Other operations**: `target` is not used or is passed through to `func` (default: `predict`), which does not use it.

### 5.5 Bug in `*_annotations` functions with tuple outputs

Both `marginalize_annotations` (`marginalize.py:175-178`) and `ablate_annotations` (`ablate.py:201-204`) contain a bug in their tuple handling:

```python
# marginalize.py:175 (ablate.py:201 is identical)
y_befores = [torch.stack([x[i] for x in y_befores]) for i in range(len(y_befores))]
```

The iteration uses `range(len(y_befores))` — the number of annotations — instead of `range(len(y_befores[0]))` — the number of tuple elements. This works by coincidence when the number of annotations equals the number of tuple elements, but crashes otherwise:
- 2 annotations, 2 tuple elements → works (coincidence)
- 3+ annotations, 2 tuple elements → `IndexError` at `x[2]` since each `x` has only 2 elements
- 1 annotation, 2 tuple elements → produces only 1 output instead of 2

Compare with `space.py:116` and `saturation_mutagenesis.py:201` which correctly use `zip(*y_...)` to transpose the list structure.

**Impact for cerberus:** `marginalize_annotations` and `ablate_annotations` should not be used with `output_mode="both"` unless there happen to be exactly 2 annotations. This is an upstream tangermeme bug.

### 5.6 Implication for cerberus integration

Since tangermeme handles tuples natively for predict/marginalize/ablate/variant_effect/space, a wrapper can expose `output_mode="both"` returning `(logits, log_counts)` as a raw tuple. This preserves all information without requiring the user to choose upfront. However:

- `output_mode="both"` is **incompatible** with `deep_lift_shap` (requires single 2D tensor)
- `output_mode="both"` **crashes** `saturation_mutagenesis` in default mode (only works with `raw_outputs=True`)
- `output_mode="both"` **crashes** `marginalize_annotations` and `ablate_annotations` except when annotation count equals 2 (upstream bug)
- `output_mode="both"` is **incompatible** with `design` functions

For these operations, users must use a single-tensor output mode with appropriate reduction.

### 5.7 Internal acknowledgment in tangermeme

The internal debugging function `_captum_deep_lift_shap` (`deep_lift_shap.py:527-528`) explicitly mentions the logits + counts pattern:

> "It assumes that the model returns 'logits' in the first output, not softmax probabilities, and count predictions in the second output."

This confirms tangermeme's author is aware of this model pattern but chose not to support it in the public API — the public `deep_lift_shap` requires a single `(B, n_targets)` tensor.

---

## 6. Wrapper Design Specification

### 6.1 Design principles

1. **Thin adapter** — the wrapper only converts output format, no model modification
2. **Configurable output** — user selects which "view" of the prediction to expose
3. **DeepLIFT/SHAP ready** — optional spatial reduction for 2D output requirement
4. **Generic** — works with any cerberus model returning `ProfileCountOutput`
5. **Composable** — can be further wrapped by users for custom transformations

### 6.2 Draft implementation

```python
"""Tangermeme compatibility wrapper for cerberus models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from cerberus.output import ProfileCountOutput


class TangermemeWrapper(nn.Module):
    """Wraps a cerberus model to produce tangermeme-compatible tensor output.

    Cerberus models return ProfileCountOutput dataclasses. Tangermeme expects
    forward() to return a torch.Tensor or tuple of tensors. This wrapper
    converts between the two interfaces.

    Parameters
    ----------
    model : nn.Module
        A cerberus model that returns ProfileCountOutput from forward().

    output_mode : str
        Which output representation to return:
        - "counts": softmax(logits) * expm1(log_counts) → (B, C, L)
          Reconstructed predicted counts. The full biological signal.
        - "logits": raw profile logits → (B, C, L)
          Shape prediction without magnitude. Unnormalized.
        - "profile": softmax(logits) → (B, C, L)
          Normalized shape prediction (probabilities).
        - "log_counts": scalar log counts → (B, C)
          Magnitude prediction only.
        - "both": returns (logits, log_counts) as a tuple.
          Compatible with predict, marginalize, ablate, variant_effect, space.
          NOT compatible with deep_lift_shap or saturation_mutagenesis.

    reduce : str or None
        Optional spatial reduction for DeepLIFT/SHAP compatibility.
        Applied after output_mode when output is 3D:
        - None: no reduction, return full spatial output
        - "sum": sum over spatial dim → (B, C)
        - "mean": mean over spatial dim → (B, C)
        - "max": max over spatial dim → (B, C)

    channel : int or None
        If set, select a single output channel (index into dim=1).
        Applied after output_mode and reduce.
        Useful for single-task analysis with DeepLIFT/SHAP.

    Examples
    --------
    >>> from cerberus.models.bpnet import BPNet
    >>> model = BPNet()
    >>>
    >>> # For tangermeme.predict — full reconstructed counts
    >>> wrapper = TangermemeWrapper(model, output_mode="counts")
    >>> y = predict(wrapper, X)  # y: (B, 1, 1000)
    >>>
    >>> # For tangermeme.deep_lift_shap — need 2D output
    >>> wrapper = TangermemeWrapper(model, output_mode="counts", reduce="sum")
    >>> attr = deep_lift_shap(wrapper, X, target=0)  # attr: (B, 4, 2114)
    >>>
    >>> # For ISM on profile shape only
    >>> wrapper = TangermemeWrapper(model, output_mode="profile")
    >>> attr = saturation_mutagenesis(wrapper, X)
    >>>
    >>> # Get both outputs as tuple (for predict, marginalize, ablate, space)
    >>> wrapper = TangermemeWrapper(model, output_mode="both")
    >>> y = predict(wrapper, X)  # y: [logits (B, 1, 1000), log_counts (B, 1)]
    """

    def __init__(
        self,
        model: nn.Module,
        output_mode: Literal["counts", "logits", "profile", "log_counts", "both"] = "counts",
        reduce: Literal["sum", "mean", "max"] | None = None,
        channel: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.output_mode = output_mode
        self.reduce = reduce
        self.channel = channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)

        # Extract tensor based on output_mode
        if self.output_mode == "counts":
            # Reconstructed counts: softmax(logits) * expm1(log_counts)
            profile = F.softmax(output.logits, dim=-1)
            counts = torch.expm1(output.log_counts)  # (B, C) or (B, 1)
            y = profile * counts.unsqueeze(-1)  # (B, C, L)

        elif self.output_mode == "logits":
            y = output.logits  # (B, C, L)

        elif self.output_mode == "profile":
            y = F.softmax(output.logits, dim=-1)  # (B, C, L)

        elif self.output_mode == "log_counts":
            y = output.log_counts  # (B, C) or (B, 1)

        elif self.output_mode == "both":
            # Return raw tuple — tangermeme handles tuples natively for
            # predict, marginalize, ablate, variant_effect, space.
            # NOT compatible with deep_lift_shap or saturation_mutagenesis.
            return (output.logits, output.log_counts)

        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        # Apply spatial reduction if requested (for 3D → 2D)
        if self.reduce is not None and y.dim() == 3:
            if self.reduce == "sum":
                y = y.sum(dim=-1)   # (B, C)
            elif self.reduce == "mean":
                y = y.mean(dim=-1)  # (B, C)
            elif self.reduce == "max":
                y = y.max(dim=-1).values  # (B, C)
            else:
                raise ValueError(f"Unknown reduce: {self.reduce}")

        # Select single channel if requested
        if self.channel is not None:
            y = y[:, self.channel : self.channel + 1]  # preserve dim

        return y

    @property
    def input_len(self) -> int:
        """The required input sequence length for the wrapped model."""
        return self.model.input_len

    @property
    def output_len(self) -> int:
        """The output profile length of the wrapped model."""
        return self.model.output_len
```

---

## 7. Per-Operation Compatibility Notes

### 7.1 `predict(model, X)`

**Compatibility:** Works with wrapper, any output_mode.

```python
wrapper = TangermemeWrapper(model, output_mode="counts")
y = predict(wrapper, X, batch_size=32, device='cuda')
# y: (N, 1, 1000) for BPNet — full reconstructed counts
```

No spatial reduction needed. The 3D output `(B, C, L)` is a valid single tensor.

### 7.2 `marginalize(model, X, motif)`

**Compatibility:** Works with wrapper. Motif substitution does not change sequence length.

```python
wrapper = TangermemeWrapper(model, output_mode="counts")
y_before, y_after = marginalize(wrapper, X, "GATA", device='cuda')
# Compare y_before vs y_after to see marginal effect of GATA
# Both: (N, 1, 1000)
```

**Note on output_mode choice:**
- `"counts"` shows the effect on predicted read counts (shape × magnitude)
- `"logits"` shows the effect on predicted shape only
- `"log_counts"` shows the effect on predicted total counts only

### 7.3 `saturation_mutagenesis(model, X)` (ISM)

**Compatibility:** Works with wrapper using single-tensor output modes. Does NOT work with `output_mode="both"` in default mode.

When `raw_outputs=False` (default), ISM computes attribution scores via `_attribution_score` which does:
```python
attr = y_hat[:, :, :, target] - y0[:, None, None, target]  # saturation_mutagenesis.py:37
```
This requires `y0` and `y_hat` to be single tensors. If the model returns a tuple, `y0` and `y_hat` become lists, and this line crashes with a `TypeError`.

With `raw_outputs=True`, tuple outputs work correctly — each element is reshaped independently via `zip(*y_hat)`, and the raw `(y0, y_hat)` are returned for custom analysis.

The `target` parameter indexes the last dimension of the output. For 3D outputs `(B, C, L)`, the last dimension is the spatial dimension — so `target` selects a spatial position, which may not be the intended semantics.

**Recommendation for ISM:**
- Use `output_mode="log_counts"` with no reduce for count-level ISM → `target=0` selects the single count output
- Or use `output_mode="counts"` with `reduce="sum"` for total-count ISM
- For position-level ISM on profile shape, use `output_mode="profile"` with no reduce — but be aware that `target` indexes positions, not channels
- Use `output_mode="both"` ONLY with `raw_outputs=True` for custom attribution analysis

```python
# Count-level ISM: which positions affect total binding?
wrapper = TangermemeWrapper(model, output_mode="counts", reduce="sum")
attr = saturation_mutagenesis(wrapper, X, device='cuda')
# attr: (B, 4, L) — attribution at each input position for total counts

# Raw dual-output ISM for custom analysis:
wrapper = TangermemeWrapper(model, output_mode="both")
y0, y_hat = saturation_mutagenesis(wrapper, X, raw_outputs=True, device='cuda')
# y0:    [logits (B, C, L), counts (B, 1)]
# y_hat: [logits (B, 4, L, C, L), counts (B, 4, L, 1)]
```

### 7.4 `deep_lift_shap(model, X)`

**Compatibility:** Requires wrapper with spatial reduction. Output must be `(B, n_targets)`.

```python
# Attribution for total binding signal
wrapper = TangermemeWrapper(model, output_mode="counts", reduce="sum", channel=0)
attr = deep_lift_shap(wrapper, X, target=0, n_shuffles=20, device='cuda')
# attr: (B, 4, 2114) — per-nucleotide attribution scores

# Attribution for predicted shape at a log-count level
wrapper = TangermemeWrapper(model, output_mode="log_counts", channel=0)
attr = deep_lift_shap(wrapper, X, target=0, device='cuda')
# attr: (B, 4, 2114) — what drives total count prediction
```

**Warning:** Using `output_mode="counts"` with `reduce="sum"` is equivalent to `expm1(log_counts)` only when summing `softmax * count` (the softmax sums to 1 over spatial positions, so `sum(softmax * count) = count`). Thus `reduce="sum"` on `"counts"` and `"log_counts"` give related but not identical results — `sum(counts)` = `expm1(log_counts)` exactly, but the attribution through the softmax pathway differs from direct attribution on log_counts.

### 7.5 `ablate(model, X, start, end)`

**Compatibility:** Works with wrapper. Ablation shuffles a region without changing length.

```python
wrapper = TangermemeWrapper(model, output_mode="counts")
results = ablate(wrapper, X, start=500, end=600, n=20, device='cuda')
# results[0]: y_before, results[1:]: y_after_shuffle_1, ...
```

### 7.6 `substitution_effect(model, X, substitutions)`

**Compatibility:** Works with wrapper. Single-nucleotide substitutions preserve length.

```python
wrapper = TangermemeWrapper(model, output_mode="counts")
subs = torch.tensor([[0, 500, 2]])  # example 0, position 500, change to G
y_before, y_after = substitution_effect(wrapper, X, subs, device='cuda')
```

### 7.7 `deletion_effect` and `insertion_effect`

**Compatibility:** Problematic. These operations change sequence length, conflicting with cerberus models' fixed input length requirement.

- `deletion_effect`: deletes nucleotides then trims from one end to restore length. The trimmed sequence is shorter at one end, which changes the genomic context.
- `insertion_effect`: inserts nucleotides then pads from one end. The padded sequence is longer at one end.

Both operations produce sequences of the original length, so they technically work. However, the biological interpretation is complicated by the asymmetric trimming/padding — one flank is shortened while the other is preserved.

```python
# Works mechanically but interpretation requires care
wrapper = TangermemeWrapper(model, output_mode="counts")
dels = torch.tensor([[0, 500, 503]])  # delete 3bp at position 500
y_before, y_after = deletion_effect(wrapper, X, dels, device='cuda')
```

### 7.8 `space(model, X, motifs, spacing)`

**Compatibility:** Works with wrapper. Motif spacing analysis substitutes motifs at different positions without changing length.

```python
wrapper = TangermemeWrapper(model, output_mode="counts", reduce="sum")
spacings = torch.tensor([[10], [20], [50], [100]])  # test 4 spacings
y = space(wrapper, X, ["GATA", "CAAT"], spacings, device='cuda')
```

### 7.9 `design` (greedy_substitution, screen)

**Compatibility:** Works with wrapper. Sequence design modifies nucleotides without changing length.

```python
wrapper = TangermemeWrapper(model, output_mode="counts")
X_designed = greedy_substitution(wrapper, X, motifs=["GATA"], y=target_y)
```

---

## 8. Output Mode Trade-offs

Each output mode answers a different biological question. No single mode is universally correct.

### `"counts"` — Reconstructed read counts

**Formula:** `softmax(logits, dim=-1) * expm1(log_counts)`
**Shape:** `(B, C, L)`
**When to use:**
- Comparing predicted signal to observed BigWig tracks
- Marginalization: "how does this motif change the predicted ChIP signal?"
- Variant effects: "how does this SNP change predicted binding?"

**Considerations:**
- Couples shape and magnitude — changes in either component appear in the output
- Most biologically interpretable representation
- Non-linear interaction between the two heads may complicate attribution methods

### `"logits"` — Raw profile logits

**Formula:** Direct `logits` output from model
**Shape:** `(B, C, L)`
**When to use:**
- Analyzing the shape prediction head in isolation
- When magnitude (total counts) is not relevant to the question
- Debugging model behavior

**Considerations:**
- Unnormalized — scale is arbitrary, not comparable across samples with different count predictions
- Does not reflect magnitude changes (a motif could change counts dramatically without affecting logits)
- Simpler gradient path for attribution methods

### `"profile"` — Normalized probability profile

**Formula:** `softmax(logits, dim=-1)`
**Shape:** `(B, C, L)`
**When to use:**
- Shape-only analysis: "where does the model predict the peak?"
- Comparing shape across conditions without magnitude confounds
- ISM on the multinomial distribution

**Considerations:**
- Probabilities sum to 1 over spatial positions (per channel)
- Like logits, ignores magnitude changes
- Softmax can mask subtle shape changes when the distribution is sharply peaked

### `"log_counts"` — Scalar count prediction

**Formula:** Direct `log_counts` output from model
**Shape:** `(B, C)` or `(B, 1)`
**When to use:**
- Count-level analysis: "does this motif increase total binding?"
- DeepLIFT/SHAP attribution for total binding (already 2D, no reduce needed)
- When spatial profile is not relevant

**Considerations:**
- Ignores spatial distribution entirely
- Single scalar per sample — limited information
- Simplest gradient path for attribution

### `"both"` — Raw tuple of (logits, log_counts)

**Formula:** Returns `(output.logits, output.log_counts)` as a Python tuple
**Shape:** `((B, C, L), (B, C))` or `((B, C, L), (B, 1))`
**When to use:**
- Exploratory analysis where you want to inspect both shape and magnitude
- `predict()`: get both outputs in a single call, returned as `list[Tensor]` — `y[0]` = logits, `y[1]` = log_counts
- `marginalize()`: compare how a motif affects both shape and counts simultaneously — `y_before[0]` vs `y_after[0]` for logit diff, `y_before[1]` vs `y_after[1]` for count diff
- `ablate()`, `variant_effect()`, `space()`: same dual-output analysis
- `saturation_mutagenesis(raw_outputs=True)`: get raw before/after predictions for custom analysis

**Incompatible with:**
- **`deep_lift_shap`** — crashes at `y = model(X_)[:, target]` (cannot index a tuple)
- **`saturation_mutagenesis` default mode** (`raw_outputs=False`) — crashes in `_attribution_score` which does `y_hat[:, :, :, target]` on a list. Only works with `raw_outputs=True`.
- **`marginalize_annotations` / `ablate_annotations`** — upstream bug: iterates `range(len(y_befores))` (annotation count) instead of `range(len(y_befores[0]))` (tuple element count). Crashes when annotation count ≠ 2.
- **`design`** functions — loss function receives list instead of tensor

**Considerations:**
- Preserves all model information without forcing a choice
- Downstream analysis must handle the `list[Tensor]` return type from `predict` and similar — `y[0]` is always the first tuple element, `y[1]` the second
- Tuple elements are never combined, averaged, or selected by tangermeme — each element is processed independently through all stages
- `reduce` and `channel` parameters are ignored in this mode (they would only apply to a single tensor)

---

## 9. Future Considerations

### Multi-channel models

Current BPNet/Pomeranian models use `output_channels=["signal"]` (single channel). If models are extended to multi-channel (e.g., predicting multiple marks simultaneously), the wrapper handles this transparently — `C > 1` in all output shapes. The `channel` parameter allows selecting specific channels.

### Ensemble wrapping

Cerberus has `ModelEnsemble` for multi-fold aggregation. An ensemble wrapper could:
1. Wrap each fold model in `TangermemeWrapper`
2. Average predictions across folds
3. Return the averaged tensor

This is a natural extension but not required for initial integration.

### ProfileLogRates output type

Some cerberus models (e.g., Gopher) return `ProfileLogRates` instead of `ProfileCountOutput`. The wrapper should be extended to handle this output type:
```python
# ProfileLogRates: log_rates (B, C, L) → exp(log_rates) for counts
```

### Tangermeme version compatibility

This analysis is based on tangermeme v1.0.3. The model interface contract (`forward() → Tensor or tuple`) is fundamental to tangermeme's design and unlikely to change, but specific function signatures may evolve.

---

## 10. Summary of Required Changes

| Priority | Change | Files | Resolves |
|----------|--------|-------|----------|
| P0 | Create `TangermemeWrapper` class | `src/cerberus/tangermeme.py` (new) | Issues 1, 2, 3 |
| P1 | Add usage documentation / examples | `docs/` or `examples/` | Issue 4 (guidance) |
| P2 | Add tests for wrapper with tangermeme ops | `tests/` | Validation |
| P3 | Extend wrapper for `ProfileLogRates` | `src/cerberus/tangermeme.py` | Generality |
