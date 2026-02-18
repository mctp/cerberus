# Cerberus-Tangermeme Integration: Compatibility Report & Strategy

## 1. Executive Summary

This document analyzes the compatibility between cerberus (genomic sequence-to-function models) and tangermeme (post-hoc sequence analysis toolkit), identifies blocking incompatibilities, and proposes a wrapper-based integration strategy.

**Bottom line:** The two packages are compatible at the input level (encoding, tensor format) but incompatible at the output level. Cerberus models return structured dataclass outputs (`ProfileCountOutput`) that tangermeme cannot process. A thin `TangermemeWrapper(nn.Module)` resolves all incompatibilities while keeping the packages decoupled.

---

## 2. Tangermeme Capability Overview

Tangermeme provides post-hoc analysis tools for any PyTorch sequence model. Not all modules are equally relevant for studying how inputs affect outputs. This section classifies each module by its role in model interpretation.

### 2.1 Attribution methods — per-position input importance

These methods answer: **"which input positions drive the model's output?"** They produce per-nucleotide scores across the full input sequence, directly revealing the input→output relationship.

| Module | Function | How it works | Gradient-based? | Output shape |
|--------|----------|-------------|----------------|-------------|
| `deep_lift_shap` | `deep_lift_shap(model, X)` | Compares activations between input and shuffled references through all layers; decomposes output into per-input contributions | Yes — modified backprop with custom hooks on nonlinearities | `(B, 4, L_input)` |
| `saturation_mutagenesis` | `saturation_mutagenesis(model, X)` | Exhaustively substitutes every nucleotide at every position; measures how each change affects output | No — forward-pass only, runs model 4×L times per sequence | `(B, 4, L_input)` |
| `pisa` | `pisa(model, X)` | Runs DeepLIFT/SHAP once per output position; builds a pairwise matrix of which input positions influence which output positions | Yes — N_output × DeepLIFT/SHAP runs | `(B, 4, L_input)` per output position |

**deep_lift_shap** requires the model to return a single 2D tensor `(B, n_targets)`. The `target` parameter selects which output to attribute. Attributions are additive: they sum to the difference between the model's prediction on the input and on the reference.

**saturation_mutagenesis** (ISM) is gradient-free but computationally expensive. By default it internally computes attribution scores via `_attribution_score` (difference + position-normalization + task-averaging). With `raw_outputs=True` it returns the raw before/after predictions for custom scoring.

**pisa** (Pairwise Influence by Sequence Attribution) is specific to profile-output models. It answers not just "which inputs matter" but "which inputs matter *for which output positions*" — directly relevant for cerberus profile models. Like `deep_lift_shap`, it requires the model to return a single tensor. It uses `predict(model, X[:1]).shape[-1]` to determine the number of output positions, then runs DeepLIFT/SHAP once per position.

### 2.2 Perturbation experiments — hypothesis-driven input→output testing

These methods answer: **"how does a specific sequence modification change the model's output?"** They return raw predictions before and after the modification. The user computes the comparison (see Section 5.7).

| Module | Function | What it modifies | What hypothesis it tests |
|--------|----------|-----------------|------------------------|
| `marginalize` | `marginalize(model, X, motif)` | Inserts a motif into background sequences | Does this motif affect the model? (gain-of-function) |
| `ablate` | `ablate(model, X, start, end)` | Shuffles a region to destroy signal | Is this region necessary for the prediction? (loss-of-function) |
| `space` | `space(model, X, motifs, spacing)` | Inserts multiple motifs at varying spacings | Do these motifs cooperate? At what distance? |
| `variant_effect` | `substitution_effect(model, X, subs)` | Applies specific point mutations | Does this SNP/variant change the prediction? |
| `variant_effect` | `deletion_effect(model, X, dels)` | Deletes nucleotides (with edge trimming) | Does this deletion change the prediction? |
| `variant_effect` | `insertion_effect(model, X, ins)` | Inserts nucleotides (with edge trimming) | Does this insertion change the prediction? |

All perturbation methods return `(y_before, y_after)` — raw predictions, not scores. tangermeme explicitly avoids computing comparisons for these operations (see Section 5.7). The user decides how to score the effect (difference, fold change, divergence, etc.).

**marginalize vs ablate** are conceptual opposites. Marginalize adds a motif to neutral background → gain-of-function. Ablate removes signal from an active sequence → loss-of-function. Together they bracket the isolated contribution of a motif.

**space** is unique in that it tests *interactions* between motifs, not individual motifs. It reveals cooperativity, competition, and preferred spacing — phenomena invisible to single-motif methods.

### 2.3 Post-attribution analysis — interpreting attribution scores

These modules process the per-position attribution scores from Section 2.1 into biologically meaningful patterns:

| Module | Function | What it does |
|--------|----------|-------------|
| `seqlet` | `seqlet.extract_seqlets(attr)` | Discovers recurring high-attribution subsequences (seqlets) using TF-MoDISCo-inspired algorithm with Laplacian null fitting |
| `annotate` | `annotate.annotate_seqlets(seqlets)` | Matches discovered seqlets against known motif databases (TOMTOM) to assign biological identity |
| `kmers` | `kmers.kmers(X, scores)` | Extracts k-mer features optionally weighted by attribution scores — converts attributions to interpretable feature vectors |

These form a pipeline: **attributions → seqlets → annotation → biological interpretation**.

### 2.4 Sequence design — reverse engineering desired outputs

| Module | Function | What it does |
|--------|----------|-------------|
| `design` | `screen(model, X, y, loss)` | Random screening: generates sequences, evaluates against target |
| `design` | `greedy_substitution(model, X, motifs, y)` | Iteratively inserts motifs to optimize toward target output |
| `design` | `greedy_marginalize(model, X, motifs, y)` | Builds optimal motif construct via sequential marginalizations |

Design operates in the reverse direction — given a desired output, find an input that produces it. This tests the model's generative capacity rather than interpreting existing predictions.

### 2.5 Infrastructure — prediction, I/O, sequence manipulation

| Module | Function | Role |
|--------|----------|------|
| `predict` | `predict(model, X)` | Batched GPU inference with memory management, dtype conversion, device transfer |
| `ersatz` | `substitute()`, `shuffle()`, `insert()`, etc. | Sequence manipulation primitives used by perturbation methods |
| `utils` | `one_hot_encode()`, `reverse_complement()`, etc. | Encoding, validation, coordinate conversion |
| `io` | `extract_loci()`, `read_meme()`, `read_vcf()` | Loading genomic sequences, signals, motifs, variants |
| `product` | `apply_pairwise()`, `apply_product()` | Apply any function across Cartesian products of inputs (e.g., sequences × cell types) |
| `match` | GC-matching utilities | Generate GC-content matched control sequences |
| `plot` | Logo visualization | Visualize attribution scores as sequence logos |

### 2.6 Model interface tangermeme expects

```python
# Minimum contract (all operations except deep_lift_shap/pisa):
y = model(X)  # X: (B, 4, L) → y: torch.Tensor or tuple of torch.Tensors

# For deep_lift_shap and pisa:
y = model(X)  # Must return (B, n_targets) — single 2D tensor only
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
| `pisa` | **No** | N/A | N/A | `predict(model, X[:1]).shape[-1]` fails on list; internally calls `deep_lift_shap` |
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
- `output_mode="both"` is **incompatible** with `pisa` (requires single tensor; internally uses `deep_lift_shap`)
- `output_mode="both"` **crashes** `saturation_mutagenesis` in default mode (only works with `raw_outputs=True`)
- `output_mode="both"` **crashes** `marginalize_annotations` and `ablate_annotations` except when annotation count equals 2 (upstream bug)
- `output_mode="both"` is **incompatible** with `design` functions

For these operations, users must use a single-tensor output mode with appropriate reduction.

### 5.7 How tangermeme outputs are consumed — tangermeme returns raw predictions, not scores

A critical design point: for most operations, **tangermeme does not compute any comparison between y_before and y_after**. It returns the raw predictions and leaves all scoring, differencing, and interpretation to the user. This is an explicit design choice — the README states:

> "Because `tangermeme` aims to be assumption-free, these functions take in a batch of examples that you specify, and return the predictions before and after adding the motif in for each example."

And for variant effect/ISM:

> "There are numerous ways to combine the predictions of each variant with the predictions on the original sequence and tangermeme allows you to use whichever approach you would like."

#### Which operations compute scores internally vs return raw

| Operation | What it returns | Internal scoring? |
|-----------|----------------|-------------------|
| `predict` | Single prediction | N/A — no before/after |
| `marginalize` | `(y_before, y_after)` — raw predictions | **No** — user computes difference |
| `ablate` | `(y_before, y_after)` — raw predictions | **No** — user computes difference |
| `variant_effect` | `(y_before, y_after)` — raw predictions | **No** — user computes difference |
| `space` | `(y_before, y_afters)` — raw predictions | **No** — user computes difference |
| `saturation_mutagenesis` | Attribution tensor (default) or `(y0, y_hat)` raw | **Yes** (default) — `_attribution_score` computes difference + position-normalization + task-averaging |
| `deep_lift_shap` | Attribution tensor | **Yes** — DeepLIFT/SHAP algorithm computes attributions internally |

#### Concrete example: `variant_effect` with single-tensor output

Using `output_mode="counts"` (single tensor):

```python
wrapper = TangermemeWrapper(model, output_mode="counts")
subs = torch.tensor([[0, 500, 2]])  # example 0, position 500, change to G

y_before, y_after = substitution_effect(wrapper, X, subs, device='cpu')
# y_before: (B, 1, 1000) — predicted counts with ref allele
# y_after:  (B, 1, 1000) — predicted counts with alt allele
```

tangermeme returns and is done. The user must then compute their own effect score:

```python
# User computes difference (tangermeme does NOT do this)
delta = y_after - y_before                    # (B, 1, 1000) — change in counts per position
log_fc = torch.log2(y_after / y_before)       # log fold change
total_effect = (y_after - y_before).sum(dim=-1)  # scalar effect per sample
```

#### Concrete example: `variant_effect` with tuple output (`"both"` mode)

Using `output_mode="both"` (tuple of logits + log_counts):

```python
wrapper = TangermemeWrapper(model, output_mode="both")
subs = torch.tensor([[0, 500, 2]])

y_before, y_after = substitution_effect(wrapper, X, subs, device='cpu')
```

Here, `y_before` and `y_after` are each a `list[Tensor]` (because `predict` concatenated the tuple elements separately — see Section 5.1):

```python
# y_before is a list of 2 tensors:
y_before[0]  # logits:     (B, 1, 1000) — profile shape with ref allele
y_before[1]  # log_counts: (B, 1)       — total count with ref allele

# y_after is a list of 2 tensors:
y_after[0]   # logits:     (B, 1, 1000) — profile shape with alt allele
y_after[1]   # log_counts: (B, 1)       — total count with alt allele
```

tangermeme returns and is done. It did NOT compare `y_before[0]` with `y_after[0]`, nor `y_before[1]` with `y_after[1]`. It did NOT combine the tuple elements in any way. The user must compute their own scores from the raw predictions:

```python
# Shape change: did the variant alter the predicted profile?
logit_delta = y_after[0] - y_before[0]            # (B, 1, 1000)
profile_before = F.softmax(y_before[0], dim=-1)
profile_after  = F.softmax(y_after[0], dim=-1)
jsd = ...  # Jensen-Shannon divergence between profiles

# Count change: did the variant alter total binding?
count_delta = y_after[1] - y_before[1]             # (B, 1) — in log1p scale
count_fc = torch.expm1(y_after[1]) / torch.expm1(y_before[1])  # fold change

# Combined: reconstruct counts and compare
counts_before = F.softmax(y_before[0], dim=-1) * torch.expm1(y_before[1]).unsqueeze(-1)
counts_after  = F.softmax(y_after[0], dim=-1) * torch.expm1(y_after[1]).unsqueeze(-1)
full_delta = counts_after - counts_before           # (B, 1, 1000)
```

#### Why this matters for the wrapper design

The fact that tangermeme returns raw predictions means:

1. **The `output_mode` choice determines what the user can analyze**, not what tangermeme will automatically score. With `"counts"` you get one perspective; with `"both"` you get the raw components for richer analysis.

2. **`"both"` mode is most flexible** for these raw-return operations — the user can independently analyze shape changes and count changes, or reconstruct full counts. The trade-off is more complex downstream code.

3. **`"counts"` mode is simplest** — one tensor, one subtraction, biologically interpretable. But it conflates shape and count changes.

4. **For `saturation_mutagenesis` and `deep_lift_shap`**, which DO compute scores internally, the `output_mode` choice determines what signal those algorithms attribute to — there is no downstream user step except visualization.

### 5.8 Internal acknowledgment in tangermeme

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

### 7.5 `pisa(model, X)` (Pairwise Influence by Sequence Attribution)

**Compatibility:** Requires wrapper with spatial reduction, like `deep_lift_shap`. Output must be a single 2D tensor `(B, n_targets)`.

PISA runs DeepLIFT/SHAP once per output position. It determines the number of output positions via `predict(model, X[:1]).shape[-1]` (`pisa.py:196`), so the model must return a single tensor whose last dimension is the number of outputs. Tuple outputs are not supported.

PISA is particularly relevant for cerberus profile models because it reveals which input nucleotides influence which output profile positions — a pairwise input→output matrix rather than a scalar attribution. However, running it on the full 1000-position profile output would require 1000 DeepLIFT/SHAP iterations per sequence.

```python
# PISA on total binding (reduced to scalar → 1 output position)
wrapper = TangermemeWrapper(model, output_mode="counts", reduce="sum", channel=0)
attr = pisa(wrapper, X, n_shuffles=20, device='cuda')
# attr: (B, 4, L_input) — same as deep_lift_shap for scalar output

# PISA on profile shape (each output bin is a target)
# Caution: very expensive — runs DeepLIFT/SHAP 1000× per sequence
wrapper = TangermemeWrapper(model, output_mode="profile", channel=0)
# This would need output shape (B, 1000) — requires squeezing channel dim
```

**Note:** PISA's `predict(model, X[:1]).shape[-1]` call expects the last dimension to be the spatial/target dimension. For cerberus outputs that are `(B, C, L)` after the wrapper, the last dimension is `L` (spatial) which is correct for profile analysis, but the channel dimension `C` must be squeezed to 1 or selected via the `channel` parameter.

### 7.6 `ablate(model, X, start, end)`

**Compatibility:** Works with wrapper. Ablation shuffles a region without changing length.

```python
wrapper = TangermemeWrapper(model, output_mode="counts")
results = ablate(wrapper, X, start=500, end=600, n=20, device='cuda')
# results[0]: y_before, results[1:]: y_after_shuffle_1, ...
```

### 7.7 `substitution_effect(model, X, substitutions)`

**Compatibility:** Works with wrapper. Single-nucleotide substitutions preserve length.

```python
wrapper = TangermemeWrapper(model, output_mode="counts")
subs = torch.tensor([[0, 500, 2]])  # example 0, position 500, change to G
y_before, y_after = substitution_effect(wrapper, X, subs, device='cuda')
```

### 7.8 `deletion_effect` and `insertion_effect`

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

### 7.9 `space(model, X, motifs, spacing)`

**Compatibility:** Works with wrapper. Motif spacing analysis substitutes motifs at different positions without changing length.

```python
wrapper = TangermemeWrapper(model, output_mode="counts", reduce="sum")
spacings = torch.tensor([[10], [20], [50], [100]])  # test 4 spacings
y = space(wrapper, X, ["GATA", "CAAT"], spacings, device='cuda')
```

### 7.10 `design` (greedy_substitution, screen)

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

## 9. Reimplementation vs. Wrapping: Analysis

### 9.1 The question

Should cerberus reimplement tangermeme's analysis functions natively rather than wrapping the two packages together? The goal is to make cerberus self-contained, well-tested, and powerful.

### 9.2 Implementation complexity varies enormously across modules

Tangermeme's modules fall into three complexity tiers:

**Tier 1 — Trivial (5-20 lines of actual logic):**

| Module | What it does | Core logic |
|--------|-------------|-----------|
| `predict` | Batched inference | `for` loop + `torch.no_grad()` + `torch.cat()` |
| `marginalize` | Insert motif, compare | Two `predict()` calls with `substitute()` between |
| `ablate` | Shuffle region, compare | Two `predict()` calls with `shuffle()` between |
| `variant_effect` | Mutate, compare | Two `predict()` calls with point mutation between |
| `space` | Insert motifs at spacings, compare | Loop of `predict()` calls with `multisubstitute()` |

These functions are thin wrappers around sequence manipulation + prediction. Their entire value is in the API design (standardized before/after pattern), not in algorithmic complexity. Reimplementing them in cerberus is straightforward and eliminates the output-type mismatch entirely.

**Tier 2 — Moderate (~100-200 lines):**

| Module | What it does | Core logic |
|--------|-------------|-----------|
| `saturation_mutagenesis` | Exhaustive single-nt mutation scan | Generate edit-distance-1 variants (numba), predict all, reshape. Optional `_attribution_score` for default scoring. |
| `pisa` | Per-output-position DeepLIFT/SHAP | Loop calling `deep_lift_shap` once per output position |

ISM is conceptually simple but has performance-critical components (numba-accelerated variant generation). PISA is a thin loop over DeepLIFT/SHAP — its complexity is inherited from deep_lift_shap.

**Tier 3 — Complex (~500 lines, subtle correctness requirements):**

| Module | What it does | Core logic |
|--------|-------------|-----------|
| `deep_lift_shap` | Per-nucleotide attribution via modified backpropagation | Hook registration on all nonlinear modules, custom backward pass implementing DeepLIFT rescale rule, convergence checking, hypothetical attribution correction for one-hot data, dinucleotide-shuffled reference generation |

### 9.3 Critical finding: tangermeme's deep_lift_shap is incompatible with cerberus models

Tangermeme's `deep_lift_shap` works by registering `register_forward_hook`, `register_forward_pre_hook`, and `register_full_backward_hook` on `nn.Module` subclasses (`nn.ReLU`, `nn.GELU`, etc.). These hooks override the backward pass to implement the DeepLIFT rescale rule instead of standard gradients.

**Problem:** Cerberus BPNet uses `F.relu()` (functional form) — not `nn.ReLU()` (module form):

```python
# bpnet.py:105
x = F.relu(self.iconv(x))   # functional — invisible to module hooks

# layers.py:227 (BPNet residual blocks)
out = F.relu(out)            # functional — invisible to module hooks
```

Functional activations have no associated module, so tangermeme's hooks never fire. The backward pass falls through to standard PyTorch autograd, which computes the standard ReLU gradient (step function: 0 for x<0, 1 for x≥0) instead of the DeepLIFT rescale rule (delta_out / delta_in relative to reference). This produces **silently incorrect attribution scores** — the code runs without error but the results violate the DeepLIFT additivity property.

**Impact by model:**
- **BPNet**: ALL ReLU activations use `F.relu()` → attributions silently incorrect throughout
- **Pomeranian**: ConvNeXtBlock uses `self.act = nn.GELU()` (hookable), but residual blocks inherit from `layers.py` which uses `F.relu()` → attributions partially incorrect
- **Gopher**: Uses `nn.ReLU()` in `nn.Sequential` blocks → hookable, attributions correct

**This is not specific to the wrapper approach** — the same `F.relu` issue exists regardless of whether we wrap tangermeme or call it directly.

### 9.4 Options for deep_lift_shap

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: Use tangermeme + fix models** | Change `F.relu()` → `self.relu = nn.ReLU()` in cerberus models | Minimal code; leverages tested implementation | Requires model changes; retraining needed if weights depend on exact gradient flow; adds tangermeme dependency |
| **B: Use Captum directly** | Write cerberus-specific adapter around `captum.attr.DeepLiftShap` | Captum handles functional activations via its own hooking system; maintained by Meta/PyTorch | Adds Captum dependency; still need genomics-specific code (dinucleotide shuffling, hypothetical attributions, convergence checking) |
| **C: Reimplement in cerberus** | Full reimplementation of DeepLIFT/SHAP from scratch | Full control; no dependencies; can handle `F.relu` natively | High risk of subtle bugs; the rescale rule for nonlinearities has edge cases; convergence checking is important for correctness validation; significant engineering effort with no unique scientific value |
| **D: Hybrid** | Reimplement Tier 1/2 natively; use tangermeme or Captum for DeepLIFT/SHAP | Best of both worlds; simple ops are self-contained; complex algorithm uses proven code | Two code styles; still has one external dependency |

### 9.5 What makes deep_lift_shap genuinely hard to reimplement

The core DeepLIFT algorithm has several subtle aspects that make reimplementation risky:

1. **Rescale rule for nonlinearities** (`deep_lift_shap.py:116-133`): Computes `delta_out / delta_in` with special handling when `delta_in ≈ 0` (falls back to standard gradient). The threshold (`1e-6`) and the fallback behavior affect attribution quality.

2. **Softmax-specific correction** (`deep_lift_shap.py:136-157`): Softmax requires a modified rescale rule that subtracts the mean attribution to maintain the zero-sum property. Getting this wrong produces attributions that don't sum to zero across the alphabet at each position.

3. **MaxPool correction** (`deep_lift_shap.py:160-198`): MaxPool requires unpooling with tracked indices and a cross-reference maximum calculation. This is the most complex single operation handler.

4. **Hypothetical attribution correction** (`deep_lift_shap.py:15-80`): For one-hot encoded DNA, changing one nucleotide simultaneously adds one base and removes another. The hypothetical attribution computation accounts for this by computing attributions for all possible bases at each position, not just the observed one.

5. **Convergence checking** (`deep_lift_shap.py:459-462`): The sum of attributions should equal the difference in prediction between input and reference. Checking this property validates the implementation. If convergence deltas are large, the attributions are unreliable.

6. **Batched reference management** (`deep_lift_shap.py:411-434`): Each input sequence is paired with multiple dinucleotide-shuffled references. The batching logic must correctly pair sequences with their references even when batch_size < n_shuffles.

These components interact — a bug in any one of them can produce attributions that look plausible but are quantitatively wrong. Tangermeme includes a Captum-based cross-validation function (`_captum_deep_lift_shap`, line 521) specifically to catch such bugs.

### 9.6 What is trivially reimplementable

The Tier 1 operations (predict, marginalize, ablate, variant_effect, space) have no algorithmic complexity. Their entire implementation pattern is:

```python
# The pattern shared by ALL perturbation methods:
y_before = predict(model, X)
X_modified = modify_sequence(X, ...)  # substitute, shuffle, mutate, etc.
y_after = predict(model, X_modified)
return y_before, y_after
```

The sequence manipulation functions (substitute, shuffle, dinucleotide_shuffle) are also straightforward.

**Advantages of cerberus-native reimplementation for these:**
- Work directly with `ProfileCountOutput` — no wrapper needed
- Can return structured results (e.g., a `MarginalizeResult` dataclass with `.logit_delta`, `.count_delta`, `.reconstructed_delta`)
- Can integrate with cerberus's existing interval/coordinate system
- Tests are self-contained
- No dependency version issues

**Saturation mutagenesis** is also reimplementable. The core algorithm is: generate all edit-distance-1 variants → predict → reshape. The numba-accelerated `_edit_distance_one` function is a nice optimization but not essential (a pure PyTorch version works too, just slower). The `_attribution_score` function is 6 lines.

### 9.7 Recommendation

**Reimplement natively in cerberus (Tier 1 + Tier 2):**
- `predict` — batched inference with ProfileCountOutput handling
- `marginalize` — motif insertion effect
- `ablate` — region ablation effect
- `variant_effect` — substitution/deletion/insertion effects
- `space` — motif spacing interaction
- `saturation_mutagenesis` — ISM with direct support for profile+count decomposition
- Sequence manipulation utilities (substitute, shuffle, dinucleotide_shuffle)

These operations benefit from cerberus-native implementation because they can work directly with `ProfileCountOutput`, return richer result types, and integrate with the interval system. The total implementation effort is modest (each is 20-50 lines of core logic) and the testing burden is manageable.

**Use an external implementation for DeepLIFT/SHAP (Tier 3):**

DeepLIFT/SHAP should NOT be reimplemented from scratch. The algorithm is complex, subtle, and the risk of silent correctness bugs is high. Instead:

1. **Fix cerberus models**: Change `F.relu()` → `nn.ReLU()` module in BPNet and residual blocks. This is a prerequisite regardless of which DeepLIFT implementation is used. It does NOT require retraining — `nn.ReLU()` and `F.relu()` produce identical forward/backward results under standard autograd; the difference only matters when hooks override the backward pass.

2. **Choose between tangermeme and Captum for the DeepLIFT engine:**
   - tangermeme: Already genomics-aware (dinucleotide shuffling, hypothetical attributions, convergence checking). Can be used via a thin wrapper after fixing `F.relu`.
   - Captum: More robust hook system, maintained by Meta. Would need a cerberus adapter for the genomics-specific parts (~100 lines for reference generation, hypothetical attributions, convergence checking).

3. **PISA** follows whichever DeepLIFT implementation is chosen (it's a loop over DeepLIFT runs).

### 9.8 Summary table

| Module | Recommendation | Rationale |
|--------|---------------|-----------|
| predict | **Reimplement** | Trivial; direct ProfileCountOutput support |
| marginalize | **Reimplement** | Trivial; richer result types |
| ablate | **Reimplement** | Trivial; richer result types |
| variant_effect | **Reimplement** | Trivial; richer result types |
| space | **Reimplement** | Trivial; richer result types |
| saturation_mutagenesis | **Reimplement** | Moderate; can handle profile+count natively |
| ersatz (substitute, shuffle) | **Reimplement** | Trivial sequence manipulation |
| deep_lift_shap | **Do NOT reimplement** | Complex algorithm; high risk of silent bugs; use tangermeme or Captum with `F.relu` → `nn.ReLU` fix |
| pisa | **Do NOT reimplement** | Thin loop over deep_lift_shap; follows its choice |
| seqlet | **Use tangermeme** | Independent post-processing; no model interface |
| annotate | **Use tangermeme** | Independent post-processing; no model interface |

### 9.9 Impact on the wrapper strategy

If Tier 1 and Tier 2 operations are reimplemented natively, the `TangermemeWrapper` is only needed for DeepLIFT/SHAP and PISA. This simplifies the wrapper significantly — it only needs to handle the `reduce` parameter (since DeepLIFT requires 2D output), and the `"both"` output mode becomes irrelevant (the native cerberus functions can return structured results directly).

---

## 10. Future Considerations

### Multi-channel models

Current BPNet/Pomeranian models use `output_channels=["signal"]` (single channel). If models are extended to multi-channel (e.g., predicting multiple marks simultaneously), the native cerberus analysis functions and wrapper handle this transparently — `C > 1` in all output shapes.

### Ensemble wrapping

Cerberus has `ModelEnsemble` for multi-fold aggregation. For DeepLIFT/SHAP (where the wrapper is still needed), an ensemble wrapper could:
1. Wrap each fold model in `TangermemeWrapper`
2. Average predictions across folds
3. Return the averaged tensor

For native cerberus operations (Tier 1+2), ensemble support can be built directly into the analysis functions.

### ProfileLogRates output type

Some cerberus models (e.g., Gopher) return `ProfileLogRates` instead of `ProfileCountOutput`. Both the native analysis functions and the DeepLIFT wrapper should handle this output type:
```python
# ProfileLogRates: log_rates (B, C, L) → exp(log_rates) for counts
```

### Tangermeme version compatibility

This analysis is based on tangermeme v1.0.3. The model interface contract (`forward() → Tensor or tuple`) is fundamental to tangermeme's design and unlikely to change, but specific function signatures may evolve. Since the hybrid approach (Section 9) limits tangermeme dependency to DeepLIFT/SHAP only, version compatibility concerns are confined to a narrow interface.

---

## 11. Summary of Required Changes

Based on the hybrid recommendation from Section 9 (reimplement Tier 1+2 natively, use external for DeepLIFT/SHAP):

| Priority | Change | Files | Resolves |
|----------|--------|-------|----------|
| P0a | Fix `F.relu()` → `nn.ReLU()` in cerberus models | `src/cerberus/models/bpnet.py`, `src/cerberus/layers.py` | Prerequisite for any DeepLIFT/SHAP (Section 9.3) |
| P0b | Implement native cerberus analysis functions (predict, marginalize, ablate, variant_effect, space, ISM) | `src/cerberus/analysis.py` (new) | Issues 1, 3; eliminates wrapper for Tier 1+2 ops |
| P0c | Implement sequence manipulation utilities (substitute, shuffle) | `src/cerberus/analysis.py` or `src/cerberus/ersatz.py` (new) | Required by native analysis functions |
| P1 | Create `TangermemeWrapper` for DeepLIFT/SHAP only | `src/cerberus/tangermeme.py` (new) | Issues 1, 2; needed only for Tier 3 |
| P2 | Add tests for native analysis functions | `tests/` | Validation |
| P3 | Add tests for DeepLIFT/SHAP wrapper | `tests/` | Validation |
| P4 | Extend for `ProfileLogRates` output type | Analysis functions + wrapper | Generality |
