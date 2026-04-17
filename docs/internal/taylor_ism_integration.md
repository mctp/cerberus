# Taylor ISM (TISM) Integration Plan

## Overview
Integrate Taylor-approximated In Silico Mutagenesis (TISM, Sasse et al., *iScience* 2024) into Cerberus as a first-class attribution method next to exact ISM. TISM replaces `3 * L` forward passes with a single forward + single backward pass, yielding a `(B, 4, L)` attribution tensor whose values approximate the raw single-base ISM deltas `f(s_ref_with_b_at_l) − f(s_ref)`.

Both methods must:

1. Accept the same `AttributionTarget` wrapper (scalar target per batch element).
2. Share the same output contract: `(B, 4, L)` tensor with the observed-base convention toggleable via a `tf_modisco_format: bool` flag.
3. Produce **bit-identical outputs on models with a linear input-to-target map** (verified by `torch.allclose`).

Reference implementation: `../s2f-models/repos/TISM/tism/torch_grad.py` (Sasse et al.). We port the math directly into `src/cerberus/attribution.py`; we do **not** vendor the repo.

## Mathematical Foundation

### Per-nucleotide TISM (paper Eq. 7 / TISM reference `output='tism'`)

Given a trained model `f`, one-hot input `s0 ∈ {0,1}^{4×L}`, and input gradient `g = ∂f/∂s0`:

```
TISM(l, b) = g[b, l] − g[ref(l), l]
```

where `ref(l) = argmax_b s0[b, l]`. At the reference base, `TISM(l, ref(l)) = 0` exactly, matching raw ISM's zero baseline. This is the formulation Cerberus uses.

TISM reference code (`correct_multipliers`, `output='tism'` branch):

```python
grad = grad - np.expand_dims(np.sum(grad * x[:, None], axis=channel_axis), channel_axis)
```

`np.sum(grad * x, axis=channel_axis)` selects `g[ref(l), l]` via the one-hot mask, so this is identical to the formula above. Cerberus should reproduce this exactly.

### Attribution map / off-simplex correction (paper Eq. 8 / TISM `output='corrected'`)

The paper also defines a mean-centered "attribution map":

```
ATISM(l, b) = g[b, l] − mean_j g[j, l]
```

which equals Majdandzic et al. 2023's off-simplex gradient correction. ATISM is related to TISM by a per-position constant:

```
ATISM(l, b) = TISM(l, b) + (g[ref(l), l] − mean_j g[j, l]) = TISM(l, b) − mean_j TISM(l, j)
```

Cerberus already ships `mean_center_attributions` which performs exactly this mean-centering on a `(B, 4, L)` tensor. Users who want ATISM can pipe raw TISM through it; we do **not** need a second kernel.

### Cerberus TF-MoDISco format (hybrid convention)

Cerberus ISM emits a `(B, 4, L)` tensor where:

* **Non-reference channels** hold the raw delta `ISM(l, b)` (unchanged).
* **Reference channel** is overwritten with `−mean_j raw_delta(l, j) = −(1/4) Σ_j ISM(l, j)`, matching the paper's Eq. 5 definition of AISM at the reference.

This is a **hybrid** convention: fully centered at the reference (where TF-MoDISco's `contrib_scores = one_hot * hypothetical_contribs` reads), but raw deltas elsewhere (preserving the "effect of mutation" interpretation at alternative bases). It is *not* the paper's fully centered AISM (Eq. 4) and *not* the raw TISM either. TISM must follow the **same** hybrid convention to remain a drop-in replacement for ISM downstream (notably in `export_tfmodisco_inputs.py`).

For a linear model `f(x) = Σ w[b, l] x[b, l]`:

* Raw ISM: `ISM(l, b) = w[b, l] − w[ref, l]` (no error).
* TISM: `g[b, l] = w[b, l]`, so `TISM(l, b) = w[b, l] − w[ref, l]` — *identical to raw ISM*.
* Both TF-MoDISco-formatted outputs are therefore bit-identical, not merely close. This is the basis of the parity test.

## Key Insights from the Manuscript

* **Speedup**. `3·L / batch_size` forward passes → one forward + one backward. Measured: ~25× at 251 bp, ~160× at 1 kb, ~8000× at 20 kb. The gain grows with sequence length and dominates Yuzu and fastISM.
* **Concordance with ISM**. Mean Pearson ~0.7 against exact ISM; 87% of sequences ≥ 0.6. TISM↔ISM correlation from the same model (0.7) *exceeds* ISM↔ISM between re-initialized models (0.58), i.e. TISM sits inside model-level uncertainty.
* **Non-linearity matters**. Under-trained models, smaller training sets, or early epochs produce *higher* TISM↔ISM correlation because they rely on near-linear effects; fully trained models develop non-linear base interactions Taylor cannot capture. Suggests a diagnostic use: TISM↔ISM gap flags regions/models with strong base-pair interactions.
* **Architecture sensitivity**. ReLU (R~0.61) and max-pool (R~0.66) degrade concordance; GELU + weighted-mean pooling + residual + batch-norm (the Cerberus default) sit comfortably near the 0.7 mean. This matters: Cerberus's pooling choice is favorable for TISM.
* **Low-accessibility regions**. Sequences with low predicted counts / high CV across cell types show worse TISM↔ISM agreement. Warn users that attribution at quiet loci should lean on exact ISM.
* **Majdandzic link**. Centered TISM ≡ Majdandzic off-simplex correction ≡ hypothetical attributions with uniform 0.25 baseline. The paper closes this theoretical gap.
* **Limitations**. First-order only: saturation effects, long-range non-linear interactions, and conditional-epistasis bases are invisible to TISM. DeepLIFT/DeepSHAP remain necessary when those matter.

## Structural Review: TISM vs `attribution.py`

A side-by-side inspection of `tism/torch_grad.py` + `tism/utils.py` against `src/cerberus/attribution.py` surfaces structural gaps that go beyond "add a TISM function." These motivate Phase 6 (Structural Consolidation) below, but several touch Phase 1 decisions and are flagged inline. Items partially or fully addressed by the prerequisite refactors are marked **[LANDED]**.

### 1. No compute / post-process separation — **[PARTIAL]**
TISM cleanly factors `takegrad(...)` → `correct_multipliers(grad, output, x, baseline, channel_axis)` as a pipeline, with `correct_multipliers` reused by **all three** TISM methods (`takegrad`, `ism`, `deepliftshap`) in [torch_grad.py:140](../../s2f-models/repos/TISM/tism/torch_grad.py#L140) and [utils.py:125](../../s2f-models/repos/TISM/tism/utils.py#L125). One pure-function post-process, five attribution modes, zero duplication.

The TF-MoDISco ref-override half of the split is **landed**: `_apply_tf_modisco_ref_override(attrs, inputs, span_start, span_end)` is a private helper, already reused by `compute_ism_attributions` and ready for TISM in Phase 1. The full multi-mode post-process (`apply_attribution_mode`) plus `compute_attributions` facade remains Phase 6.

### 2. No `AttributionMode` enum for output formulation
TISM's `output=` taxonomy in `correct_multipliers`:

| TISM `output` | Formula | Paper / reference |
|---|---|---|
| `'local'` | `grad` | raw gradient |
| `'global'` | `grad * (x − baseline)` | Ancona et al. 2017 |
| `'corrected'` | `grad − mean_j grad` | Majdandzic 2023 / paper Eq. 8 |
| `'hypothetical'` | `grad − Σ_j baseline_j * grad_j` | TF-MoDISco upstream |
| `'tism'` | `grad − grad_ref` | paper Eq. 7 |

Cerberus effectively exposes only a **sixth** hybrid mode (mutants raw + ref `= −mean`), not in TISM's list, implemented implicitly via the `tf_modisco_format` flag. No enum; users can't pick a formulation without re-reading source.

### 3. Naming collision: `ATTRIBUTION_MODES` — **[LANDED]**
The old `ATTRIBUTION_MODES` constant named `AttributionTarget` **reductions** (`log_counts`, `profile_bin`, ...) — *what scalar we attribute to*, not *how we post-process the attribution*. Renamed to `TARGET_REDUCTIONS` in the rename commit (`53d0d4e`) so the `AttributionMode` enum can land in Phase 6 without colliding.

### 4. No `baseline` parameter
TISM accepts baseline as `None` → uniform `1/N`, `(N,)` per-base frequencies, `(N, L)` per-position, or full `(B, N, L)` per-sample ([torch_grad.py:76-95](../../s2f-models/repos/TISM/tism/torch_grad.py#L76-L95)). Required for `hypothetical`, `global`, and for DeepLiftShap semantics. Cerberus has **no** baseline anywhere in `attribution.py`; DLS baselines live only inside `tools/export_tfmodisco_inputs.py` (shuffle vs zero strategy flags). Any future port of `hypothetical` or `global` modes needs a baseline surface in the library.

### 5. No native multi-track
`AttributionTarget` reduces to a scalar by construction; multi-track attribution means looping `AttributionTarget(channel=k)` for `k in tracks` and redoing the forward each time. TISM's `takegrad(tracks=[...])` and `ism(tracks=[...])` iterate tracks sharing one cached forward per sample. Phase 4 addresses this computationally, but the **class design** (one target = one scalar) is the root constraint.

### 6. No `multiply_by_inputs` switch
Standard "gradient × input" visualization is one line in TISM (`ismout * x.unsqueeze(1)`). Cerberus has no hook for it — it's usable only via users re-implementing the multiply outside the library.

### 7. No internal batching / device management
TISM's `ism` and `deepliftshap` take `batch_size` and `device` ([utils.py:139](../../s2f-models/repos/TISM/tism/utils.py#L139), [utils.py:17](../../s2f-models/repos/TISM/tism/utils.py#L17)). Cerberus offloads both to the caller. `export_tfmodisco_inputs.py` reimplements batching; any new tool will do the same. This is a deliberate Cerberus convention (library stays stateless w.r.t. device/batch) but deserves a one-line docstring note rather than being implicit.

### 8. Captum-GELU footgun acknowledged in TISM, unexamined here
TISM explicitly rejects Captum's DeepLiftShap because it doesn't support GELU and several other non-linearities, and switches to `tangermeme.deep_lift_shap` ([utils.py:24-27](../../s2f-models/repos/TISM/tism/utils.py#L24-L27)). Cerberus uses GELU throughout *and* routes DLS through Captum. Recent commit `80c4d73 fix: wrap captum DeepLift forward_func in nn.Module` is circumstantial evidence this is already causing issues. Investigation item (independent of TISM integration): validate Cerberus's Captum-DLS output against `tangermeme.deep_lift_shap` on a GELU-rich checkpoint; if they disagree materially, migrate to tangermeme the way TISM did.

### 9. Plot helper colocation (minor)
TISM's `plot_attribution` lives next to the compute code ([utils.py:266](../../s2f-models/repos/TISM/tism/utils.py#L266)); Cerberus's Phase 1 plan puts `plot_seqlogo` in `plots.py`. Structural observation only — Cerberus's placement is consistent with its existing `save_count_scatter` organization and is correct.

---

## Implementation Plan

The plan is staged. Phase 1 lands a minimal, correct, regression-safe TISM. Later phases add robustness and speed one change at a time, each gated by a regression check (parity on `_WeightedScalarTarget` + Pearson ≥ 0.6 on a fixed biasnet fixture) before the next phase starts.

### Phase 1 — Minimal Port (land first, ship-ready)

#### 1. `src/cerberus/attribution.py`

Replace the existing `compute_taylor_ism_attributions` with an implementation that parallels `compute_ism_attributions` and exposes `tf_modisco_format`.

```python
def compute_taylor_ism_attributions(
    target_model: torch.nn.Module,
    inputs: torch.Tensor,
    span: IsmSpan,
    *,
    tf_modisco_format: bool = True,
) -> torch.Tensor:
    """First-order Taylor approximation of `compute_ism_attributions`.

    Replaces 3·L forward passes with one forward + one backward. On models
    whose input-to-target map is linear (e.g. `_WeightedScalarTarget` in the
    tests), the returned tensor is bit-identical to `compute_ism_attributions`.

    Parameters
    ----------
    target_model:
        Module wrapped by `AttributionTarget` (scalar per batch element).
        Must accept `(B, C, L)` input with C ≥ `N_NUCLEOTIDES` and first
        `N_NUCLEOTIDES` channels = DNA.
    inputs:
        `(B, C, L)` one-hot (or soft) input. Reference base at each position
        is taken as `argmax` over the first `N_NUCLEOTIDES` channels.
    span:
        `(start, end)` tuple; `None` on either side defaults to the sequence
        endpoint. Positions outside the resolved span remain zero.
    tf_modisco_format:
        If True (default), the reference channel receives `−mean_j raw_delta_j`,
        matching the TF-MoDISco `one_hot * hypothetical_contribs` convention
        used by `compute_ism_attributions`. If False, raw TISM deltas are
        returned unchanged (reference base = 0), matching TISM reference
        `output='tism'`.
    """
```

Body steps:

1. **Validation.** `inputs.shape[1] >= N_NUCLEOTIDES`. Resolve span via `resolve_ism_span(seq_len, span)`.
2. **Leaf with grad.** `x = inputs.detach().clone().requires_grad_(True)`.
3. **Forward.** `out = target_model(x).reshape(batch_size)` — requires scalar target, matching ISM's contract.
4. **Backward.** `(grads,) = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=False, retain_graph=False)`. Summed-output trick is correct because batch elements are independent.
5. **Raw deltas via dot product** (already Phase-2a-compatible, see below). `grad_ref = (grads[:, :N_NUCLEOTIDES, :] * inputs[:, :N_NUCLEOTIDES, :]).sum(dim=1)  # (B, L)`; `raw = grads[:, :N_NUCLEOTIDES, :] - grad_ref.unsqueeze(1)`. Matches `correct_multipliers` in [torch_grad.py:198-199](../../s2f-models/repos/TISM/tism/torch_grad.py#L198-L199) exactly and is well-defined on soft inputs.
6. **Span zeroing.** Allocate zero-initialized `(B, N_NUCLEOTIDES, L)`; write `raw[:, :, span_start:span_end]` into it.
7. **TF-MoDISco override** (when `tf_modisco_format=True`): call the existing `_apply_tf_modisco_ref_override(attrs, inputs, span_start, span_end)` helper. When False, skip this call — reference channel stays at zero (raw TISM mode).

Keep `compute_ism_attributions` behaviorally identical but add the same `tf_modisco_format` flag. When False, skip the `_apply_tf_modisco_ref_override` call and leave the raw zero delta at reference. **Default stays True** — no change for existing callers.

Also: do **not** call `model.eval()` inside either function. Matches current ISM behavior; caller sets eval (documented in docstring).

Export `compute_taylor_ism_attributions` from `src/cerberus/__init__.py`.

**Prerequisite refactors (already landed on `marcin-feature`).** Four small changes landed ahead of Phase 1 to remove naming collisions and DRY up the ISM / TISM overlap before the second method arrives:

*Rename commit* (`53d0d4e`):
* `ATTRIBUTION_MODES` → `TARGET_REDUCTIONS` (the set names reductions, not modes — clears the collision with the future `AttributionMode` enum).
* `AttributionTarget(mode=..., ...).mode` → `reduction=..., .reduction`.
* `apply_off_simplex_gradient_correction()` → `mean_center_attributions()` (the operation is equivalent to Majdandzic off-simplex correction *and* paper ATISM *and* hypothetical uniform-baseline; "mean-center" names what it does).

*Architecture refactor (this commit)*:
* `N_NUCLEOTIDES = 4` module constant + `IsmSpan = tuple[int | None, int | None]` type alias. Replaces six magic-4 literals and makes the DNA-alphabet assumption greppable.
* `compute_ism_attributions(..., ism_start, ism_end)` → `(..., span: IsmSpan)`. `resolve_ism_span(seq_len, start, end)` → `(seq_len, span)`. Halves the span surface area on every signature that grows with TISM.
* Hoisted the TF-MoDISco reference override out of the per-position ISM loop into one vectorized post-pass.
* Extracted `_apply_tf_modisco_ref_override(attrs, inputs, span_start, span_end)` private helper. TISM reuses this directly — no duplication.

Hard break on all of the above (no deprecation shims). In-repo callers (tests + `tools/export_tfmodisco_inputs.py`) updated atomically. CLI flags `--target-mode`, `--ism-start`, `--ism-end` unchanged (external contracts; the tool packs `(args.ism_start, args.ism_end)` into a tuple at the boundary).

#### 2. Silent assumptions to make explicit

Each needs a docstring line or a `raise`:

| Assumption | Current state | Proposed |
|---|---|---|
| First 4 channels = ACGT | Implicit in `[:, :4, :]` slice | Documented in docstring |
| Reference = argmax over 4 channels | Silent; wrong on soft / PWM inputs | Add NOTE in docstring; not enforced (breaks biasnet ISM use cases if enforced) |
| Target is scalar per batch | `.reshape(batch_size)` will raise | Document "scalar output required" + catch non-scalar with a clearer `ValueError` |
| `model.eval()` not called | Silent; shared convention with ISM | Document: "caller must set eval mode" |
| Runs under grad context | Fails under `torch.no_grad()` | `torch.enable_grad()` wrap inside the function |
| Ignores conditioning channels (C > 4) | Implicit via slice | Documented |

#### 3. `src/cerberus/plots.py` — `plot_seqlogo`

Add a thin `logomaker`-based helper (optional import, mirroring `save_count_scatter`):

```python
def plot_seqlogo(
    ax,
    attrs: np.ndarray,      # (4, L) signed attributions
    *,
    alphabet: str = "ACGT",
    as_ic: bool = False,
) -> None:
```

* `as_ic=False` (default, matches the TISM reference `plot_attribution`): build a pandas DataFrame `{A, C, G, T: attrs[i]}` and call `logomaker.Logo(df, ax=ax)`. Letter heights track signed attribution; positive/negative stacking is handled by logomaker.
* `as_ic=True`: convert attributions to probabilities (softmax along axis 0) and scale by per-position information content (2 + Σ p log2 p clipped to [0, 2]).
* Raise `ImportError` with a clear install hint (`pip install cerberus[extras]`) if `logomaker` / `pandas` are unavailable.

#### 4. `tools/plot_biasnet_ism.py`

Delete the local `plot_logo` (lines ~403–500) and replace call sites with `from cerberus.plots import plot_seqlogo`. Preserve the stacked logo-over-heatmap layout. This removes ~100 lines of matplotlib patch-effect code.

#### 5. `pyproject.toml`

Add `"logomaker"` and `"pandas"` to `[project.optional-dependencies].extras`. `pandas` is a hard `logomaker` dep; listing it explicitly makes the requirement visible. No change to the hard deps.

#### 6. `tools/export_tfmodisco_inputs.py`

* Extend `--attribution-method` choices: `["integrated_gradients", "deep_lift_shap", "ism", "taylor_ism"]`.
* Re-use the existing `--ism-start` / `--ism-end` flags for `taylor_ism`; add a note that they define the window in which deltas are computed (positions outside are zero, same as ISM).
* Dispatch to `compute_taylor_ism_attributions(..., tf_modisco_format=True)` so the downstream TF-MoDISco `shap.npz` / `ohe.npz` shape contract is unchanged.

#### 7. Tests (`tests/test_attribution.py`)

Add:

1. **Linear parity.** On `_WeightedScalarTarget`, assert `torch.allclose(compute_taylor_ism_attributions(...), compute_ism_attributions(...))` with `atol=1e-6` across several spans (full, partial window, edge positions).
2. **Raw mode.** With `tf_modisco_format=False`, assert `attrs[:, ref, :]` is exactly zero within the span; ISM and TISM both satisfy this.
3. **ATISM equivalence.** `mean_center_attributions(raw_tism)` should equal `grads - grads.mean(dim=1, keepdim=True)` (within the span), proving the Majdandzic bridge.
4. **Span zeroing.** Positions outside the resolved `span` remain zero.
5. **Non-scalar target raises.** Wrap a model that returns `(B, 2)` and assert a clear `ValueError`.
6. **Shape & dtype.** `(B=3, 4, L=16)` input → `(3, 4, 16)` output, matching input dtype.

Do **not** test against a fully non-linear model for an exact value; correlation against ISM there is a property test (~0.7) not a unit test.

#### 8. Docs + changelog

* Add a "Taylor ISM" section to [docs/attribution.md](docs/attribution.md) (or whichever page currently documents `compute_ism_attributions` — grep before editing).
* Mention `plot_seqlogo` in the plotting page.
* `CHANGELOG.md` → `[Unreleased]`: "Added `compute_taylor_ism_attributions` and `plot_seqlogo`; `tools/export_tfmodisco_inputs.py` now accepts `--attribution-method taylor_ism`."
* Regenerate LLM context: `python tools/generate_llms_txt.py`.

### Regression Gate (runs between every phase)

Before starting any phase ≥ 2, capture a reference artifact from the current `main`:

1. **Linear parity.** `torch.allclose(compute_taylor_ism_attributions(...), compute_ism_attributions(...))` on `_WeightedScalarTarget` — must stay `True` with `atol=1e-6` through all phases.
2. **Non-linear fidelity fixture.** Pick one trained biasnet checkpoint and 32 sequences from a held-out chromosome. Store baseline `(32, 4, L)` TISM tensor + per-sequence Pearson(TISM, ISM) as `tests/data/tism_reference.npz`. New phase passes if:
   * `torch.allclose(new_tism, baseline_tism, atol=1e-5)` on the *same* seed, OR
   * Per-sequence Pearson(new_tism, ISM) drops by ≤ 0.01 on average AND no single sequence drops by > 0.05.
3. **Runtime floor.** Benchmark one forward+backward pass on a 10k-bp synthetic CNN; runtime must not regress > 10% phase-over-phase (except Phase 4, which explicitly targets speed).

Each phase lands in its own PR, the regression gate is re-run, the fixture is refreshed only when the change is expected to shift values (documented in the PR).

### Phase 2 — Numerical & API Hardening

Ordering chosen so failures surface loudly rather than silently returning wrong attributions.

#### 2a. Soft-input-robust reference via dot product
Replace `argmax` + fancy indexing with TISM's own formulation:
```python
grad_ref = (grads[:, :4, :] * inputs[:, :4, :]).sum(dim=1)   # (B, L)
raw      = grads[:, :4, :] - grad_ref.unsqueeze(1)
```
On one-hot inputs this is bit-identical to Phase 1 (regression gate #1 + #2 must stay green). On soft / PWM inputs it's well-defined where `argmax` is not. Matches `correct_multipliers` in [torch_grad.py:198-199](../../s2f-models/repos/TISM/tism/torch_grad.py#L198-L199). The TF-MoDISco override path still needs a one-hot reference — keep `argmax` there, but add a warning if `inputs[:, :4, :].max(dim=1).values.min() < 1 - 1e-4` (i.e. "soft input passed with `tf_modisco_format=True`").

#### 2b. Defensive gradient hygiene
Three independent guards, each its own commit:

1. Wrap the body in `with torch.enable_grad():` so callers inside `@torch.no_grad()` (common in eval loops) don't silently fail.
2. After `autograd.grad`, assert `torch.isfinite(grads).all()` with an actionable message ("non-finite input gradient — check for BN-in-eval on a path with zero variance, or dead ReLUs").
3. If `inputs.shape[1] > 4`, assert that `grads[:, 4:, :]` is zero (conditioning channels should not leak into the DNA attribution). If it isn't, raise — silently slicing is how subtle bugs become load-bearing.

#### 2c. Freeze parameters for input-only gradient
Save each parameter's `requires_grad` on entry, set all to `False`, restore on exit:
```python
saved = [(p, p.requires_grad) for p in target_model.parameters()]
try:
    for p, _ in saved: p.requires_grad_(False)
    # ... forward + autograd.grad(out, x) ...
finally:
    for p, flag in saved: p.requires_grad_(flag)
```
We only want `∂f/∂x`; freezing params shrinks the autograd graph and lets PyTorch drop activations that only flow to parameter leaves. Regression gate #1 must stay bit-identical; expect a modest memory win (measure on the 10k-bp fixture and note in the PR).

### Phase 3 — Scaling

#### 3a. Batch / span chunking
Add `chunk_size: int | None = None` that loops sub-batches and calls the Phase 2 kernel per chunk, concatenating along dim 0. If `None`, behavior is identical to Phase 2 (regression gate bit-identical). Picked first because export-style runs over thousands of peaks hit memory limits long before single-sequence runs do.

Consider a secondary `span_chunk_size` only if profiling on the 10k+ bp fixture shows the backward pass dominates — span chunking requires re-running the backward per chunk and is not a free win.

#### 3b. Reverse-complement averaging
Add `reverse_complement: bool = False`. When True:
```python
tism_fwd = _tism_core(model, x, ...)
tism_rc  = _rc(_tism_core(model, _rc(x), ...))
attrs    = 0.5 * (tism_fwd + tism_rc)
```
where `_rc` reverses along the last axis and swaps `[A,C,G,T] → [T,G,C,A]`. Cerberus models are trained with RC augmentation, so the attribution should respect that symmetry. Costs exactly 2× a single TISM call. Regression check: on the biasnet fixture, RC-averaged TISM should correlate with exact ISM *as well or better* than single-strand TISM on ≥ 80% of sequences.

### Phase 4 — Multi-Track (biggest real-world speedup)

Today, attributing one sequence across 81 cell types means 81 `(forward + backward)` passes because `AttributionTarget` always reduces to a scalar. With activations cached once and PyTorch's `is_grads_batched=True` (or `torch.func.jacrev` / `vmap`), we can do 1 forward + T cheap backwards that share the stored activations.

Proposed API:
```python
def compute_taylor_ism_attributions(
    target_model,            # now may return (B,) or (B, T)
    inputs,
    span: IsmSpan,
    *,
    tracks: Sequence[int] | None = None,   # None → all output tracks
    tf_modisco_format: bool = True,
) -> torch.Tensor:                         # (B, T, 4, L) if tracks else (B, 4, L)
```
`AttributionTarget` grows a `tracks` constructor arg (or the caller passes a multi-head target directly). When `tracks is None` (Phase 1/2/3 behavior), the return shape stays `(B, 4, L)` — regression gate #1 + #2 still pass.

Implementation order:
1. Prototype with a manual loop over `tracks` to validate correctness against the scalar path.
2. Swap the loop for `torch.autograd.grad(..., is_grads_batched=True)` with a stacked `grad_outputs` tensor; verify identical result.
3. Benchmark against `torch.func.vmap(torch.func.jacrev(target_model))(x)` on the 81-cell-type biasnet — pick whichever wins. `jacrev` tends to win for dense multi-track, `is_grads_batched` for sparse track subsets.

Regression gate additions for this phase:
* When `tracks is None`, output must remain bit-identical to Phase 3 on the linear fixture.
* When `tracks=[k]`, output must equal `compute_taylor_ism_attributions(track_k_scalar_target, ...)` from Phase 3 for every `k`.
* Expected wall-clock: ≥ 10× faster than the looped-scalar baseline on 81 tracks (paper-level speedup, but more conservative because PyTorch's batched backward overhead is real).

### Phase 5 — Performance Polish

Only pursue if profiling demands it:

* `torch.compile` wrapping of the core forward+backward combo.
* Diagnostic mode returning `(attrs, grad_norm, raw_deltas)` as a `NamedTuple` — useful for the paper's hypothesis that TISM–ISM discordance flags non-linear loci.

### Phase 6 — Structural Consolidation

Once Phases 1–4 are stable and the regression fixtures lock down behavior, refactor `attribution.py` so new attribution methods compose cleanly and the three inlined copies of "post-process" collapse into one. Motivated by Structural Review §1, §2, §4 above.

#### 6a. Introduce `AttributionMode`
```python
from enum import StrEnum

class AttributionMode(StrEnum):
    RAW            = "raw"             # as-computed, ref channel = 0
    TF_MODISCO     = "tf_modisco"      # current Cerberus hybrid (ref := −mean_j raw_j)
    MEAN_CENTERED  = "mean_centered"   # raw − mean_j raw_j   (TISM 'corrected', paper Eq. 8)
    HYPOTHETICAL   = "hypothetical"    # raw − Σ_j baseline_j · raw_j
    GLOBAL         = "global"          # raw · (x − baseline)
```
`TF_MODISCO` stays the default so no caller migrates on day one. The `ATTRIBUTION_MODES` / `AttributionTarget.mode` / `apply_off_simplex_gradient_correction` aliases were already dropped in the prerequisite rename commit (no deprecation shims), so there is nothing left to clean up in this phase.

#### 6b. Extract `apply_attribution_mode`
```python
def apply_attribution_mode(
    raw: torch.Tensor,          # (B, 4, L)  — ref channel = 0
    inputs: torch.Tensor,       # (B, ≥4, L)
    mode: AttributionMode,
    *,
    baseline: torch.Tensor | None = None,   # None → uniform 0.25
) -> torch.Tensor:
```
Pure function. Replaces the inlined ref-override in `compute_ism_attributions` and `compute_taylor_ism_attributions`, and the separate post-processing in `tools/export_tfmodisco_inputs.py`. `MEAN_CENTERED` is exactly the existing `mean_center_attributions` function — the existing helper becomes the `MEAN_CENTERED` branch of `apply_attribution_mode` and can be deleted once callers migrate.

Baseline broadcasting follows TISM ([torch_grad.py:76-95](../../s2f-models/repos/TISM/tism/torch_grad.py#L76-L95)): accept `(4,)`, `(4, L)`, or `(B, 4, L)`; `None` → uniform `0.25`.

#### 6c. Split compute from post-process
Rename the core kernels to reflect their output contract:
```python
compute_ism_raw(target, x, span) -> Tensor         # (B, 4, L), ref = 0
compute_taylor_ism_raw(target, x, span) -> Tensor  # (B, 4, L), ref = 0
```
And add a facade that composes:
```python
def compute_attributions(
    method: Literal["ism", "taylor_ism", "deep_lift_shap", "integrated_gradients"],
    target: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    mode: AttributionMode = AttributionMode.TF_MODISCO,
    baseline: torch.Tensor | None = None,
    tracks: Sequence[int] | None = None,    # Phase 4 API
    span: tuple[int | None, int | None] = (None, None),
    chunk_size: int | None = None,          # Phase 3a
    reverse_complement: bool = False,       # Phase 3b
    multiply_by_inputs: bool = False,       # Structural Review §6
) -> Tensor:
```
`compute_ism_attributions` and `compute_taylor_ism_attributions` stay as thin wrappers (`compute_attributions(method="ism", mode=TF_MODISCO, ...)` etc.) so callers don't churn. `tools/export_tfmodisco_inputs.py` collapses its per-method dispatch into a single `compute_attributions(method=args.attribution_method, ...)` call — ~40 lines deleted.

#### 6d. Validate paper identities as tests
Once the pipeline is factored, the paper's theoretical claims become one-line pytest assertions:

1. `mean_centered(taylor_ism_raw(x))` ≈ `hypothetical(gradient, baseline=uniform_0.25)` — paper Eq. 11 / 15.
2. `mean_centered(ism_raw(x))` ≈ `mean_centered(taylor_ism_raw(x))` on a linear model — bit-identical.
3. `tf_modisco(ism_raw(x))` at the reference channel equals `mean_centered(ism_raw(x))` at the reference channel — Cerberus hybrid consistency.

These should be correlation-based unit tests (`pearsonr > 0.99` on small-CNN) not `allclose`, because numerical float order matters.

#### 6e. Reassess Captum DLS vs `tangermeme`
Independent investigation (Structural Review §8). Deliverable is a benchmark script, not a refactor decision:

* On a GELU-rich Cerberus checkpoint (the biasnet fixture), compute DLS attributions via (a) current Captum path with `nn.Module`-wrapped `forward_func`, (b) `tangermeme.deep_lift_shap`.
* Report per-sequence Pearson correlation between (a) and (b), and convergence-delta statistics from both.
* If (a) and (b) disagree materially (Pearson < 0.95 on > 10% of sequences, or Captum deltas exceed the threshold that tangermeme flags), file a migration proposal. Otherwise close the item.

Regression gate carries over from Phase 4; this is additive structural work, not a values change. Adds one new gate: `compute_attributions(method="taylor_ism", mode=TF_MODISCO)` must be bit-identical to the post-Phase-4 `compute_taylor_ism_attributions`.

## Correctness Matrix

| Property | ISM (Cerberus) | TISM (Cerberus, `tf_modisco_format=True`) | Notes |
|---|---|---|---|
| Raw delta at mutant `b` | `f(s[b@l]) − f(s)` | `g[b,l] − g[ref,l]` | Exact for linear `f`; first-order for non-linear |
| Value at reference | `−mean_j raw_delta_j` (Eq. 5) | `−mean_j raw_delta_j` (same formula) | Equal on linear models |
| Span behavior | Zero outside `[start, end)` | Zero outside `[start, end)` | Same |
| Forward passes | `3·L / batch` | 1 forward + 1 backward | Backward ≈ 2× forward |
| Matches paper ATISM (Eq. 8) | After `mean_center_attributions` | After `mean_center_attributions` | `grad − mean(grad)` bridge |
| Matches TISM reference `output='tism'` | — | Yes (with `tf_modisco_format=False`) | Bit-identical up to float precision |
| Matches TISM reference `output='corrected'` | — | Yes (raw → `mean_center_attributions`) | Majdandzic identity |

## Success Criteria

1. `compute_taylor_ism_attributions` is exported from `cerberus` and documented.
2. `torch.allclose(ism, tism)` on `_WeightedScalarTarget` — the parity test is the load-bearing check.
3. `python tools/export_tfmodisco_inputs.py --attribution-method taylor_ism ...` produces `ohe.npz` and `shap.npz` with the same shape / dtype contract as the `ism` path, and TF-MoDISco runs on the output without schema changes.
4. `tools/plot_biasnet_ism.py` renders identical-looking output after switching to `plot_seqlogo` (visual parity, no regression).
5. `pytest -v tests/` and `npx pyright tests/ src/` pass.
6. Docs rebuild cleanly (`mkdocs build --strict`); `llms.txt` regenerated.

## Known Non-Issues (Don't "Fix" These)

* The hybrid TF-MoDISco convention (raw deltas at mutants, `−mean` at reference) is deliberate and matches Cerberus ISM. Do *not* silently switch to paper Eq. 4's fully centered form — it would break downstream TF-MoDISco contribution-score extraction that relies on `one_hot * hypothetical` semantics.
* Per-sample loop in the TISM reference (`for i in range(x.size(0))`) is a legacy workaround; Cerberus's batched `torch.autograd.grad` on a summed scalar is the correct and faster equivalent for independent batch elements.
* `model.eval()` is intentionally not called inside the attribution functions — the caller is responsible (same convention as `compute_ism_attributions`).
