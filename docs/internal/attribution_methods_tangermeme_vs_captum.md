# Attribution Methods: tangermeme vs. captum

An internal technical reference comparing attribution methods in
[tangermeme](https://github.com/jmschrei/tangermeme) and
[captum](https://github.com/meta-pytorch/captum), with a focus on
DeepLIFTSHAP implementation differences.

---

## 1. Overview of Available Methods

### tangermeme

tangermeme is genomics-specific; its attribution methods assume one-hot encoded
DNA sequences of shape `(N, 4, L)`.

| Method | Module | Notes |
|---|---|---|
| DeepLIFT/SHAP | `tangermeme/deep_lift_shap.py` | Primary attribution; custom reimplementation optimized for genomics |
| ISM (In-Silico Mutagenesis) | `tangermeme/saturation_mutagenesis.py` | Single-nucleotide perturbation scan; uses Numba-JIT inner loop |
| Marginalization | `tangermeme/marginalize.py` | Measures prediction change before/after motif insertion |
| Ablation | `tangermeme/ablate.py` | Shuffles out genomic regions to remove information |
| PISA | `tangermeme/pisa.py` | Higher-level interaction analysis |
| Variant Effect | `tangermeme/variant_effect.py` | SNP/indel effect prediction |

There is no standalone vanilla DeepLIFT (only the SHAP variant), and no
integrated gradients, saliency, GradCAM, or LRP.

### captum

captum is a general-purpose attribution library for arbitrary PyTorch models.

**Gradient-based:**
- `Saliency` — `|∂output/∂input|`
- `InputXGradient` — `input * gradient`
- `IntegratedGradients` — path integral of gradients from baseline to input
- `GradientShap` — stochastic SHAP approximation via noisy gradients

**Propagation-based:**
- `DeepLift` — Rescale rule; single baseline
- `DeepLiftShap` — DeepLIFT averaged over a distribution of baselines
- `LRP` — Layer-wise Relevance Propagation
- `GuidedBackprop`, `Deconvolution`, `GuidedGradCam`

**Perturbation-based:**
- `FeatureAblation`, `Occlusion` (sliding window ablation)
- `FeaturePermutation`
- `KernelShap`, `ShapleyValues`, `ShapleyValueSampling`

**Layer/Neuron variants:** Nearly every method has `Layer*` and `Neuron*`
variants (e.g., `LayerDeepLift`, `NeuronDeepLiftShap`, `LayerIntegratedGradients`,
`LayerGradCam`, `LayerConductance`).

**LLM support:** `LLMAttribution`, `LLMGradientAttribution`.

**Noise tunnel:** `NoiseTunnel` wraps any method with SmoothGrad-style
averaging.

---

## 2. DeepLIFTSHAP: Core Algorithm

Both libraries implement the same core concept: run DeepLIFT against multiple
reference sequences and average the resulting contribution multipliers. The
underlying rule is the **Rescale Rule**:

```
multiplier_i = grad_output_i * (f(x)_i - f(x_ref)_i) / (x_i - x_ref_i)
```

When `|x_i - x_ref_i| < eps`, fall back to the regular gradient to avoid
division by zero.

Both implementations register PyTorch backward hooks on nonlinear modules. The
input and reference are concatenated as a batch `[input | reference]` and
passed through the model together. Inside each hook, activations are recovered
by splitting the batch with `chunk(2)`, then `delta_out / delta_in` is
computed.

Despite this shared skeleton, the two implementations diverge in every major
design decision.

---

## 3. Key Differences: DeepLIFTSHAP

### 3.1 Reference/Baseline Construction

**tangermeme:**
- Default: `references=dinucleotide_shuffle`, `n_shuffles=20`
- Each input sequence gets its own set of n shuffled versions of itself,
  preserving dinucleotide composition. This is the standard practice in the
  genomics attribution community (tf-modisco, Kundaje lab).
- References can also be an explicit `torch.Tensor` of shape `(N, n_shuffles, 4, L)`.
- Pair iteration is explicit: the function loops over all `N * n_shuffles`
  (example, reference) pairs in batches of size `batch_size`.

**captum:**
- `DeepLiftShap` requires a pre-built baseline tensor; there is no built-in
  reference generation.
- The baseline distribution is typically global — shared across all inputs
  (e.g., all-zeros, random noise, or a GC-matched background).
- Baselines can be a callable `f(inputs) -> baselines` for dynamic generation,
  but per-example shuffling requires external setup.
- `_expand_inputs_baselines_targets()` uses `repeat_interleave` (inputs) and
  `repeat` (baselines) to build an `(N * n_baselines)`-sized batch, then runs
  a single forward+backward pass.

**Critical difference:** tangermeme generates per-example references by design;
captum uses a shared baseline distribution applied uniformly to all inputs.
This distinction matters in genomics: dinucleotide composition varies between
sequences, so a global reference may introduce baseline-induced artifacts.

---

### 3.2 How SHAP Values Are Computed from DeepLIFT Multipliers

This is the most important algorithmic difference between the two.

**captum (standard `(x - x_ref) * m`):**

After the DeepLIFT backward pass, multipliers `m` are computed. The default
attribution is:

```
attribution = (input - baseline) * m
```

The final SHAP approximation is the mean of these attributions across all
baselines:

```
SHAP(x_i) ≈ mean_j [ m(x, x_ref_j) * (x_i - x_ref_j_i) ]
```

This is mathematically correct for continuous, independent features but is
insufficient for one-hot encoded data where nucleotide channels at a position
are mutually exclusive.

**tangermeme (`hypothetical_attributions`):**

After the backward pass, raw multipliers `m` (same shape as input) are
available. By default, tangermeme applies `hypothetical_attributions()` before
averaging. For each possible nucleotide `i ∈ {A, C, G, T}`:

```
hypothetical_input[:, i] = 1  (one-hot for nucleotide i)
hypothetical_diffs = hypothetical_input - reference
hypothetical_contribs = hypothetical_diffs * multipliers
projected_contribs[:, i] = sum(hypothetical_contribs, dim=channel)
```

This produces a `(4, L)` tensor where each entry answers: **"what would the
contribution be if position p were nucleotide n, regardless of what it actually
is?"** The formula (simplified):

```
SHAP_hypothetical(x_i) ≈ mean_j [ sum_k( m_k(x, x_ref_j) * (1[i=k] - x_ref_j_k) ) ]
```

After averaging across `n_shuffles` references:
- If `hypothetical=True`: return the full `(4, L)` hypothetical attribution
  map.
- If `hypothetical=False` (default): mask the result by the actual one-hot
  sequence `X`, selecting only the values at observed nucleotides. This gives
  the standard attribution track used for visualization and tf-modisco input.

**Why this matters:** In a one-hot encoding, choosing nucleotide A at position p
is not independent of not choosing C, G, or T at that position. A plain
`(x - x_ref) * m` computation at position p gives different values depending on
which nucleotide happened to be observed, making comparisons across positions
misleading. The `hypothetical_attributions` projection fixes this by computing
contributions in a uniform hypothetical space. This is the correct
genomics-aware treatment, and is the same approach used by DeepSHAP in the
Kundaje lab toolchain.

If you want to use captum on one-hot sequence data correctly, you must pass
`custom_attribution_func=hypothetical_attributions`. This is exactly what
tangermeme's internal `_captum_deep_lift_shap()` wrapper does.

---

### 3.3 Hook Registration

**captum:**
- OOP class; hooks stored in `self.forward_handles` / `self.backward_handles`.
- Registration via `model.apply(self._register_hooks)`.
- `SUPPORTED_NON_LINEAR` is a class-level dict; subclassing overrides it.
- Rule functions take explicit `(module, inputs, outputs, grad_input, grad_output, eps)`.
- `eps` is a constructor parameter (default `1e-10`).
- Hooks are cleaned up in a `finally` block; emits a `warnings.warn` on registration.

**tangermeme:**
- Functional design; hooks registered in a standalone `_register_hooks()` function.
- `_NON_LINEAR_OPS` dict is attached directly to each module as `module._NON_LINEAR_OPS`.
- Rule functions take `(module, grad_input, grad_output)`; inputs/outputs come
  from `module.input` / `module.output` stored during the forward hook.
- `eps` is hardcoded: `1e-6` for nonlinear ops, `1e-7` for maxpool.
- Hooks cleared after attribution; `_NON_LINEAR_OPS` is deleted from each module.
- No warning emitted.

**Epsilon comparison:** captum's default `1e-10` is much smaller than
tangermeme's `1e-6`. On low-precision inputs (e.g., `float16`), captum's
tighter threshold may cause numerical instability that tangermeme avoids.

---

### 3.4 Supported Nonlinear Operations

| Op | tangermeme | captum |
|---|---|---|
| ReLU | yes | yes |
| ELU, LeakyReLU | yes | yes |
| Sigmoid, Tanh | yes | yes |
| Softplus | yes | yes |
| Softmax | yes | yes |
| MaxPool1d, MaxPool2d | yes | yes |
| MaxPool3d | no | yes |
| GELU | yes | no |
| SiLU (Swish) | yes | no |
| Mish | yes | no |
| GLU | yes | no |
| ReLU6, RReLU | yes | no |
| SELU, CELU | yes | no |
| Softshrink, LogSigmoid, PReLU | yes | no |

tangermeme covers more modern activation types (GELU, SiLU, Mish, GLU).
captum adds MaxPool3d. For transformers and recent genomic models using GELU/
SiLU, tangermeme's hooks will fire correctly while captum will fall back to
the standard gradient for those ops.

---

### 3.5 Batch Processing and Memory

**tangermeme:**
- Iterates over all `N * n_shuffles` (example, reference) pairs in a `trange`
  loop with batch size `batch_size` controlling total pairs per forward pass.
- Accumulates per-pair multipliers; flushes completed example's multipliers
  to output list after all `n_shuffles` references are processed.
- Supports `verbose=True` with tqdm progress bar.
- Supports `dtype` and `torch.autocast` for mixed-precision inference.
- Additional model arguments passed via `args`.

**captum:**
- Expands everything into one `(N * n_baselines)`-sized batch; single
  forward+backward pass.
- No chunking or memory management — memory must fit the full expanded batch.
- No mixed-precision support.
- Additional model arguments via `additional_forward_args`.

For large `n_shuffles` or large `N`, tangermeme's incremental loop is
considerably more memory-efficient.

---

### 3.6 Convergence Delta

Both compute convergence deltas (`output_diff - input_diff` where
`input_diff = sum((X - ref) * multipliers)`).

**tangermeme:** Computed per batch; warns if any delta exceeds
`warning_threshold` (default `0.001`); optionally printed via
`print_convergence_deltas=True`. Does not raise an exception.

**captum:** Returned as an optional second value when
`return_convergence_delta=True`. Shape is `(N * n_baselines,)` for
`DeepLiftShap`. No automatic warning.

---

## 4. API Summary

| Aspect | tangermeme | captum |
|---|---|---|
| Design pattern | Functional (standalone functions) | OOP (classes with `.attribute()`) |
| Input shape | `(N, 4, L)` assumed | Any `(N, ...)` or tuple of tensors |
| Reference generation | Built-in (`dinucleotide_shuffle`) or user tensor | User must supply baseline tensor or callable |
| Per-example references | Yes | No (shared baseline set) |
| Hypothetical attributions | Built-in, default behavior | Requires `custom_attribution_func` |
| Mixed precision | Yes (`dtype`, `torch.autocast`) | No |
| Memory control | Fine-grained via `batch_size` over pairs | Single batch, no chunking |
| Return raw multipliers | `raw_outputs=True` | Via `custom_attribution_func` interception |
| Convergence delta | Automatic warning | Optional return value |
| Layer/neuron attribution | Not supported | `LayerDeepLift`, `NeuronDeepLiftShap`, etc. |
| Multiple input tensors | Not natively supported | Yes (tuple of inputs) |
| LLM support | No | Yes (`LLMAttribution`) |
| Extensibility | `additional_nonlinear_ops` dict | Subclass and override `SUPPORTED_NON_LINEAR` |

---

## 5. Using captum Correctly for Genomics

If you use captum's `DeepLiftShap` on one-hot sequences, you need to replicate
tangermeme's behavior manually:

```python
from captum.attr import DeepLiftShap
from tangermeme.utils import hypothetical_attributions

dls = DeepLiftShap(model)

# Generate per-example references externally (e.g., dinucleotide shuffle)
# baselines shape: (N * n_shuffles, 4, L) — requires repeat_interleave on inputs

attributions = dls.attribute(
    inputs,
    baselines,
    custom_attribution_func=hypothetical_attributions,
    return_convergence_delta=False,
)

# Mask by observed nucleotides if you don't want hypothetical scores
attributions = attributions * inputs
```

This is exactly what tangermeme's `_captum_deep_lift_shap()` internal wrapper
does.

---

## 6. When to Use Each

| Situation | Recommendation |
|---|---|
| DNA/genomic one-hot sequences | tangermeme — correct hypothetical attributions by default |
| Models with GELU/SiLU activations | tangermeme — has hooks for these; captum falls back to gradient |
| Many shuffles, memory-constrained | tangermeme — iterative batching |
| Non-genomic data (images, tabular) | captum — more general, no genomics assumptions |
| Multi-tensor inputs | captum — native tuple support |
| Layer/neuron-level attribution | captum — `LayerDeepLift` etc. |
| Transformer/LLM attribution | captum — dedicated `LLMAttribution` |
| Integrated gradients | captum — not in tangermeme |

---

*Written 2026-04-08. Source: tangermeme `deep_lift_shap.py`, captum `_core/deep_lift.py`.*
