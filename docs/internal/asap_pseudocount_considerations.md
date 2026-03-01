# ASAP Pseudocount and Scatter Plot Considerations

## Background

This document records a design investigation into `count_pseudocount` behaviour for
ASAP models, prompted by a visible discontinuity in the validation count scatter plot.
No code changes were made.  The issue and its options are captured here for future
reference.

See also: `docs/internal/count_log_spaces.md` for the general two-space framework.

---

## The Scatter Plot Discontinuity

The validation count scatter plot (`save_count_scatter`) plots predicted log-counts
vs target log-counts.  Targets are computed as:

```
target_lc = log(total_count + pseudocount)
```

With the ASAP default of `count_pseudocount = 1.0`:

| Reads in region | total_count | target_lc |
|---|---|---|
| 0 | 0 | log(1) = 0 |
| 1 (100 bp fragment, 4 bp bins) | ~25 | log(26) ≈ 3.26 |
| 10 | ~250 | log(251) ≈ 5.52 |

There is a large gap between zero-read regions (clustered at 0) and even sparsely
covered regions.  This is a visual discontinuity on the X axis, not a numerical
error, but it makes the plot difficult to interpret.

### Why BPNet uses `pseudocount = 150`

ChIP-seq reads are ~150 bp and BPNet uses 1 bp bins, so one read contributes ~150
to `total_count`.  With `pseudocount = 150`:

| Reads | target_lc |
|---|---|
| 0 | log(150) ≈ 5.01 |
| 1 | log(300) ≈ 5.70 |

The zero-read cluster is embedded in the distribution rather than isolated at 0.
The gap between 0 and 1 read is `log(2) ≈ 0.69` in both cases, but the cluster is
no longer at an extreme.

### Correct pseudocount choice

The right pseudocount is approximately the total count contribution of one read:

```
pseudocount ≈ read_length / bin_size    (in raw coverage units)
```

For ASAP with `output_bin_size = 4` and ~100 bp ATAC fragments: `~100 / 4 = 25`.

---

## The Prediction–Target Mismatch for ASAP

A second, more fundamental issue exists beyond the pseudocount choice.

**ASAP uses `ProfilePoissonNLLLoss`.**  The Poisson objective trains:

```
exp(log_rates) ≈ counts   (per bin)
→  logsumexp(log_rates)  ≈  log(total_count)       [pure-log space]
```

**Targets in metrics and scatter plot** are computed as:

```
log(total_count + pseudocount)                     [offset-log space]
```

So predictions and targets are in different spaces.  The systematic offset is:

```
target - pred ≈ log(total_count + pseudocount) - log(total_count)
             = log(1 + pseudocount / total_count)
```

This offset is large for low-count regions and shrinks for high counts.  For
zero-count regions, predictions can go very negative while targets sit at
`log(pseudocount)`.

Consequently `mse_log_counts` and `pearson_log_counts` metrics are systematically
biased for ASAP.  Pearson correlation is partially robust because the ordering of
examples is preserved by a monotone transform, but the MSE is not.

---

## `ProfileLogRates` Semantics: ASAP vs Gopher

`ProfileLogRates` is returned by both ASAP and Gopher models, but carries
**different semantics** depending on the training loss.

### ASAP — `ProfilePoissonNLLLoss`

```
exp(log_rates) ≈ counts          (Poisson rates, raw)
logsumexp(log_rates) ≈ log(total_count)
```

Pseudocount is **not** embedded in the rates.

### Gopher — `CoupledMSEMultinomialLoss`

The count loss is:

```
MSE( logsumexp(log_rates),  log(total_count + pseudocount) )
```

The model is therefore trained to produce rates where:

```
logsumexp(log_rates) ≈ log(total_count + pseudocount)
```

Pseudocount **is** baked into the rates by training.

### Consequence for shared metrics and `compute_total_log_counts`

The metrics `LogCountsMeanSquaredError` and `LogCountsPearsonCorrCoef` and the
function `compute_total_log_counts` currently apply `logsumexp(log_rates)` for
`ProfileLogRates` inputs without any pseudocount adjustment.

| Model | `logsumexp(log_rates)` | target | metric pred space |
|---|---|---|---|
| Gopher / CoupledMSE | `log(count + p)` | `log(count + p)` | consistent ✓ |
| ASAP / Poisson | `log(count)` | `log(count + p)` | offset by `log(1 + p/count)` ✗ |

A naive fix — applying pseudocount to `logsumexp` for all `ProfileLogRates` —
would break Gopher:

```
log( exp(log(count + p)) + p ) = log(count + 2p)   ≠   log(count + p)
```

Any generic fix therefore requires distinguishing the two model types, either by
loss class (`isinstance` check) or by a flag stored on the output object.

---

## Why Pseudocount Cannot Be Baked into Poisson NLL Training

The Poisson NLL loss is:

```
L = exp(log_rate) - count * log_rate
```

The count enters as a raw integer observation, not log-transformed.  There is no
structural place for a pseudocount offset.  Adding an MSE count loss alongside
Poisson NLL:

```
L_total = L_poisson + λ * MSE( logsumexp(log_rates), log(count + p) )
```

creates a contradiction: Poisson drives `logsumexp → log(count)` while the count
MSE drives `logsumexp → log(count + p)`.  For any `p > 0` these are different
targets and the two terms impose opposing gradients.

**Conclusion:** Pseudocount-adjusted count representation is **fundamentally
incompatible** with `ProfilePoissonNLLLoss` as the primary loss.

---

## Option: Switch ASAP to `CoupledMSEMultinomialLoss`

`CoupledMSEMultinomialLoss` accepts `ProfileLogRates` (same architecture) and
trains:

- **Profile loss**: multinomial cross-entropy on `softmax(log_rates)` — shape only
- **Count loss**: `MSE(logsumexp(log_rates), log(count + pseudocount))`

After training, `logsumexp(log_rates) ≈ log(count + pseudocount)`.  All metrics
and the scatter plot would then be self-consistent without any special-casing.

### Tradeoffs

| Aspect | `ProfilePoissonNLLLoss` | `CoupledMSEMultinomialLoss` |
|---|---|---|
| Statistical basis | Poisson likelihood (principled for counts) | Heuristic MSE on log-counts |
| Overdispersion | Not modelled | Not modelled |
| Zero regions | Handled via Poisson likelihood | Handled via pseudocount floor |
| Count representation | `logsumexp = log(count)` | `logsumexp = log(count + p)` |
| Metrics / scatter plot | Mismatch (see above) | Self-consistent ✓ |
| Practical performance | Good for sparse ATAC | Good; used by Basenji / Enformer style models |

ATAC-seq data is overdispersed relative to Poisson in practice, so the Poisson
assumption is already approximate.

---

## Current Status

No code changes were made.  The following options remain open:

1. **Switch ASAP training to `CoupledMSEMultinomialLoss`** (recommended if
   scatter plot / metric consistency is the priority).  Pseudocount should be set
   to approximately `read_length / bin_size` (e.g. `25` for 100 bp fragments at
   4 bp bins).

2. **Fix only the scatter plot** (`_accumulate_log_counts` in `module.py`):
   detect `isinstance(criterion, ProfilePoissonNLLLoss)` and apply
   `log(total_rate + pseudocount)` for predictions only.  Metrics remain biased
   but the visual discontinuity is resolved.

3. **Accept the status quo** and document `mse_log_counts` / `pearson_log_counts`
   as unreliable for ASAP models when `count_pseudocount` is non-trivial.

---

**Date:** 2026-02-28
**Related:**
- `docs/internal/count_log_spaces.md` — two-space framework and `log_counts_include_pseudocount` flag
- `docs/internal/logsumexp_analysis.md` — multi-channel logsumexp correctness
