# Pseudocount Audit — v0.9.5 (March 2026)

Exhaustive analysis and fix record for pseudocount handling across training,
metrics, inference, and tooling.  Supersedes the scattered notes in
`count_log_spaces.md`, `asap_pseudocount_considerations.md`, and
`logsumexp_analysis.md` where those documents overlap with this one.

---

## 1. Architecture Summary

### The two log-spaces

| Space | Flag | Losses | Transform | Inverse |
|---|---|---|---|---|
| Offset-log | `uses_count_pseudocount = True` | MSEMultinomialLoss, CoupledMSEMultinomialLoss, BPNetLoss, DalmatianLoss | `log(count + p)` | `exp(x) - p` |
| Pure-log | `uses_count_pseudocount = False` | PoissonMultinomialLoss, CoupledPoissonMultinomialLoss, NegBinMultinomialLoss, CoupledNegBinMultinomialLoss, ProfilePoissonNLLLoss | `log(count)` | `exp(x)` |

### Canonical dispatch

```python
# config.py:809 — the ONE correct way to detect the space
log_counts_include_pseudocount, count_pseudocount = get_log_count_params(model_config)
```

Reads `loss_cls.uses_count_pseudocount` (class attribute), returns the scaled
pseudocount from `loss_args`.  Works for all loss families including
`DalmatianLoss`.

### Pseudocount lifecycle

```
CLI --count-pseudocount (raw units, e.g. 150 for ChIP read length)
  → data_config["count_pseudocount"]
  → propagate_pseudocount() multiplies by target_scale
  → loss_args["count_pseudocount"] (scaled units)
  → loss_args["log_counts_include_pseudocount"] (bool, also injected)
  → metrics_args gets both via setdefault
```

`propagate_pseudocount` is called once from `instantiate()` in `module.py`.
It returns a new dict (immutable — safe for multi-fold reuse).

---

## 2. Training-Time Correctness

All loss computations verified correct:

**MSEMultinomialLoss** (`loss.py:157-164`):
```
count_loss = MSE(pred_log_counts, log(target_counts + pseudocount))
```
Global and per-channel paths both correct.  Model learns offset-log space.

**CoupledMSEMultinomialLoss** (`loss.py:195-205`):
```
pred_log_counts = logsumexp(log_rates)      # log(sum(exp(rates)))
target_log_counts = log(counts + pseudocount)
count_loss = MSE(pred, target)
```
At convergence `sum(exp(rates)) ≈ count + pseudocount` — pseudocount is baked
into the rates by training.  This is intentional.

**PoissonMultinomialLoss** (`loss.py:279-285`):
```
count_loss = PoissonNLLLoss(log_input=True)(pred_log_counts, target_counts)
```
Standard Poisson NLL.  No pseudocount.  `count_pseudocount` accepted in
constructor for config compatibility but never stored as instance attribute.

**NegativeBinomialMultinomialLoss** (`loss.py:360-369`):
```
nb_logits = pred_log_counts - log(r)
NB(total_count=r, logits=nb_logits)
```
Parameterization: `logits = log(mu/r)`, mean = mu.  No pseudocount.

**DalmatianLoss** (`loss.py:436-508`):
Wraps a base loss (typically MSE).  `count_pseudocount` forwarded via
`args.setdefault(...)` into `self.base_loss`.  Not stored on self.

---

## 3. Inference-Time Correctness

### `compute_total_log_counts` (`output.py:266-302`)

Multi-channel `ProfileCountOutput` with offset-log:
```python
# Invert per-channel, sum in linear space, reapply once
total = (exp(log_counts) - pseudocount).clamp_min(0.0).sum(dim=1)
return log(total + pseudocount)     # log(c0 + c1 + p), NOT log(c0 + c1 + 2p)
```

Multi-channel `ProfileCountOutput` with pure-log:
```python
return logsumexp(log_counts, dim=1)  # log(c0 + c1)
```

`ProfileLogRates` (both spaces): always `logsumexp(log_rates.flatten)`.
Ignores `log_counts_include_pseudocount`.  For CoupledMSE the result IS in
offset-log space (pseudocount baked into rates); for Poisson it IS in pure-log
space.  Works for symmetric comparisons (pred vs obs) but cannot distinguish
the spaces — see Known Limitation 1.

### `_reconstruct_linear_signal` (`predict_bigwig.py:149-187`)

`ProfileCountOutput`: `softmax(logits) * (exp(log_counts) - pseudocount)` — correct.

`ProfileLogRates`: `exp(log_rates)` — for CoupledMSE, sums to `count + pseudocount`
(small distributed background).  See Known Limitation 2.

### `compute_obs_log_counts` (`output.py:305-331`)

Offset-log: `log(obs_total + pseudocount)`.  Pure-log: `log(obs_total.clamp_min(1.0))`.

### Prediction utilities

`predict_log_counts` (predict_misc.py), `export_predictions.py`, `export_bigwig.py`
all use `get_log_count_params` for dispatch.  Verified correct.

---

## 4. Metrics Correctness

### Profile reconstruction metrics — correct

`CountProfilePearsonCorrCoef` and `CountProfileMeanSquaredError` both do:
```python
total_counts = (exp(log_counts) - self.count_pseudocount).clamp_min(0.0)
preds_counts = softmax(logits) * total_counts
```

### Log-count metrics — fixed in v0.9.5

`LogCountsMeanSquaredError` and `LogCountsPearsonCorrCoef` aggregate
multi-channel `ProfileCountOutput` predictions to a global count.

**Before (bug):** used plain `logsumexp` regardless of space:
```
logsumexp([log(c0+p), log(c1+p)]) = log(c0 + c1 + 2p)    ← wrong
target = log(c0 + c1 + p)                                  ← one p
```

**After (v0.9.5):** with `log_counts_include_pseudocount=True`, inverts
per-channel, sums, reapplies — matching `compute_total_log_counts`:
```python
total = (exp(log_counts) - pseudocount).clamp_min(0.0).sum(dim=1)
log(total + pseudocount)                                    ← correct
```

The `log_counts_include_pseudocount` flag is injected by `propagate_pseudocount`
into `metrics_args` and threaded through `DefaultMetricCollection`,
`BPNetMetricCollection`, and `PomeranianMetricCollection`.

---

## 5. Validation Scatter Plot — fixed in v0.9.5

The scatter plot (`save_count_scatter`) shows predicted vs observed log-counts
at the end of each validation epoch.

### Before (two bugs)

`_accumulate_log_counts` in `module.py` used:
```python
pseudocount = getattr(self.criterion, "count_pseudocount", 1.0)
log_counts_include_pseudocount = hasattr(self.criterion, "count_pseudocount")
```

**Bug A — Poisson/NB losses:** `count_pseudocount` is NOT stored as an instance
attribute on these losses.  `hasattr` returned False (correct flag), but
`getattr` fell through to default 1.0.  Target was `log(count + 1.0)` while
prediction was in `log(count)` space — systematic offset for low counts.

**Bug B — DalmatianLoss:** `count_pseudocount` is forwarded to `self.base_loss`,
not stored on self.  `hasattr` returned False (WRONG — Dalmatian uses offset-log)
and pseudocount defaulted to 1.0 (WRONG — actual scaled value lives in base_loss).

### After (v0.9.5)

Removed `_accumulate_log_counts` entirely.  The scatter plot now reads directly
from `LogCountsPearsonCorrCoef`'s accumulated `preds_list` / `targets_list`.
This metric already receives the correct `log_counts_include_pseudocount` and
`count_pseudocount` via `propagate_pseudocount`, so the values are guaranteed
consistent with the loss function.

---

## 6. Tool Parameter Defaults

| Tool | `--count-pseudocount` | Typical loss | Rationale |
|---|---|---|---|
| `train_pomeranian.py` | 150.0 | BPNetLoss (MSE) | ChIP-seq ~150bp read at 1bp bins |
| `train_bpnet.py` | 150.0 | BPNetLoss (MSE) | Same |
| `train_dalmatian.py` | 1.0 | DalmatianLoss(MSE) | ATAC pseudobulk, log1p convention |
| `train_asap.py` | 1.0 | ProfilePoissonNLLLoss | No effect (Poisson); affects metrics only |
| `train_biasnet.py` | 1.0 | MSEMultinomialLoss | log1p convention |
| `train_gopher.py` | 1.0 | CoupledMSEMultinomialLoss | log1p, baked into rates |

The pseudocount should approximate the total count contribution of one read:
`pseudocount ≈ read_length / bin_size` (in raw coverage units).

For ASAP (Poisson NLL), the pseudocount has no effect on the loss but
influences the `LogCounts*` validation metrics.  See
`asap_pseudocount_considerations.md` for the detailed tradeoff analysis.

---

## 7. Multi-Channel Global-Count Reconstruction — fixed in v0.9.5

When `predict_total_count=True` and `n_output_channels > 1`, the count head
outputs `(B, 1)` — a single global total.  The `CountProfile*` metrics and
`_reconstruct_linear_signal` multiply each channel's softmax probabilities by
this total.  Since each channel's softmax sums to 1 over length, the
reconstructed signal sums to `C * total` instead of `total`.

**Fix:** When `log_counts` has 1 output but there are C > 1 channels, divide
`total_counts` by C before per-channel multiplication.  Applied in:

- `CountProfilePearsonCorrCoef.update()` (metrics.py) — Pearson is scale-
  invariant so the fix doesn't change the metric value, but keeps the
  intermediate reconstruction correct.
- `CountProfileMeanSquaredError.update()` (metrics.py) — MSE is scale-sensitive;
  without the fix, MSE is inflated and never reaches 0 for perfect predictions.
- `_reconstruct_linear_signal()` (predict_bigwig.py) — BigWig signal would be
  C× too large.

The fix is a no-op when `log_counts` is per-channel `(B, C)` or when C == 1
(the common case for all current training tools).

---

## 8. Known Limitations (not bugs, documented intentionally)


### Limitation 1: `compute_total_log_counts` ignores flags for ProfileLogRates

The `ProfileLogRates` branch always returns `logsumexp(log_rates)` regardless
of `log_counts_include_pseudocount`.  For CoupledMSE, the result IS in offset-log
space (pseudocount baked into rates); for Poisson it IS in pure-log space.

This works for symmetric comparisons (export_predictions.py applies the same
transform to both predicted and observed).  It would be wrong if a caller needed
to extract an absolute count from a CoupledMSE model's ProfileLogRates output.

### Limitation 2: CoupledMSE BigWig background offset

`_reconstruct_linear_signal` for `ProfileLogRates` returns `exp(log_rates)`.
For CoupledMSE-trained models, `sum(exp(log_rates)) ≈ count + pseudocount`,
so the exported BigWig signal integrates to `count + pseudocount` per window.
The pseudocount distributes as ~`p / n_bins` per bin.  For p=1, n_bins=1024:
~0.001 per bin — negligible.  For p=150: ~0.15 per bin — visible in silent
regions but small relative to active signal.

### Limitation 3: CoupledMSE per-channel + global aggregation

When CoupledMSE trains with `count_per_channel=True`, each channel learns
`sum_l exp(logits[c,l]) ≈ count_c + p`.  Global aggregation via logsumexp
gives `log(total + C*p)` instead of `log(total + p)`.  Offset of `(C-1)*p`.
Only matters for the unusual combination of per-channel CoupledMSE with global
inference aggregation.  Not triggered by any current training tool.

---

## 9. Deprecated Patterns

### `hasattr(criterion, "count_pseudocount")`

Used in the old `_accumulate_log_counts` and documented in earlier versions
of `count_log_spaces.md`.  Unreliable because DalmatianLoss and Poisson/NB
losses do not store `count_pseudocount` as an instance attribute.

**Use instead:** `get_log_count_params(model_config)` or
`loss_cls.uses_count_pseudocount`.

### `_accumulate_log_counts` on CerberusModule

Removed in v0.9.5.  Duplicated work already done by `LogCountsPearsonCorrCoef`.
The scatter plot now reads from the metric's state directly.

---

## 10. Key File Reference

| File | Key lines | Purpose |
|---|---|---|
| `config.py` | `get_log_count_params` | Canonical pseudocount dispatch |
| `config.py` | `propagate_pseudocount` | Scales and injects pseudocount into loss/metrics args |
| `loss.py` | `uses_count_pseudocount` on each class | Declares which space the loss uses |
| `output.py` | `compute_total_log_counts` | Multi-channel aggregation with correct inversion |
| `output.py` | `compute_obs_log_counts` | Observed-count log transform |
| `metrics.py` | `LogCountsMeanSquaredError`, `LogCountsPearsonCorrCoef` | Metrics with `log_counts_include_pseudocount` flag |
| `module.py` | `on_validation_epoch_end` | Scatter plot reads from metric state |
| `predict_bigwig.py` | `_reconstruct_linear_signal` | Pseudocount inversion for BigWig export |
| `predict_misc.py` | `predict_log_counts` | High-level inference with auto-detection |

---

## 11. Related Documents

- `count_log_spaces.md` — defines the two-space framework, naming conventions,
  and pitfalls.  Updated in v0.9.5 to deprecate the `hasattr` dispatch pattern.
- `asap_pseudocount_considerations.md` — detailed analysis of the Poisson metric
  mismatch for ASAP models and options for resolution.  No code changes made.
- `logsumexp_analysis.md` — historical analysis of `log1p`/`expm1` vs `logsumexp`
  for count targets.  Predates `count_pseudocount`; line references are stale.

---

**Date:** March 20, 2026
**Version:** 0.9.5
