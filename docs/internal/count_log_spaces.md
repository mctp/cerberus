# Count Log-Spaces in Cerberus

## Overview

Cerberus uses two distinct log-space conventions for count quantities, depending
on the loss function.  Understanding which space a value lives in is critical for
correct multi-channel aggregation, metric computation, and count reconstruction.

This document defines the two spaces, explains how `target_scale` and
`count_pseudocount` interact, and proposes a naming convention for the
`log_counts_include_pseudocount` flag that currently distinguishes them.

---

## The Two Log-Spaces

### 1. Offset-log space (MSE losses)

**Transform:**

```
log_count = log(count + pseudocount)
```

**Inverse:**

```
count = exp(log_count) - pseudocount
```

Used by: `MSEMultinomialLoss`, `CoupledMSEMultinomialLoss` (and the legacy
`BPNetLoss` which inherits from `MSEMultinomialLoss`).

The MSE count loss trains the count head to predict `log(total + pseudocount)`.
When `pseudocount = 1.0` this is identical to `log1p` / `expm1`.

**Why an offset?**  The pseudocount prevents `log(0)` for silent regions and
encodes the minimum meaningful signal level.  It is analogous to a Bayesian
prior count.

### 2. Pure-log space (Poisson / Negative Binomial losses)

**Transform:**

```
log_count = log(count)          # i.e. log(mu), the Poisson mean
```

**Inverse:**

```
count = exp(log_count)
```

Used by: `PoissonMultinomialLoss`, `CoupledPoissonMultinomialLoss`,
`NegativeBinomialMultinomialLoss`, `CoupledNegativeBinomialMultinomialLoss`.

These losses feed `log_count` directly into `PoissonNLLLoss(log_input=True)` or
a Negative Binomial distribution, which interprets it as `log(expected_count)`.
There is no pseudocount offset.

---

## The `log_counts_include_pseudocount` Flag

### Current name and locations

The flag is called `log_counts_include_pseudocount` in:

| Location | Purpose |
|---|---|
| `output.compute_total_log_counts(log_counts_include_pseudocount=...)` | Multi-channel aggregation |
| `tools/export_predictions.py` (local variable) | Observed-count log transform |

The name `log_counts_include_pseudocount` is misleading because *both* spaces are "log" spaces.
What the flag actually means is: **"the log-count values include a pseudocount
offset"**.

### What it controls

When aggregating **multi-channel** `ProfileCountOutput.log_counts` into a
single total:

| `log_counts_include_pseudocount` | Aggregation | Correct for |
|---|---|---|
| `False` (default) | `logsumexp(log_counts)` = `log(sum(counts))` | Pure-log (Poisson/NB) |
| `True` | `log(sum(exp(lc) - p) + p)` | Offset-log (MSE) |

The naive `logsumexp` is wrong in offset-log space because:

```
logsumexp([log(c0+p), log(c1+p)]) = log((c0+p) + (c1+p)) = log(c0 + c1 + 2p)
```

but the correct total is:

```
log(c0 + c1 + p)       # one pseudocount, not N
```

For **single-channel** outputs the flag has no effect — `log_counts` is returned
as-is.

### How to detect the space

Every loss class defines a `uses_count_pseudocount` **class attribute** (True for
MSE/Dalmatian, False for Poisson/NB).  The canonical dispatch is:

```python
# config.py — preferred entry point
log_counts_include_pseudocount, count_pseudocount = get_log_count_params(model_config)
```

This reads `loss_cls.uses_count_pseudocount` and pulls the scaled pseudocount from
`loss_args`.  It works correctly for all loss families including `DalmatianLoss`
(which forwards `count_pseudocount` to its inner base loss).

> **Deprecated pattern:** Earlier code used `hasattr(criterion, "count_pseudocount")`
> on loss *instances* as a duck-typing signal.  This is unreliable because
> `DalmatianLoss` and Poisson/NB losses do not store `count_pseudocount` as an
> instance attribute.  Do not use `hasattr` for this purpose — use
> `get_log_count_params` or `loss_cls.uses_count_pseudocount` instead.

### Proposed rename

The name `log_counts_include_pseudocount` should eventually be renamed to something that conveys
"the log-counts include a pseudocount offset".  Candidates:

| Name | Pros | Cons |
|---|---|---|
| `log_counts_include_pseudocount` | Most explicit, self-documenting | Long |
| `offset_log_space` | Short, describes the space | Less obvious what "offset" is |
| `pseudocount_in_log_counts` | Clear about what is in there | Slightly verbose |
| `mse_log_space` | Ties to loss family | Fragile if new losses added |

**Recommendation:** `log_counts_include_pseudocount`.  It reads naturally at call
sites:

```python
compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=p)
```

When renaming, update:
- `output.compute_total_log_counts` parameter
- `tools/export_predictions.py` local variable
- `metrics.py` `LogCountsMeanSquaredError` / `LogCountsPearsonCorrCoef` parameter
- All test files referencing `log_counts_include_pseudocount`

---

## `count_pseudocount` and `target_scale`

### Two different jobs for the same field

`count_pseudocount` does two very different jobs depending on the loss:

| Phase / loss | Job | Right order of magnitude |
|---|---|---|
| Phase 1 absolute / single-task (`MSE*`, `Dalmatian*`, `Poisson*`, `NB*`) | Prevent `log(0)` for silent regions; embed the zero-reads cluster into the log-count distribution. | "One read's worth" of coverage — i.e. `read_length / bin_size` in raw units, adjusted for CPM and `target_scale`. |
| Phase 2 differential (`DifferentialCountLoss`) | Empirical-Bayes shrinkage prior on the log-fold change: pull `log2((c_b+pc)/(c_a+pc))` toward 0 for peaks in the low-count tail. | The chosen low quantile of per-condition training-region total counts (e.g. 10th percentile). |

The library exposes two helpers that compute the scaled value correctly for
each role:

```python
from cerberus import (
    resolve_reads_equivalent_pseudocount,  # Phase 1 / single-task
    resolve_quantile_pseudocount,          # Phase 2 differential
)
```

See the helper docstrings in `src/cerberus/pseudocount.py` for arguments.
Phase 1 / single-task training tools expose this via
`--pseudocount-reads`, `--read-length`, `--input-scale {raw,cpm}`, and
`--total-reads`. `train_multitask_differential_bpnet.py` uses the quantile
helper in Phase 2, controlled by `--phase2-pseudocount-quantile` (default
`0.10`) and `--phase2-pseudocount-samples`.

### User-facing config

In `DataConfig`:

```yaml
count_pseudocount: 1.0    # specified in raw coverage units (legacy path)
target_scale: 0.01        # multiplicative scaling applied to all targets
```

### Propagation

`parse_hparams_config` multiplies the two before injecting into loss/metrics:

```python
scaled_pseudocount = data_conf["count_pseudocount"] * data_conf["target_scale"]
model_conf["loss_args"]["count_pseudocount"] = scaled_pseudocount
model_conf["metrics_args"]["count_pseudocount"] = scaled_pseudocount
```

**Why?**  The dataset applies `target_scale` to raw counts before the model sees
them.  If raw counts are `[0, 1, 2, ...]` and `target_scale = 0.01`, the model
sees `[0.0, 0.01, 0.02, ...]`.  The pseudocount must be in the same units as
these scaled targets so that `log(scaled_count + scaled_pseudocount)` is
consistent.

### Example

```
raw_count          = 100
target_scale       = 0.01
count_pseudocount  = 1.0   (raw units)

scaled_count       = 100 * 0.01 = 1.0
scaled_pseudocount = 1.0 * 0.01 = 0.01

loss target        = log(1.0 + 0.01) = log(1.01)
```

Without scaling the pseudocount, the loss target would be `log(1.0 + 1.0) =
log(2.0)` — a very different value that would dominate small counts.

---

## Where Each Space Is Used

### Offset-log space (`log(count + pseudocount)`)

| Component | File | What it does |
|---|---|---|
| `MSEMultinomialLoss` count target | `loss.py` | `log(target_count + self.count_pseudocount)` |
| `CoupledMSEMultinomialLoss` count target | `loss.py` | Same |
| `LogCountsMeanSquaredError` target | `metrics.py` | `log(target_count + self.count_pseudocount)` |
| `LogCountsPearsonCorrCoef` target | `metrics.py` | Same |
| `compute_total_log_counts` (log_counts_include_pseudocount=True) | `output.py` | Inverts per-channel, sums, reapplies |
| `CountProfile*` metrics (reconstruction) | `metrics.py` | `exp(log_counts) - count_pseudocount` |
| `module.on_validation_epoch_end` scatter plot | `module.py` | Reads from `LogCountsPearsonCorrCoef` state |
| `export_predictions` observed counts | `export_predictions.py` | `log(obs + pseudocount)` |

### Pure-log space (`log(count)`)

| Component | File | What it does |
|---|---|---|
| `PoissonMultinomialLoss` | `loss.py` | `PoissonNLLLoss(log_input=True)` against raw counts |
| `CoupledPoissonMultinomialLoss` | `loss.py` | Same, counts derived via logsumexp |
| `NegativeBinomialMultinomialLoss` | `loss.py` | NB distribution with `log_counts - log(r)` |
| `CoupledNegativeBinomialMultinomialLoss` | `loss.py` | Same, counts derived via logsumexp |
| `compute_total_log_counts` (log_counts_include_pseudocount=False) | `output.py` | `logsumexp(log_counts)` |
| `export_predictions` observed counts | `export_predictions.py` | `log(obs.clamp_min(1))` |

### Neither space (per-position target transform)

The `log_counts_include_pseudocount_targets` flag is **unrelated** to `count_pseudocount`.  It
indicates that per-position target values are stored as `log1p(count)` (a
dataset-level transform).  All loss/metric classes that support it invert via
`torch.expm1(target).clamp_min(0.0)`.  This is always `expm1` regardless of
`count_pseudocount` because the dataset transform is always `log1p`.

---

## Common Pitfalls

1. **Confusing `log_counts_include_pseudocount_targets` with `log_counts_include_pseudocount`.**
   - `log_counts_include_pseudocount_targets`: per-position values stored as `log1p(count)`.
     Always inverted with `expm1`.
   - `log_counts_include_pseudocount`: count-head values include a pseudocount offset.
     Inverted with `exp(x) - pseudocount`.

2. **Using `expm1` to invert log-counts.**
   Only correct when `pseudocount = 1.0`.  The general inversion is
   `exp(x) - pseudocount`.

3. **Using `logsumexp` for multi-channel offset-log counts.**
   Gives `log(N * pseudocount + total)` instead of `log(pseudocount + total)`.
   Must invert per-channel, sum, then reapply.

4. **Forgetting to scale `count_pseudocount` by `target_scale`.**
   `parse_hparams_config` handles this automatically, but manual instantiation
   of losses/metrics must use the scaled value.

---

**Date:** February 28, 2026 (updated March 20, 2026)
**Related:**
- `docs/internal/pseudocount_audit_v095.md` — comprehensive audit and fix record for v0.9.5
- `docs/internal/asap_pseudocount_considerations.md` — Poisson metric mismatch analysis
- `docs/internal/logsumexp_analysis.md` (predecessor, partially outdated)
