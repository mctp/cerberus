# Adaptive Counts Loss Weight for BPNet-Style Models

## Problem Statement

`BPNetLoss` (and `MSEMultinomialLoss` generally) combines two loss terms with a fixed
scalar weight `alpha`:

```
L_total = beta * L_profile + alpha * L_count
```

The default `alpha=1.0, beta=1.0` is uninformed and almost certainly wrong for any real
dataset. The two terms operate on fundamentally different scales that are determined by
the depth of the ChIP-seq/ATAC-seq signal. Without a data-derived `alpha`, training is
either dominated by the profile term (model ignores count magnitude) or the count term
(model ignores profile shape), depending on dataset depth and model initialization.

---

## Mathematical Analysis

### Profile Loss Scale

`BPNetLoss` uses the **full multinomial NLL** (including factorial terms), averaged over
batch and channels (`average_channels=True`):

```
L_profile_c = -log P(y_c | pi_c)
            = -log(N_c!) + sum_j log(y_j!) - sum_j y_j * log(pi_j)
```

where `N_c = sum_j y_j` is the total count for channel `c`, `y_j` is the raw target
signal at position `j`, and `pi_j = softmax(logits)_j`.

The dominant term is the cross-entropy: `-sum_j y_j * log(pi_j) = N_c * H(y_c/N_c, pi_c)`.

At initialization with uniform logits (`pi_j = 1/L`):

```
H_cross = log(L)    (e.g. log(1000) ≈ 6.9 nats for L=1000)
```

The factorial terms partially cancel the cross-entropy term. Empirically, for sparse
ChIP-seq profiles (`N_c ~ 500`, `L = 1000`), the full NLL at initialization is
O(hundreds to low thousands) of nats.

**Key dependence:** `L_profile ~ O(N̄ * log L)` where `N̄` is mean per-channel count.

### Count Loss Scale

The count loss is MSE in log-count space:

```
L_count = MSE(log_counts_pred, log1p(N_total))
        = (log(1 + N_pred) - log(1 + N_true))^2
```

At bad initialization (`N_pred ≈ 0`):

```
L_count ~ (log(1 + N_total))^2    (e.g. (log 501)^2 ≈ 38 for N=500)
```

**Key dependence:** `L_count ~ O((log N_total)^2)` at initialization.

### Imbalance

The ratio `L_profile / L_count` scales with the data depth. As `N` grows:

- `L_profile` grows linearly in `N`
- `L_count` grows as `(log N)^2`, far more slowly

| Mean peak depth `N̄` | `L_profile` (approx) | `L_count` (bad init) | Ratio |
|---------------------:|---------------------:|---------------------:|------:|
| 50                   | ~200                 | ~16                  | ~12×  |
| 200                  | ~600                 | ~28                  | ~21×  |
| 500                  | ~850                 | ~38                  | ~22×  |
| 2000                 | ~2600                | ~54                  | ~48×  |

With `alpha=1, beta=1` the profile term increasingly dominates as depth grows. The count
head receives proportionally weaker gradient signal, causing it to underfit and produce
poor total count predictions. Setting `alpha` from data restores the balance.

---

## The Adaptive Alpha Formula

The chrombpnet-pytorch codebase derives `alpha` as:

```
alpha = median_total_counts / scale     (scale=10 by default)
```

This is a practical heuristic: `alpha` grows linearly with data depth, counteracting the
linear growth of `L_profile`. The divisor `scale=10` was empirically validated in
chrombpnet and is the recommended starting point.

The bpnet-refactor codebase uses the BPNet supplementary formula:

```
lambda = alpha_factor * n_obs
```

where `n_obs` is the average total count per peak. For `alpha_factor=1`, this gives
`lambda = n_obs`, which is ~10× larger than the chrombpnet formula for the same dataset.
Both are data-adaptive; the chrombpnet formula is more conservative.

**Recommended formula for cerberus:**

```python
alpha = median_total_counts / 10.0
```

### Why Linear Scaling, Not sqrt or log

The right functional form for `alpha` follows from the goal of balancing the two loss
terms: `alpha ~ L_profile / L_count`. From the scaling analysis:

```
alpha_ideal ~ N * log(L) / (log N)^2
```

Over practical depth ranges (N = 100–2000), the growth of `(log N)^2` is weak enough
that `alpha_ideal` is close to linear in N. Comparing the growth of candidate formulas
as N increases 20× from 100 to 2000:

| Formula        | Growth (100 → 2000) | Notes |
|----------------|--------------------:|-------|
| `N / c`        | **20×**             | Matches growth of `L_profile`. Recommended. |
| `N / (log N)²` | ~7×                 | Theoretically precise but close to linear in practice. |
| `sqrt(N) / c`  | 4.5×                | Under-compensates at high depth. |
| `log(N) / c`   | 1.65×               | Severely under-compensates at high depth. |

`sqrt` and `log` both fail to track the linear growth of `L_profile`: at high depths the
profile term dominates increasingly, defeating the purpose of adaptive weighting.
`N / (log N)²` is theoretically the most precise but indistinguishable from linear over
practical ranges — the empirical constant `scale=10` absorbs the difference. Linear
scaling (`N / scale`) is therefore the right choice.

---

## When to Apply

Apply adaptive alpha to any loss class that combines multinomial profile NLL with a
scalar count loss:

| Loss class | Apply adaptive alpha? | Notes |
|---|---|---|
| `BPNetLoss` | **Yes** | Primary target. Uses full multinomial NLL. |
| `MSEMultinomialLoss` | **Yes** | Same loss, more general interface. |
| `CoupledMSEMultinomialLoss` | **Yes** | Mathematically equivalent; same scaling. |
| `PoissonMultinomialLoss` | **Yes (different scale)** | Count loss is Poisson NLL, not MSE. Scale analysis differs but data-derived `count_weight` is still appropriate. |
| `NegativeBinomialMultinomialLoss` | **Yes (different scale)** | Count loss is NB NLL. Same principle applies. |

Do **not** hard-code `alpha=1.0` in any training script targeting BPNet-style models.

---

## Complicating Factor 1: `target_scale`

`DataConfig["target_scale"]` is a multiplicative scaling factor applied to all target
signals by the `Scale` transform during data loading. `get_raw_targets()` bypasses all
transforms and returns unscaled signal.

If `target_scale = s`, the training targets are `s * raw_targets`. The total counts seen
by the loss are therefore `s * N_raw`. The count loss target becomes:

```
log1p(s * N_raw)  ≈  log(s) + log(N_raw)    for large N_raw
```

This shifts the count loss target upward by `log(s)`, increasing the MSE at initialization
if the model still predicts counts at the `N_raw` scale. More importantly, the relevant
depth for computing `alpha` is the **scaled** depth `s * N_raw`, not the raw depth.

**Implementation:** `compute_median_counts()` reads raw targets and multiplies by
`target_scale` before returning:

```python
def compute_median_counts(self, n_samples: int = 2000) -> float:
    dataset = self.train_dataset
    n = len(dataset)
    indices = random.sample(range(n), min(n_samples, n))
    counts = []
    for i in indices:
        raw = dataset.get_raw_targets(dataset.sampler[i])   # (C, L), bypasses transforms
        counts.append(float(raw.sum()))
    raw_median = float(np.median(counts))
    target_scale = self.data_config["target_scale"]
    scaled_median = raw_median * target_scale
    logger.info(
        f"Computed median_counts={scaled_median:.1f} "
        f"(raw={raw_median:.1f} × target_scale={target_scale}) "
        f"from {len(indices)} training intervals."
    )
    return scaled_median
```

The returned value represents what the loss actually sees, so callers use it directly:

```python
alpha = compute_counts_loss_weight(datamodule.compute_median_counts())
```

---

## Complicating Factor 2: `implicit_log_targets`

`implicit_log_targets=True` signals that the dataset stores targets in `log1p` space
and the loss should recover raw counts via `expm1` before computing the loss:

```python
# In MSEMultinomialLoss.forward():
if self.implicit_log_targets:
    targets = torch.expm1(targets).clamp_min(0.0)   # recover raw counts
```

After `expm1`, the loss sees the same raw counts as when `implicit_log_targets=False`.
The count loss target is still `log1p(raw_counts * target_scale)`. Therefore:

**`implicit_log_targets` does not change the alpha formula.** The same
`compute_median_counts()` (which reads raw signal and applies `target_scale`) gives the
correct statistics in both cases.

The data flow in both configurations:

```
implicit_log_targets=False:
  dataset → raw_counts × target_scale → loss (L_count on log1p(scaled_raw))

implicit_log_targets=True:
  dataset → log1p(raw_counts × target_scale) → expm1() in loss → raw_counts × target_scale
          → loss (L_count on log1p(scaled_raw))
```

Both paths produce the same count loss. `compute_median_counts()` correctly captures
the effective depth in both cases.

---

## Implementation

### `compute_counts_loss_weight` in `src/cerberus/models/bpnet.py`

```python
def compute_counts_loss_weight(median_counts: float, scale: float = 10.0) -> float:
    """
    Compute the count loss weight (alpha) from training data statistics.

    Implements the formula from chrombpnet-pytorch:
        alpha = median_total_counts / scale    (default scale=10)

    This counteracts the linear growth of the full multinomial NLL profile loss
    with peak depth, which would otherwise cause the profile term to dominate
    as dataset depth increases.

    Args:
        median_counts: Median total signal counts per peak from the training fold,
            scaled by target_scale. Obtain from CerberusDataModule.compute_median_counts().
        scale: Divisor. Default 10 matches chrombpnet-pytorch. Use 5 for stronger
            count supervision, 20 for stronger profile supervision.

    Returns:
        alpha value to pass as loss_args["alpha"].
    """
    if median_counts <= 0:
        raise ValueError(f"median_counts must be positive, got {median_counts}")
    return median_counts / scale
```

### `compute_median_counts` in `src/cerberus/datamodule.py`

See full implementation in Complicating Factor 1 above.

### Usage in training scripts

After `datamodule.setup(...)` and before constructing `model_config`:

```python
from cerberus.models.bpnet import compute_counts_loss_weight

if args.alpha is None:
    median_counts = datamodule.compute_median_counts()
    alpha = compute_counts_loss_weight(median_counts)
    logging.info(f"Auto alpha={alpha:.4f} (median_counts={median_counts:.1f} / 10)")
else:
    alpha = args.alpha
    logging.info(f"Manual alpha={alpha:.4f}")

loss_args = {"alpha": alpha}    # always explicit, never implicit
```

The `--alpha` argument default should be changed from `1.0` to `None` in all BPNet
example and tool scripts:

```python
parser.add_argument(
    "--alpha", type=float, default=None,
    help="Count loss weight. If not set, computed from training data as "
         "median_counts/10 (recommended)."
)
```

---

## Summary

| Factor | Effect on alpha | How `compute_median_counts` handles it |
|---|---|---|
| Peak depth (`N̄`) | `alpha` scales linearly with `N̄` | Sampled directly from training intervals |
| `target_scale` | `alpha` scales linearly with `target_scale` | Multiplied into returned median |
| `implicit_log_targets` | No effect on alpha | Loss recovers raw counts; formula unchanged |
| `count_per_channel=True` | Per-channel counts used | `compute_median_counts` sums all channels; divide by `n_channels` if per-channel loss |

The only case where `compute_median_counts` is insufficient is when targets undergo
**non-linear transforms** beyond `target_scale` and `log1p`. In that case, the effective
count scale must be computed analytically or by sampling the transformed dataloader
directly.
