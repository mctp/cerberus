# Code Audit: Bugs, Mathematical Errors, and Edge Cases

Exhaustive review of `src/cerberus/` performed 2026-03-22.
Only logical errors, mathematical errors, and dangerous edge cases are listed — not code smells, style, or missing features.

---

## Bugs

### 1. Jitter mutates cached sampler intervals in-place — Jitter augmentation silently dies out during training

**File:** `transform.py:104-106`, `dataset.py:359`

```python
# Jitter.__call__
interval.start = interval.start + start
interval.end = interval.start + self.input_len
```

`Interval` is a mutable dataclass (`frozen` is not set). When `__getitem__` is called, `ListSampler` returns a *reference* to the stored interval object. Jitter then mutates `.start` and `.end` in-place. On subsequent access to the same index (by the same worker), the interval is already length `input_len` with slack=0, and **Jitter becomes a no-op**.

**Practical impact analysis:**

All training scripts and configs use `reload_dataloaders_every_n_epochs=0`, meaning the DataLoader (and its workers) persist for the entire training run.

- **`num_workers > 0` (standard production):** Each worker forks the process and gets copy-on-write copies of `sampler._intervals`. Worker mutations are isolated from each other and from the main process. However, within a single persistent worker, re-accessing the same index in a later epoch (inevitable over ~50 epochs: probability ≈ 1−(1−1/W)^K per index) yields an already-cropped interval where Jitter has no slack remaining. Over a full training run, Jitter augmentation progressively diminishes as more indices are "frozen" in each worker's memory.

- **`num_workers=0` (debugging, CI tests):** All mutations happen in the main process. After epoch 1, **every interval** is permanently jittered to its first-epoch random position. Jitter is completely dead for all subsequent epochs.

- **Multi-GPU DDP:** Each rank creates independent DataModule/Dataset/workers. The bug affects each rank identically with no cross-rank interference. Same diminishing-Jitter pattern on each rank.

**Why training doesn't crash and results appear reasonable:** The first epoch always works correctly (all intervals are fresh). The frozen positions still cover the correct genomic regions — they're just fixed rather than randomly shifted. The model effectively trains on the standard center-cropped positions after epoch 1 (as if `max_jitter=0`). This silently degrades augmentation diversity but doesn't produce wrong predictions.

**`ReverseComplement` has the same mutation pattern** (`transform.py:182`: `interval.strand = ...`) but strand only has two values (+/−), so re-mutation just flips it again — no progressive drift.

**Test safety of a fix:** No existing tests call `dataset[idx]` on the same index twice. All interval-tracking tests call `ds[0]` exactly once. The equivalence test (`test_get_interval_equivalence_to_getitem`) uses `padded_size == input_len` (slack=0, Jitter is a no-op), so no actual mutation occurs — it passes with or without a fix. A `copy.copy()` in `__getitem__` before calling `_get_interval` would fix the bug with zero test breakage.

**Fix:**
```python
# dataset.py __getitem__, before calling _get_interval:
import copy
interval = copy.copy(self.sampler[idx])
```
or make `Interval` a frozen dataclass and have Jitter/ReverseComplement return new instances via `dataclasses.replace()`.

---

### 2. PeakSampler / NegativePeakSampler: off-by-one in peak exclusion zones

**File:** `samplers.py:1330, 1447`

```python
neg_excludes[interval.chrom].add((interval.start, interval.end))
```

InterLap stores **closed** intervals `[start, end]`. Cerberus intervals are **half-open** `[start, end)`. The correct closed form is `(interval.start, interval.end - 1)`. As written, the exclusion zone extends one base beyond each peak on the right side (effectively `[start, end]` inclusive = `[start, end+1)` half-open).

**Impact:** Background candidate intervals that begin exactly at a peak's exclusive end position are incorrectly excluded. One extra base-pair of exclusion per peak. Minor for large genomes, but systematically biases the background pool.

---

### 3. Metrics reconstruction wrong when `predict_total_count=True` with multiple output channels

**File:** `metrics.py:119-126` (`CountProfilePearsonCorrCoef`), `metrics.py:167-177` (`CountProfileMeanSquaredError`)

```python
probs = F.softmax(logits, dim=-1)          # (B, C, L) — each channel sums to 1
total_counts = torch.exp(log_counts.float()) - self.count_pseudocount  # (B, 1) when total
preds_counts = probs * total_counts.unsqueeze(-1)  # (B, C, L)
```

When `predict_total_count=True`, `log_counts` is `(B, 1)` and represents the log of the **global** count (sum across all channels). `softmax(logits, dim=-1)` produces per-channel distributions that each sum to 1. Multiplying by the global count means each channel receives the full global count. The reconstructed total `= sum_C(probs * total)` = `C * total`, overcounting by a factor of C (number of output channels).

**Impact:** Profile Pearson and MSE metrics are computed on signals that are C× too large. Metric values are biased/incorrect when `n_output_channels > 1` and `predict_total_count=True`.

---

### 4. `predict_bigwig` inversion assumes sum-pooling but default `Bin` is max-pooling

**File:** `predict_bigwig.py:281`, `transform.py:297`

```python
# predict_bigwig.py
track_data = track_data / target_scale / output_bin_size

# transform.py Bin default
def __init__(self, bin_size: int, method: str = "max", ...):
```

The BigWig reconstruction divides by `output_bin_size` to recover per-bp signal. This inversion is only valid for **sum** pooling (where binned_value = sum of per-bp values). The default `Bin` method is `"max"`, for which dividing by bin size produces incorrect values (it would turn max-pooled values into values that are `1/bin_size` of the true peak).

**Impact:** When `output_bin_size > 1` and the default max-pooling is used, BigWig predictions are systematically wrong by a factor of `1/output_bin_size`.

---

## Mathematical Errors

### 5. `PoissonMultinomialLoss` profile loss uses raw counts instead of probabilities

**File:** `loss.py:319-333`

```python
def _compute_profile_loss(self, logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    loss_shape = -torch.sum(targets * log_probs, dim=-1).mean()
```

The docstring calls this "Cross Entropy (Multinomial NLL form)" but targets are raw counts, not probabilities. The true cross-entropy would normalize targets: `target_probs = targets / targets.sum(dim=-1, keepdim=True)`. As written, the loss computes `-sum(count_i * log(p_i))` which equals `N * H(target_probs, pred_probs)` where N is total count. This means the profile loss scales **linearly with peak depth**.

**Impact:** Deeper peaks contribute proportionally more to the profile gradient. In contrast, `MSEMultinomialLoss` uses the full multinomial NLL with `lgamma` terms and averages over channels, partially mitigating this effect. The scaling behavior is different between the two loss classes despite similar names. This may be intentional but contradicts the docstring claim of equivalence with a standard cross-entropy.

### 6. Multinomial NLL with non-integer targets

**File:** `loss.py:153-154, 165-166`

```python
log_fact_sum = torch.lgamma(profile_counts + 1)
log_prod_fact = torch.sum(torch.lgamma(targets + 1), dim=-1)
```

The multinomial distribution requires integer counts. `lgamma(x+1)` is the continuous extension of `log(x!)` but for non-integer `x` it does not correspond to any valid combinatorial quantity. Targets can be non-integer when:
- BigWig signals have fractional coverage values
- `target_scale != 1.0` is applied before the loss sees the data
- Binning with `"avg"` method is used

**Impact:** The `log_fact_sum` and `log_prod_fact` terms become meaningless constants that still affect the loss magnitude and gradients. The profile loss gradient (`-target * (1 - softmax)`) is still correct, but the count loss gradient receives incorrect offsets from the lgamma terms. For typical scaled-integer data this is negligible; for continuously-valued targets the error can be significant.

---

## Edge Cases and Robustness Issues

### 7. `dataset.get_raw_targets` crop assumes interval width equals `input_len`

**File:** `dataset.py:252-257`

```python
crop_start = (self.data_config.input_len - self.data_config.output_len) // 2
crop_end = crop_start + self.data_config.output_len
if raw_target.shape[-1] > self.data_config.output_len:
    raw_target = raw_target[..., crop_start:crop_end]
```

`crop_start` is computed relative to `input_len`, but the extracted signal spans the full query interval which may be `padded_size` (wider than `input_len`) or an arbitrary user-provided interval. If the interval is wider than `input_len`, the crop extracts the wrong sub-region.

**Impact:** When called with intervals of size `!= input_len`, the returned "output-length" region is misaligned with the model's actual output region.

### 8. `count_per_channel=True` with `predict_total_count=True` produces shape mismatch

**File:** `loss.py:190-193`

```python
if self.count_per_channel:
    target_counts = targets.sum(dim=2)              # (B, C)
    target_log_counts = torch.log(target_counts + self.count_pseudocount)  # (B, C)
    count_loss = F.mse_loss(pred_log_counts, target_log_counts)  # (B,1) vs (B,C)
```

When `predict_total_count=True`, `pred_log_counts` is `(B, 1)`. With `count_per_channel=True`, `target_log_counts` is `(B, C)`. PyTorch broadcasting expands `(B, 1)` to `(B, C)`, computing MSE between the single global prediction and each per-channel target independently. The gradient pushes the total-count prediction toward the **mean** of per-channel log-counts, not the log of their **sum**.

**Impact:** Silently trains the count head on the wrong objective. The combination `count_per_channel=True + predict_total_count=True` is a configuration error that produces no exception but yields mathematically incorrect training.

### 9. `DUST` score normalization includes ambiguous-base k-mers

**File:** `complexity.py:69-90`

```python
lookup = np.full(256, 4, dtype=np.int8)  # N and all non-ACGT → index 4
# ... k-mers are computed over a 5-symbol alphabet
# but normalization uses 4-base expectation:
exp_random = max((seq_len - k + 1) / (2 * 4**k), 1e-9)
```

K-mers containing ambiguous bases (N) are mapped to index 4 and counted as valid k-mers in the `bincount`. The observed score thus includes N-containing k-mers, but the expected score is computed for a 4-letter alphabet. For N-rich regions (common near telomeres, centromeres, and assembly gaps), the ratio `observed/expected` is inflated because: (a) repeated NNN k-mers increase the observed score, and (b) the expected count is computed as if N doesn't exist.

**Impact:** Complexity-matched sampling over-estimates repetitiveness for N-rich regions, biasing the background interval selection.

### 10. `aggregate_tensor_track_values` drops trailing bins from non-divisible spans

**File:** `output.py:196`

```python
n_bins = span_bp // output_bin_size
```

If `span_bp` (= `max_end - min_start`) is not evenly divisible by `output_bin_size`, the integer division silently drops `span_bp % output_bin_size` bases from the right end of the merged interval. The `merged_interval.end` reported in the output still reflects the full span, but the data array is shorter.

**Impact:** The returned numpy array has fewer bins than the merged interval implies. Downstream consumers that compute `expected_bins = (interval.end - interval.start) // output_bin_size` will get a matching count, but any consumer using `interval.end` directly for coordinate calculations will be off.

### 11. `compute_obs_log_counts` clamps zero-count regions to `log(1) = 0`

**File:** `output.py:401`

```python
return torch.log(obs_total.clamp_min(1.0))
```

For Poisson/NB losses (`log_counts_include_pseudocount=False`), silent regions with zero total counts are clamped to 1.0 before taking log, yielding 0.0. The model's prediction for these regions may be very negative (e.g., −10 for near-zero predicted rates). The floor at `log(1)=0` creates a discontinuity in the target distribution and biases evaluation: the MSE/Pearson metrics see a cluster of target values at exactly 0 that doesn't reflect the true data.

**Impact:** Evaluation metrics (log-count MSE and Pearson) are biased for datasets with many silent regions.

### 12. `predict_bigwig` region-based window generation with negative start

**File:** `predict_bigwig.py:86-94`

```python
pos = region.start - offset
while pos + input_len <= region.end + offset + input_len:
    win_start = max(0, pos)
    win_end = win_start + input_len
```

When `region.start < offset` (region near chromosome start), `pos` starts negative. The `max(0, pos)` clamp shifts the window right, but the output region (computed as `win_start + offset` to `win_start + offset + output_len`) no longer aligns with the intended region start. The first window's output may extend beyond the region boundary.

**Impact:** Predictions near chromosome starts for region-based queries may have misaligned coordinates. Genome-wide mode (SlidingWindowSampler) is not affected because it starts from position 0.

### 13. `merge_intervals` discards strand information

**File:** `interval.py:225`

```python
merged.append(Interval(current_chrom, current_start, current_end))
```

Merged intervals always receive the default strand `"+"`. If input intervals have strand `"-"`, the strand is silently discarded.

**Impact:** Any downstream processing that relies on strand after `merge_intervals` will treat all regions as forward-strand.

### 14. `Interval.center()` can produce out-of-bounds coordinates

**File:** `interval.py:42-56`

```python
offset = (current_len - width) // 2
new_start = self.start + offset
new_end = new_start + width
```

No bounds checking against chromosome sizes. When `width > current_len`, `offset` is negative, potentially producing `new_start < 0`. Even with `width <= current_len`, if the interval is near chromosome boundaries, the centered interval may extend beyond chromosome limits.

**Impact:** Out-of-bounds intervals cause silent truncation or errors in downstream extractors (pyfaidx, pybigtools).

### 15. `CoupledMSEMultinomialLoss` count-profile coupling constraint

**File:** `loss.py:240-253`

In the coupled losses, the same logits tensor is used for both profile shape (via softmax, which is shift-invariant) and absolute count (via logsumexp, which is NOT shift-invariant). This creates a mathematical tension: the profile loss is satisfied by any shifted version of the logits, but the count loss constrains the absolute scale. The model must learn to satisfy both simultaneously through a single logit tensor.

**Impact:** Not a bug per se, but the coupling means count loss gradients modify the logit scale, which can slow profile convergence. The decoupled variants (separate log_counts head) avoid this tension.

### 16. `_worker_init_fn` truncates torch seed to 32 bits

**File:** `datamodule.py:114`

```python
worker_seed = torch.initial_seed() % 2**32
```

`torch.initial_seed()` returns a 64-bit value. The modulo truncation means workers with seeds differing only in the upper 32 bits get identical numpy/random states.

**Impact:** In practice this is unlikely with PyTorch's seeding strategy, but it reduces the effective seed space for numpy operations in workers.

### 17. `ScaledSampler.split_folds` may produce zero-size splits

**File:** `samplers.py:956-958`

```python
train_size = int(len(train) * ratio)
val_size = int(len(val) * ratio)
test_size = int(len(test) * ratio)
```

When `ratio < 1` and a split is small, `int(len(split) * ratio)` truncates to 0. A `ScaledSampler` with `num_samples=0` produces an empty sampler, which is handled gracefully downstream but may surprise users expecting at least 1 sample per split.

**Impact:** Silent empty validation/test sets when using heavy subsampling with small folds.
