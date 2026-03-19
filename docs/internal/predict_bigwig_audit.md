# predict_bigwig.py — Error Analysis

Audit of `src/cerberus/predict_bigwig.py` for correctness, resource safety, and scalability.

**File:** `src/cerberus/predict_bigwig.py`
**Date:** 2026-03-18

---

## Critical

### 1. Unbounded island growth causes OOM on whole chromosomes

**Location:** `predict_to_bigwig` stream_generator, lines 106-132

The island break condition:

```python
if current_island and (window.start >= prev_input_start + output_len):
```

only fires when there is a gap of at least `output_len` between consecutive window starts. With the default stride of `output_len // 2`, consecutive windows are separated by `stride < output_len`, so the condition is **never true** for contiguous genomic regions.

For a blacklist-free chromosome (e.g. human chr1, ~249 Mbp), ALL windows are accumulated into `current_island` before `_process_island` is called. With `stride=512`, that is ~486K windows. Inside `_process_island`, the function then:

1. Builds a `linear_signals` list of ~486K tensors, each `(C, output_len)`.
2. Allocates a numpy accumulator of shape `(C, ~249M / bin_size)`.

For `C=2`, `output_len=1024`, `bin_size=1`: the signals list alone is ~486K * 2 * 1024 * 4 bytes = ~4 GB. The accumulator is ~2 GB. This will OOM on most hardware.

**Fix:** Chunk islands by a maximum window count or bp span. When the chunk limit is reached, flush to `_process_island` and yield results before continuing. Alternatively, stream predictions directly to the accumulator without buffering all signals.

---

## Medium

### 2. Missing pseudocount inversion in `_reconstruct_linear_signal`

**Location:** `_reconstruct_linear_signal`, line 161

```python
total_counts = torch.exp(log_counts).unsqueeze(-1)  # (C, 1)
```

When `log_counts` is in `log(count + pseudocount)` space (MSE loss with `count_per_channel=True`), `exp(log_counts) = count + pseudocount`, not `count`. The reconstructed signal is systematically overestimated by `pseudocount` per channel.

**Impact by count magnitude:**

| True count | Pseudocount=1 | Overestimate |
|------------|---------------|--------------|
| 1000       | 1001          | 0.1%         |
| 100        | 101           | 1%           |
| 10         | 11            | 10%          |
| 1          | 2             | 100%         |

The same problem is correctly handled in `output.py:compute_total_log_counts` via its `log_counts_include_pseudocount` parameter. The fix requires threading the pseudocount flag and value through `predict_to_bigwig` -> `_process_island` -> `_reconstruct_linear_signal`.

**Note:** For Poisson/NB losses (`log_counts_include_pseudocount=False`), `log_counts` is in natural log space and `exp(log_counts)` is correct. The bug only manifests with MSE-style losses that include a pseudocount offset.

---

### 3. BigWig file handle never closed

**Location:** `predict_to_bigwig`, lines 135-136

```python
bw = pybigtools.open(str(output_path), "w")
bw.write(genome_config["chrom_sizes"], stream_generator())
```

The `bw` handle is never closed. If `stream_generator()` raises an exception mid-write, the BigWig file may be truncated or corrupt with no cleanup. If `pybigtools` relies on `__del__` for flushing, normal completion may also leave data unflushed in edge cases (e.g., process killed by OOM).

**Fix:** Use a context manager or explicit try/finally with `bw.close()`.

---

## Low / Cleanup

### 4. `aggregation` parameter is dead code

**Location:** `predict_to_bigwig` signature (line 26) and `_process_island` (line 208)

The `aggregation` parameter is accepted by `predict_to_bigwig`, passed to `_process_island`, but never used. Line 208 hardcodes `aggregation="model"`:

```python
for batched_output, batch_intervals in model_ensemble.predict_intervals_batched(
    island_intervals,
    dataset,
    use_folds=use_folds,
    aggregation="model",   # <-- hardcoded, ignores parameter
    batch_size=batch_size,
):
```

This is intentional for correctness — per-window reconstruction via `_reconstruct_linear_signal` requires unbatched outputs, which only `"model"` aggregation provides. But the parameter on `predict_to_bigwig` misleads callers into thinking they can choose a different mode.

**Fix:** Remove the `aggregation` parameter from `predict_to_bigwig` and `_process_island`. Document that `"model"` aggregation is required because softmax normalization is only valid within a single window, not across merged intervals.

### 5. Inverse transform division order unverified

**Location:** `_process_island`, line 236

```python
track_data = track_data / target_scale / output_bin_size
```

This assumes training targets were constructed as `raw_per_bp_signal * output_bin_size * target_scale` (bin-sum first, then scale). If the transform pipeline applies scaling before binning, the inversion order would be wrong. The result would differ by a factor of `output_bin_size` when `target_scale != 1`.

**Status:** Needs verification against the actual transform chain in `cerberus.transforms`. For `output_bin_size=1` (most models), the division is a no-op and the order doesn't matter.

### 6. Dead `while track_data.ndim > 2` loop

**Location:** `_process_island`, lines 239-240

```python
while track_data.ndim > 2:
    track_data = track_data[0]
```

`aggregate_tensor_track_values` always returns a 2D array `(C, n_bins)`. This loop can never execute. Should be removed to avoid confusion about possible return shapes.

### 7. Unused `numpy` import

**Location:** line 2

```python
import numpy as np
```

Never referenced in the module.

### 8. Region tiling loop is over-engineered

**Location:** `predict_to_bigwig` stream_generator, lines 73-83

```python
pos = region.start - offset
while pos + input_len <= region.end + offset + input_len:
    ...
    pos += stride
    if win_end >= region.end + offset:
        break
```

The while condition simplifies to `while pos <= region.end + offset`, which is always true until the inner break fires. The two termination conditions are redundant. The coverage is technically correct — the break ensures the last window's output reaches `region.end` — but the logic is harder to verify than necessary.

**Fix:** Replace with a single clear loop condition, e.g.:

```python
pos = region.start - offset
while True:
    win_start = max(0, pos)
    win_end = win_start + input_len
    if win_end > chrom_sizes[region.chrom]:
        break
    windows.append(Interval(region.chrom, win_start, win_end, "+"))
    if win_end - offset >= region.end:
        break
    pos += stride
```

---

## Summary

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | **Critical** | Unbounded island growth → OOM on whole chromosomes | **Fixed** — streaming accumulator replaces tensor list |
| 2 | **Medium** | Missing pseudocount inversion in signal reconstruction | **Fixed** — `count_pseudocount` parameter added |
| 3 | **Medium** | BigWig file handle never closed | **Fixed** — try/finally with bw.close() |
| 4 | Low | `aggregation` parameter is dead code | **Fixed** — removed from API |
| 5 | Low | Inverse transform division order unverified | **Verified** correct — Scale then Bin in transforms |
| 6 | Cleanup | Dead `while ndim > 2` loop | **Fixed** — removed with its test |
| 7 | Cleanup | Unused `numpy` import | **Fixed** — now used by streaming accumulator |
| 8 | Cleanup | Region tiling loop is over-engineered | Open |
| 8 | Cleanup | Region tiling loop over-engineered | Open |
