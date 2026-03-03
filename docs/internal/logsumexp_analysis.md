# Analysis: Potential logsumexp Optimizations in Cerberus

## Executive Summary

After careful analysis, **there are NO safe replacements** of `log1p(sum(expm1(x)))` with `logsumexp(x)` that maintain numerical equivalence. The mathematical operations are fundamentally different:

- `log1p(sum(expm1(x)))` computes: **log(Σxᵢ + 1)**
- `logsumexp(x)` computes: **log(Σexp(xᵢ))**

When x = log(xᵢ + 1):
- `log1p(sum(expm1(x)))` = **log(Σxᵢ + 1)** ✓ (what we want)
- `logsumexp(x)` = **log(Σ(xᵢ + 1))** = **log(Σxᵢ + N)** ✗ (different)

## Numerical Verification

```python
targets = log1p([1, 2, 3])  # [0.6931, 1.0986, 1.3863]

Method 1 (current): log1p(sum(expm1(targets)))
  = log1p(sum([1, 2, 3])) = log1p(6) = 1.9459 ✓

Method 2 (logsumexp): logsumexp(targets)
  = log(sum([2, 3, 4])) = log(9) = 2.1972 ✗

Difference: 0.2513 (12.9% error)
```

## Pattern Locations

### Loss Functions (7 instances)

All loss functions with `log1p_targets=True` follow this pattern:

**Pattern:**
```python
if self.log1p_targets:
    targets = torch.expm1(targets).clamp_min(0.0)  # log(x+1) → x

# Later (count loss):
target_counts = targets.sum(dim=...)  # Σx
target_log_counts = torch.log1p(target_counts)  # log(Σx + 1)
```

**Locations:**
1. `MSEMultinomialLoss.forward()` - lines 106, 119-120, 124-125
2. `CoupledMSEMultinomialLoss.forward()` - lines 147, 160-161, 168-169
3. `PoissonMultinomialLoss.forward()` - lines 233, 242-244, 247-249
4. `CoupledPoissonMultinomialLoss.forward()` - lines 271, 284-285, 290-291
5. `NegativeBinomialMultinomialLoss.forward()` - lines 319, 329, 331
6. `CoupledNegativeBinomialMultinomialLoss.forward()` - lines 360, 373, 378

### Metrics (2 instances)

**LogCountsMeanSquaredError** (lines 189-197):
```python
if self.log1p_targets:
    target = torch.expm1(target)

target_counts = target.sum(dim=2)
target_log_counts = torch.log1p(target_counts)
```

**LogCountsPearsonCorrCoef** - same pattern

## Why Predictions Use logsumexp

Interestingly, **predictions** already use `logsumexp`:

```python
# Coupled models compute log-counts from log-rates:
pred_log_counts = torch.logsumexp(logits, dim=2)  # log(Σexp(logits))
```

This creates an apparent mismatch:
- **Predictions:** `log(Σexp(logits))`
- **Targets:** `log(Σcounts + 1)`

However, this is **intentional**. The model learns `logits` such that:
```
log(Σexp(logits)) ≈ log(Σcounts + 1)
Therefore: Σexp(logits) ≈ Σcounts + 1
```

The model implicitly learns to predict `(total_counts + 1)` rather than `total_counts`. This is a valid parameterization choice that:
1. Avoids log(0) issues
2. Provides better gradient behavior near zero
3. Matches the `log1p` transform used for targets

## Alternative Formulations (Would Change Semantics)

If we wanted to use `logsumexp` for targets, we'd need to:

### Option 1: Change target storage format
```python
# Instead of: targets = log1p(counts) = log(counts + 1)
# Use:        targets = log(counts)  # NOT log1p!

# Then:
target_log_counts = torch.logsumexp(targets, dim=...)  # log(Σcounts)
```

**Impact:** Breaking change - all existing models/data incompatible

### Option 2: Accept the approximation
```python
# Use logsumexp but accept it's different:
target_log_counts = torch.logsumexp(targets, dim=...)  # log(Σcounts + N)

# For large counts, log(Σcounts + N) ≈ log(Σcounts + 1)
# Error: log((Σcounts + N)/(Σcounts + 1)) ≈ log(1 + (N-1)/Σcounts)
```

**Impact:** Changes loss formulation, affects training

### Option 3: Compute correction term
```python
# Start with logsumexp
lse = torch.logsumexp(targets, dim=...)  # log(Σ(counts + 1))

# Subtract correction: log(Σ(counts + 1)) - log(Σcounts + 1)
# But this requires knowing Σcounts... which needs expm1 anyway!
```

**Impact:** Doesn't actually avoid expm1

## Recommendations

### For Current Codebase: NO CHANGES RECOMMENDED

The current implementation is **correct and optimal** given the design constraint:
- Targets stored as `log1p(counts)` for numerical stability
- Need to compute `log(Σcounts + 1)` for count loss
- Must use `expm1 → sum → log1p` sequence
- Adding `.clamp_min(0.0)` (already done) is the right approach

### For Future Consideration: Log-Space Arithmetic

If designing from scratch, consider:
1. Store targets as `log(counts)` instead of `log1p(counts)`
2. Use `logsumexp` throughout for count computations
3. Handle zero-counts with masking or smoothing

**Trade-offs:**
- ✅ Faster (no expm1)
- ✅ More numerically stable (stay in log-space)
- ✅ Simpler code
- ✗ Breaking change
- ✗ Requires special handling for zero-counts

## Performance Impact

Current overhead from `expm1 → sum → log1p`:
- **Time:** ~3 operations instead of 1 (logsumexp)
- **Memory:** Temporary tensor for expm1 result
- **Numerical precision:** One exp/log round-trip

**Estimated impact:** <1% of total training time
- Most time spent in convolutions and matrix ops
- Count loss computed once per batch
- Profile loss dominates computation

**Conclusion:** Performance impact is **negligible**, not worth breaking compatibility.

## Code Quality Note

The current code is actually **well-designed**:
- Uses `log1p` for numerical stability near zero
- Consistently applies the same transform
- Properly clamps after `expm1` to prevent negative values
- Documents the log1p_targets behavior

The pattern is **intentional, not a bug**.

---

**Analysis Date:** February 17, 2026
**Analyst:** Claude Sonnet 4.5
**Verification:** Numerical tests confirm 12.9% difference between methods
