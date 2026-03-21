# Implementation Plan: `complexity_center_size` for Complexity Matching

## Problem

When `padded_size` is large (e.g., 32kb+ for multi-scale context models like
Saluki), complexity metrics (GC, DUST, CpG) are computed over the full interval.
At 32kb+, GC content regresses toward the genome mean (~0.41) and all intervals
look identical, making complexity matching ineffective. Background negatives
become essentially random rather than matched to the local sequence composition
of the peaks.

## Solution

Add a `complexity_center_size` parameter that crops intervals to their center
N bp before computing complexity metrics. This decouples the model's input
window size from the complexity matching scale. Default behavior: use
`padded_size` (identical to current behavior).

## Files to Modify

1. `src/cerberus/complexity.py` — add `center_size` parameter
2. `src/cerberus/samplers.py` — thread parameter through 3 classes
3. `tests/test_complexity.py` — add tests for center cropping
4. `tests/test_samplers.py` — add tests for parameter threading

## Detailed Changes

### 1. `src/cerberus/complexity.py`

**Function: `compute_intervals_complexity` (line ~160)**

Current signature:
```python
def compute_intervals_complexity(
    intervals: Iterable[Interval],
    fasta_path: Path | str,
    metrics: list[str] | None = None
) -> np.ndarray:
```

New signature:
```python
def compute_intervals_complexity(
    intervals: Iterable[Interval],
    fasta_path: Path | str,
    metrics: list[str] | None = None,
    center_size: int | None = None,
) -> np.ndarray:
```

**Change in the `for interval in intervals:` loop (line ~188):**

Before the FASTA lookup (`seq_obj = fasta[interval.chrom][interval.start : interval.end]`),
add center cropping:

```python
start = interval.start
end = interval.end

if center_size is not None and (end - start) > center_size:
    mid = (start + end) // 2
    start = mid - center_size // 2
    end = start + center_size

seq_obj = fasta[interval.chrom][start:end]
```

No other changes to this file.

### 2. `src/cerberus/samplers.py`

Three classes need changes. All changes are additive (new optional parameter
with backward-compatible default).

#### 2a. `ComplexityMatchedSampler.__init__` (line ~975)

Add `center_size` parameter:

```python
def __init__(
    self,
    target_sampler: Sampler,
    candidate_sampler: Sampler,
    fasta_path: Path | str,
    chrom_sizes: dict[str, int],
    folds: list[dict[str, InterLap]] | None = None,
    exclude_intervals: dict[str, InterLap] | None = None,
    bins: int = 20,
    candidate_ratio: float = 1.0,
    metrics: list[str] | None = None,
    seed: int = 42,
    generate_on_init: bool = True,
    metrics_cache: dict[str, np.ndarray] | None = None,
    center_size: int | None = None,              # ADD THIS
):
```

Store it:
```python
self.center_size = center_size
```

#### 2b. `ComplexityMatchedSampler._get_metrics` (line ~1033)

Change the `compute_intervals_complexity` call (line ~1044) to pass
`center_size`:

```python
# Current:
new_metrics = compute_intervals_complexity(missing, self.fasta_path, self.metrics)

# New:
new_metrics = compute_intervals_complexity(
    missing, self.fasta_path, self.metrics, center_size=self.center_size
)
```

**IMPORTANT — cache key issue:** The metrics cache uses `str(iv)` as the key
(line 1038-1039, 1046). If the same interval is used with different
`center_size` values across runs, the cached metrics will be wrong. Two options:

- **Option A (simple):** Include `center_size` in the cache key:
  `key = f"{iv}:cs={self.center_size}"`. This invalidates old caches when
  `center_size` changes.
- **Option B (no change):** Accept that the cache is per-session and per
  `center_size`. Since `center_size` is set once per sampler and the cache
  is per-`CerberusDataModule`, this is safe as long as the same model config
  is used consistently.

**Recommended: Option A.** It's one line and prevents subtle bugs.

Change in `_get_metrics`:
```python
# Current:
key = str(iv)

# New:
key = f"{iv}:cs={self.center_size}" if self.center_size is not None else str(iv)
```

This preserves backward compatibility — existing caches (without `center_size`)
still work because `center_size=None` uses the old key format.

#### 2c. `PeakSampler.__init__` (line ~1212)

Add `complexity_center_size` parameter:

```python
def __init__(
    self,
    intervals_path: Path | str,
    fasta_path: Path | str,
    chrom_sizes: dict[str, int],
    padded_size: int,
    folds: list[dict[str, InterLap]] | None = None,
    exclude_intervals: dict[str, InterLap] | None = None,
    background_ratio: float = 1.0,
    min_candidates: int = 10000,
    candidate_oversample_factor: float = 5.0,
    seed: int = 42,
    prepare_cache: dict[str, np.ndarray] | None = None,
    complexity_center_size: int | None = None,   # ADD THIS
):
```

Pass it to `ComplexityMatchedSampler` (line ~1292):

```python
# Current:
self.negatives = ComplexityMatchedSampler(
    target_sampler=self.positives,
    candidate_sampler=self.candidates,
    fasta_path=fasta_path,
    ...
)

# New — add center_size kwarg:
self.negatives = ComplexityMatchedSampler(
    target_sampler=self.positives,
    candidate_sampler=self.candidates,
    fasta_path=fasta_path,
    ...,
    center_size=complexity_center_size,
)
```

#### 2d. `NegativePeakSampler.__init__` (line ~1320+)

Same change as PeakSampler — add `complexity_center_size` parameter and pass
to its `ComplexityMatchedSampler`. Check the exact class definition, it follows
the same pattern.

#### 2e. `create_sampler` factory function

Find where `PeakSampler` and `NegativePeakSampler` are instantiated in
`create_sampler`. The `sampler_args` dict is unpacked as `**sampler_args` into
the constructor. Since `complexity_center_size` is an optional kwarg with
default `None`, no changes are needed in `create_sampler` — the key passes
through automatically via `**sampler_args`.

**Verify this:** Search for `PeakSampler(` in `create_sampler` and confirm it
uses `**config["sampler_args"]` or similar unpacking. If it constructs args
explicitly (picking specific keys), then `complexity_center_size` must be added
to the explicit arg list.

### 3. Tests

#### 3a. `tests/test_complexity.py`

Add two tests:

```python
def test_compute_intervals_complexity_center_size():
    """center_size crops intervals before computing metrics."""
    interval = Interval("chr1", 10000, 42000, "+")  # 32kb interval

    # Full interval metrics
    full = compute_intervals_complexity([interval], fasta_path)

    # Center 2624bp metrics
    center = compute_intervals_complexity(
        [interval], fasta_path, center_size=2624
    )

    # They should differ (different GC at different scales)
    assert full.shape == center.shape == (1, 3)
    assert not np.allclose(full, center)


def test_compute_intervals_complexity_center_size_noop_when_smaller():
    """center_size larger than interval is a no-op."""
    interval = Interval("chr1", 10000, 12000, "+")  # 2kb interval

    full = compute_intervals_complexity([interval], fasta_path)
    center = compute_intervals_complexity(
        [interval], fasta_path, center_size=32000
    )

    np.testing.assert_array_equal(full, center)
```

#### 3b. `tests/test_samplers.py` or new `tests/test_complexity_center_size.py`

Add an integration test:

```python
def test_peak_sampler_complexity_center_size_changes_negatives():
    """Different complexity_center_size produces different negative sets."""
    # Create two PeakSamplers with same peaks but different center_size
    sampler_full = PeakSampler(
        intervals_path=peaks_path,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        padded_size=33024,
        seed=42,
    )
    sampler_center = PeakSampler(
        intervals_path=peaks_path,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        padded_size=33024,
        complexity_center_size=2624,
        seed=42,
    )

    # The negative intervals should differ
    negs_full = [iv for iv in sampler_full.negatives]
    negs_center = [iv for iv in sampler_center.negatives]
    assert negs_full != negs_center
```

### 4. Type checking

Run after all changes:
```bash
npx pyright src/cerberus/complexity.py src/cerberus/samplers.py
```

### 5. Usage

In training scripts for multi-scale models:

```python
sampler_config: SamplerConfig = {
    "sampler_type": "peak",
    "padded_size": context_len + 2 * max_jitter,  # e.g. 33024 for 32kb
    "sampler_args": {
        "intervals_path": args.peaks,
        "background_ratio": args.background_ratio,
        "complexity_center_size": 2624,  # match 2kb Pomeranian scale
    },
}
```

For standard Pomeranian training (2kb input), no changes needed — omitting
`complexity_center_size` defaults to `None` which uses the full interval.

## Summary of Changes

| File | Lines changed | Nature |
|------|--------------|--------|
| `src/cerberus/complexity.py` | ~6 added | center crop before FASTA lookup |
| `src/cerberus/samplers.py` (ComplexityMatchedSampler) | ~4 added | store and pass `center_size` |
| `src/cerberus/samplers.py` (PeakSampler) | ~2 added | accept and forward `complexity_center_size` |
| `src/cerberus/samplers.py` (NegativePeakSampler) | ~2 added | same as PeakSampler |
| `tests/test_complexity.py` | ~20 added | unit tests for center cropping |
| `tests/test_samplers.py` | ~15 added | integration test |
| Total | ~49 lines | fully backward compatible |
