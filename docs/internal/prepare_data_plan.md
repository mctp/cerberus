# Plan: Implement `prepare_data()` in CerberusDataModule

## Problem

`CerberusDataModule.setup()` runs on **every DDP rank**. The expensive initialization inside it — particularly `ComplexityMatchedSampler` computing sequence complexity metrics for hundreds of thousands of intervals — is duplicated across all ranks. With 4 GPUs, the same FASTA reads and metric computations happen 4 times independently.

Lightning provides `prepare_data()` specifically for this: it runs **once on rank 0** before `setup()` is called on any rank. Heavy preprocessing runs once, serializes results to disk, and all ranks load from disk in `setup()`.

---

## Current Init Cost Profile

| Component | Typical Cost | Runs Per Rank | Notes |
|---|---|---|---|
| `compute_intervals_complexity()` | **Minutes** (100k–600k FASTA reads + GC/DUST/CpG per interval) | Yes | Dominates total init time |
| `ComplexityMatchedSampler._initialize()` | Same as above + binning | Yes | Wraps `compute_intervals_complexity` |
| `PeakSampler.__init__()` | Same (creates `ComplexityMatchedSampler` internally) | Yes | Most expensive sampler type |
| `InMemorySequenceExtractor._load()` | **Minutes**, ~12 GB for hg38 | Yes | Only when `in_memory=True` |
| `InMemorySignalExtractor.__init__()` | **Minutes** per channel, ~12 GB each | Yes | Only when `in_memory=True` |
| `create_genome_folds()` | Microseconds | Yes | Not worth caching |
| `get_exclude_intervals()` | < 1 sec | Yes | Not worth caching |
| `IntervalSampler._load()` | Seconds | Yes | Cheap BED parsing |
| Transform creation | Negligible | Yes | Not worth caching |

The **high-ROI targets** are `compute_intervals_complexity()` and the `InMemory*` extractors. Everything else is fast enough to duplicate.

---

## Proposed Design

### Cache Directory

```
~/.cache/cerberus/<config_hash>/
    metrics_cache.npz        # interval → complexity metrics
    candidate_intervals.npy  # pre-generated random candidate intervals
    ready                    # sentinel file indicating completion
```

The `<config_hash>` is derived from the deterministic inputs: FASTA path + its mtime, sampler config, seed, chrom sizes. If any input changes, a new cache directory is used.

### `prepare_data()` — Rank 0 Only

```python
def prepare_data(self):
    cache_dir = self._resolve_cache_dir()
    if (cache_dir / "ready").exists():
        return  # Already precomputed

    # 1. Build exclude intervals + folds (cheap, needed for sampler creation)
    # 2. Create the full sampler (expensive — complexity metrics computed here)
    # 3. Serialize metrics_cache dict to cache_dir/metrics_cache.npz
    # 4. Serialize candidate interval lists to cache_dir/candidate_intervals.npy
    # 5. Write sentinel file
```

### `setup()` — Every Rank

```python
def setup(self, stage=None, ...):
    cache_dir = self._resolve_cache_dir()

    # Load precomputed metrics cache from disk
    metrics_cache = self._load_metrics_cache(cache_dir)

    # Pass to dataset/sampler creation — ComplexityMatchedSampler
    # finds all intervals already cached, skips FASTA reads entirely
    full_dataset = CerberusDataset(
        ...,
        metrics_cache=metrics_cache,
    )
```

### Threading Through the Cache

The `metrics_cache` dict needs to flow from `prepare_data()` through to `ComplexityMatchedSampler`. The path is:

```
CerberusDataModule.setup()
  → CerberusDataset.__init__(metrics_cache=...)
    → create_sampler(metrics_cache=...)
      → ComplexityMatchedSampler(metrics_cache=loaded_cache)
        → _get_metrics() finds all keys present, computes nothing
```

This requires adding a `metrics_cache` parameter to:
- `CerberusDataset.__init__()`
- `create_sampler()`
- `ComplexityMatchedSampler.__init__()` — already has this parameter

---

## What Gets Cached

### Phase 1: Complexity Metrics (highest ROI)

The `metrics_cache` dict maps `str(interval)` → `np.ndarray` of shape `(M,)` where M is the number of metrics (typically 3: GC, DUST, CpG). This is already the internal cache format used by `ComplexityMatchedSampler._get_metrics()`.

**Serialization format:** `np.savez_compressed()` with keys as interval strings and values as metric arrays. Alternatively, two parallel arrays: one for interval keys, one for the metric matrix.

**Cache key inputs:**
- FASTA path + mtime (metrics depend on sequence content)
- Metric names list (e.g., `["gc", "dust", "cpg"]`)
- Sampler config (determines which intervals exist)
- Seed (determines random candidate intervals)

**Expected file size:** For 600k intervals × 3 metrics × 4 bytes = ~7 MB uncompressed. Negligible.

**Expected speedup:** Eliminates 100k–600k FASTA reads and per-interval metric computations per additional rank. For a 4-GPU setup with 200k target + 400k candidate intervals, saves ~3× the init time (ranks 1–3 skip all computation).

### Phase 2: In-Memory Extractor Data (optional, for `in_memory=True`)

When `in_memory=True`, the `InMemorySequenceExtractor` and `InMemorySignalExtractor` each read the entire genome from FASTA/BigWig files. These could be pre-written as memory-mapped numpy files:

```
~/.cache/cerberus/<config_hash>/
    sequence/chr1.npy    # (4, chrom_size) uint8 one-hot
    sequence/chr2.npy
    ...
    signal/H3K4me3/chr1.npy   # (chrom_size,) float32
    signal/H3K4me3/chr2.npy
    ...
```

Each rank would `np.load(..., mmap_mode='r')` instead of re-reading from FASTA/BigWig. This also reduces memory: memory-mapped files are shared at the OS level across processes.

**Expected file size:** ~12 GB per genome for sequence, ~3 GB per BigWig channel. Large but saved on fast local storage.

**Expected speedup:** Eliminates genome-wide FASTA/BigWig reads on ranks 1–N. Also reduces total memory from N copies to 1 shared mmap.

**Complication:** The current code calls `tensor.share_memory_()` for DataLoader worker sharing. Memory-mapped numpy arrays are already implicitly shared across forked workers (copy-on-write), but converting to torch tensors copies the data. Would need `torch.from_numpy()` on mmap'd arrays, which shares memory without copy. The `share_memory_()` call would become unnecessary.

This phase is more invasive and can wait. Phase 1 alone addresses the most common pain point.

---

## Implementation Steps

### Step 1: Add `_resolve_cache_dir()` to `CerberusDataModule`

Computes a deterministic cache directory path from config inputs:

```python
def _resolve_cache_dir(self) -> Path:
    import hashlib, json
    key_data = json.dumps({
        "fasta_path": str(self.genome_config["fasta_path"]),
        "fasta_mtime": os.path.getmtime(self.genome_config["fasta_path"]),
        "sampler_config": self.sampler_config,
        "seed": self.seed,
        "chrom_sizes": self.genome_config["chrom_sizes"],
    }, sort_keys=True)
    h = hashlib.sha256(key_data.encode()).hexdigest()[:16]
    return Path("~/.cache/cerberus").expanduser() / h
```

### Step 2: Implement `prepare_data()` in `CerberusDataModule`

- Check for sentinel file → skip if exists
- Create full sampler (triggers `ComplexityMatchedSampler._initialize()`)
- Extract and serialize `metrics_cache` to `.npz`
- Write sentinel

### Step 3: Thread `metrics_cache` through `setup()` → `CerberusDataset` → `create_sampler()`

- Add `metrics_cache: dict | None = None` parameter to `CerberusDataset.__init__()`
- Pass through to `create_sampler()`
- `create_sampler()` passes to `ComplexityMatchedSampler` (already accepts it)
- In `setup()`, load `.npz` and pass to dataset constructor

### Step 4: Handle the No-Cache Fallback

When `prepare_data()` hasn't run (e.g., standalone `CerberusDataset` usage without `CerberusDataModule`), the code path must remain unchanged — `ComplexityMatchedSampler` computes metrics on the fly as today. The `metrics_cache=None` default preserves this.

### Step 5: Handle Sampler Types That Don't Need Caching

Only `ComplexityMatchedSampler` (and by extension `PeakSampler` which wraps it) benefits from the metrics cache. `IntervalSampler`, `RandomSampler`, `SlidingWindowSampler` are fast and don't need caching. The `prepare_data()` step can check `sampler_config["sampler_type"]` and skip if not `"peak"` or `"complexity_matched"`.

---

## Files Modified

| File | Change |
|---|---|
| `src/cerberus/datamodule.py` | Add `prepare_data()`, `_resolve_cache_dir()`, cache loading in `setup()` |
| `src/cerberus/dataset.py` | Add `metrics_cache` parameter to `__init__()`, pass to `create_sampler()` |
| `src/cerberus/samplers.py` | Add `metrics_cache` parameter to `create_sampler()`, pass to `ComplexityMatchedSampler` |
| `tests/test_prepare_data.py` | New test file |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Stale cache (FASTA changed but path didn't) | Include mtime in cache key |
| Cache directory fills up | Document cleanup; consider TTL or LRU |
| Non-Lightning usage (standalone dataset) | Fallback: `metrics_cache=None` → compute on the fly (current behavior) |
| `prepare_data()` not called in notebook/script usage | Cache is opportunistic, not required; existing code path unaffected |
| Seed mismatch between `prepare_data()` and `setup()` | Both use `self.seed` from the same config; deterministic |

---

## Verification

1. Existing tests pass unchanged (`pytest -v tests/`)
2. New tests verify:
   - Cache is created by `prepare_data()` and loaded by `setup()`
   - `ComplexityMatchedSampler` skips computation when cache is populated
   - Fallback works when no cache exists
   - Cache key changes when config changes
3. Benchmark: measure `setup()` wall time with and without cache on a realistic config
