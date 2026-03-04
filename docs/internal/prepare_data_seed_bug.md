# Bug: prepare_data() Cache Misses Due to Non-Deterministic Seeds

## Problem

`prepare_data()` computes complexity metrics and saves them to disk. `setup()` loads the cache and creates the dataset. But with `seed=None` (the default), the random candidate intervals differ between the two calls, causing near-total cache misses.

Production observation (pomeranian training, 4 GPUs):
- `prepare_data()` computes 528,212 entries and saves cache
- `setup()` loads cache (528,212 entries) but recomputes 440,144 of 440,210 candidate metrics
- Only 66 candidates found in cache (random overlap)
- All 88,042 target metrics hit cache (from deterministic BED file)

## Root Cause

The seed flows through this chain:

```
CerberusDataModule(seed=None)
  → CerberusDataset(seed=None)
    → create_sampler(seed=None)
      → PeakSampler(seed=None)
        → MultiSampler.__init__(seed=None)        # samplers.py:304
          self.rng = random.Random(None)           # seeds from system time
          self.resample(seed=None)
            → _update_seed(None)                   # samplers.py:228-234
              self.seed = self.rng.getrandbits(32)  # RANDOM VALUE from system time
              self.rng = random.Random(self.seed)
            → generate_sub_seeds(self.seed, 2)     # samplers.py:46-53
              [sub_seed_0, sub_seed_1]              # deterministic from self.seed
            → ComplexityMatchedSampler.resample(sub_seed_1)
              → _update_seed(sub_seed_1)
              → _initialize()                       # samplers.py:1040
                cand_seed = self.rng.getrandbits(32)
                self.candidate_sampler.resample(cand_seed)
                  → RandomSampler generates intervals  # deterministic from cand_seed
```

The critical step is `MultiSampler.__init__()` at [samplers.py:342](src/cerberus/samplers.py#L342):

```python
self.rng = random.Random(seed)  # When seed=None, seeds from system time
```

`random.Random(None)` uses `time.time_ns()` as entropy. Since `prepare_data()` and `setup()` create separate `CerberusDataset` instances at different wall-clock times, they get different RNG states, producing different `self.seed` values, different sub_seeds, and ultimately different random candidate intervals.

When `seed` is an explicit int (e.g., 42), `random.Random(42)` is deterministic and the chain produces identical intervals.

## Why Targets Hit Cache but Candidates Don't

- **Targets** come from `IntervalSampler` which loads from a BED file. The intervals are deterministic regardless of seed.
- **Candidates** come from `RandomSampler` which generates random genomic intervals. The intervals depend on the seed chain above.

## Why `generate_sub_seeds(None, n)` Returns `[None]*n`

At [samplers.py:46-53](src/cerberus/samplers.py#L46-L53):

```python
def generate_sub_seeds(seed, n):
    if seed is None:
        return [None] * n
    ...
```

When the MultiSampler's seed is None, sub_seeds are all None. Each child sampler then self-seeds from `random.Random(None)` — non-deterministic. However, this code path is bypassed because `_update_seed(None)` converts the None to a random int before `generate_sub_seeds` is called.

The actual flow: `_update_seed(None)` sets `self.seed = self.rng.getrandbits(32)` (a random int), then `generate_sub_seeds(self.seed, 2)` receives an int, not None. So the sub_seeds ARE deterministic from this random int — but the random int itself differs between prepare_data() and setup().

## `train_dataloader()` Depends on `self.seed` Being None

At [datamodule.py:260](src/cerberus/datamodule.py#L260):

```python
base_seed = self.seed if self.seed is not None else 0
seed = base_seed + (epoch * world_size) + rank
```

Tests in [test_datamodule_seeding.py:103-113](tests/test_datamodule_seeding.py#L103-L113) verify that when `seed=None`, `base_seed=0` and resampling uses `0 + epoch*world + rank`. Changing `self.seed` to always be an int would break this formula and its tests.

## Fix

`CerberusDataModule.seed` is now `int` (default 42, no longer `int | None`).

The seed must be deterministic across DDP ranks because Lightning DDP
re-executes the training script in each subprocess, creating a new
`CerberusDataModule` per rank.  A random default (`random.Random(None)`)
would produce different seeds per rank → different cache directories →
cache misses.

```python
# Before:
seed: int | None = None
self.seed = seed  # None → non-deterministic sampler init

# After:
seed: int = 42
self.seed = seed  # always int → deterministic across DDP ranks
```

The `base_seed` dead code in `train_dataloader()` was also removed:

```python
# Before (dead code):
base_seed = self.seed if self.seed is not None else 0
seed = base_seed + (epoch * world_size) + rank

# After:
seed = self.seed + (epoch * world_size) + rank
```
