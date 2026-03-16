# Seeding and Determinism in Cerberus

## Overview

Cerberus has **two independent randomness domains** that must both be seeded for
full reproducibility:

1. **Sampler RNG** — which genomic intervals are drawn for train/val/test
2. **Global RNG** — model weight initialization, DataLoader batch shuffling,
   and data augmentation transforms (Jitter, RandomRC)

Domain 1 is correctly seeded. Domain 2 is **not** — `pl.seed_everything()` is
never called in the library code.

---

## Domain 1: Sampler RNG (correctly seeded)

The seed flows through the full chain with default `seed=42` at every level:

```
train_single(seed=42)                       # train.py:362
  → CerberusDataModule(seed=seed)           # train.py:411
    → self.seed = seed                      # datamodule.py:88
    → CerberusDataset(seed=self.seed)       # datamodule.py:160, 243
      → self.seed = seed                    # dataset.py:99
      → create_sampler(..., seed=seed)      # dataset.py:213
        → RandomSampler(seed=seed)          # samplers.py:519-523
          self.rng = random.Random(seed)
          self.resample(seed)
```

Every sampler class (`RandomSampler`, `PeakSampler`, `NegativePeakSampler`,
`ComplexityMatchedSampler`, `MultiSampler`) initializes a **private**
`random.Random(seed)` instance in `__init__`. These are isolated from the
global `random` module state — calling `random.seed()` elsewhere has no effect
on them.

Epoch-wise resampling is also deterministic:

```python
# datamodule.py:264-269
seed = self.seed + (epoch * world_size) + rank
self.train_dataset.resample(seed=seed)
```

Sub-samplers get independent seeds via `generate_sub_seeds(seed, n)`:

```python
# samplers.py:45-51
def generate_sub_seeds(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(n)]
```

**Conclusion:** Given the same `seed` parameter, the same genomic intervals are
sampled in the same order across runs, ranks, and epochs. This is fully
deterministic.

---

## Domain 2: Global RNG (NOT seeded)

Three components use the **global** torch/numpy/random RNG rather than a
private seeded instance:

### 2a. DataLoader `shuffle=True`

```python
# datamodule.py:276
shuffle=True,
```

When `shuffle=True`, PyTorch's `RandomSampler` (the DataLoader index sampler,
not our `cerberus.samplers.RandomSampler`) calls `torch.randperm()` to generate
a random permutation of dataset indices. This uses the **global torch
generator** (`torch.default_generator`).

Without `pl.seed_everything()` or `torch.manual_seed()`, the shuffle order is
different on every run. This means the same sampled intervals are presented in a
**different batch order** across runs, producing different gradient updates.

### 2b. Jitter transform

```python
# transform.py:87
start = torch.randint(start_min, start_max + 1, (1,)).item()
```

`torch.randint` without a `generator` argument draws from
`torch.default_generator`. Without global seeding, jitter offsets are
non-reproducible.

### 2c. RandomRC (reverse complement) transform

```python
# transform.py:159
if torch.rand(1).item() > self.probability:
```

Same issue — `torch.rand` uses the global generator.

### 2d. Model weight initialization

```python
# models/bpnet.py:151
nn.init.xavier_uniform_(m.weight)
```

PyTorch's `nn.init.*` functions (except `zeros_` and `constant_`) use the
global torch generator. Without seeding, models start from different random
weights on each run.

Note: `nn.Conv1d` and `nn.Linear` default initialization (`kaiming_uniform_`)
also uses the global generator. Even models that don't call `nn.init.*`
explicitly (like Dalmatian's conv layers) get random initial weights from
PyTorch's default init.

### 2e. Worker init function

```python
# datamodule.py:109-111
worker_seed = torch.initial_seed() % 2**32
np.random.seed(worker_seed)
random.seed(worker_seed)
```

This correctly derives worker-local seeds from torch's seed — but
`torch.initial_seed()` returns the global generator's seed. If the global seed
was never set, each worker gets a non-reproducible seed. With
`pl.seed_everything(42, workers=True)`, each worker gets a deterministic seed.

---

## What `pl.seed_everything(seed)` does

```python
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)
```

This single call seeds **all four** global RNG sources:

| RNG | Call | Affects |
|-----|------|---------|
| `random.seed(42)` | Python stdlib | Not used by cerberus samplers (they use private instances) |
| `np.random.seed(42)` | NumPy | Not directly used, but may affect dependencies |
| `torch.manual_seed(42)` | PyTorch CPU | `shuffle=True`, `torch.randint`, `torch.rand`, `nn.init.*` |
| `torch.cuda.manual_seed_all(42)` | PyTorch GPU | GPU-side random ops |

With `workers=True`, it also sets `os.environ["PL_SEED_WORKERS"] = "1"` which
tells Lightning's internal worker init to seed each worker deterministically.

---

## Impact assessment

| Component | Without `seed_everything` | With `seed_everything` |
|-----------|---------------------------|------------------------|
| **Which intervals are sampled** | Deterministic | Deterministic |
| **Interval order within epoch** | Deterministic | Deterministic |
| **Batch shuffle order** | Non-deterministic | Deterministic |
| **Jitter offsets** | Non-deterministic | Deterministic |
| **RC augmentation** | Non-deterministic | Deterministic |
| **Model initial weights** | Non-deterministic | Deterministic |
| **Worker RNG state** | Non-deterministic | Deterministic |

The sampler seed ensures the same train/val/test **split** and the same set of
intervals. But the batch ordering, augmentation, and weight init differ between
runs. In practice this means:

- **Training curves are not exactly reproducible** across runs with the same
  config. Final val_loss will vary by the normal stochastic training noise.
- **Val/test evaluation IS reproducible** within a single run (val uses
  `shuffle=False` and deterministic transforms).
- **Cross-experiment comparisons are valid** — the same val intervals are
  evaluated with the same deterministic transforms. Only the training path
  differs.

---

## Where to add it

The fix is a single line in `train.py:_train()`, before any model or data
construction:

```python
# train.py, inside _train(), before line 258
pl.seed_everything(seed, workers=True)
```

This requires threading the `seed` parameter from `train_single`/`train_multi`
through to `_train`. Currently `_train` does not receive `seed` — it only flows
to `CerberusDataModule`.

Alternatively, add it to each training script's entry point. The docs
(`docs/usage.md:227-235`) already recommend this but the library doesn't
enforce it.

---

## Caveats

Even with `pl.seed_everything()`, full bitwise reproducibility requires:

1. `torch.use_deterministic_algorithms(True)` — forces deterministic CUDA
   kernels (slower). Without this, `cudnn.benchmark=True` and certain ops
   (scatter, atomicAdd) are non-deterministic.
2. Single GPU — multi-GPU introduces non-deterministic communication ordering.
3. Same hardware — different GPU architectures may produce different float
   results.

For the exp4/exp5 comparison experiments, `seed_everything` alone is sufficient
— we don't need bitwise reproducibility, just consistent initialization and
augmentation.
