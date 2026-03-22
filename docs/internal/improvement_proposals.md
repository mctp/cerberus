# Cerberus Improvement Proposals

*Date: 2025-03-22*

This document proposes high-value improvements to the Cerberus codebase,
prioritized by impact on **simplicity**, **correctness**, and **user
experience**. Each proposal includes motivation, scope, and suggested
approach.

---

## Tier 1 — High Impact

### 1. Deduplicate loss module boilerplate

**Problem.** `loss.py` (514 lines) contains six loss classes that repeat
nearly identical logic. Every class re-implements:

- `log1p_targets` undo (`torch.expm1(...).clamp_min(0.0)`)
- `forward()` that just calls `loss_components()` and does a weighted sum
- `_compute_profile_loss()` (duplicated in `MSEMultinomialLoss` and
  `PoissonMultinomialLoss` with minor differences)

The "Coupled" variants (`CoupledMSEMultinomialLoss`,
`CoupledPoissonMultinomialLoss`, `CoupledNegativeBinomialMultinomialLoss`)
exist only to derive `log_counts` from `logsumexp(log_rates)` instead of
from a dedicated head. Their `loss_components` methods are copy-paste of
the parent with a different count source.

**Proposal.** Extract shared mechanics into a base class:

```python
class ProfileCountLossBase(nn.Module):
    """Shared profile + count loss mechanics."""

    def _undo_log1p(self, targets): ...
    def _compute_profile_loss(self, logits, targets): ...
    def _extract_counts(self, outputs, targets):
        """Override in coupled variants to use logsumexp."""
        ...
    def loss_components(self, outputs, targets, **kwargs):
        targets = self._maybe_undo_log1p(targets)
        self._check_output_type(outputs)
        logits, pred_log_counts = self._extract_predictions(outputs)
        ...
    def forward(self, outputs, targets, **kwargs):
        c = self.loss_components(outputs, targets, **kwargs)
        return self.profile_weight * c["profile_loss"] + self.count_weight * c["count_loss"]
```

The three "Coupled" classes become thin overrides of
`_extract_predictions`. `NegativeBinomialMultinomialLoss` overrides only
the count loss computation. This would cut ~150 lines and eliminate the
most dangerous form of duplication: a bug fixed in one loss but not
another.

**Risk.** Moderate — loss functions are critical to correctness.
Regression tests already cover this well (23 loss tests + 34 pseudocount
tests + 50 since-0.9.2 tests).

---

### 2. Split `samplers.py` into a subpackage

**Problem.** `samplers.py` is 1,564 lines containing 9 classes, 3
protocols, and several free functions. It mixes fundamentally different
concerns:

- Base protocols and fold-splitting logic
- Simple samplers (Interval, SlidingWindow, Random)
- Complex algorithmic samplers (ComplexityMatched, Peak)
- Multi-sampler composition

This makes the file hard to navigate, hard to review, and the
`ComplexityMatchedSampler` (~200 lines of nested logic) is entangled with
interval helpers that other samplers also use.

**Proposal.** Convert to a subpackage:

```
samplers/
  __init__.py          # re-export public names (no behavior change)
  _protocol.py         # BaseSampler, ProxySampler, SamplerProtocol
  _partition.py        # partition_intervals_by_fold, match_bin_counts, helpers
  interval.py          # IntervalSampler, SlidingWindowSampler, RandomSampler
  peak.py              # PeakSampler, NegativePeakSampler
  complexity.py        # ComplexityMatchedSampler
  multi.py             # MultiSampler
```

The `__init__.py` re-exports preserve the existing public API
(`from cerberus.samplers import IntervalSampler` keeps working).

**Risk.** Low — pure file reorganization, no behavior change. `__init__`
re-exports mean zero downstream breakage.

---

### 3. Eager file validation at `CerberusDataModule.setup()`

**Problem.** `GenomeConfig.fasta_path`, `DataConfig.inputs`, and
`DataConfig.targets` are `Path` fields but Pydantic only validates the
*type*, not file existence. If a user specifies a wrong path, the error
surfaces late — during the first `__getitem__` call inside a DataLoader
worker, producing a traceback like:

```
RuntimeError: Error opening file '/wrong/path.bw'
  [in pybigtools C extension]
```

This is confusing because the traceback originates in a subprocess and
doesn't point to the misconfigured field.

**Proposal.** Add an eager check in `CerberusDataModule.setup()` (or
`prepare_data()`), before workers are spawned:

```python
def _validate_paths(self):
    """Verify all configured file paths exist before spawning workers."""
    for name, path in [("fasta_path", self.genome_config.fasta_path)]:
        if not path.exists():
            raise FileNotFoundError(f"Genome FASTA not found: {path}")
    for label, channel_map in [("input", self.data_config.inputs),
                                ("target", self.data_config.targets)]:
        for channel, path in channel_map.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Data {label} channel '{channel}' file not found: {path}"
                )
```

This gives users an immediate, obvious error with the field name and path
that's wrong.

**Risk.** Very low — adds a check, doesn't change behavior. Could be
skipped via a flag for testing scenarios with synthetic data.

---

### 4. Export `TrainConfig`, `ModelConfig`, and training entrypoints from `__init__`

**Problem.** Users must import from internal submodules for the most
common operations:

```python
from cerberus import GenomeConfig, DataConfig, SamplerConfig  # OK
from cerberus.config import TrainConfig, ModelConfig           # why not __init__?
from cerberus.train import train_single, train_multi           # why not __init__?
from cerberus.module import instantiate, CerberusModule        # why not __init__?
```

`TrainConfig` and `ModelConfig` are not less important than
`GenomeConfig` — they're required for every training run. The split
between "configs you can import from cerberus" and "configs you must
import from cerberus.config" is arbitrary and confusing.

**Proposal.** Add to `__init__.py`:

```python
from .config import TrainConfig, ModelConfig, PretrainedConfig, CerberusConfig
from .module import CerberusModule, instantiate, instantiate_model
from .train import train_single, train_multi
```

This lets users write:

```python
from cerberus import (
    GenomeConfig, DataConfig, SamplerConfig,
    TrainConfig, ModelConfig, CerberusConfig,
    CerberusModule, instantiate, train_single,
)
```

**Risk.** None — purely additive.

---

### 5. Add tests for `predict_bigwig.py` and `pretrained.py`

**Problem.** These two user-facing modules have zero direct test
coverage:

- `predict_bigwig.py` (284 lines) — generates genome-wide BigWig
  prediction files, a critical output artifact.
- `pretrained.py` (125 lines) — loads pretrained weights with prefix
  extraction and optional freezing.

Both are tested only indirectly through integration tests. A regression in
either module (e.g., an off-by-one in tiling, or a key-mismatch in
weight loading) would go undetected by the test suite.

**Proposal.** Write targeted unit tests:

For `pretrained.py`:
- Test prefix extraction with matching/non-matching keys
- Test freeze behavior (all params frozen after load)
- Test error message when prefix not found

For `predict_bigwig.py`:
- Test tiling logic (correct overlap handling, no gaps)
- Test output shape/value for a mock model on a small chromosome
- Test aggregation (mean/max) over overlapping tiles

**Risk.** None — pure addition.

---

## Tier 2 — Medium Impact

### 6. Validate output-type / loss-type compatibility at instantiation

**Problem.** If a user pairs a model that returns `ProfileLogRates` with
`MSEMultinomialLoss` (which requires `ProfileCountOutput`), the error
occurs at the first training step:

```
TypeError: MSEMultinomialLoss requires ProfileCountOutput
```

By this point the user has already downloaded data, built the dataset,
and started the trainer. The mismatch was knowable at config time.

**Proposal.** Add an optional static compatibility check to
`instantiate()` or `CerberusConfig.cross_validate()`. This requires a
light convention: each model class declares its output type, and each
loss class declares its expected input type. A compatibility matrix could
be checked at instantiation.

One approach: add a class attribute `output_type` to each model class and
`expected_output_type` to each loss, then compare in `instantiate()`.

**Risk.** Low. The fallback (runtime TypeError) is already clean.

---

### 7. Consolidate scattered test files

**Problem.** The test suite has 160 files, many of which contain a single
test function. Examples:

- `test_asap_equivalence.py` — 1 test
- `test_batch_timing.py` — 1 test
- `test_binning.py` — 1 test
- `test_cre_loading.py` — 1 test

There are 12 dataset-related test files (`test_dataset_*.py`,
`test_mock_dataset.py`, `test_prepare_data.py`, etc.) and 6
config-related test files. This fragmentation makes it hard to assess
coverage at a glance and increases maintenance burden.

**Proposal.** Merge related test files:

- `test_dataset_*.py` (12 files) -> `test_dataset.py` with clear
  `TestDatasetCreation`, `TestDatasetIteration`, etc. classes
- `test_config_*.py` (6 files) -> single-test smoke tests folded into
  `test_config.py`
- Single-test files that only verify "code runs without error" should
  either be upgraded with assertions or removed

**Risk.** Low — test-only changes. Run the full suite to verify.

---

### 8. Document the `model_config_` / `model_config` alias prominently

**Problem.** The `CerberusConfig.model_config_` attribute uses
`Field(alias="model_config")` because Pydantic V2 reserves the bare
name. This means:

- In Python: `cerberus_config.model_config_` (with underscore)
- In YAML/dict: `"model_config"` (without underscore)
- `cerberus_config.model_config` silently returns Pydantic's internal
  `ConfigDict`, not the `ModelConfig` object

The docstring explains this, but the public docs (`configuration.md`) and
examples never highlight the pitfall. Users coming from other Pydantic
projects will assume `.model_config` works.

**Proposal.** Add a visible warning box to `configuration.md`:

```markdown
!!! warning "model_config_ attribute"
    Due to Pydantic V2 reserving `model_config`, the attribute is named
    `model_config_` in Python. Using `cerberus_config.model_config` will
    return Pydantic's internal `ConfigDict` — not the `ModelConfig` object.
```

Also consider adding a `__getattr__` guard on `CerberusConfig` that
raises a helpful error if someone accesses `.model_config` expecting a
`ModelConfig`:

```python
def __getattr__(self, name):
    if name == "model_config" and not isinstance(getattr(type(self), name, None), property):
        raise AttributeError(
            "Use 'model_config_' (with underscore) to access the ModelConfig. "
            "Pydantic V2 reserves 'model_config' for its own ConfigDict."
        )
    raise AttributeError(name)
```

**Risk.** Very low.

---

### 9. Add loss/model selection guide to public docs

**Problem.** Users must choose between 8 loss classes and 6 model
architectures, but the docs provide no guidance on which combination to
use for a given experimental design. The `models.md` page describes
architectures but doesn't explain when to prefer one loss over another
(MSE vs. Poisson vs. NB, coupled vs. decoupled, factorized vs.
standard).

**Proposal.** Add a "Choosing a Loss Function" section to
`components.md` or a new `docs/loss_selection.md` page:

| Scenario | Recommended Loss | Why |
|---|---|---|
| ChIP-seq, moderate depth | `MSEMultinomialLoss` | Standard BPNet loss, robust |
| Low-coverage / high-variance | `NegativeBinomialMultinomialLoss` | Accounts for overdispersion |
| Single output head (no count head) | `CoupledMSEMultinomialLoss` | Derives counts from profile |
| Bias-factorized model (Dalmatian) | `DalmatianLoss` | Handles bias/signal decomposition |

**Risk.** None — documentation only.

---

### 10. Add `cache.py` robustness and tests

**Problem.** `cache.py` (111 lines) uses a SHA-256 hash of the config
JSON + FASTA file `mtime` to determine the cache directory. Two issues:

1. If a FASTA file is replaced with a file of the same modification
   time, stale cache is used (edge case, but possible with `cp -p` or
   rsync).
2. Only 2 tests cover this module.

**Proposal.**
- Include FASTA file *size* in the hash (catches most replacement
  scenarios without the cost of hashing the file).
- Add tests for cache hit/miss/invalidation scenarios.

**Risk.** Very low.

---

## Tier 3 — Polish

### 11. Consistent type annotation style

The codebase mixes `Optional[X]` (PEP 484) and `X | None` (PEP 604).
Both are valid, but inconsistency across modules is distracting. Since
the project uses `from __future__ import annotations` in most files, the
PEP 604 style (`X | None`) works everywhere.

**Proposal.** Adopt `X | None` uniformly. Can be done incrementally,
module by module.

---

### 12. Expand `multi_gpu.md`

The multi-GPU documentation is 5 lines linking to internal docs. Users
launching distributed training need at least one concrete example showing
`accelerator="gpu"`, `devices=N`, `strategy="ddp"`, and the
`--multi` flag in the training tools.

**Proposal.** Add a minimal working example and note the interaction with
`num_workers` and `reload_dataloaders_every_n_epochs`.

---

### 13. Add `model_copy(update=...)` examples to `configuration.md`

The docs show config creation but never demonstrate mutation via
`model_copy()`. Since configs are frozen, users need to know how to
derive a modified config (e.g., changing `learning_rate` for a sweep).

**Proposal.** Add a 5-line example:

```python
# Derive a config with different learning rate
new_train = train_config.model_copy(update={"learning_rate": 1e-4})
```

---

### 14. `InterLap` coordinate wrapping helpers

The codebase correctly handles the conversion between half-open
`[start, end)` intervals (used everywhere) and closed `[start, end-1]`
intervals (used by InterLap). But the conversions are inline in
`exclude.py` and `samplers.py`, making them easy to get wrong during
future edits.

**Proposal.** Extract two small helpers:

```python
def to_interlap(start: int, end: int) -> tuple[int, int]:
    """Convert half-open [start, end) to InterLap closed [start, end-1]."""
    return (start, end - 1)

def from_interlap(start: int, end: int) -> tuple[int, int]:
    """Convert InterLap closed [start, end] to half-open [start, end+1)."""
    return (start, end + 1)
```

---

### 15. Remove or upgrade single-assertion smoke tests

Several test files exist only to verify that code executes without
raising an exception (`test_batch_timing.py`, `test_binning.py`,
`test_cre_loading.py`). These provide minimal value — they pass even if
the output is garbage. Either add meaningful assertions or fold them into
the module's main test file as a smoke-test section.

---

## Summary Matrix

| # | Proposal | Area | Impact | Effort | Risk |
|---|---|---|---|---|---|
| 1 | Deduplicate loss boilerplate | Simplicity | High | Medium | Medium |
| 2 | Split samplers.py into subpackage | Simplicity | High | Low | Low |
| 3 | Eager file path validation | UX | High | Low | Very Low |
| 4 | Export all configs/entrypoints from `__init__` | UX | High | Very Low | None |
| 5 | Add predict_bigwig + pretrained tests | Correctness | High | Medium | None |
| 6 | Output/loss type compatibility check | Correctness | Medium | Medium | Low |
| 7 | Consolidate scattered test files | Simplicity | Medium | Medium | Low |
| 8 | Document model_config_ alias prominently | UX | Medium | Very Low | Very Low |
| 9 | Loss/model selection guide | UX | Medium | Low | None |
| 10 | Cache robustness + tests | Correctness | Medium | Low | Very Low |
| 11 | Consistent type annotation style | Simplicity | Low | Low | None |
| 12 | Expand multi_gpu.md | UX | Low | Very Low | None |
| 13 | model_copy() examples in docs | UX | Low | Very Low | None |
| 14 | InterLap coordinate helpers | Correctness | Low | Very Low | Low |
| 15 | Upgrade smoke tests | Correctness | Low | Low | None |
