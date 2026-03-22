# Pydantic V2 Migration for config.py

## Executive Summary

Replace the 7 TypedDict classes and ~500 lines of hand-written validation in
`src/cerberus/config.py` (982 lines) with Pydantic V2 frozen `BaseModel` classes.

Key changes:

- **TypedDict → BaseModel** with `ConfigDict(frozen=True, extra="forbid")`.
- **Bracket access → attribute access** (`config["key"]` → `config.key`) across the entire codebase.
- **Typed sub-models** for `fold_args` (`FoldArgs`) and `sampler_args` (discriminated union per `sampler_type`).
- **Pseudocount reparameterization**: `count_pseudocount` moves from `DataConfig` to `ModelConfig` as a first-class field in scaled units. `propagate_pseudocount()` is deleted entirely. Single source of truth.
- **Big-bang migration** in one atomic PR.

Primary benefit: eliminates ~500 lines of manual `isinstance`/range-check boilerplate, gains structured `ValidationError`, IDE autocomplete, and enforced immutability.

Primary cost: new runtime dependency, ~250 attribute-access migration sites in `src/`, 218 `cast()` migrations in tests, 14 tool scripts updated.

---

## Current State

### Schema Definitions

Seven TypedDict classes in `config.py:26-214`:

| TypedDict | Fields | Lines |
|-----------|--------|-------|
| `GenomeConfig` | name, fasta_path, exclude_intervals, allowed_chroms, chrom_sizes, fold_type, fold_args | 26-51 |
| `SamplerConfig` | sampler_type, padded_size, sampler_args | 53-86 |
| `DataConfig` | inputs, targets, input_len, output_len, max_jitter, output_bin_size, encoding, log_transform, reverse_complement, use_sequence, target_scale, count_pseudocount | 89-125 |
| `TrainConfig` | batch_size, max_epochs, learning_rate, weight_decay, patience, optimizer, scheduler_type, scheduler_args, filter_bias_and_bn, reload_dataloaders_every_n_epochs, adam_eps, gradient_clip_val | 128-158 |
| `PretrainedConfig` | weights_path, source, target, freeze | 161-179 |
| `ModelConfig` | name, model_cls, loss_cls, loss_args, metrics_cls, metrics_args, model_args, pretrained | 182-203 |
| `CerberusConfig` | train_config, genome_config, data_config, sampler_config, model_config | 206-214 |

### Validation Layer

Five per-section validators (~400 lines of `isinstance` + range checks):

- `validate_genome_config()` (288-383) — required keys, types, path resolution, chrom_sizes filtering
- `validate_data_config()` (386-481) — required keys, positive ints, cross-field `reverse_complement` + `use_sequence` check
- `validate_sampler_config()` (484-568) — required keys, sampler_type-specific arg checks, **mutates via `setdefault`** (lines 554, 562)
- `validate_train_config()` (601-686) — required keys, positive values
- `validate_model_config()` (689-772) — required keys, optional model_args validation

Two cross-validators:

- `validate_data_and_sampler_compatibility()` (571-598) — padded_size vs input_len + 2*max_jitter
- `validate_data_and_model_compatibility()` (775-813) — channel matching

### The Pseudocount Problem (Dual Source of Truth)

`propagate_pseudocount()` (849-887) is a cross-config mutation function:

1. Reads `data_config.count_pseudocount` (raw units) and `data_config.target_scale`
2. Dynamically imports `loss_cls` to check `uses_count_pseudocount`
3. Computes `scaled_pseudocount = raw * target_scale`
4. Injects into `model_config.loss_args` and `model_config.metrics_args` via spread+setdefault

This creates two sources of truth:
- `data_config["count_pseudocount"]` — raw units
- `loss_args["count_pseudocount"]` — scaled units (derived copy)

The `setdefault` means an explicit value in `loss_args` silently wins over the derived value, making it unclear which is authoritative. Additionally, `log_counts_include_pseudocount` (a boolean derived from the loss class) is stored in `metrics_args` as if it were user-specified config.

### Mutation Patterns

| Pattern | Location | Description |
|---------|----------|-------------|
| `{**config, "key": val}` | config.py:887, train.py:101 | Spread to create new dict with overrides |
| `.copy()` + `["key"] = val` | train.py:427-430 | Shallow copy + bracket assignment for fold override |
| `setdefault()` in validator | config.py:554, 562 | Injects `complexity_center_size` default during validation |
| `setdefault()` on spread copy | config.py:883-886 | Injects pseudocount into loss_args/metrics_args copies |
| `{**config["sampler_args"]}` + override | predict_misc.py:109-114 | Creates modified sampler config for evaluation |

### Serialization

- **YAML load**: `yaml.safe_load()` → dict → validation pipeline (`parse_hparams_config`)
- **JSON dump**: `json.dump(payload, f, indent=2, default=str)` (train.py:140)
- **Lightning hparams**: `save_hyperparameters(_sanitize_config(config))` (module.py:67-73)
- **Path sanitization**: `_sanitize_config()` recursively converts `Path` → `str`

### Consumer Convention

Per CLAUDE.md: all consumers use bracket access `config["key"]`. No `.get("key", default)` allowed. Defaults injected via `setdefault` in validators (the normalization boundary).

---

## Proposed Architecture

### BaseModel Classes

Each TypedDict becomes a frozen BaseModel:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class GenomeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    fasta_path: Path
    exclude_intervals: dict[str, Path]
    allowed_chroms: list[str]
    chrom_sizes: dict[str, int]
    fold_type: str
    fold_args: FoldArgs

    @field_validator("fasta_path", mode="before")
    @classmethod
    def resolve_fasta(cls, v, info):
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_path(v, "Genome file", search_paths=search_paths)
```

### Typed Sub-Models

**FoldArgs** (currently `dict[str, Any]`):

```python
class FoldArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(ge=0)
    test_fold: int | None = Field(default=None, ge=0)
    val_fold: int | None = Field(default=None, ge=0)
```

**Sampler args** as discriminated union:

```python
class IntervalSamplerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    intervals_path: Path

class SlidingWindowSamplerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    stride: int = Field(gt=0)

class RandomSamplerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    num_intervals: int = Field(gt=0)

class PeakSamplerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    intervals_path: Path
    background_ratio: float = Field(default=1.0, gt=0)
    complexity_center_size: int | None = None  # resolves the setdefault TODO

class NegativePeakSamplerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    intervals_path: Path
    background_ratio: float = Field(default=1.0, gt=0)
    complexity_center_size: int | None = None

class ComplexityMatchedSamplerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    target_sampler: "SamplerConfig"  # forward ref — recursive
    candidate_sampler: "SamplerConfig"
    bins: int = Field(gt=0)
    candidate_ratio: float = Field(gt=0)
    metrics: list[str]

SamplerArgs = Annotated[
    IntervalSamplerArgs | SlidingWindowSamplerArgs | RandomSamplerArgs
    | PeakSamplerArgs | NegativePeakSamplerArgs | ComplexityMatchedSamplerArgs,
    Discriminator("sampler_type_discriminator"),  # or use a model_validator
]
```

### Pseudocount Reparameterization

**Remove** `count_pseudocount` from `DataConfig`.
**Add** `count_pseudocount: float` as a first-class field on `ModelConfig` in scaled units:

```python
class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    model_cls: str
    loss_cls: str
    loss_args: dict[str, Any]
    metrics_cls: str
    metrics_args: dict[str, Any]
    model_args: dict[str, Any]
    pretrained: list[PretrainedConfig] = Field(default_factory=list)
    count_pseudocount: float = Field(default=0.0, ge=0)  # scaled units, single source of truth
```

**Delete** `propagate_pseudocount()`. At instantiation time:

```python
def instantiate_metrics_and_loss(model_config: ModelConfig) -> tuple[MetricCollection, CerberusLoss]:
    loss_cls = import_class(model_config.loss_cls)
    loss_args = {**model_config.loss_args, "count_pseudocount": model_config.count_pseudocount}
    criterion = loss_cls(**loss_args)

    metrics_cls = import_class(model_config.metrics_cls)
    metrics_args = {
        **model_config.metrics_args,
        "count_pseudocount": model_config.count_pseudocount,
        "log_counts_include_pseudocount": loss_cls.uses_count_pseudocount,
    }
    metrics = metrics_cls(**metrics_args)
    return metrics, criterion
```

**Tools** compute the scaled value at config construction:

```python
# tools/train_bpnet.py
model_config = ModelConfig(
    ...,
    count_pseudocount=raw_pseudocount * target_scale,
)
```

**`get_log_count_params()`** reads directly:

```python
def get_log_count_params(model_config: ModelConfig) -> tuple[bool, float]:
    loss_cls = import_class(model_config.loss_cls)
    if loss_cls.uses_count_pseudocount:
        return True, model_config.count_pseudocount
    return False, 0.0
```

### Remaining Untyped Bags

`model_args`, `loss_args`, `metrics_args`, `scheduler_args` stay as `dict[str, Any]`. These are intentionally open-ended — their schemas are defined by user-supplied model/loss/metric classes, not by cerberus config.

### Serialization

- `_sanitize_config()` → `config.model_dump(mode="json")` (Pydantic converts Path→str automatically)
- `parse_hparams_config()` → `CerberusConfig.model_validate(data, context={"search_paths": paths})`
- `save_hyperparameters` → `config.model_dump(mode="json") if config is not None else None`

### Mutations

All mutations become `model_copy(update=...)`:

```python
# Before (train.py:427-430):
genome_config = genome_config.copy()
genome_config["fold_args"] = genome_config["fold_args"].copy()
genome_config["fold_args"]["test_fold"] = test_fold

# After:
genome_config = genome_config.model_copy(update={
    "fold_args": genome_config.fold_args.model_copy(update={
        "test_fold": test_fold, "val_fold": val_fold
    })
})
```

---

## Feature Mapping Table

| Current Pattern | Pydantic V2 Equivalent |
|----------------|----------------------|
| `TypedDict` class | `BaseModel` subclass with `ConfigDict(frozen=True, extra="forbid")` |
| Manual `isinstance()` checks | Automatic type coercion + `ValidationError` |
| Manual range checks (`> 0`, `>= 0`) | `Field(gt=0)`, `Field(ge=0)` |
| `required_keys` set difference | Automatic — missing fields raise `ValidationError` |
| `setdefault()` in validator | `Field(default=...)` on the model |
| `_validate_path()` + `FileNotFoundError` | `@field_validator` with `ValidationInfo.context` |
| `_validate_file_dict()` | `@field_validator` on `dict[str, Path]` fields |
| Cross-validators | `@model_validator(mode="after")` on `CerberusConfig` |
| `_sanitize_config()` (Path→str) | `model_dump(mode="json")` |
| `{**config, "key": val}` | `config.model_copy(update={"key": val})` |
| `config.copy()` + bracket assign | `config.model_copy(update={...})` |
| `config["key"]` bracket access | `config.key` attribute access |
| `propagate_pseudocount()` | Deleted — `count_pseudocount` is a first-class `ModelConfig` field |

---

## Benefits

1. **~500 lines of validation boilerplate eliminated** — type checking, range checks, required-key checks all expressed declaratively via Field() and type annotations.
2. **Structured `ValidationError`** — field-level error paths instead of generic TypeError/ValueError. Users see exactly which field failed and why.
3. **IDE autocomplete on attribute access** — `config.input_len` is fully typed by pyright; `config["input_len"]` is opaque.
4. **Enforced immutability** — `frozen=True` makes the codebase's desired immutability a hard guarantee, resolving the acknowledged TODO at lines 550-562 about validators mutating config.
5. **Typed sampler_args** — catches YAML typos per sampler_type at parse time (e.g., `intervlas_path` raises immediately instead of failing at runtime).
6. **Pseudocount single source of truth** — eliminates the dual-source problem where `data_config.count_pseudocount` and `loss_args["count_pseudocount"]` could diverge.
7. **JSON Schema export** — `CerberusConfig.model_json_schema()` generates a complete schema for documentation.

---

## Risks and Costs

1. **New runtime dependency** — Pydantic V2 (~5MB installed, compiled Rust core via pydantic-core). Adds to the dependency tree alongside numpy/torch/pytorch-lightning. May complicate HPC environments with restricted `pip install`.

2. **Attribute-access migration** — ~250 `config["key"]` sites across 14 source files in `src/cerberus/` must become `config.key`. Mechanical but high churn.

3. **Test migration** — 218 `cast(XConfig, {...})` calls across 45 test files must become `XConfig(...)` or `XConfig.model_validate({...})`. Tests that pass intentionally invalid/incomplete configs need separate handling (use plain dicts, not model constructors).

4. **Tools migration** — 14 training scripts construct configs as dict literals. Must become model constructors. Tools also need to compute `count_pseudocount * target_scale` at construction time.

5. **Sampler args discriminated union complexity** — The union type adds modeling complexity, especially for `complexity_matched` which recursively contains `SamplerConfig`. Pydantic handles recursive models but requires forward-ref resolution.

6. **Lightning hparams roundtrip** — When Lightning loads `hparams.yaml` from a checkpoint, it produces plain dicts. `parse_hparams_config` must accept dicts and produce Pydantic models via `model_validate`. Needs integration testing.

7. **int→float silent coercion** — Pydantic V2 lax mode coerces `int` to `float`. YAML values like `target_scale: 1` (int) that currently raise TypeError will silently become `1.0`. This is likely desirable but changes the error surface.

8. **`extra="forbid"` strictness** — Test configs with extra keys (e.g., mock objects) will fail. These must be cleaned up.

9. **Pseudocount reparameterization** — semantic config change. Existing YAML files with `data_config.count_pseudocount` need migration. Old `hparams.yaml` files become incompatible unless a backward-compat validator is added.

10. **Performance** — Pydantic model construction is ~10-100x slower than plain dict construction. Negligible for config parsing (once at startup) but may slow test suites that construct hundreds of configs.

---

## target_scale Cross-Cutting Concern

`target_scale` is currently in `DataConfig` but is consumed by:

- **Data transforms** (`transform.py:366-368`) — applies `Scale(factor=target_scale)` to targets
- **Pseudocount scaling** — tools compute `raw_pseudocount * target_scale` at config construction
- **Adaptive loss weights** (`datamodule.py:405-406`) — scales median raw counts
- **Inference denormalization** (`predict_bigwig.py:265-266`) — divides by target_scale to recover raw signal

This is a cross-cutting concern that arguably belongs at the top level or in a shared config. However, it is primarily a data scaling parameter, and moving it would be a breaking change with limited benefit. Recommend keeping in `DataConfig` for now, with a note that it could be promoted in a future refactor if the cross-cutting nature causes further problems.

---

## Migration Scope (Big-Bang Single PR)

| Step | Scope | Files |
|------|-------|-------|
| 1. Add dependency | `pydantic>=2.0` to pyproject.toml | 1 |
| 2. Rewrite config.py | TypedDicts → BaseModels, typed sub-models, pseudocount reparameterization | 1 |
| 3. Delete dead code | `propagate_pseudocount()`, `_sanitize_config()`, all `validate_*` functions | 1 |
| 4. Update instantiation | `instantiate_metrics_and_loss()`, `get_log_count_params()` | 2 (module.py, config.py) |
| 5. Migrate src/ consumers | `config["key"]` → `config.key`, spread → `model_copy` | 14 |
| 6. Migrate tests/ | `cast(Config, {...})` → `Config(...)` | 45 |
| 7. Migrate tools/ | Dict literals → model constructors, pseudocount scaling | 14 |
| 8. Update conventions | CLAUDE.md (attribute access), docs/configuration.md | 2 |
| **Total** | | **~80 files** |

---

## Effort Estimate

| Phase | Description | Estimate |
|-------|-------------|----------|
| Schema conversion | BaseModels, typed sub-models, pseudocount reparameterization | 1 day |
| src/ consumer migration | Attribute access, model_copy, serialization | 1 day |
| Test migration | 218 cast() calls, invalid-config test fixes | 1-2 days |
| Tools migration | 14 train scripts, pseudocount computation | 0.5 day |
| Cleanup & docs | CLAUDE.md, configuration.md, pyproject.toml | 0.5 day |
| **Total** | | **4-5 days** |

Test migration (Phase 3) is the dominant cost and is almost entirely mechanical — a good candidate for automated find-and-replace.

---

## Recommendation

**Go.** The migration is worth doing because:

1. The validation boilerplate is already a maintenance burden — the TODO comments at config.py:550-562 acknowledge that the current architecture is wrong (validators mutating config).
2. The pseudocount dual-source-of-truth problem is a latent bug source that the reparameterization cleanly eliminates.
3. The codebase is still small enough (982 lines config, 45 test files) that the migration is tractable in a single PR.
4. Pydantic V2 is a mainstream dependency in the Python ML ecosystem (used by FastAPI, LangChain, vLLM, etc.) and does not conflict with existing deps.
5. `frozen=True` gives the immutability guarantee the codebase already wants but doesn't consistently enforce.

---

## Outcome (Post-Implementation)

The migration was completed. Key metrics:

| Metric | Estimated | Actual |
|--------|-----------|--------|
| Files changed | ~80 | 158 |
| config.py net lines | -500 | -323 (982 → 659) |
| Test files migrated | 45 | 62 |
| Test count after | — | 1473 pass, 0 fail |
| New regression tests | — | 76 (test_pydantic_config.py) |
| Examples verified on GPU | — | 14/14 pass (A100) |
| Pyright errors | — | 4 (all pre-existing) |

### What went as planned
- TypedDict → BaseModel conversion was straightforward
- Typed sampler args with discriminated union worked well
- FoldArgs typing was clean
- Pseudocount reparameterization eliminated `propagate_pseudocount` entirely
- `frozen=True` enforcement caught all mutation sites
- Backward-compat migration in `parse_hparams_config` handles legacy YAML

### Unexpected issues
- **Context propagation in model_validator** (problem #3 in problems doc): manual model
  construction inside `@model_validator` does not propagate `ValidationInfo.context`.
  Required `model_validate(args, context=ctx)` instead of `Model(**args)`.
- **Test scope was 2x larger than estimated**: 62 files vs 45, because many test files
  had config construction even though they weren't in the initial grep for `cast()`.
- **mock.patch cleanup**: ~40 test files patched deleted validation functions. The
  volume of mock removal was not anticipated in the plan.
- **`model_construct()` needed pervasively in tests**: almost all test fixtures use fake
  paths, requiring `model_construct()` to skip path validation. This was anticipated
  in gotcha #21 but the scale (hundreds of call sites) was larger than expected.
