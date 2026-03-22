# Pydantic Migration: Breaking Changes for Users

This document covers every breaking change introduced by the Pydantic V2 migration
in cerberus. Each section shows what changed, why, and how to update your code.

---

## 1. Config objects are Pydantic models, not dicts

All configuration types (`GenomeConfig`, `DataConfig`, `SamplerConfig`, `TrainConfig`,
`ModelConfig`, `CerberusConfig`) are now frozen Pydantic `BaseModel` classes instead
of `TypedDict` / plain dict objects.

### What this means

- **Attribute access replaces bracket access.** Every `config["key"]` must become
  `config.key`.
- **Configs are immutable.** Direct assignment (`config.key = val` or
  `config["key"] = val`) raises a `ValidationError`. Use `model_copy(update=...)`
  to produce a new config with changes.
- **Extra keys are forbidden.** Passing unknown fields to a config constructor raises
  `ValidationError` (previously silently ignored).
- **Type coercion is automatic.** Pydantic coerces compatible types (e.g. `int` to
  `float` for `target_scale`). Incompatible types raise at construction time with
  structured error messages.

### Migration

**Reading config fields:**

```python
# Before
input_len = data_config["input_len"]
targets = data_config["targets"]
fold_k = genome_config["fold_args"]["k"]

# After
input_len = data_config.input_len
targets = data_config.targets
fold_k = genome_config.fold_args.k
```

**Modifying a config (creating a modified copy):**

```python
# Before
genome_config = genome_config.copy()
genome_config["fold_args"] = genome_config["fold_args"].copy()
genome_config["fold_args"]["test_fold"] = 2
genome_config["fold_args"]["val_fold"] = 3

# After
genome_config = genome_config.model_copy(update={
    "fold_args": genome_config.fold_args.model_copy(update={
        "test_fold": 2,
        "val_fold": 3,
    })
})
```

**Spread-merge patterns:**

```python
# Before
new_config = {**model_config, "count_pseudocount": 150.0}

# After
new_config = model_config.model_copy(update={"count_pseudocount": 150.0})
```

---

## 2. count_pseudocount moved from DataConfig to ModelConfig

`count_pseudocount` has been removed from `DataConfig` and added to `ModelConfig` as
a first-class field. The value is now in **scaled units** (raw pseudocount multiplied
by `target_scale`), eliminating the old dual-source-of-truth problem.

### What this means

- `data_config.count_pseudocount` no longer exists. Accessing it raises
  `AttributeError`.
- `model_config.count_pseudocount` is the single source of truth, in scaled units.
- `propagate_pseudocount()` has been deleted. There is nothing to call and nothing
  to import.
- Training scripts must compute the scaled value at construction time.

### Migration

**Constructing ModelConfig in training scripts:**

```python
# Before
data_config = DataConfig(
    ...,
    count_pseudocount=150.0,  # raw units
    target_scale=10.0,
)
model_config = ModelConfig(
    ...,
)
# propagate_pseudocount(data_config, model_config) was called later

# After
raw_pseudocount = 150.0
target_scale = 10.0
data_config = DataConfig(
    ...,
    # count_pseudocount is gone from DataConfig
    target_scale=target_scale,
)
model_config = ModelConfig(
    ...,
    count_pseudocount=raw_pseudocount * target_scale,  # scaled units
)
```

**Reading the pseudocount value:**

```python
# Before
pseudocount = data_config["count_pseudocount"]

# After
pseudocount = model_config.count_pseudocount  # already in scaled units
```

### Backward compatibility for existing YAML checkpoints

`parse_hparams_config` automatically migrates old `hparams.yaml` files that still
have `count_pseudocount` under `data_config`. It computes
`raw_pseudocount * target_scale`, moves the value to `model_config`, and logs a
deprecation warning. No manual YAML editing is required for loading old checkpoints.

---

## 3. sampler_args is now a typed model

`SamplerConfig.sampler_args` is no longer a `dict[str, Any]`. It is a typed union
of Pydantic models, dispatched by `sampler_type`.

### Sampler args types

| `sampler_type` | Args class | Fields |
|---|---|---|
| `"interval"` | `IntervalSamplerArgs` | `intervals_path` |
| `"sliding_window"` | `SlidingWindowSamplerArgs` | `stride` |
| `"random"` | `RandomSamplerArgs` | `num_intervals` |
| `"peak"` | `PeakSamplerArgs` | `intervals_path`, `background_ratio`, `complexity_center_size` |
| `"negative_peak"` | `NegativePeakSamplerArgs` | `intervals_path`, `background_ratio`, `complexity_center_size` |
| `"complexity_matched"` | `ComplexityMatchedSamplerArgs` | `target_sampler`, `candidate_sampler`, `bins`, `candidate_ratio`, `metrics` |

### Migration

**Constructing SamplerConfig:**

```python
# Before
sampler_config = SamplerConfig(
    sampler_type="peak",
    padded_size=2626,
    sampler_args={
        "intervals_path": "/path/to/peaks.bed",
        "background_ratio": 1.0,
    },
)

# After
from cerberus.config import PeakSamplerArgs

sampler_config = SamplerConfig(
    sampler_type="peak",
    padded_size=2626,
    sampler_args=PeakSamplerArgs(
        intervals_path="/path/to/peaks.bed",
        background_ratio=1.0,
    ),
)
```

> **Note:** When constructing `SamplerConfig` from a plain dict (e.g., from YAML),
> Pydantic's `model_validator` on `SamplerConfig` automatically routes the
> `sampler_args` dict to the correct typed model based on `sampler_type`. You only
> need to use the explicit constructor when building configs in Python code.

**Accessing sampler args fields:**

```python
# Before
intervals_path = sampler_config["sampler_args"]["intervals_path"]
bg_ratio = sampler_config["sampler_args"]["background_ratio"]

# After
intervals_path = sampler_config.sampler_args.intervals_path
bg_ratio = sampler_config.sampler_args.background_ratio
```

**Typo detection:** A misspelled field like `intervlas_path` now raises
`ValidationError` at construction time instead of silently creating a dict key
that is never read.

---

## 4. fold_args is now a FoldArgs model

`GenomeConfig.fold_args` is no longer a `dict[str, Any]`. It is a `FoldArgs`
Pydantic model with typed fields.

### FoldArgs fields

| Field | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | (required) | Number of folds (>= 0) |
| `test_fold` | `int \| None` | `None` | Test fold index |
| `val_fold` | `int \| None` | `None` | Validation fold index |

### Migration

**Constructing GenomeConfig:**

```python
# Before
genome_config = GenomeConfig(
    ...,
    fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
)

# After
from cerberus.config import FoldArgs

genome_config = GenomeConfig(
    ...,
    fold_args=FoldArgs(k=5, test_fold=0, val_fold=1),
)
```

> As with sampler_args, `FoldArgs` is automatically constructed from a plain dict
> when parsing YAML, so only Python-side constructors need updating.

**Accessing fold_args fields:**

```python
# Before
k = genome_config["fold_args"]["k"]
test_fold = genome_config["fold_args"]["test_fold"]

# After
k = genome_config.fold_args.k
test_fold = genome_config.fold_args.test_fold
```

---

## 5. validate_* functions are deleted

The following functions no longer exist and must not be imported:

- `validate_genome_config`
- `validate_data_config`
- `validate_sampler_config`
- `validate_train_config`
- `validate_model_config`
- `validate_data_and_sampler_compatibility`
- `validate_data_and_model_compatibility`

### Where validation happens now

- **Per-field validation** (types, ranges, required fields) happens automatically
  when you construct any config model. Invalid values raise
  `pydantic.ValidationError` with structured, field-level error messages.
- **Cross-config validation** (data/sampler compatibility, data/model channel
  matching) happens in `CerberusConfig`'s `model_validator` when you construct
  the combined config.

### Migration

```python
# Before
validate_genome_config(genome_config)
validate_data_config(data_config)
validate_sampler_config(sampler_config)
validate_data_and_sampler_compatibility(data_config, sampler_config)

# After
# All validation happens at construction time. If you reach this point
# without a ValidationError, the configs are valid.
genome_config = GenomeConfig(...)   # validates on construction
data_config = DataConfig(...)       # validates on construction

# Cross-validation happens when constructing CerberusConfig:
cerberus_config = CerberusConfig(
    train_config=train_config,
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    model_config=model_config,      # note: key is "model_config", not "model_config_"
)
```

---

## 6. _sanitize_config is deleted

The `_sanitize_config()` function (which recursively converted `Path` objects to
strings for JSON serialization) has been removed.

### Migration

```python
# Before
from cerberus.config import _sanitize_config
sanitized = _sanitize_config(config_dict)
json.dump(sanitized, f, indent=2)

# After
# Pydantic's model_dump(mode="json") handles Path -> str conversion automatically.
json.dump(config.model_dump(mode="json"), f, indent=2)
```

For Lightning's `save_hyperparameters`:

```python
# Before
self.save_hyperparameters(_sanitize_config({
    "train_config": train_config,
    "data_config": data_config,
    ...
}))

# After
self.save_hyperparameters({
    "train_config": train_config.model_dump(mode="json") if train_config is not None else None,
    "data_config": data_config.model_dump(mode="json") if data_config is not None else None,
    ...
})
```

---

## 7. CerberusConfig.model_config_ naming

Pydantic V2 reserves the name `model_config` for its own `ConfigDict`. Because
cerberus already had a field called `model_config`, the Python attribute is renamed
to `model_config_` with an alias of `"model_config"`.

### What this means

- In **Python code**, use `cerberus_config.model_config_` (with trailing underscore).
- In **YAML/dict serialization**, the key is still `"model_config"` (no underscore).
- When **constructing** `CerberusConfig` from a dict, use the key `"model_config"`.

### Migration

```python
# Accessing the model config from CerberusConfig
model_cfg = cerberus_config.model_config_     # Python attribute (note underscore)

# Constructing CerberusConfig from a dict (e.g., from YAML)
cerberus_config = CerberusConfig(
    train_config=...,
    genome_config=...,
    data_config=...,
    sampler_config=...,
    model_config=model_config,  # dict key / alias (no underscore)
)

# Serialization round-trips correctly
data = cerberus_config.model_dump(mode="json")
assert "model_config" in data  # key is "model_config" in the dict
```

---

## 8. New dependency: pydantic>=2.0

`pydantic>=2.0` has been added to `pyproject.toml` as a runtime dependency. This
pulls in `pydantic-core` (compiled Rust extension).

### Action required

```bash
pip install --upgrade cerberus
# or, for development:
pip install -e ".[dev]"
```

If you are in a restricted HPC environment, ensure that `pydantic>=2.0` and
`pydantic-core` can be installed. Pre-built wheels are available for all major
platforms on PyPI.

---

## 9. Serialization changes

All config serialization now goes through Pydantic's `model_dump`.

### Migration

**Saving configs to JSON:**

```python
# Before
import json
payload = {
    "model_config": dict(model_config),
    "data_config": dict(data_config),
}
json.dump(payload, f, indent=2, default=str)

# After
import json
payload = {
    "model_config": model_config.model_dump(mode="json"),
    "data_config": data_config.model_dump(mode="json"),
}
json.dump(payload, f, indent=2)
# No default=str needed -- model_dump(mode="json") converts Path to str.
```

**Loading configs from dicts:**

```python
# Before (TypedDict — just a dict, no validation)
model_config: ModelConfig = raw_dict["model_config"]

# After (Pydantic — validates and constructs a frozen model)
model_config = ModelConfig.model_validate(raw_dict["model_config"])
# or equivalently:
model_config = ModelConfig(**raw_dict["model_config"])
```

---

## 10. Existing hparams.yaml backward compatibility

`parse_hparams_config` handles legacy YAML files produced before the Pydantic
migration. The following transformations are applied automatically:

| Legacy pattern | Automatic fix |
|---|---|
| Missing `pretrained` field in `model_config` | Defaults to `[]` with a deprecation warning |
| `count_pseudocount` in `data_config` | Migrated to `model_config.count_pseudocount` (raw x target_scale) with a warning |
| Extra Lightning-injected keys at top level | Stripped before validation |

No manual editing of old `hparams.yaml` files is needed for **inference**. However,
the deprecation warnings indicate that retraining the model will produce an updated
YAML file that no longer requires these shims.

---

## Quick reference: import changes

```python
# New imports you may need:
from cerberus.config import (
    FoldArgs,
    PeakSamplerArgs,
    NegativePeakSamplerArgs,
    IntervalSamplerArgs,
    SlidingWindowSamplerArgs,
    RandomSamplerArgs,
    ComplexityMatchedSamplerArgs,
    PretrainedConfig,
)

# Deleted imports (will raise ImportError):
# from cerberus.config import validate_genome_config
# from cerberus.config import validate_data_config
# from cerberus.config import validate_sampler_config
# from cerberus.config import validate_train_config
# from cerberus.config import validate_model_config
# from cerberus.config import validate_data_and_sampler_compatibility
# from cerberus.config import validate_data_and_model_compatibility
# from cerberus.config import _sanitize_config
# from cerberus.config import propagate_pseudocount
```

---

## Quick reference: full before/after for a training script

```python
# ============================================================
# BEFORE (TypedDict era)
# ============================================================
from cerberus.config import (
    GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig,
    validate_genome_config, validate_data_config, validate_sampler_config,
    validate_data_and_sampler_compatibility, propagate_pseudocount,
    _sanitize_config,
)

data_config: DataConfig = {
    "inputs": {},
    "targets": {"signal": "/path/to/signal.bw"},
    "input_len": 2114,
    "output_len": 1000,
    "max_jitter": 256,
    "output_bin_size": 1,
    "encoding": "ACGT",
    "log_transform": False,
    "reverse_complement": True,
    "use_sequence": True,
    "target_scale": 10.0,
    "count_pseudocount": 150.0,
}

sampler_config: SamplerConfig = {
    "sampler_type": "peak",
    "padded_size": 2626,
    "sampler_args": {
        "intervals_path": "/path/to/peaks.bed",
        "background_ratio": 1.0,
    },
}

model_config: ModelConfig = {
    "name": "BPNet",
    "model_cls": "cerberus.models.bpnet.BPNet",
    "loss_cls": "cerberus.models.bpnet.BPNetLoss",
    "loss_args": {"alpha": 1.0},
    "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
    "metrics_args": {},
    "model_args": {...},
}

validate_data_config(data_config)
validate_sampler_config(sampler_config)
validate_data_and_sampler_compatibility(data_config, sampler_config)
propagate_pseudocount(data_config, model_config)

# Reading config
input_len = data_config["input_len"]
intervals = sampler_config["sampler_args"]["intervals_path"]

# Serialization
sanitized = _sanitize_config({"data_config": data_config})
json.dump(sanitized, f, indent=2, default=str)


# ============================================================
# AFTER (Pydantic era)
# ============================================================
from cerberus.config import (
    GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig,
    PeakSamplerArgs, FoldArgs,
)

raw_pseudocount = 150.0
target_scale = 10.0

data_config = DataConfig(
    inputs={},
    targets={"signal": "/path/to/signal.bw"},
    input_len=2114,
    output_len=1000,
    max_jitter=256,
    output_bin_size=1,
    encoding="ACGT",
    log_transform=False,
    reverse_complement=True,
    use_sequence=True,
    target_scale=target_scale,
)

sampler_config = SamplerConfig(
    sampler_type="peak",
    padded_size=2626,
    sampler_args=PeakSamplerArgs(
        intervals_path="/path/to/peaks.bed",
        background_ratio=1.0,
    ),
)

model_config = ModelConfig(
    name="BPNet",
    model_cls="cerberus.models.bpnet.BPNet",
    loss_cls="cerberus.models.bpnet.BPNetLoss",
    loss_args={"alpha": 1.0},
    metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
    metrics_args={},
    model_args={...},
    count_pseudocount=raw_pseudocount * target_scale,
)

# Validation happens automatically at construction.
# Cross-validation happens when building CerberusConfig.

# Reading config
input_len = data_config.input_len
intervals = sampler_config.sampler_args.intervals_path

# Serialization
payload = {"data_config": data_config.model_dump(mode="json")}
json.dump(payload, f, indent=2)
```

---

## Verification Status

All breaking changes were verified end-to-end:

- **1473 unit tests pass** (0 failures, 24 skipped)
- **76 new Pydantic regression tests** in `test_pydantic_config.py`
- **14/14 training examples pass on A100 GPU** (BPNet, ASAP, Gopher, Pomeranian,
  BiasNet, Dalmatian across ChIP-seq, ATAC-seq, and scATAC-seq datasets)
- **Pyright**: 4 errors (all pre-existing, none from migration)
- **Legacy hparams.yaml**: auto-migrated by `parse_hparams_config` with deprecation warning
