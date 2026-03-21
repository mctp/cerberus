# Pydantic Migration Gotchas — Detailed Technical Reference

Each gotcha includes: the current code with file/line reference, what breaks and why,
and the recommended fix with a code snippet.

---

## 1. Bracket Access Does Not Work on BaseModel

**Current** (everywhere in `src/cerberus/`):
```python
input_len = data_config["input_len"]        # ~250 sites across 14 files
scheduler_args = self.train_config["scheduler_args"]  # module.py:105
```

**What breaks**: Pydantic `BaseModel` does not implement `__getitem__`. Every
`config["key"]` raises `TypeError`.

**Fix**: Migrate to attribute access.
```python
input_len = data_config.input_len
scheduler_args = self.train_config.scheduler_args
```

**Scope**: ~250 bracket-access sites across `module.py`, `train.py`, `dataset.py`,
`datamodule.py`, `transform.py`, `samplers.py`, `predict_bigwig.py`, `predict_misc.py`,
`pretrained.py`, `model_ensemble.py`, `config.py`, `output.py`.

---

## 2. Dict Spread `{**config}` Produces a Plain Dict

**Current** (`config.py:887`, `train.py:101`):
```python
# config.py:887
return {**model_config, "loss_args": loss_args, "metrics_args": metrics_args}

# train.py:101
return {**model_config, "loss_args": resolved_loss_args}
```

**What breaks**: `{**pydantic_model}` iterates key-value pairs and produces a plain
`dict`, not a model instance. The result loses type safety and frozen guarantees.

**Fix**: Use `model_copy(update=...)`.
```python
# config.py — but propagate_pseudocount is deleted by reparameterization

# train.py:101
return model_config.model_copy(update={"loss_args": resolved_loss_args})
```

---

## 3. `.copy()` + Bracket Assignment Breaks

**Current** (`train.py:427-430`):
```python
genome_config = genome_config.copy()
genome_config["fold_args"] = genome_config["fold_args"].copy()
genome_config["fold_args"]["test_fold"] = test_fold
genome_config["fold_args"]["val_fold"] = val_fold
```

**What breaks**: `frozen=True` means:
- `.copy()` does not exist (it's `model_copy()`)
- `model["key"] = val` raises `TypeError` (no `__setitem__`)
- `model.key = val` raises `ValidationError` (frozen)

**Fix**: Single `model_copy` with nested update.
```python
genome_config = genome_config.model_copy(update={
    "fold_args": genome_config.fold_args.model_copy(update={
        "test_fold": test_fold,
        "val_fold": val_fold,
    })
})
```

---

## 4. `frozen=True` Blocks All Mutation

**Every mutation site in the codebase** must be converted. Complete list:

| Location | Current Pattern | Fix |
|----------|----------------|-----|
| `train.py:427-430` | `.copy()` + bracket assign | `model_copy(update=...)` (see gotcha 3) |
| `train.py:101` | `{**model_config, "loss_args": ...}` | `model_copy(update=...)` (see gotcha 2) |
| `predict_misc.py:109-114` | `{**sampler_config, ...}` | `model_copy(update=...)` (see gotcha 19) |
| `model_ensemble.py:62-66` | `self.cerberus_config["model_config"] = ...` | `model_copy(update=...)` (see gotcha 14) |
| `config.py:554,562` | `setdefault()` in validator | `Field(default=None)` (see gotcha 7) |
| `config.py:883-886` | spread + `setdefault()` in propagate | Deleted by reparameterization (see gotcha 8) |

---

## 5. Sampler Args Discriminated Union Design

**Current** (`config.py:84-86`):
```python
class SamplerConfig(TypedDict):
    sampler_type: str
    padded_size: int
    sampler_args: dict[str, Any]  # untyped bag, schema depends on sampler_type
```

**Challenge**: Pydantic V2 discriminated unions require a discriminator field on the
sub-model itself, but the discriminator (`sampler_type`) lives on the parent
(`SamplerConfig`), not inside `sampler_args`.

**Recommended approach**: Use a `@model_validator(mode="before")` on `SamplerConfig`
to route to the correct args type:

```python
from typing import Annotated, Union
from pydantic import Discriminator

SamplerArgsUnion = Union[
    IntervalSamplerArgs, SlidingWindowSamplerArgs, RandomSamplerArgs,
    PeakSamplerArgs, NegativePeakSamplerArgs, ComplexityMatchedSamplerArgs,
]

class SamplerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sampler_type: str
    padded_size: int = Field(gt=0)
    sampler_args: SamplerArgsUnion

    @model_validator(mode="before")
    @classmethod
    def resolve_sampler_args(cls, data):
        """Route sampler_args to the correct typed model based on sampler_type."""
        if isinstance(data, dict):
            sampler_type = data.get("sampler_type")
            args = data.get("sampler_args", {})
            type_map = {
                "interval": IntervalSamplerArgs,
                "sliding_window": SlidingWindowSamplerArgs,
                "random": RandomSamplerArgs,
                "peak": PeakSamplerArgs,
                "negative_peak": NegativePeakSamplerArgs,
                "complexity_matched": ComplexityMatchedSamplerArgs,
            }
            if sampler_type in type_map and isinstance(args, dict):
                data["sampler_args"] = type_map[sampler_type](**args)
        return data
```

**Recursive nesting**: `ComplexityMatchedSamplerArgs` contains
`target_sampler: "SamplerConfig"` and `candidate_sampler: "SamplerConfig"`.
Pydantic V2 handles recursive models natively but requires
`SamplerConfig.model_rebuild()` after all classes are defined.

---

## 6. `FoldArgs` Typing — Optional Fields

**Current** (`config.py:37-41`):
```python
# fold_args: Arguments for the folding strategy.
#            For 'chrom_partition', required keys: 'k' (int),
#            'test_fold' (int), 'val_fold' (int).
#            test_fold and val_fold can be omitted if passed directly
#            to CerberusDataModule or train_single.
```

**Gotcha**: `test_fold` and `val_fold` are conditionally required — present in the
final config but sometimes omitted at construction time and injected later (by
`train_single` in `train.py`).

**Fix**: Use `int | None` with `Field(default=None)`:
```python
class FoldArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(ge=0)
    test_fold: int | None = Field(default=None, ge=0)
    val_fold: int | None = Field(default=None, ge=0)
```

The fold override in `train.py:427-430` becomes:
```python
genome_config = genome_config.model_copy(update={
    "fold_args": genome_config.fold_args.model_copy(update={
        "test_fold": test_fold, "val_fold": val_fold
    })
})
```

---

## 7. `setdefault()` in Validators → `Field(default=...)`

**Current** (`config.py:550-554`):
```python
# TODO: setdefault in a validate_ function is wrong — validators should
# not mutate config. Defaults belong in a dedicated normalize/parse step
# or at the call site. Keeping this temporarily so create_sampler can use
# bracket access. Remove once a proper config normalization layer exists.
config["sampler_args"].setdefault("complexity_center_size", None)
```

**Fix**: The default moves into the typed model field:
```python
class PeakSamplerArgs(BaseModel):
    intervals_path: Path
    background_ratio: float = Field(default=1.0, gt=0)
    complexity_center_size: int | None = None  # was setdefault(None)

class NegativePeakSamplerArgs(BaseModel):
    intervals_path: Path
    background_ratio: float = Field(default=1.0, gt=0)
    complexity_center_size: int | None = None
```

This resolves the acknowledged TODO — the default is declarative and lives in the
schema definition, not in a validator.

---

## 8. Pseudocount Reparameterization

**Current** (`config.py:849-887`):
```python
def propagate_pseudocount(data_config, model_config):
    loss_cls = import_class(model_config["loss_cls"])
    raw_pseudocount = data_config["count_pseudocount"]
    scaled_pseudocount = raw_pseudocount * data_config["target_scale"]
    loss_args = {**model_config["loss_args"]}
    loss_args.setdefault("count_pseudocount", scaled_pseudocount)
    metrics_args = {**model_config["metrics_args"]}
    metrics_args.setdefault("count_pseudocount", scaled_pseudocount)
    metrics_args.setdefault("log_counts_include_pseudocount", loss_cls.uses_count_pseudocount)
    return {**model_config, "loss_args": loss_args, "metrics_args": metrics_args}
```

**What changes**: This function is **deleted entirely**. `count_pseudocount` becomes a
first-class `ModelConfig` field in scaled units.

**Before** (dual source of truth):
```
data_config.count_pseudocount = 150.0          ← raw units
  ↓ propagate_pseudocount() mutates model_config
loss_args["count_pseudocount"] = 150.0          ← scaled copy
metrics_args["count_pseudocount"] = 150.0       ← another copy
metrics_args["log_counts_include_pseudocount"]  ← derived, stored as config
```

**After** (single source of truth):
```
model_config.count_pseudocount = 150.0   ← single field, scaled units
  ↓ injected at instantiation time only
loss_cls(count_pseudocount=150.0, **loss_args)
metrics_cls(count_pseudocount=150.0, log_counts_include_pseudocount=..., **metrics_args)
```

**Files affected**:
- `config.py` — delete `propagate_pseudocount()`, remove `count_pseudocount` from `DataConfig`
- `module.py:333` — remove the `propagate_pseudocount()` call from `instantiate()`
- `module.py:353-379` — update `instantiate_metrics_and_loss()` to inject pseudocount
- `config.py:816-846` — update `get_log_count_params()` to read `model_config.count_pseudocount`
- `predict_bigwig.py:55` — update to read from `model_config.count_pseudocount` directly
- `predict_misc.py:168` — same
- All `tools/train_*.py` — compute `raw_pseudocount * target_scale` at config construction
- `tests/test_propagate_pseudocount.py` — rewrite or delete

---

## 9. `log_counts_include_pseudocount` No Longer Stored in Config

**Current**: `propagate_pseudocount()` injects `log_counts_include_pseudocount` (a boolean
derived from `loss_cls.uses_count_pseudocount`) into `metrics_args`. This value is then
stored in `hparams.yaml` as if it were user-specified config.

**After**: This value is derived at instantiation time from the loss class:
```python
# In instantiate_metrics_and_loss():
loss_cls = import_class(model_config.loss_cls)
metrics_args = {
    **model_config.metrics_args,
    "log_counts_include_pseudocount": loss_cls.uses_count_pseudocount,
}
```

**Files affected**:
- `predict_misc.py:168-177` — currently calls `get_log_count_params()` which returns
  this value. The function still works (it imports the loss class).
- `predict_bigwig.py:55` — same pattern.

No change needed at these consumption sites — `get_log_count_params()` already derives
the value from the loss class.

---

## 10. Path Resolution via `ValidationInfo.context`

**Current** (`config.py:220-268`):
```python
def _resolve_path(path, search_paths=None):
    # Tries path as-is, then relative to each search_path, then suffix matching
    ...

def _validate_path(path, description, check_exists=True, search_paths=None):
    p = Path(path)
    if check_exists:
        resolved = _resolve_path(p, search_paths)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(...)
    return p
```

**Challenge**: `search_paths` is runtime context (depends on where `hparams.yaml` lives),
not part of the config schema. Pydantic field validators don't have access to external
state by default.

**Fix**: Pass `search_paths` via Pydantic V2's `ValidationInfo.context`:
```python
class GenomeConfig(BaseModel):
    fasta_path: Path

    @field_validator("fasta_path", mode="before")
    @classmethod
    def resolve_fasta(cls, v, info: ValidationInfo):
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_path(v, "Genome file", search_paths=search_paths)
```

Callers must pass context:
```python
CerberusConfig.model_validate(data, context={"search_paths": [hparams_dir]})
```

**Important**: `_resolve_path()` and `_validate_path()` remain as utility functions.
They are not eliminated — they're called from within `@field_validator` methods.

---

## 11. Lightning `save_hyperparameters` Compatibility

**Current** (`module.py:67-73`):
```python
self.save_hyperparameters({
    "train_config": _sanitize_config(train_config),
    "genome_config": _sanitize_config(genome_config),
    "data_config": _sanitize_config(data_config),
    "sampler_config": _sanitize_config(sampler_config),
    "model_config": _sanitize_config(model_config),
})
```

**Fix**: Replace `_sanitize_config()` with `model_dump(mode="json")`:
```python
self.save_hyperparameters({
    "train_config": train_config.model_dump(mode="json"),
    "genome_config": genome_config.model_dump(mode="json") if genome_config else None,
    "data_config": data_config.model_dump(mode="json"),
    "sampler_config": sampler_config.model_dump(mode="json") if sampler_config else None,
    "model_config": model_config.model_dump(mode="json"),
})
```

`model_dump(mode="json")` converts `Path` → `str` automatically. The `_sanitize_config()`
function can be deleted.

**Gotcha**: `genome_config` and `sampler_config` can be `None` (they're optional in
`CerberusModule.__init__`). Guard with ternary.

---

## 12. `json.dump` with `default=str`

**Current** (`train.py:140`):
```python
json.dump(payload, f, indent=2, default=str)
```

**What breaks**: If `payload` contains Pydantic models, `json.dump` with `default=str`
produces `"ModelConfig(name='BPNet', ...)"` string representations instead of dicts.

**Fix**: Ensure payload contains dicts, not models:
```python
payload = {
    "model_config": model_config.model_dump(mode="json"),
    "data_config": data_config.model_dump(mode="json"),
    "train_config": train_config.model_dump(mode="json"),
}
json.dump(payload, f, indent=2)
```

The `default=str` fallback is no longer needed since `model_dump(mode="json")` produces
a pure Python dict tree.

---

## 13. Test `cast()` Pattern Migration

**Current** (218 occurrences across 45 test files):
```python
config = cast(SamplerConfig, {
    "sampler_type": "random",
    "padded_size": 1000,
    "sampler_args": {"num_intervals": 100}
})
```

**After**:
```python
config = SamplerConfig(
    sampler_type="random",
    padded_size=1000,
    sampler_args=RandomSamplerArgs(num_intervals=100),
)
```

**Gotcha — intentionally invalid configs**: Several tests pass incomplete or invalid
configs to test error handling (e.g., `test_config_validation.py`). These must use
`pytest.raises(ValidationError)` instead of the old `pytest.raises(ValueError/TypeError)`:

```python
# Before:
config = cast(SamplerConfig, {"sampler_type": "interval", "padded_size": 1000, "sampler_args": {}})
with pytest.raises(ValueError, match="missing required keys"):
    validate_sampler_config(config)

# After:
with pytest.raises(ValidationError):
    SamplerConfig(sampler_type="interval", padded_size=1000, sampler_args={})
```

**Gotcha — extra keys**: Tests like `test_datamodule.py` that pass mock extra keys
(e.g., `cast(SamplerConfig, {"mock": "sampler", ...})`) will fail with `extra="forbid"`.
These must be cleaned up.

**Volume**: 218 occurrences across 45 files. Most are mechanical find-and-replace.
Files with highest counts: `test_config_validation.py` (26), `test_train_wrapper.py` (20),
`test_train_coverage.py` (14), `test_datamodule.py` (12), `test_prepare_data.py` (11).

---

## 14. `model_ensemble.py` Bracket Assignment

**Current** (`model_ensemble.py:61-66`):
```python
if model_config is not None:
    self.cerberus_config["model_config"] = model_config
if data_config is not None:
    self.cerberus_config["data_config"] = data_config
if genome_config is not None:
    self.cerberus_config["genome_config"] = genome_config
```

**What breaks**: `frozen=True` + no `__setitem__`.

**Fix**:
```python
overrides = {}
if model_config is not None:
    overrides["model_config"] = model_config
if data_config is not None:
    overrides["data_config"] = data_config
if genome_config is not None:
    overrides["genome_config"] = genome_config
if overrides:
    self.cerberus_config = self.cerberus_config.model_copy(update=overrides)
```

---

## 15. Validator Idempotency

**Current**: Validators are called at multiple sites. `validate_model_config` is called
in both `instantiate()` (module.py:327) and `instantiate_model()` (module.py:264). This
is documented as "idempotent" for dicts.

**With Pydantic**: `ModelConfig.model_validate(already_a_model)` returns the same instance
if the input is already a `ModelConfig` (in lax mode). This is effectively free.

**Recommendation**: Consumer functions should type-hint `ModelConfig` (not `ModelConfig | dict`).
The validation happens at construction time (in `parse_hparams_config` or tool scripts),
not at every consumer entry point. Remove the redundant `validate_*` calls from consumers.

---

## 16. `int` vs `float` Coercion

**Current** (`config.py:453`):
```python
if not isinstance(config["target_scale"], float) or config["target_scale"] <= 0:
    raise ValueError("target_scale must be a positive number")
```

YAML `target_scale: 1` (parsed as `int(1)`) currently **raises** because
`isinstance(1, float)` is `False`.

**With Pydantic**: Lax mode (default) silently coerces `int(1)` → `float(1.0)`.
YAML files with `target_scale: 1` will start working.

**Impact**: This is almost certainly desirable behavior — it's a UX improvement. But
tests that assert specific `TypeError`/`ValueError` for integer inputs to float fields
will need updating.

**Affected fields**: `target_scale`, `learning_rate`, `weight_decay`, `adam_eps`,
`gradient_clip_val`, `count_pseudocount`, `background_ratio`.

---

## 17. `extra="forbid"` vs Lightning Extras

**Current** (`config.py:934`):
```python
# We allow extra keys (like other PL hparams), but ensure we have ours
```

`parse_hparams_config` reads a YAML file that may contain Lightning-injected extra keys
beyond the 5 config sections.

**With Pydantic**: `extra="forbid"` on `CerberusConfig` would reject these extra keys.

**Fix**: Extract only the known keys before validating:
```python
def parse_hparams_config(path, search_paths=None):
    data = yaml.safe_load(f)
    known_keys = {"train_config", "genome_config", "data_config", "sampler_config", "model_config"}
    config_data = {k: data[k] for k in known_keys}
    return CerberusConfig.model_validate(config_data, context={"search_paths": search_paths})
```

---

## 18. `PretrainedConfig` Backfill Shim

**Current** (`config.py:948-955`):
```python
if "pretrained" not in raw_model_config:
    logger.warning("hparams.yaml ... missing 'pretrained' field ...")
    raw_model_config["pretrained"] = []
```

**Fix**: Use `Field(default_factory=list)` on `ModelConfig.pretrained`:
```python
class ModelConfig(BaseModel):
    pretrained: list[PretrainedConfig] = Field(default_factory=list)
```

Pydantic handles the missing field automatically. The warning can move to a
`@field_validator("pretrained", mode="before")` if the deprecation notice is still
desired.

---

## 19. `predict_misc.py` Spread Override

**Current** (`predict_misc.py:109-114`):
```python
sampler_args = {**sampler_config["sampler_args"]}
full_sampler_config = {
    **sampler_config,
    "sampler_args": sampler_args,
    "padded_size": data_config["input_len"],
}
```

**What breaks**: `{**pydantic_model}` produces a dict, and the result is passed to
`create_sampler(config, ...)` which would need a model.

**Fix**:
```python
full_sampler_config = sampler_config.model_copy(update={
    "padded_size": data_config.input_len,
})
```

Note: the inner `sampler_args` copy is unnecessary with frozen models — `model_copy`
creates a new instance and the inner `sampler_args` (itself a frozen model) is safely
shared.

---

## 20. `propagate_pseudocount` Elimination — Migration Path

**Current call site** (`module.py:329-333`):
```python
# Propagate scaled count_pseudocount from data_config into loss_args and
# metrics_args. This is the single call site.
model_config = propagate_pseudocount(data_config, model_config)
```

**After**: This line is simply deleted. The `instantiate_metrics_and_loss()` function
handles injection (see gotcha 8). The `model_config` passed to `instantiate()` already
has `count_pseudocount` set correctly as a first-class field.

**Deep copy concern**: The old code created spread copies of `loss_args` and
`metrics_args` to avoid mutating the original. With frozen models, this concern
disappears — there's nothing to mutate.

---

## 21. Performance

Pydantic model construction is ~10-100x slower than plain dict construction due to
type validation, coercion, and field processing.

**For config parsing** (once at startup): negligible. Even 100x slower on a ~5μs
dict construction is still <1ms.

**For test suites**: Tests that construct hundreds of config objects in parametrized
fixtures may see slower test startup. The 218 `cast()` → `Config()` migrations add
real validation overhead per construction.

**Mitigation**: Use `model_construct()` (skips validation) in test fixtures where
you're building known-good configs for non-validation tests:
```python
# For tests that don't test validation:
config = SamplerConfig.model_construct(
    sampler_type="random", padded_size=1000,
    sampler_args=RandomSamplerArgs.model_construct(num_intervals=100),
)
```

---

## 22. `**model_args` Unpacking at Call Sites

**Current** (`module.py:279-283`):
```python
model_args = model_config["model_args"]
model = model_cls(
    input_len=input_len,
    output_len=output_len,
    output_bin_size=output_bin_size,
    **model_args
)
```

**No change needed**: `model_args` is typed as `dict[str, Any]` on `ModelConfig`.
Accessing `model_config.model_args` returns a plain Python dict. Unpacking `**dict`
works as before.

Same applies to:
- `module.py:112`: `**scheduler_args` (scheduler_args is `dict[str, Any]`)
- `module.py:369`: `metrics_cls(**metrics_args)` (metrics_args is `dict[str, Any]`)
- `module.py:373`: `loss_cls(**loss_args)` (loss_args is `dict[str, Any]`)

---

## 23. Recursive SamplerConfig in `complexity_matched`

**Current** (`config.py:70-75`):
```python
# 'complexity_matched':
#     - target_sampler: Configuration for the target sampler.
#     - candidate_sampler: Configuration for the candidate sampler.
```

**Pydantic design**:
```python
class ComplexityMatchedSamplerArgs(BaseModel):
    target_sampler: "SamplerConfig"      # forward reference
    candidate_sampler: "SamplerConfig"   # forward reference
    bins: int = Field(gt=0)
    candidate_ratio: float = Field(gt=0)
    metrics: list[str]

# After all classes defined:
SamplerConfig.model_rebuild()
```

**Gotcha**: The `@model_validator(mode="before")` on `SamplerConfig` (see gotcha 5)
that routes `sampler_args` to the correct typed model handles the recursion naturally —
when Pydantic validates `target_sampler`, it triggers `SamplerConfig`'s validator, which
routes the nested `sampler_args` to the appropriate type.

**Testing**: Ensure round-trip `model_validate` → `model_dump` works for nested
complexity_matched configs.

---

## 24. Backward Compatibility with Existing `hparams.yaml`

**Problem**: Existing saved `hparams.yaml` files (from trained models) contain:
- `data_config.count_pseudocount` (removed in reparameterization)
- `loss_args.count_pseudocount` (now redundant — value lives on `model_config.count_pseudocount`)
- `metrics_args.count_pseudocount` (same)
- `metrics_args.log_counts_include_pseudocount` (no longer stored)

**Options**:

**A. Clean break** (recommended): Bump to v1.0.0. Old hparams files require re-training
or a one-time migration script.

**B. Backward-compat validator**: Add a `@model_validator(mode="before")` on
`CerberusConfig` that detects old-format YAML and transforms it:
```python
@model_validator(mode="before")
@classmethod
def migrate_legacy_pseudocount(cls, data):
    if isinstance(data, dict):
        dc = data.get("data_config", {})
        mc = data.get("model_config", {})
        # If old-style: data_config has count_pseudocount but model_config doesn't
        if "count_pseudocount" in dc and "count_pseudocount" not in mc:
            target_scale = dc.get("target_scale", 1.0)
            mc["count_pseudocount"] = dc["count_pseudocount"] * target_scale
            logger.warning("Migrated legacy count_pseudocount from data_config to model_config")
    return data
```

**C. Migration script**: A standalone `tools/migrate_hparams.py` that reads old YAML,
transforms, and writes new YAML. Run once per trained model.

Recommendation: **Option B** for `parse_hparams_config` (load-time migration with
warning), combined with **Option C** for bulk migration of existing checkpoints.
