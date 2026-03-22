# Pydantic Migration — Problems Encountered

Problems discovered and solved during the actual implementation.

## 1. `model_config` Name Clash with Pydantic V2

**Problem**: Pydantic V2 reserves `model_config` as a class-level `ConfigDict` variable.
`CerberusConfig` has a field named `model_config` (of type `ModelConfig`). This is
a hard name clash — Pydantic will not allow a field with this name.

**Solution**: Use `model_config_` as the Python attribute name with `Field(alias="model_config")`
and `populate_by_name=True`. YAML serialization/deserialization uses `"model_config"` key,
but Python code accesses via `cerberus_config.model_config_`.

**Impact**: All Python code that accessed `cerberus_config["model_config"]` became
`cerberus_config.model_config_`. Aesthetic regression but unavoidable.

## 2. DataConfig: Removed `count_pseudocount` Field

**Problem**: The pseudocount reparameterization removed `count_pseudocount` from `DataConfig`.
Existing YAML config files and 218 test fixtures included this field. With
`extra="forbid"`, any YAML with `count_pseudocount` in `data_config` fails validation.

**Solution**: `parse_hparams_config` has a backward-compat migration that reads legacy
`count_pseudocount` from `data_config`, multiplies by `target_scale`, migrates the
scaled value to `model_config.count_pseudocount`, strips it from `data_config`, and
logs a deprecation warning. For tests, all fixtures were updated to remove the field.

## 3. Sampler Args Context Propagation (Critical Bug)

**Problem**: `SamplerConfig.resolve_sampler_args` (`@model_validator(mode="before")`)
originally constructed typed sampler args via bare `__init__()`:
```python
data["sampler_args"] = _SAMPLER_ARGS_TYPE_MAP[sampler_type](**args)
```
This did **not** propagate `ValidationInfo.context` (containing `search_paths`) to the
nested model's field validators. Result: `PeakSamplerArgs.resolve_intervals_path` received
`info.context = None`, and relative `intervals_path` from `hparams.yaml` could not be
resolved → `FileNotFoundError` when loading models via `ModelEnsemble`.

**Fix**: Changed to `model_validate(args, context=ctx)` which explicitly forwards context:
```python
ctx = info.context if info and info.context else None
data["sampler_args"] = _SAMPLER_ARGS_TYPE_MAP[sampler_type].model_validate(args, context=ctx)
```

**Key insight**: Pydantic V2 automatically propagates context to nested model **fields**
during `model_validate()`, but does NOT propagate context when you manually call
`Model(**dict)` or `Model(field=val)` inside a validator. Only `model_validate(..., context=...)`
forwards context.

Also required adding `info: ValidationInfo` as a second parameter to the
`@model_validator(mode="before")` signature.

## 4. Consumer validate_* Calls Removed

**Problem**: `dataset.py`, `datamodule.py`, and `module.py` all called `validate_*` functions
on config dicts. These functions no longer exist — validation happens at Pydantic model
construction time.

**Solution**: Removed all `validate_*` calls from consumers. Consumers now accept
already-validated Pydantic model instances. Validation happens at the boundary
(`parse_hparams_config`, tool scripts, or explicit `Config(...)` construction).

## 5. Test Mock Patches of Deleted Functions

**Problem**: ~40 test files used `@mock.patch("cerberus.datamodule.validate_data_and_sampler_compatibility")`
and similar patches for the deleted validation functions. These caused `AttributeError`
at test collection time because the target module no longer has the patched name.

**Solution**: Removed all `mock.patch` decorators targeting deleted functions, along with
their corresponding mock parameters from test function signatures. Since validation now
happens at construction time (and tests use `model_construct()` to skip it), the mocking
was no longer necessary.

**Scope**: 41 test files had mock patches removed, affecting ~150 test functions.

## 6. `model_construct()` Required for Test Fixtures with Fake Paths

**Problem**: Many tests create config fixtures with non-existent file paths (e.g.,
`fasta_path=Path("/fake/genome.fa")`). With Pydantic, `GenomeConfig(fasta_path=...)`
triggers `@field_validator("fasta_path")` which calls `_validate_path()` and raises
`FileNotFoundError`.

**Solution**: Use `GenomeConfig.model_construct(...)` in test fixtures — this creates
the model instance without running any validators. Used consistently across all 62
test files for configs that reference non-existent paths.

**Tradeoff**: `model_construct()` skips ALL validation, not just path checking. Tests
that specifically test validation use the normal `Config(...)` constructor.

## 7. Pydantic int→float Coercion Changes Error Surface

**Problem**: The old `validate_data_config` had `isinstance(config["target_scale"], float)`,
which rejected `int` values. Pydantic V2 lax mode silently coerces `int(1)` to `float(1.0)`.
Some tests that asserted `ValueError` for integer inputs to float fields needed updating.

**Solution**: Tests updated to use truly invalid types (e.g., `batch_size=[1,2,3]`) instead
of relying on the int/float distinction. The coercion is actually desirable UX — YAML
`target_scale: 1` now works instead of requiring `target_scale: 1.0`.

## 8. `model_copy(update=...)` for Frozen Model Mutations

**Problem**: Five mutation patterns in the codebase (`train.py` fold override,
`propagate_pseudocount`, `resolve_adaptive_loss_args`, `predict_misc.py` sampler override,
`model_ensemble.py` config override) all used dict spread `{**config, "key": val}` or
`.copy()` + bracket assignment. Both are illegal on frozen Pydantic models.

**Solution**: All five patterns converted to `config.model_copy(update={...})`. For nested
updates (e.g., fold_args inside genome_config), chained `model_copy` calls:
```python
genome_config = genome_config.model_copy(update={
    "fold_args": genome_config.fold_args.model_copy(update={
        "test_fold": test_fold, "val_fold": val_fold
    })
})
```

## 9. `extra="forbid"` vs Lightning Extra hparams Keys

**Problem**: `parse_hparams_config` reads YAML files that may contain extra keys injected
by Lightning (beyond the 5 config sections). `extra="forbid"` on `CerberusConfig` would
reject these.

**Solution**: Extract only the 5 known keys before passing to `CerberusConfig.model_validate()`:
```python
config_data = {k: data[k] for k in required_keys}
```

## 10. Actual Effort vs Estimate

**Estimated**: 4-5 dev days.
**Actual**: Completed in one session with extensive agent parallelization. The mechanical
migrations (bracket access, cast→constructor, mock.patch removal) were highly parallelizable.
The hardest parts were the config.py rewrite itself and the context propagation bug (#3).

**Files changed**: 158 (vs estimated ~80). The test migration was larger than anticipated
because many test files that weren't in the initial count also had config-related code.
