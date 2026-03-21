# Pydantic Migration — Problems Encountered

## 1. `model_config` Name Clash with Pydantic V2

**Problem**: Pydantic V2 reserves `model_config` as a class-level `ConfigDict` variable.
Our `CerberusConfig` has a field named `model_config` (of type `ModelConfig`). This is
a hard name clash — Pydantic will not allow a field with this name.

**Solution**: Use `model_config_` as the Python attribute name with `Field(alias="model_config")`
and `populate_by_name=True`. YAML serialization/deserialization uses `"model_config"` key,
but Python code accesses via `cerberus_config.model_config_`.

**Impact**: All Python code that accesses `cerberus_config["model_config"]` must become
`cerberus_config.model_config_`. This is an aesthetic regression but unavoidable with
Pydantic's naming conventions.

## 2. DataConfig: Removed `count_pseudocount` Field

**Problem**: The pseudocount reparameterization removes `count_pseudocount` from `DataConfig`.
However, existing YAML config files and many test fixtures include this field. With
`extra="forbid"`, any YAML with `count_pseudocount` in `data_config` will fail validation.

**Solution**: `parse_hparams_config` has a backward-compat validator that reads legacy
`count_pseudocount` from `data_config`, scales it, migrates to `model_config`, and strips
it before Pydantic validation. For tests, fixtures must be updated to remove the field.

## 3. Sampler Args Path Resolution

**Problem**: `IntervalSamplerArgs`, `PeakSamplerArgs`, `NegativePeakSamplerArgs` all have
`intervals_path: Path` fields that need search_paths resolution. But the `ValidationInfo.context`
is only available on the **root** model's validators, not on nested models validated
during construction.

**Solution**: Added `@field_validator("intervals_path", mode="before")` on each sampler args
model that reads `info.context`. Pydantic V2 propagates context to nested model validators
when using `model_validate()` on the root model. This works correctly.

## 4. Consumer validate_* Calls Removed

**Problem**: `dataset.py`, `datamodule.py`, and `module.py` all called `validate_*` functions
on config dicts. These functions no longer exist — validation happens at Pydantic model
construction time.

**Solution**: Removed all `validate_*` calls from consumers. Consumers now accept
already-validated Pydantic model instances. If a consumer receives a plain dict, Python's
type system will catch it (pyright). At runtime, passing a dict where a BaseModel is
expected will cause AttributeError on the first attribute access, which is a clear error.
