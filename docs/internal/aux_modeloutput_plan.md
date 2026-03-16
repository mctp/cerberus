# Add `aux` dict to ModelOutput for generic auxiliary tensor passing

## Context

Models like `PWMBiasModel` (debug/pwm_model/) need to pass training-time auxiliary tensors (e.g., regularization losses) from the model to the loss/training loop. Currently this requires:
1. A custom output subclass (`RegularizedProfileCountOutput`) just to carry one extra tensor
2. A custom loss wrapper (`RegularizedMSEMultinomialLoss`) just to add that tensor to the loss

This doesn't scale. Every new model with regularization needs 2 new files and a new output type. The fix: add a generic `aux: dict[str, Tensor] | None` to `ModelOutput` and consume `aux["reg_loss"]` in `CerberusModule._shared_step`.

## Design Decisions

1. **`aux` on ModelOutput base class** — generic dict for training-only sideband tensors
2. **`reg_loss` consumed in `_shared_step`, NOT in losses** — regularization is a model-level concern; losses stay pure data-fitting objectives
3. **`FactorizedProfileCountOutput` stays as-is** — its fields are first-class (used at inference by DalmatianLoss and interpretation tools), not training-only aux data
4. **`detach()` drops `aux`** — all existing detach methods already construct explicitly and `aux` defaults to `None`, so no changes needed in subclass detach methods

## Changes

### 1. `src/cerberus/output.py` — Add `aux` field

- Add `aux: dict[str, torch.Tensor] | None = None` to `ModelOutput` base dataclass
- No changes to subclass `detach()` methods (they already construct explicitly, `aux` defaults to `None`)
- Update `unbatch_modeloutput()`: pop `"aux"` from the asdict output before iterating (it's a dict, not a tensor — would break the unbind logic)
- `aggregate_intervals` and `aggregate_models`: **no changes needed** — existing `isinstance(val, torch.Tensor)` filters already exclude `None` and `dict` values

### 2. `src/cerberus/module.py` — Consume `aux["reg_loss"]` in `_shared_step`

After `loss = self.criterion(outputs, targets, **batch_context)`, add:

```python
if hasattr(outputs, "aux") and outputs.aux is not None:
    reg_loss = outputs.aux.get("reg_loss")
    if reg_loss is not None:
        loss = loss + reg_loss
        self.log(f"{prefix}reg_loss", reg_loss.detach(), batch_size=batch_size,
                 sync_dist=(prefix == "val_"))
```

This goes BEFORE logging `{prefix}loss` so the logged loss reflects what the optimizer minimizes.

### 3. `src/cerberus/loss.py` — No changes

Losses remain pure. This is the whole point.

### 4. Migrate debug/pwm_model/ (validates the design)

- **Delete** `debug/pwm_model/output.py` (`RegularizedProfileCountOutput` no longer needed)
- **Delete** `debug/pwm_model/loss.py` (`RegularizedMSEMultinomialLoss` no longer needed)
- **Update** `debug/pwm_model/pwm_bias.py`: return `ProfileCountOutput(..., aux={"reg_loss": reg_loss})` instead of `RegularizedProfileCountOutput(..., reg_loss=reg_loss)`
- Update training configs/scripts to use plain `MSEMultinomialLoss` instead of `RegularizedMSEMultinomialLoss`

### 5. Tests

- Test `aux` defaults to `None` on all output types
- Test `aux` is dropped after `detach()`
- Test `unbatch_modeloutput` excludes `aux`
- Test `aggregate_models` works with `aux` present
- Test `_shared_step` adds `reg_loss` from aux to total loss
- Test `_shared_step` works unchanged when `aux` is None

### 6. Docs & Changelog

- Update `docs/models.md` with `aux` convention
- Add changelog entry

## Backwards Compatibility

| Component | Impact |
|---|---|
| Existing models (BPNet, Dalmatian, Pomeranian) | None — `aux` defaults to `None` |
| Existing losses | None — never see `aux` |
| `detach()` methods | None — already construct explicitly |
| `unbatch_modeloutput` | One-line pop of `aux` key |
| `aggregate_*` functions | None — type filters already exclude non-tensors |
| isinstance checks in losses | Unaffected |
| Config/YAML | No changes — `aux` is runtime, not configured |

## Verification

1. `pytest -v tests/` — all existing tests pass
2. `npx pyright tests/ src/` — no type errors
3. Manual: run PWMBiasModel training with `aux`-based reg_loss, verify `train_reg_loss` appears in logs
