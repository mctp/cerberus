# Code Quality Audit

Comprehensive audit of `src/cerberus/` for code complexity, bugs, dead code, poor design, and other issues. Issues are grouped by priority.

---

## Critical / High Priority

### 1. BUG: Variable shadowing corrupts model cache (`model_ensemble.py:568,580`) (FIXED)

In `_load_model`, the parameter `key` (the model name, e.g. `"fold_0"`) is overwritten inside the state_dict processing loop: `key = k[6:]`. After the loop, `self.cache[key]` uses the last loop value of `key`, not the original model identifier. The cache stores models under wrong keys, so `if key in self.cache` never matches and models are reloaded every time.

### 2. BUG: `.bed.gz` detection is broken (`signal.py:137`) (FIXED)

`path.suffix.lower()` returns only the last suffix (`.gz` for `file.bed.gz`), never `.bed.gz`. The `.bed.gz` string in the suffix tuple is dead code. Furthermore, bare `.gz` files that are not BED files are incorrectly routed to `BedMaskExtractor`.

### 3. BUG: `PoissonMultinomialLoss` / `NegativeBinomialMultinomialLoss` crash with `propagate_pseudocount` (`loss.py`, `config.py`) (FIXED)

`propagate_pseudocount` always injects `count_pseudocount` into `loss_args` via `setdefault`. But only `MSEMultinomialLoss` accepts this kwarg. Using Poisson or NegativeBinomial loss classes with pseudocount propagation will crash with `TypeError` at module instantiation.

### 4. BUG: `BPNet` counts head uses uncropped feature map (`models/bpnet.py:157`) (ORIGINAL, NOT FIXED)

`BPNet.forward` pools over the full `x` tensor for counts (`x.mean(dim=-1)`) without cropping to `target_len` first. All other models (`GemiNet`, `LyraNet`, `Pomeranian`) crop before pooling. BPNet's count prediction includes flanking context excluded from the profile, creating an inconsistency that may affect training.

### 5. `BPNetLoss` silently discards caller-provided arguments (`models/bpnet.py:249-254`) (FIXED)

`BPNetLoss.__init__` pops `average_channels`, `flatten_channels`, `count_per_channel`, `log1p_targets`, `count_weight`, and `profile_weight` from `**kwargs` without warning. A caller who passes `log1p_targets=True` gets `False` silently. Configuration errors are swallowed.

### 6. Catching `BaseException` in signal extraction (`signal.py:98-104`)

`except (Exception, BaseException)` is redundant (`BaseException` already covers `Exception`) and dangerous. Can silently swallow unexpected fatal errors. The re-raise logic for `KeyboardInterrupt`/`SystemExit`/`GeneratorExit` is correct in intent but the broad catch is fragile.

### 7. Silent exception swallowing in `_accumulate_log_counts` (`module.py:192-206`) (PARTIALLY FIXED)

`except (ValueError, AttributeError, TypeError, IndexError): pass` silently swallows five exception types. Bugs in `compute_total_log_counts` or `self.criterion` will produce no scatter plot with zero indication of why.

### 8. Missing model exports (`models/__init__.py`) (FIXED)

`Pomeranian`, `PomeranianK5`, `ConvNeXtDCNN`, `BPNet1024`, all `MetricCollection` and `Loss` subclasses are not exported. The `pomeranian` module is entirely invisible from the public API.

### 9. `PGC` layers in `LyraNet` have no residual connection (`models/lyra.py:290-291`) (ORIGINAL, NOT FIXED)

PGC layers are applied sequentially without residuals: `for pgc in self.pgc_layers: x = pgc(x)`. The `PGC` class itself has no residual. Meanwhile, S4D layers in the same model use explicit residuals. Missing residual connections in 4-7 stacked PGC layers may cause gradient flow problems.

### 10. Missing required keys cause raw `KeyError` (`config.py:618,621`) (FIXED)

`adam_eps` and `gradient_clip_val` are not in `required_keys` for `validate_train_config`, so their absence produces a raw `KeyError` instead of the helpful "missing required keys" `ValueError`.

---

## Medium Priority

### 11. Massive code duplication in loss classes (`loss.py`)

6 loss classes with massive duplication. The `Coupled*` variants differ only in how `pred_log_counts` is derived. `_compute_profile_loss` is duplicated between `MSEMultinomialLoss` and `PoissonMultinomialLoss` with subtle mathematical differences (full multinomial NLL vs. cross-entropy) hidden behind identical method names.

### 12. Duplicated `MetricCollection` subclasses across 4 model files

`GemiNetMetricCollection`, `LyraNetMetricCollection`, `BPNetMetricCollection`, `PomeranianMetricCollection` are byte-for-byte identical except for the class name. Should be a single shared class.

### 13. Size-variant model subclasses are pure boilerplate (~350 lines)

`GemiNetMedium`, `GemiNetLarge`, `GemiNetExtraLarge`, `LyraNetMedium`/`Large`/`ExtraLarge`/`ExtraExtraLarge`, `BPNet1024` do nothing except change default argument values. Each is a verbatim copy of the parent's `__init__` signature. Factory functions or class methods would eliminate this.

### 14. Redundant validation across the entire call chain (`config.py`, `dataset.py`, `datamodule.py`, `module.py`, `train.py`) (PARTIALLY FIXED)

Configs are validated 3-4 times redundantly during a standard training flow:
1. `parse_hparams_config` validates everything
2. `CerberusDataModule.__init__` re-validates genome, data, sampler
3. `CerberusDataset.__init__` re-validates again
4. `instantiate()` re-validates train, genome, sampler, model, data
5. `instantiate_model()` re-validates model and data

Unclear ownership of validation responsibility. Wasted computation (includes filesystem checks).

### 15. Repetitive validation boilerplate in config.py (~400 lines)

All five `validate_*_config` functions follow an identical pattern (check dict, compute required_keys, check presence, validate types, return). Could be a generic schema-driven validator.

### 16. `propagate_pseudocount` called in two different places (`config.py:848`, `train.py:326`) (FIXED)

Called in both `parse_hparams_config` and `_train`. Since it uses `setdefault`, the second call is a no-op if the first already ran. Unclear ownership. Modifying `count_pseudocount` between the two calls silently keeps the stale value. Fixed by moving the single call into `instantiate()` where the loss and metrics are actually constructed.

### 17. `propagate_pseudocount` mutates argument in-place (`config.py:767-785`) (FIXED)

Modifies `model_config["loss_args"]` and `model_config["metrics_args"]` dicts in-place via `setdefault`. Combined with `resolve_adaptive_loss_args` (which creates a new top-level dict but shares nested dicts), mutation can affect the original `model_config` across folds. Fixed by making `propagate_pseudocount` return a new `ModelConfig` with shallow-copied `loss_args` and `metrics_args` dicts, matching the pattern used by `resolve_adaptive_loss_args`.

### 18. Duplicated `PGC` class (`lyra.py:132-175` vs `layers.py:101-206`) (ORIGINAL, NOT FIXED)

Two similarly-named PGC implementations exist: `PGC` in lyra.py operates in `(B, L, D)` format with `nn.Linear`; `PGCBlock` in layers.py operates in `(B, D, L)` format with `nn.Conv1d`. Confusing and increases maintenance burden.

### 19. `config.get()` violations of project convention (FIXED)

All `config.get()` calls replaced with bracket access. Keys `use_sequence`, `scheduler_type`, `scheduler_args`, `reload_dataloaders_every_n_epochs` made required in validators. `fold_args["test_fold"]` and `fold_args["val_fold"]` now use bracket access. All test configs updated with required keys (`test_fold`/`val_fold` in fold_args, `max_jitter` replacing `jitter`, `use_sequence` added where missing).

### 20. Mutable default arguments (systemic across all models) (FIXED)

All mutable list defaults in `__init__` signatures changed to `None` with defaults set in the method body. Affected files: `asap.py`, `bpnet.py`, `geminet.py`, `gopher.py`, `lyra.py`, `pomeranian.py`, `predict_bigwig.py`, `samplers.py`. Uses `_UNSET` sentinel in `pomeranian.py` where `None` has a separate semantic meaning (auto-compute dilations).

### 21. Broad `try/except` swallowing errors (PARTIALLY FIXED)

Multiple locations silently catch and discard exceptions. Added `logger.warning` to the silent ones:
- `datamodule.py:170-180` — `except (AttributeError, RuntimeError): pass` during resampling (FIXED — now warns)
- `train.py:142-143` — `except Exception` during config dump (already logged warning)
- `mask.py:101` — `except Exception` returns zeros (already logged debug)
- `complexity.py:186-204` — `except Exception` returns NaN (FIXED — now warns)
- `sequence.py:93-99` — `except Exception` returns 0.0 (FIXED — now warns)
- `model_ensemble.py:597-601` — `except yaml.YAMLError: pass` overwrites corrupt metadata (FIXED in #38)

### 22. Duplicated cropping logic across models and layers

The center-crop-with-error-check pattern is copy-pasted in:
- `layers.py:89-96, 173-180, 197-204, 230-237`
- `geminet.py:109-129`, `lyra.py:320-340`, `bpnet.py:131-152`, `pomeranian.py:169-186`

Should be a shared utility function.

### 23. Duplicated FASTA reading / lazy-loading patterns

- FASTA open-iterate-extract pattern duplicated between `complexity.py:183-204` and `sequence.py:89-101`
- Lazy-loading + pickle (`_load`, `__getstate__`) copy-pasted across `SignalExtractor`, `BigBedMaskExtractor`, `InMemoryBigBedMaskExtractor`
- BED file reading duplicated between `samplers.py:708-722`, `mask.py:210+`, and `exclude.py`

### 24. Dead parameters and dead code in LyraNet

- `s4_lr` parameter documented as "Ignored" but still accepted and passed through (`lyra.py:210,227,265`)
- `self.prenorm = True` hardcoded, `if not self.prenorm:` branch is unreachable (`lyra.py:237,300,312-314`)
- `return_embeddings` parameter accepted but never referenced (`lyra.py:281`)

### 25. `num_channels` parameter stored but never used in metrics (`metrics.py:42-44, 87-89`) (FIXED)

`ProfilePearsonCorrCoef` and `CountProfilePearsonCorrCoef` accept and store `self.num_channels` but never use it. Misleads users. Fixed by removing `num_channels` from both metric classes, all MetricCollection subclasses, and all callers.

### 26. Duplicated Pearson computation and log-count extraction in metrics (`metrics.py`)

- `ProfilePearsonCorrCoef` and `CountProfilePearsonCorrCoef` have identical `update`/`compute` patterns
- `LogCountsMeanSquaredError` and `LogCountsPearsonCorrCoef` have ~30-line copy-pasted count extraction logic

### 27. Truthiness checks on `Compose` objects and empty lists (`dataset.py:306,311,179`) (FIXED)

`if self.transforms:` always evaluates True (non-None object). `if transforms and deterministic_transforms:` would be False for empty lists `[]`, silently creating defaults instead of using the empty list. Fixed by using `is not None` for the init check and removing the always-true guards in `__getitem__` and `create_subset`.

### 28. `setup_logging` modifies root logger (`logging.py:14-33`) (FIXED)

A library should configure its own logger hierarchy (`logging.getLogger("cerberus")`), not the root logger. This interferes with logging configuration of the host application. Fixed by changing to `logging.getLogger("cerberus")` so only the cerberus hierarchy is configured.

### 29. `ConvNeXtDCNN` swallows unknown `**kwargs` silently (`models/asap.py:113,132`) (FIXED)

Any typo in a keyword argument (e.g., `filtr0=256`) is silently ignored. Fixed by validating kwargs keys against the known config keys before updating, raising `ValueError` for unknown arguments.

### 30. `LogCountsPearsonCorrCoef` accumulates unbounded lists (`metrics.py:261-262`)

`preds_list` and `targets_list` grow without bound throughout an epoch. For large validation sets, memory usage scales linearly.

### 31. `dataclasses.asdict` loses type information (`output.py:65,237`)

Recursively converts nested dataclasses (including `Interval`) to plain dicts. Callers expecting `Interval` objects get dicts instead.

### 32. `_process_island` only processes first channel (`predict_bigwig.py:145`) (PARTIALLY FIXED)

`values = track_data[0]` silently discards all channels except the first. Multi-channel model outputs are truncated without warning. Partially fixed by adding a `logger.warning` when multiple channels are detected. Full fix (adding a `channel` parameter) is tracked as a TODO in the code.

### 33. `n_dilated_layers` parameter ignored when `dilations` is provided (`models/pomeranian.py:58,66`) (FIXED)

When `dilations` is provided (which is the default), `n_dilated_layers` is completely ignored. User could pass `n_dilated_layers=16` and get 8 layers. Fixed by adding validation that raises `ValueError` when both are provided and disagree on length.

### 34. `Bin._bin` does not validate `method` (`transform.py:299-311`) (FIXED)

If `self.method` is not "max", "avg", or "sum", the method silently returns the tensor without pooling. No validation in `__init__`. Fixed by adding validation in `__init__` that raises `ValueError` for invalid methods.

### 35. `_forward_models` has deeply nested control flow (`model_ensemble.py:185-263`)

4-5 levels of nesting with multiple code paths. Hard to understand, test, and maintain.

### 36. Duplicated `create_human_genome_config` / `create_mouse_genome_config` (`genome.py:204-303`)

Nearly identical functions differing only in filenames, species string, and error messages.

### 37. O(n^2) lookup in `UniversalExtractor.extract` (`signal.py:185,188,191`)

`bw_keys.index(name)` is O(n) linear search within a loop over all channels. Should use a pre-computed dict.

### 38. `YAML.YAMLError` silently overwrites corrupt metadata (`model_ensemble.py:597-601`) (FIXED)

`except yaml.YAMLError: pass` silently ignores corrupt metadata files and overwrites them, losing all previously recorded fold information. Fixed by adding a `logger.warning` that alerts the user their existing fold information will be lost.

---

## Low Priority

### 39. Commented-out code in `models/asap.py:83,100,102`

Dead code artifacts from refactoring. Should be removed (lives in version control).

### 40. `r_tensor` re-created every forward pass (`loss.py:359,401`)

`torch.tensor(self.total_count, ...)` in NegativeBinomial losses should be a registered buffer.

### 41. `grn` parameter name shadows attribute in `ConvNeXtV2Block` (`layers.py:65-68`)

Constructor `bool` parameter and `nn.Module` attribute share the same name.

### 42. `DilatedResidualBlock` does not raise on `diff < 0` unlike other blocks (`layers.py:230-239`)

Inconsistent error handling compared to `ConvNeXtV2Block` and `PGCBlock`.

### 43. No `__all__` definitions in any module

Makes public API unclear. Wildcard imports export internal helpers.

### 44. `calculate_dust_score` recreates lookup table on every call (`complexity.py:67-73`)

Should be a module-level constant.

### 45. `ModelOutput.detach()` raises `NotImplementedError` but class is not abstract (`output.py:16-18`)

Should use `ABC` and `@abstractmethod` for static analysis.

### 46. `download._download_file` has no timeout, retry, or progress indication (`download.py:11-18`)

Network downloads of potentially multi-GB files with no error handling.

### 47. Unused import `torch` in `predict_bigwig.py:1`

### 48. Unused `numpy` import in `models/asap.py:1`

`np.round()` used once; could use builtin `round()`.

### 49. Unreachable dead code in `datamodule.py:174`

`world_size = self.trainer.world_size if self.trainer else 1` — already inside `if self.trainer:`, so `else 1` is unreachable.

### 50. Falsy check instead of `None` check in `_BasenjiCoreBlock` (`models/asap.py:57`)

`if not filters3:` treats `filters3=0` same as `None`. Should be `if filters3 is None:`.

### 51. `compute_total_log_counts` has convoluted branching (`output.py:289-304`)

Single vs. multi-channel handling with `isinstance` checks and different `dim` arguments. Correct but fragile.

### 52. File descriptor leaks in FASTA handling (`sequence.py:90`, `complexity.py:183`)

`pyfaidx.Fasta` opened but never explicitly closed. Can cause "too many open files" errors.

### 53. Docstring mentions nonexistent `device` parameter (`predict_bigwig.py:39`)

### 54. `samplers.py` is 1422 lines with 9 classes

Could be split into separate modules for better organization.

### 55. `hasattr` duck-typing in `dataset.py:349-353` and `samplers.py:316-332`

`hasattr(self.sampler, "get_peak_status")` bypasses type system with `# type: ignore`. `MultiSampler` probes first sub-sampler with `hasattr` checks.

---

## Summary

| Priority | Count | Top issues |
|----------|-------|------------|
| Critical/High | 10 | Cache key bug (#1), .bed.gz detection (#2), pseudocount crash (#3), BPNet uncropped counts (#4), silent kwarg suppression (#5) |
| Medium | 28 | Loss duplication (#11), redundant validation (#14), mutable defaults (#20), broad except (#21), dead code (#24) |
| Low | 17 | Commented-out code (#39), missing `__all__` (#43), minor imports (#47-48) |
