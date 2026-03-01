# Validation Call Graph

Analysis of where each `validate_*` function is called and whether the calls are redundant or serve distinct code paths.

---

## Validator Functions

| Validator | Defined in |
|-----------|------------|
| `validate_train_config` | `config.py:554` |
| `validate_genome_config` | `config.py:258` |
| `validate_data_config` | `config.py:356` |
| `validate_sampler_config` | `config.py:450` |
| `validate_model_config` | `config.py:643` |
| `validate_data_and_sampler_compatibility` | `config.py:524` |

---

## Call Sites (Complete)

### `validate_genome_config`

| # | File:Line | Called From | Entry Point |
|---|-----------|-------------|-------------|
| 1 | `config.py:841` | `parse_hparams_config()` | `ModelEnsemble.__init__()` |
| 2 | `datamodule.py:56` | `CerberusDataModule.__init__()` | `train_single()` |
| 3 | `dataset.py:92` | `CerberusDataset.__init__()` | `CerberusDataModule.setup()` -> `CerberusDataset()` |
| 4 | `module.py:348` | `instantiate()` | `_train()` -> `instantiate()` |

### `validate_data_config`

| # | File:Line | Called From | Entry Point |
|---|-----------|-------------|-------------|
| 1 | `config.py:842` | `parse_hparams_config()` | `ModelEnsemble.__init__()` |
| 2 | `datamodule.py:57` | `CerberusDataModule.__init__()` | `train_single()` |
| 3 | `dataset.py:93` | `CerberusDataset.__init__()` | `CerberusDataModule.setup()` -> `CerberusDataset()` |
| 4 | `module.py:293` | `instantiate_model()` | `_train()` -> `instantiate()` -> `instantiate_model()` |

### `validate_sampler_config`

| # | File:Line | Called From | Entry Point |
|---|-----------|-------------|-------------|
| 1 | `config.py:843` | `parse_hparams_config()` | `ModelEnsemble.__init__()` |
| 2 | `datamodule.py:58` | `CerberusDataModule.__init__()` | `train_single()` |
| 3 | `dataset.py:99` | `CerberusDataset.__init__()` | conditional on `sampler_config is not None` |
| 4 | `module.py:350` | `instantiate()` | conditional on `sampler_config is not None` |

### `validate_model_config`

| # | File:Line | Called From | Entry Point |
|---|-----------|-------------|-------------|
| 1 | `config.py:844` | `parse_hparams_config()` | `ModelEnsemble.__init__()` |
| 2 | `module.py:292` | `instantiate_model()` | `_train()` -> `instantiate()` -> `instantiate_model()` |
| 3 | `module.py:356` | `instantiate()` | `_train()` -> `instantiate()` |

### `validate_train_config`

| # | File:Line | Called From | Entry Point |
|---|-----------|-------------|-------------|
| 1 | `config.py:840` | `parse_hparams_config()` | `ModelEnsemble.__init__()` |
| 2 | `module.py:346` | `instantiate()` | conditional on `train_config is not None` |

### `validate_data_and_sampler_compatibility`

| # | File:Line | Called From | Entry Point |
|---|-----------|-------------|-------------|
| 1 | `config.py:847` | `parse_hparams_config()` | `ModelEnsemble.__init__()` |
| 2 | `datamodule.py:59` | `CerberusDataModule.__init__()` | `train_single()` |
| 3 | `dataset.py:100` | `CerberusDataset.__init__()` | conditional on `sampler_config is not None` |

---

## Call Chains by Entry Point

### Path A: Training (`train_single` / `train_multi`)

```
train_single()
  └─ CerberusDataModule.__init__()
  │    └─ validate_genome_config          ← 1st call
  │    └─ validate_data_config            ← 1st call
  │    └─ validate_sampler_config         ← 1st call
  │    └─ validate_data_and_sampler_compatibility  ← 1st call
  │
  └─ _train()
       └─ CerberusDataModule.setup()
       │    └─ CerberusDataset.__init__()
       │         └─ validate_genome_config          ← 2nd call (REDUNDANT)
       │         └─ validate_data_config            ← 2nd call (REDUNDANT)
       │         └─ validate_sampler_config         ← 2nd call (REDUNDANT)
       │         └─ validate_data_and_sampler_compatibility  ← 2nd call (REDUNDANT)
       │
       └─ instantiate()
            └─ validate_train_config                ← 1st call (not redundant)
            └─ validate_genome_config               ← 3rd call (REDUNDANT)
            └─ validate_sampler_config              ← 3rd call (REDUNDANT)
            └─ validate_model_config                ← 1st call
            └─ instantiate_model()
                 └─ validate_model_config           ← 2nd call (REDUNDANT)
                 └─ validate_data_config            ← 3rd call (REDUNDANT)
```

**Redundancy count in training path:**
- `validate_genome_config`: called **3x** (DataModule, Dataset, instantiate)
- `validate_data_config`: called **3x** (DataModule, Dataset, instantiate_model)
- `validate_sampler_config`: called **3x** (DataModule, Dataset, instantiate)
- `validate_data_and_sampler_compatibility`: called **2x** (DataModule, Dataset)
- `validate_model_config`: called **2x** (instantiate, instantiate_model)
- `validate_train_config`: called **1x** (not redundant)

### Path B: Inference (`ModelEnsemble`)

```
ModelEnsemble.__init__()
  └─ parse_hparams_config()
  │    └─ validate_train_config                   ← 1st call
  │    └─ validate_genome_config                  ← 1st call
  │    └─ validate_data_config                    ← 1st call
  │    └─ validate_sampler_config                 ← 1st call
  │    └─ validate_model_config                   ← 1st call
  │    └─ validate_data_and_sampler_compatibility ← 1st call
  │
  └─ _ModelManager._load_model()
       └─ instantiate_model()
            └─ validate_model_config              ← 2nd call (REDUNDANT)
            └─ validate_data_config               ← 2nd call (REDUNDANT)
```

**Redundancy count in inference path:**
- `validate_model_config`: called **2x** (parse_hparams, instantiate_model)
- `validate_data_config`: called **2x** (parse_hparams, instantiate_model)
- All others: called **1x** (not redundant)

### Path C: Export Predictions (`tools/export_predictions.py`)

```
export_predictions.main()
  └─ ModelEnsemble()
  │    └─ parse_hparams_config()               ← validates ALL 6
  │
  └─ CerberusDataset()
       └─ validate_genome_config               ← 2nd call (REDUNDANT)
       └─ validate_data_config                 ← 2nd call (REDUNDANT)
```

### Path D: Direct `CerberusDataset` construction (user code)

```
CerberusDataset()
  └─ validate_genome_config                    ← 1st call (NOT redundant)
  └─ validate_data_config                      ← 1st call (NOT redundant)
  └─ validate_sampler_config                   ← 1st call (NOT redundant, if sampler_config given)
  └─ validate_data_and_sampler_compatibility   ← 1st call (NOT redundant, if sampler_config given)
```

### Path E: Direct `instantiate()` call (user code)

```
instantiate()
  └─ validate_train_config                     ← 1st call (NOT redundant)
  └─ validate_genome_config                    ← 1st call (NOT redundant)
  └─ validate_sampler_config                   ← 1st call (NOT redundant)
  └─ validate_model_config                     ← 1st call
  └─ instantiate_model()
       └─ validate_model_config                ← 2nd call (REDUNDANT)
       └─ validate_data_config                 ← 1st call (NOT redundant)
```

---

## Verdict: Which Calls Are Truly Redundant?

### Always redundant (same data, same code path)

1. **`validate_model_config` in `instantiate()` line 356** — always redundant with the call inside `instantiate_model()` line 292, which is called just above. The validated result from `instantiate_model` is discarded (it only returns the model, not the validated config).

2. **`validate_genome_config` and `validate_data_config` in `CerberusDataset.__init__()`** — redundant when called from `CerberusDataModule.setup()`, since `CerberusDataModule.__init__()` already validated these same configs. NOT redundant when `CerberusDataset` is constructed directly by users or tools.

3. **`validate_sampler_config` and `validate_data_and_sampler_compatibility` in `CerberusDataset.__init__()`** — same as above: redundant via DataModule, not redundant for direct construction.

4. **`validate_genome_config` and `validate_sampler_config` in `instantiate()`** — redundant in the training path (already validated in DataModule and Dataset), but NOT redundant when `instantiate()` is called directly.

### Never redundant (distinct code paths)

1. **`validate_train_config` in `instantiate()`** — only called here and in `parse_hparams_config()`, which is a different code path (inference only).

2. **`validate_data_config` in `instantiate_model()`** — needed because `instantiate_model()` can be called independently from `_ModelManager._load_model()`.

3. **All validators in `CerberusDataModule.__init__()`** — needed because DataModule can be constructed independently of `parse_hparams_config()`.

4. **All validators in `CerberusDataset.__init__()`** — needed because Dataset can be constructed independently of DataModule.

---

## Summary

The redundancy exists because each component (`DataModule`, `Dataset`, `instantiate`, `instantiate_model`) defensively validates its own inputs, which is correct for standalone use but wasteful when composed together in the training pipeline.

**The only unconditionally redundant call** is `validate_model_config` at `module.py:356`, which duplicates the call 8 lines earlier inside `instantiate_model()` with the exact same argument.

All other redundancies are **context-dependent**: redundant in the training pipeline (`train_single` -> `_train`) but necessary for standalone use of each component. The tradeoff is:
- **Current approach**: ~15 redundant validations per training run (including filesystem checks in `_validate_path`). Cost is small but non-zero.
- **Alternative**: Validate once at the entry point, pass a `validated=True` flag or use validated wrapper types to skip re-validation downstream. This would reduce runtime cost but add API complexity.

The most impactful fix with the least disruption would be removing the single unconditionally redundant `validate_model_config` call in `instantiate()`.
