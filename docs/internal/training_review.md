# Training Infrastructure Review

**Date:** 2025-12-23
**Scope:** `src/cerberus/train.py`, `src/cerberus/module.py`, `src/cerberus/config.py`, `src/cerberus/optim.py`

## Overview
This document summarizes the review of the training infrastructure in Cerberus. The codebase separates concerns between configuration, model definition (`LightningModule`), and execution (`Trainer`). While the structure is sound, several areas show tight coupling or hardcoded assumptions that may hinder future extensibility and robustness.

## Detailed Review

### 1. `src/cerberus/config.py`

**Strengths:**
-   **Explicit Schemas:** Use of `TypedDict` provides clear documentation of expected configuration structures.
-   **Validation Logic:** Comprehensive manual validation functions (`validate_train_config`, etc.) ensure data integrity before training starts.

**Weaknesses:**
-   **Verbosity:** Manual validation functions are verbose and repetitive.
-   **Type Enforcement:** `TypedDict` does not enforce types at runtime. Mixing `TypedDict` with runtime validation code is less efficient than using modern validation libraries.

**Recommendations:**
-   **Migrate to Pydantic:** Replacing `TypedDict` and manual validation functions with `pydantic.BaseModel` would significantly reduce code volume, provide automatic type validation/coercion, and improve serialization/deserialization capabilities.

### 2. `src/cerberus/train.py`

**Strengths:**
-   **Clean Interface:** The `train()` function provides a high-level entry point wrapping `pl.Trainer`.
-   **Defaults:** Sensible default callbacks (`ModelCheckpoint`, `EarlyStopping`) are provided.

**Weaknesses:**
-   **Type Hinting Inconsistency:** The signature `train_config: dict | TrainConfig` implies `TrainConfig` might be a class, but it is treated as a dict (subscript access). Since `TrainConfig` is a `TypedDict`, this is technically correct but can be confusing if one expects object-attribute access.
-   **Callback Flexibility:** Default callbacks are instantiated inside the function. While additional callbacks can be passed, modifying or removing the default ones (e.g., changing `ModelCheckpoint` naming convention) requires modifying the function code.

**Recommendations:**
-   Allow overriding default callbacks or configuration of their parameters via arguments.
-   Standardize on a config object (e.g., Pydantic model) to avoid ambiguity in access patterns (`config.param` vs `config['param']`).

### 3. `src/cerberus/module.py`

**Strengths:**
-   **Standard Structure:** Follows standard `pl.LightningModule` patterns.
-   **Metric Collection:** Uses `torchmetrics` for modular metric tracking.

**Weaknesses:**
-   **Tight Coupling in `configure_optimizers`:**
    ```python
    dataset_len = len(self.trainer.datamodule.train_dataset)
    ```
    This line assumes:
    1.  The module is always run with a `Trainer`.
    2.  The trainer has a `datamodule` attached.
    3.  The datamodule has a `train_dataset` attribute.
    4.  The dataset supports `__len__`.
    This fragility can break inference or simple test runs where a full datamodule isn't present.
-   **Metric Flattening:**
    ```python
    outputs_flat = outputs.detach().flatten()
    targets_flat = targets.detach().flatten()
    metric_collection.update(outputs_flat, targets_flat)
    ```
    Flattening the entire batch calculates the **global correlation** across all bins/pixels in the batch. In sequence-to-function tasks, it is often more important to track the **mean per-sequence correlation** (Pearson per sample, averaged). If the batch size is large, the global correlation might mask poor performance on individual sequences.

**Recommendations:**
-   **Decouple Optimizer Setup:** Pass `steps_per_epoch` or `total_steps` in `train_config` or calculate it safely using `self.trainer.estimated_stepping_batches` (if available and reliable).
-   **Refine Metrics:** Verify if "global correlation" is the intended metric. Consider adding a metric that computes correlation per sample and averages it, as this is often the standard for genomic signal prediction.

### 4. `src/cerberus/optim.py`

**Strengths:**
-   **Custom Logic:** Provides specialized optimizer setups (AdamW with decay filtering) and schedulers (Warmup + Cosine Annealing with Restarts).

**Weaknesses:**
-   **Hardcoded Module Types:**
    ```python
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, torch.nn.Conv1d, torch.nn.LSTM)
    ```
    This manual whitelist for weight decay is fragile. If a user introduces a new layer type (e.g., `TransformerEncoderLayer`, `GRU`, or custom layers), they might inadvertently exclude/include parameters from weight decay.
-   **Scheduler Complexity:** The custom scheduler implementation (`WarmupLinearScheduleCosineDecay` + `CosineAnnealingWithDecay`) is complex. PyTorch now offers native composable schedulers (`SequentialLR`, `LinearLR`, `CosineAnnealingWarmRestarts`) that might achieve similar results with standard, maintained code.

**Recommendations:**
-   **Generalize Decay Filtering:** Instead of hardcoding module types, consider filtering by parameter name (e.g., `weight` vs `bias`/`norm`) which is more robust across different architectures. Alternatively, allow passing `whitelist_modules` as an argument.
-   **Simplify Schedulers:** Evaluate if standard PyTorch schedulers can replace the custom implementations to reduce maintenance burden.

## Summary of Action Items

1.  **Refactor Configs:** Plan a migration to Pydantic for robust configuration management.
2.  **Fix Coupling:** Update `CerberusModule` to not rely on `self.trainer.datamodule` for scheduler constants.
3.  **Review Metrics:** Confirm metric definitions (Global vs Per-Sample).
4.  **Robustify Optimizer:** Remove hardcoded layer types in weight decay filtering.
