# Optimization and Configuration Testing Strategy

This document outlines the testing strategy for the recently implemented optimization logic, scheduler configurations, and training configuration validation in `cerberus`.

## Scope

The following components require testing:

1.  **Configuration (`src/cerberus/config.py`)**:
    *   `validate_train_config`: Ensure strict validation of optimizer names, scheduler types, and types/values of all fields.

2.  **Optimization Logic (`src/cerberus/optim.py`)**:
    *   `configure_adamw_with_decay_filtering`: Verify correct parameter grouping (decay vs. no decay) for whitelisted/blacklisted modules.
    *   `WarmupLinearScheduleCosineDecay` & `CosineAnnealingWithDecay`: Verify learning rate curves.

3.  **Module Integration (`src/cerberus/module.py`)**:
    *   `CerberusModule.configure_optimizers`: Verify that the correct optimizer and scheduler are initialized based on `train_config`.

4.  **Training Wrapper (`src/cerberus/train.py`)**:
    *   `train()`: Verify that the function correctly initializes the `pl.Trainer` with provided callbacks and configuration.

## Testing Plan & Progress

### 1. Configuration Tests (`tests/test_train_config.py`)
- [x] **Valid Config**: Test with a fully valid dictionary.
- [x] **Required Parameter**: Ensure `filter_bias_and_bn` is required.
- [x] **Invalid Optimizer**: Test with an unsupported optimizer string.
- [x] **Invalid Scheduler**: Test with an unsupported scheduler type.
- [x] **Type Checking**: Test with incorrect types for numeric fields (e.g., string for batch_size).
- [x] **Missing Keys**: Test with missing required keys.

### 2. Optimizer Unit Tests (`tests/test_optim.py`)
- [x] **Parameter Filtering**: Create a dummy model (Linear, LayerNorm, Conv1d, Bias) and verify `configure_adamw_with_decay_filtering` assigns correct weight decay groups.
- [x] **Scheduler Curve**: Step through `WarmupLinearScheduleCosineDecay` and assert LR increases then decays.
- [x] **Cosine Annealing**: Step through `CosineAnnealingWithDecay` and verify cyclical behavior.

### 3. Module Integration Tests (`tests/test_module_optim.py`, `tests/test_module_steps.py`)
- [x] **Configure Optimizers (Standard)**: Test `adamw` + `default`.
- [x] **Configure Optimizers (ASAP)**: Test `adamw_asap` + `cosine`. Mock `self.trainer.datamodule` to provide dataset length.
- [x] **Training Step**: Verify forward pass, loss calculation, and logging.
- [x] **Validation Step**: Verify forward pass, loss calculation, and logging.
- [x] **Validation Epoch End**: Verify metrics logging and reset.

### 4. Train Wrapper Tests (`tests/test_train_wrapper.py`)
- [x] **Train Function Setup**: Mock `pl.Trainer` and `fit` to verify `train()` passes correct arguments and callbacks.

## Execution
Run `pytest` on the new files as they are created.
