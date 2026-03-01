# Proposal: Fixing Redundant Configuration Validation

## Problem Statement

During a standard training lifecycle, configuration dictionaries are validated redundantly up to 5 times. Because validation functions like `validate_genome_config` and `validate_data_config` inherently verify the existence of files on the disk (`path.exists()`), this causes a cascade of unnecessary I/O operations. On distributed setups utilizing network-attached storage (NFS), this causes significant latency and wasted computational overhead.

The redundant validation occurs in the following sequence:
1. `parse_hparams_config`
2. `CerberusDataModule.__init__`
3. `CerberusDataset.__init__`
4. `instantiate()`
5. `instantiate_model()`

## Proposed Fixes

### 1. Centralize Deep Validation at the Entry Point (Recommended)

Deep validation involving filesystem I/O should only happen exactly once when the configurations are first parsed or assembled. 

**Action Items:**
- Remove all calls to `validate_*_config` from:
  - `CerberusDataModule.__init__`
  - `CerberusDataset.__init__`
  - `instantiate()`
  - `instantiate_model()`
- Update the docstrings for these classes and functions to explicitly state that they expect **pre-validated** configuration dictionaries.
- Ensure that the primary API entry points (like `parse_hparams_config` or main training scripts) always run the full validation before passing the configs downstream.

### 2. Introduce a Fast Structural Validation Mode

If maintaining defensive programming deep in the call stack is a strict requirement (e.g., supporting users who instantiate `CerberusDataModule` manually without using `parse_hparams_config`), we should decouple type/structure validation from I/O validation.

**Action Items:**
- Add a parameter `check_fs: bool = True` to `validate_genome_config`, `validate_data_config`, and `_validate_path`.
- When calling these validators from inside `__init__` methods or factory functions, set `check_fs=False`.
- This ensures that if a user passes a malformed dictionary it fails immediately, but avoids repeating the expensive `os.path.exists()` checks for configurations that were already fully vetted upstream.

### 3. Long-term Refactoring: Migrate to Pydantic or Dataclasses

The root cause of this lack of trust in the call chain is the reliance on `TypedDict`, which provides static typing but zero runtime guarantees.

**Action Items:**
- Refactor `GenomeConfig`, `DataConfig`, `ModelConfig`, etc., from `TypedDict` to `pydantic.BaseModel` or Python `@dataclass`.
- Perform the validation directly within the object's `__post_init__` or Pydantic's native validators.
- Once a `GenomeConfig` object is instantiated, its structural integrity is guaranteed. Downstream components like `CerberusDataset` can simply rely on Python's type system (`isinstance(config, GenomeConfig)`) and inherently trust the data, eliminating the need for scattered validation functions.