# ModelEnsemble Refactor: Portability & Structure Standardization

## Overview
This refactor addresses three critical areas:
1.  **Portability**: Enabling model restoration on systems with different file paths via path sanitization and runtime resolution.
2.  **Sanitization**: Removing pickled Python objects (classes) from configuration files, replacing them with string references to ensure environment independence.
3.  **Structure Standardization**: Unifying single-fold and multi-fold model directory structures and enforcing explicit metadata.

## Architectural Changes

### Unified Directory Structure
All training runs now enforce a consistent directory structure rooted in an "Experiment Directory". Whether training a single fold or multiple folds, each model instance resides in a `fold_X` subdirectory.

```
Experiment_Root/
├── ensemble_metadata.yaml  # Manifesto of available folds
├── fold_0/                 # Subdirectory for Fold 0 (even for single models)
│   ├── checkpoints/
│   │   └── ...
│   └── lightning_logs/
│       └── ...
│           └── hparams.yaml
├── fold_1/
│   └── ...
```

### Ensemble Metadata
An `ensemble_metadata.yaml` file is now mandatory at the Experiment Root.
-   **Format**: `folds: [0, 1, 2]`
-   **Role**: Serves as the source of truth for `ModelEnsemble`. Directory scanning heuristics (`_detect_multifold`) have been removed in favor of strict metadata reading.

## Implementation Details

### 1. Configuration (`src/cerberus/config.py`)

#### Path & Type Sanitization (`_sanitize_config`)
-   **Paths**: Attempts to convert absolute `Path` objects to paths relative to the current working directory. Falls back to absolute if not possible.
-   **Types**: Converts Python classes (e.g., `cerberus.models.BPNet`) to their fully qualified string names. This ensures `hparams.yaml` is pure text.

#### Path Resolution (`parse_hparams_config`)
-   Introduced `search_paths` argument.
-   Implemented `_resolve_path` and `_rebase_config_paths`. When loading a config, if a file path does not exist, the system searches for it in `search_paths` (checking both relative structure and flattened filename).

#### Strict Validation (`validate_model_config`)
-   `model_cls`, `loss_cls`, and `metrics_cls` in `ModelConfig` **must** be strings. Passing raw Python types is no longer allowed.

#### Utilities
-   `_import_class(name)`: Helper to dynamically load a class from a dotted string.
-   `update_ensemble_metadata(root, fold_id)`: Centralized logic to update the metadata file.

### 2. Entrypoints (`src/cerberus.train.py`)

#### Instantiation (`instantiate`)
-   Now expects `ModelConfig` to contain strings for class names.
-   Uses `_import_class` to resolve these strings to types before initialization.

#### Training Flow (`train_single`, `train_multi`)
-   **`train_single`**:
    -   Now treats `root_dir` as the **Experiment Root**.
    -   Automatically appends `fold_{test_fold}` to the path for the actual run directory.
    -   Calls `update_ensemble_metadata` to register the fold.
-   **`train_multi`**:
    -   Passes the Experiment Root to `train_single`, allowing it to manage subdirectories.
    -   (Simplification): No longer pre-initializes full metadata; relies on incremental updates from `train_single`.

### 3. Model Ensemble (`src/cerberus/model_ensemble.py`)

#### Strict Loading
-   **Removed**: `_detect_multifold` and fallback logic.
-   **Enforced**: `load_models_and_folds` now strictly requires `ensemble_metadata.yaml` to exist and contain a `folds` list.
-   **Validation**: Raises `FileNotFoundError` immediately if metadata or fold directories are missing.

#### Portability Support
-   `ModelEnsemble.__init__` accepts `search_paths`.
-   Passes `search_paths` to `parse_hparams_config` to enable loading on new systems.

## Breaking Changes
1.  **Configuration Objects**: Users defining `ModelConfig` manually in Python scripts **must** use string class names (e.g., `"cerberus.models.bpnet.BPNet"`) instead of class objects (`BPNet`).
2.  **Legacy Checkpoints**: Existing model directories lacking `ensemble_metadata.yaml` or `fold_X` structure will fail to load with the new `ModelEnsemble`.
3.  **Validation**: `validate_model_config` will raise `TypeError` if non-string types are encountered.

## Next Steps
1.  **Update Notebooks**: Revise demo notebooks (e.g., `model_ensemble_demo2.py`) to use string-based configs.
2.  **Migration Script**: (Optional) Create a script to convert legacy checkpoint structures to the new format (create `fold_0` subdir, move files, generate metadata).
3.  **Testing**: Verify `search_paths` functionality with a test case simulating moved data.

## Recent Updates (Bug Fixes & Test Alignments)

### 1. Model Loading Logic Fix (`src/cerberus/model_ensemble.py`)
-   **Issue**: `_load_model` previously treated the `checkpoint_path` argument as a directory and attempted to `rglob("*.ckpt")` inside it. However, `load_models_and_folds` was already resolving the best checkpoint *file* and passing that file path. This caused a crash (trying to glob inside a file).
-   **Fix**: `_load_model` has been updated to strictly accept a **checkpoint file path**.
-   **API Change**: Renamed argument `path` to `ckpt_file` in `_load_model` to be explicit. Removed directory scanning logic from this method.

### 2. Test Suite Alignment
-   **`tests/test_model_ensemble_loader.py`**:
    -   Renamed `test_load_model_directory` to `test_load_model_file` to reflect the API change.
    -   Updated `test_load_models_and_folds` to create the required `ensemble_metadata.yaml` mock.
-   **`tests/test_model_ensemble_hparams.py`**:
    -   Updated tests to initialize `ensemble_metadata.yaml` and proper fold structures, as `ModelEnsemble` now throws `FileNotFoundError` if metadata is missing.
-   **`tests/test_train_multi.py`**:
    -   Added mocking for `cerberus.train.update_ensemble_metadata` to prevent File I/O errors during tests where the filesystem is mocked.
-   **`tests/test_config_validation.py` & `tests/test_predict.py`**:
    -   Updated to pass **string class names** (e.g., `"tests.test_predict.DummyModel"`) instead of class objects in `ModelConfig`, complying with the new strict validation.
-   **`tests/test_hparams_parsing.py`**:
    -   Updated `test_parse_hparams_config_success` to create a mock `hparams.yaml` with valid string-based configs, rather than relying on an obsolete file in `tests/data`.

### 3. Test Data Cleanup
-   **`tests/data/models/`**:
    -   Removed legacy model binaries (`bpnet-chip_ar_mdapca2b`, etc.) to reduce repository bloat.
    -   Purged these files from git history using `git-filter-repo`.
    -   Tests relying on these files (e.g., `test_hparams_parsing.py`) now gracefully skip if data is missing.

### 4. Refactor Completion (Architecture Simplification)
-   **Architecture Simplified**: `_ModelManager` in `src/cerberus/model_ensemble.py` has been refactored to strictly follow `ensemble_metadata.yaml`. It no longer guesses directory structures (`_detect_multifold` removed). Single-fold models are treated identically to multi-fold (iterating over `folds` list).
-   **Config Enhancements**:
    -   Added `import_class` utility in `src/cerberus/config.py`.
    -   Updated `src/cerberus.train.py` to use `import_class` for dynamic class loading from string configs.
    -   Updated `validate_model_config` to strictly enforce string types for class names.
-   **Example Updates**: Updated `examples/chip_ar_mdapca2b_bpnet.py` to use the new string-based configuration format.

### 5. Cohesion & Architecture Refactor (January 2026)
-   **Aggregation Logic Moved**: Static aggregation methods (`_unbatch_modeloutput`, `_aggregate_tensor_track_values`, etc.) were moved from `src/cerberus/model_ensemble.py` to `src/cerberus/output.py` to improve cohesion (logic stays with data structures).
-   **Instantiation Logic Moved**: Factory functions `instantiate` and `instantiate_model` were moved from `src/cerberus.train.py` to `src/cerberus/module.py`. `entrypoints.py` now re-exports them but focuses on high-level training workflows.

## ⚠️ Important: Repository History Rewrite

As part of the cleanup (January 2026), the git history was rewritten to remove large binary files in `tests/data/models/`. This requires all developers to update their local repositories.

**Option 1: Fresh Clone (Recommended)**
```bash
cd ..
rm -rf cerberus
git clone https://github.com/mctp/cerberus.git
```

**Option 2: Reset Local Branch**
If you want to keep your existing directory:
```bash
git fetch origin
git reset --hard origin/main
```
*Warning: This will discard any uncommitted local changes.*
