# Training Curricula Design for Cerberus

## 1. Overview & Outline

### 1.1. Overview
This document outlines the design for implementing **Training Curricula** in the Cerberus framework. The goal is to allow dynamic adjustment of training data distributions (e.g., easy-to-hard examples) and loss functions (e.g., profile-to-count emphasis) over the course of training.

### 1.2. Outline
1.  **Overview & Outline**: Executive summary.
2.  **Introduction**: Definition of data and loss curricula.
3.  **Current Capabilities**: Analysis of existing codebase support and limitations.
4.  **Design Proposal A: Callback-Driven (Ad-Hoc)**: Initial proposal using direct attribute modification.
5.  **Learning Strategies**: Concrete examples of curricula with feasibility analysis.
6.  **Implementation Plan**: Step-by-step implementation guide.
7.  **Code Example (Draft)**: Draft implementation of key components (Ad-Hoc approach).
8.  **Design Proposal B: Unified Dynamic Sampler Protocol**: Alternative, more robust proposal.
9.  **Trade-off Analysis & Recommendation**: Comparison and final recommendation (Proposal B).
10. **Affected Files**: List of files requiring modification.
11. **Context for Agents**: Summary of design decisions and exploration history.

## 2. Introduction

Training curricula involve dynamically adjusting the training process over time to improve model convergence, stability, and generalization. In the context of sequence-to-function models like Cerberus, this typically involves two main axes:

1.  **Data Curricula**: Changing the distribution of training examples (e.g., shifting from high-signal peaks to genome-wide background).
2.  **Loss Curricula**: Changing the objective function (e.g., shifting emphasis from shape to count, or annealing constraints).

This document evaluates the current codebase's support for these paradigms and proposes a modular design for implementation.

## 3. Current Capabilities & Constraints

### 3.1. Loss Functions
**Status**: Well-supported.
*   Loss functions (e.g., `MSEMultinomialLoss`, `PoissonMultinomialLoss`) are standard `torch.nn.Module` classes.
*   Key parameters (weights, total_count) are instance attributes.
*   The loss instance is accessible via `CerberusModule.criterion`.
*   **Modification**: Parameters can be modified in-place during training (e.g., inside a Callback).

### 3.2. Data Sampling
**Status**: Partially supported (requires specific configuration).
*   **`MultiSampler`**: Designed to mix different data sources (e.g., "peaks" vs "random") with configurable `scaling_factors`.
*   **Resampling**: `MultiSampler.resample(seed)` re-draws indices based on the scaling factors. This allows changing the ratio of positive/negative examples.
*   **Constraint - `GCMatchedSampler`**: This sampler pre-computes GC content matching based on the *initial* state of the candidate sampler. Dynamic changes to the underlying candidate sampler (e.g., changing its size) are **not** automatically propogated and would require expensive re-computation.
    *   *Recommendation*: Avoid wrapping dynamic samplers inside `GCMatchedSampler`. Use `MultiSampler` as the top-level aggregator.

### 3.3. Data Loading
**Status**: Supported with overhead.
*   Changing sampler scaling factors changes the total length (`__len__`) of the dataset.
*   PyTorch `DataLoader` caches length and iterators.
*   **Mechanism**: To support dynamic dataset length, the `Trainer` must be configured with `reload_dataloaders_every_n_epochs=1`.
*   **Hook**: `CerberusDataModule.train_dataloader()` already includes a hook to call `dataset.resample(seed)` when loaders are reloaded.

## 4. Design Proposal A: Callback-Driven (Ad-Hoc)

We propose a `CurriculumCallback` to manage training schedules. This keeps the core logic (Module/DataModule) clean and isolates curriculum logic.

### 4.1. `CurriculumCallback`

**Location**: `src/cerberus/callbacks.py` (New file)

**Responsibilities**:
1.  Parse a curriculum schedule configuration.
2.  Update Sampler `scaling_factors` at the end of each epoch (preparing for next epoch's dataloader reload).
3.  Update Loss parameters at the start of each epoch.

**Configuration Schema**:
```python
curriculum_config = {
    "schedule": {
        0: { # Epoch 0
            "sampler": {"peaks": 1.0, "random": 0.1},
            "loss": {"count_weight": 0.0, "profile_weight": 1.0}
        },
        10: { # Epoch 10
            "sampler": {"peaks": 1.0, "random": 0.5}, # Increase difficult negatives
            "loss": {"count_weight": 0.5} # Introduce count loss
        },
        20: { # Epoch 20
            "loss": {"count_weight": 1.0} # Full count weight
        }
    }
}
```

### 4.2. Implementation Logic

**`on_train_epoch_end(trainer, pl_module)`**:
*   Determine next epoch (`trainer.current_epoch + 1`).
*   Check if schedule exists for next epoch.
*   **Sampler Update**:
    *   Access `trainer.datamodule.train_dataset.sampler`.
    *   Verify it is a `MultiSampler`.
    *   Update `scaling_factors` based on config.
    *   *Note*: Actual resampling happens in `CerberusDataModule.train_dataloader()` which is called *after* this callback (due to `reload_dataloaders_every_n_epochs=1`).

**`on_train_epoch_start(trainer, pl_module)`**:
*   Determine current epoch.
*   **Loss Update**:
    *   Access `pl_module.criterion`.
    *   Update attributes (e.g., `count_weight`) using `setattr`.
    *   Log changes to `pl_module.log`.

## 5. Learning Curricula Strategies

Drawing from experience with Imaging CNNs and Sequence-to-Function models, the following strategies are relevant:

### 5.1. Data Curricula (Easy vs. Hard Examples)
*   **Lesson**: Models learn "easy" features (high SNR peaks) quickly. "Hard" examples (low SNR, background with GC bias) require more training time but are crucial for specificity.
*   **Strategy: Peak-Centric to Genome-Wide**:
    *   *Phase 1 (Warmup)*: High ratio of Peaks vs. Random background (e.g., 1:1). Allows model to learn the motif/profile grammar quickly without being overwhelmed by empty space.
    *   *Phase 2 (Annealing)*: Gradually decrease Peak ratio or increase Random ratio (e.g., towards 1:10 or 1:100). Forces the model to discriminate signal from noise, reducing False Positives.
    *   *Feasibility*: **High**. Supported via `MultiSampler` scaling factors.
*   **Strategy: Signal Magnitude**:
    *   Start with only the highest 20% of peaks (strongest signal). Gradually add lower-strength peaks.
    *   *Feasibility*: **Medium**. Requires pre-splitting data into "bins" of signal strength and treating them as separate samplers in `MultiSampler`.
*   **Strategy: Hard Negative Mining**:
    *   *Description*: Dynamically identify false positives during training and oversample them in subsequent epochs.
    *   *Feasibility*: **Low (Current) / High (DynamicSampler)**. Currently difficult as samplers are static. With the **Unified Dynamic Sampler**, the `update_curriculum` method could accept a list of new intervals (mined hard negatives) and update the internal sampling pool.
*   **Strategy: Resolution/Complexity**:
    *   *Description*: Start with coarse resolution (e.g., 128bp bins) to learn broad patterns, refine to 1bp.
    *   *Feasibility*: **Low**. Requires changing model architecture (output heads) and tensor shapes on the fly. `DynamicSampler` does not solve this; it would require a "DynamicModule".

### 5.2. Loss Curricula (Task Complexity)
*   **Lesson**: Profile shape (local dependencies) is often easier to learn than total count (global scaling). Joint training can sometimes lead to instability if gradients compete.
*   **Strategy: Profile First**:
    *   *Phase 1*: `count_weight = 0.0`, `profile_weight = 1.0`. Focus on shape.
    *   *Phase 2*: Ramp up `count_weight` linearly. Adds magnitude constraint.
    *   *Feasibility*: **High**. Supported via Loss parameters.
*   **Strategy: Dispersion Annealing (for Negative Binomial)**:
    *   Start with a high `total_count` (low dispersion, closer to Poisson).
    *   Decrease `total_count` to allow for over-dispersion as the model refines its predictions.
    *   *Feasibility*: **High**. Supported via Loss parameters.
*   **Strategy: Noise Injection**:
    *   *Description*: Gradually introduce input noise or dropout to improve robustness.
    *   *Feasibility*: **High**. Can be implemented by `CurriculumCallback` modifying `dataset.transforms` (e.g., increasing `std` of GaussianNoise) or model dropout rates. `DynamicSampler` is not strictly required here, but could coordinate if noise is sample-dependent.

## 6. Implementation Plan

1.  **Create `src/cerberus/callbacks.py`**:
    *   Implement `CurriculumCallback`.
    *   Implement `CurriculumConfig` validation.

2.  **Update `CerberusDataModule`**:
    *   Ensure `train_dataloader` logic is robust to dynamic sampler sizes. (Already verified: calls `resample` which handles index regeneration).

3.  **Update `CerberusModule`**:
    *   Expose `configure_callbacks` to accept curriculum config.

4.  **Configuration**:
    *   Add `curriculum` section to `TrainConfig`.

## 7. Code Example (Draft)

```python
# src/cerberus/callbacks.py

from pytorch_lightning.callbacks import Callback
import logging

class CurriculumCallback(Callback):
    def __init__(self, schedule: dict):
        self.schedule = schedule # {epoch: {type: config}}
        # Sort keys to find active schedule efficiently? 
        # Or just check exact match for discrete steps.

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch in self.schedule:
            step_config = self.schedule[epoch]
            
            # Loss Update
            if "loss" in step_config:
                criterion = pl_module.criterion
                for param, value in step_config["loss"].items():
                    if hasattr(criterion, param):
                        setattr(criterion, param, value)
                        pl_module.log(f"curriculum/loss/{param}", value)
                        logging.info(f"Curriculum: Set loss.{param} to {value}")

    def on_train_epoch_end(self, trainer, pl_module):
        # Prepare sampler for NEXT epoch
        next_epoch = trainer.current_epoch + 1
        if next_epoch in self.schedule:
            step_config = self.schedule[next_epoch]
            
            # Sampler Update
            if "sampler" in step_config:
                # Assuming top-level MultiSampler
                dataset = trainer.datamodule.train_dataset
                sampler = dataset.sampler
                
                if hasattr(sampler, "scaling_factors"):
                    # Logic to map 'sampler_name' to index or use dict-based config if supported
                    # For now, assume ordered list or dict mapping
                    updates = step_config["sampler"]
                    # ... update logic ...
                    logging.info(f"Curriculum: Updated sampler scaling for epoch {next_epoch}")
```

## 8. Design Proposal B: Unified Dynamic Sampler Protocol

To avoid ad-hoc manipulation of sampler internals (like `scaling_factors`) via callbacks, we can introduce a formal abstraction for dynamic samplers.

### 8.1. Proposed Abstraction: `DynamicSampler` Protocol

Extend the `Sampler` protocol to explicitly support state updates.

```python
class DynamicSampler(Sampler, Protocol):
    def update_curriculum(self, epoch: int, config: dict[str, Any]) -> None:
        """
        Updates the sampler's internal state based on the training epoch and configuration.
        This method should be called before `resample()` or at the epoch boundary.
        """
        ...
```

### 8.2. Refactoring `MultiSampler`

Instead of the callback modifying `scaling_factors` directly, `MultiSampler` would implement `update_curriculum`.

```python
class MultiSampler(BaseSampler):
    # ... existing methods ...

    def update_curriculum(self, epoch: int, config: dict):
        """
        Looks for 'scaling' or 'weights' in the config and updates internal factors.
        """
        if "scaling" in config:
             # Logic to parse config and update self.scaling_factors
             # e.g., config['scaling'] = {'peaks': 0.5, 'random': 1.0}
             self._update_scaling(config["scaling"])
```

### 8.3. Benefits

1.  **Encapsulation**: The logic for *how* a sampler changes (e.g., whether it changes scaling factors, or switches internal strategies, or re-computes GC matches) remains inside the Sampler class.
2.  **Simplicity**: The `CurriculumCallback` becomes generic:
    ```python
    if hasattr(sampler, "update_curriculum"):
        sampler.update_curriculum(epoch, step_config)
    ```
3.  **Extensibility**: New samplers (e.g., `HardExampleMiningSampler`) can implement their own curriculum logic (e.g., updating the threshold for "hard" examples) without changing the callback code.

### 8.4. Handling `GCMatchedSampler`

With this abstraction, `GCMatchedSampler` could implement `update_curriculum` to intelligently handle updates. For example, it could check if the underlying `candidate_sampler` has changed significantly and decide whether to trigger a potentially expensive GC re-computation, or use an approximation.

## 9. Trade-off Analysis & Recommendation

### 9.1. Callback-Driven Ad-Hoc Modification (Proposal A)
*   **Pros**:
    *   **Speed**: Faster to implement initially as it requires minimal changes to existing `Sampler` classes.
    *   **Flexibility**: The callback can perform arbitrary operations without being constrained by a protocol.
*   **Cons**:
    *   **Brittleness**: Tightly coupled to the internal implementation details (e.g., `scaling_factors` attribute) of specific samplers. If the sampler implementation changes, the callback may break silently or require updates.
    *   **Maintainability**: Logic is split between the callback and the sampler's state.
    *   **Testing**: Harder to unit test the curriculum logic in isolation from the full training loop.

### 9.2. Unified Dynamic Sampler Protocol (Proposal B)
*   **Pros**:
    *   **Encapsulation**: Sampler state logic is contained within the Sampler itself.
    *   **Type Safety**: A formal protocol ensures that samplers support the expected operations.
    *   **Modularity**: New sampling strategies (e.g., hard negative mining) can be added without modifying the callback.
*   **Cons**:
    *   **Effort**: Requires refactoring existing samplers (`MultiSampler`, `GCMatchedSampler`) and potentially defining a more rigorous configuration schema.

### 9.3. Recommendation
We recommend adopting the **Unified Dynamic Sampler Protocol (Proposal B)**. While it requires slightly more upfront effort to refactor `MultiSampler`, it provides a much more robust and extensible foundation for future curriculum development. It aligns with the design goal of making the system "clear, simple but extensible and modular."

## 10. Affected Files

The following files will need to be modified or created to implement the Unified Dynamic Sampler and Curriculum Callback design.

### 10.1. Source Code (`src/cerberus/`)
*   **`src/cerberus/samplers.py`**:
    *   Define `DynamicSampler` protocol.
    *   Implement `update_curriculum` in `MultiSampler` (and optionally `GCMatchedSampler`).
*   **`src/cerberus/callbacks.py`** (New):
    *   Implement `CurriculumCallback` class.
*   **`src/cerberus/config.py`**:
    *   Update `TrainConfig` to include optional `curriculum` field.
    *   Add validation for curriculum schema.
*   **`src/cerberus/module.py`**:
    *   Update `configure_callbacks` to instantiate `CurriculumCallback` if config is present.

### 10.2. Tests (`tests/`)
*   **`tests/test_multi_sampler.py`**:
    *   Add test cases for `MultiSampler.update_curriculum` verifying that `scaling_factors` update correctly and `resample()` respects them.
*   **`tests/test_curriculum.py`** (New):
    *   Integration tests for `CurriculumCallback`.
    *   Verify loss parameters are updated.
    *   Verify sampler scaling is updated at epoch boundaries.

### 10.3. Examples & Notebooks
*   **`notebooks/cerberus_basics.py`**: Update to demonstrate curriculum config if applicable.

## 11. Context for Agents

### 11.1. Objective & Exploration
The goal was to design a system for Training Curricula (dynamic data and loss schedules) without modifying the core training loop extensively.
*   **Analysis of `src/cerberus/module.py`**: Confirmed `criterion` is accessible and mutable.
*   **Analysis of `src/cerberus/samplers.py`**: Found `MultiSampler` uses `scaling_factors`. `GCMatchedSampler` is static (cached GC) and hard to update dynamically.
*   **Analysis of `src/cerberus/datamodule.py`**: Confirmed `train_dataloader()` hooks into `dataset.resample()`. Critical finding: dynamic dataset size requires `Trainer(reload_dataloaders_every_n_epochs=1)`.

### 11.2. Design Evolution
1.  **Initial Approach**: A simple Callback that modifies `dataset.sampler.scaling_factors` directly (Proposal A).
    *   *Critique*: Ad-hoc, brittle, leaky abstraction.
2.  **Refined Approach**: Introduced `DynamicSampler` protocol (Proposal B).
    *   *Benefit*: Encapsulates update logic (e.g., parsing config, updating weights, handling cache invalidation) within the Sampler.
    *   *Enabler*: Allows advanced strategies like **Hard Negative Mining** where the sampler needs to ingest new data (mined negatives), which is impossible with a simple "scaling factor" update.

### 11.3. Key Code Snippets

**Proposed `DynamicSampler` Protocol**:
```python
class DynamicSampler(Sampler, Protocol):
    def update_curriculum(self, epoch: int, config: dict[str, Any]) -> None:
        pass
```

**Curriculum Callback Structure**:
```python
class CurriculumCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # ... resolve next epoch ...
        if hasattr(sampler, "update_curriculum"):
            sampler.update_curriculum(next_epoch, config)
```

### 11.4. Future Work
*   Implement `DynamicSampler` in `samplers.py`.
*   Refactor `MultiSampler` to use it.
*   Implement `CurriculumCallback`.
*   Add unit tests in `tests/`.
