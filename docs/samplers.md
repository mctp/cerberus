# Samplers and Randomness

Cerberus implements a rigorous strategy for handling randomness and seeding across its sampling modules. This ensures that training data generation is reproducible, consistent, and correctly randomized across epochs and folds.

## The Seeding Contract

All samplers in Cerberus (`RandomSampler`, `MultiSampler`, `ComplexityMatchedSampler`, etc.) adhere to a unified contract regarding initialization, resampling, and state propagation.

### 1. Initialization
*   **Argument**: All samplers accept an optional `seed: int | None` argument in their constructor.
*   **Behavior**:
    *   If `seed` is provided (integer), the sampler initializes its internal random number generator (RNG) with this seed. This guarantees that the initial state of the sampler is deterministic.
    *   If `seed` is `None`, the sampler initializes with a random seed derived from the operating system's entropy source. This is non-reproducible unless the global random state is managed externally.

### 2. Resampling (`resample`)
The `resample(seed: int | None = None)` method is the core mechanism for advancing the sampler's state, typically invoked at the start of each training epoch.

*   **Sticky Seeds (Deterministic Progression)**:
    *   If a sampler was initialized with a seed (e.g., `seed=42`), calling `resample(None)` does **not** reset it to 42. Instead, it advances the RNG to the *next* state in the deterministic sequence.
    *   This ensures that Epoch 1, Epoch 2, etc., receive different but reproducible batches of data.
*   **Explicit Reseed**:
    *   Calling `resample(new_seed)` forces the sampler to a specific new state, resetting its RNG sequence.
*   **Generation**:
    *   Calling `resample` triggers the regeneration of intervals (for `RandomSampler`) or the reshuffling of indices (for `MultiSampler`, `ScaledSampler`).

### 3. Seed Propagation (Sub-Samplers)
Composite samplers (those that wrap other samplers) act as **coordinators** for randomness. They ensure that all components are seeded consistently without correlation artifacts.

*   **Pattern**: When a parent sampler (e.g., `MultiSampler`) is resampled, it:
    1.  Updates its own master seed.
    2.  Derives unique sub-seeds for each of its children using its master RNG.
    3.  Calls `resample(sub_seed)` on each child.
*   **Benefit**: This prevents "seed collision" where multiple sub-samplers might accidentally behave identically if initialized with the same global constant. It ensures that sub-samplers are de-correlated but fully determined by the parent's seed.

### 4. Splitting Folds (`split_folds`)
Splitting a sampler into Train/Validation/Test sets allows for cross-validation while maintaining the randomness context.

*   **Derived Identity**: The `split_folds` method does not simply slice the existing list of intervals. Instead, it creates **new sampler instances** restricted to the specific fold regions.
*   **Seed Derivation**: The new samplers for Train, Val, and Test are initialized with seeds derived from the parent sampler's *current* state.
*   **Idempotency**: Calling `split_folds` multiple times on the same sampler state will return identical splits.
*   **Dynamic Splits**: If the parent sampler is resampled (advanced to next epoch), subsequent calls to `split_folds` will produce new splits with new seeds. This allows the validation set (if stochastic) to vary across epochs if desired, though typically validation sets are kept static by using a separate, non-resampled sampler instance.
*   **Overlap Handling**: If the defined Test and Validation folds overlap (or are identical), intervals falling into the overlapping regions will be assigned to **both** the Test and Validation samplers. Intervals present in either Test or Validation folds are strictly excluded from the Training sampler.

## Specific Implementations

### IntervalSampler
*   **Function**: Loads a static list of intervals from a file (BED or narrowPeak).
*   **Seeding**: Not applicable (deterministic data).
*   **Resampling**: No-op (data is static).
*   **Splitting**: Filters the loaded intervals based on the fold definitions.

### SlidingWindowSampler
*   **Function**: Generates intervals by sliding a fixed-size window across the genome with a specified stride.
*   **Seeding**: Not applicable (deterministic generation).
*   **Resampling**: No-op.
*   **Splitting**: Filters the generated windows based on the fold definitions.

### RandomSampler
*   **Function**: Generates `num_intervals` random genomic intervals.
*   **Seeding**: The `seed` determines the sequence of random regions and coordinates selected.
*   **Resampling**: Regenerates the entire list of intervals.
*   **Splitting**: Creates new `RandomSampler` instances for each fold. The number of intervals in each split is determined by the proportion of intervals that fell into that fold in the parent's current generation, effectively preserving the density distribution.

### MultiSampler
*   **Function**: Combines multiple samplers into a single stream.
*   **Seeding**: The `seed` controls the shuffling of the combined index list and the seed propagation to sub-samplers.
*   **Resampling**: Updates self-seed, propagates derived seeds to all sub-samplers, and re-shuffles the combined index list.

### ComplexityMatchedSampler
*   **Function**: Selects a subset of intervals from a `candidate_sampler` to match the distributional properties of a `target_sampler`.
*   **Metrics**: Matches on a specified list of complexity metrics (e.g., `["gc"]`, `["gc", "dust"]`, or `["gc", "dust", "cpg"]`).
*   **Seeding**: The `seed` controls the random selection of candidates from the N-dimensional complexity bins.
*   **Resampling**:
    1.  Updates self-seed.
    2.  Selects a new subset of candidates from the *existing* candidate pool using the new seed.
    3.  Note: The `candidate_sampler` is **not** automatically resampled after initialization. The pool of candidates remains static to avoid expensive re-computation of complexity metrics. Only the selection *from* that pool changes.
    4.  Note: The `target_sampler` is also **not** automatically resampled.

### ScaledSampler
*   **Function**: Subsamples or oversamples another sampler to a fixed size.
*   **Seeding**: The `seed` controls the random selection of indices.
*   **Resampling**: Updates self-seed, propagates a derived seed to the child sampler, and re-selects indices.

### PeakSampler
*   **Function**: A high-level convenience class that combines an `IntervalSampler` (positives/peaks) with a `ComplexityMatchedSampler` (negatives/background) into a single `MultiSampler`.
*   **Seeding**: Propagates seed to the negative generation process.
*   **Resampling**: Re-samples the negative set to match the positives (while keeping positives static).
*   **Splitting**: Splits both the positive and negative samplers consistent with folds.
*   **Peak labelling**: Because `PeakSampler` extends `MultiSampler`, each batch item returned by `CerberusDataset` includes a `peak_status` field: `1` for peak intervals (from `IntervalSampler`) and `0` for background intervals (from `ComplexityMatchedSampler`). This label is always consistent with `__getitem__` — both read from the same internal index table.

## Best Practices for Users

1.  **Global Configuration**: Set a single random seed in your configuration. The `create_sampler` factory will propagate this seed correctly to all nested samplers.
2.  **Epoch Management**: Rely on the training loop (or `CerberusDataModule`) to call `resample(None)` at the start of each epoch. Do not manually manage seeds inside the training loop unless you have a specific requirement for static data.
3.  **Validation**: For validation and testing, it is common to use samplers that are *not* resampled between epochs to ensure comparable metrics. Ensure your validation dataloader does not invoke `resample()` or is initialized with a fixed, separate sampler instance.

## Randomness in Training vs. Validation

A key requirement in machine learning is that the **Validation Set must remain constant** throughout training to allow for fair comparison of model performance across epochs, while the **Training Set may change** (e.g., via random sampling of the background genome) to maximize data diversity.

Cerberus handles this automatically through the interaction of `CerberusDataModule` and `RandomSampler`:

1.  **Setup Phase**:
    *   `CerberusDataModule.setup()` initializes the full dataset and calls `split_folds()`.
    *   This creates three **separate** dataset instances: `train_dataset`, `val_dataset`, and `test_dataset`.
    *   Crucially, `split_folds()` creates independent sampler instances for each dataset. For a `RandomSampler`, the validation sampler generates its random intervals **once** at initialization and stores them.

2.  **Training Loop**:
    *   At the start of each epoch, `CerberusDataModule.train_dataloader()` is called.
    *   It explicitly calls `self.train_dataset.resample(seed=epoch)`.
    *   This triggers the **training sampler** to regenerate its intervals (e.g., picking new random negative regions), ensuring the model sees different background examples each epoch.

3.  **Validation Loop**:
    *   `CerberusDataModule.val_dataloader()` returns a loader for `self.val_dataset`.
    *   It does **not** call `resample()`.
    *   Therefore, the **validation sampler** retains the exact same list of intervals that were generated during the `setup()` phase.
    *   This guarantees that validation metrics are computed on the same fixed set of examples every epoch.
