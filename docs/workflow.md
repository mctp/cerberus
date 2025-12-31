# Cerberus Workflow: Training, Prediction, and Exploration

This document outlines the high-level workflow for using Cerberus, summarizing how the various components fit together to take you from raw data to trained models and biological insights. It is based on the patterns demonstrated in `examples/chip_ar_mdapca2b.py` and the corresponding prediction notebooks.

## The Cerberus Lifecycle

The typical Cerberus lifecycle consists of three distinct phases that share a common configuration backbone:

1.  **Training**: Learning the sequence-to-function mapping.
2.  **Prediction**: Applying the learned model to genomic intervals (inference).
3.  **Exploration**: Analyzing model outputs and visualizing predictions against ground truth.

Everything is glued together by **Configuration Objects** (`GenomeConfig`, `DataConfig`, etc.), ensuring consistency between these phases.

---

## 1. Training

The training phase is responsible for learning the mapping from DNA sequence to functional signals (e.g., ChIP-seq profiles).

### Key Components

*   **Configuration**: You define 5 key config dictionaries:
    *   `GenomeConfig`: Which genome build (hg38), exclusions (blacklists), and fold strategy to use.
    *   `DataConfig`: Input/output shapes, resolution, and transformations (e.g., log-transform targets).
    *   `SamplerConfig`: Where to select training examples (e.g., `interval` sampler for peaks, `sliding_window` for genome-wide).
    *   `TrainConfig`: Hyperparameters like batch size, learning rate, and optimizer.
    *   `ModelConfig`: Architecture (e.g., BPNet, Baseline CNN) and loss function definitions.
*   **Entrypoints**:
    *   `train_fold`: Trains a single model on a specific fold split.
    *   `train_multi`: Performs k-fold cross-validation, training multiple models.

### The Process

1.  **Data Setup**: Download/Prepare genome FASTA and signal BigWigs.
2.  **Config Definition**: Set up the configuration dictionaries.
3.  **Execution**: Call `train_fold` or `train_multi`.
4.  **Output**: Cerberus saves model checkpoints (`.ckpt`) and logs to the specified output directory.

**Example Reference**: `examples/chip_ar_mdapca2b.py` demonstrates a full training script that supports both single-fold and multi-fold training for Baseline CNN and BPNet architectures.

---

## 2. Prediction (Inference)

Once a model is trained, the prediction phase involves applying it to specific genomic regions. The workflow is streamlined to reuse the same configurations from training, with the system automatically handling the switch to deterministic behavior.

### Key Concepts

*   **Deterministic Transforms**: When performing inference, we want reproducible results without random augmentations (like jitter or reverse complement). Cerberus handles this automatically.
*   **Model Loading**: The `ModelManager` class loads the model state from a checkpoint, ensuring the architecture matches the config.

### The Process

1.  **Load Configs**: Reuse the `GenomeConfig`, `DataConfig`, and `ModelConfig` used during training.
2.  **Load Model**: Use `ModelManager(checkpoint_path=...)` to instantiate the model and load weights.
3.  **Initialize Dataset**: Create a `CerberusDataset` with `is_train=False`.
    *   This automatically switches the data pipeline to use **deterministic transforms** (e.g., center-cropping instead of random jitter, disabling reverse-complement augmentation).
    *   No manual adjustment of `DataConfig` is required.
4.  **Run Prediction**: Use `predict_intervals()` to generate predictions for a list of genomic intervals.

**Example Reference**: `notebooks/chip_ar_mdapca2b_predict_bpnet.py` shows how to load a trained BPNet model and run inference on test set peaks.

---

## 3. Model Exploration

Exploration involves interpreting the raw model outputs, which can vary by architecture, and comparing them to ground truth.

### Understanding Outputs

Different models produce different outputs:

*   **Baseline CNNs**: Typically output `logits` (log-scale predictions).
    *   *Transformation*: `exp(logits)` -> Predicted Signal.
*   **BPNet**: Outputs a tuple of `(profile_logits, log_counts)`.
    *   `profile_logits`: The shape of the signal (probability distribution over the window).
    *   `log_counts`: The total magnitude of signal in the window.
    *   *Transformation*: `softmax(profile_logits) * exp(log_counts)` -> Predicted Signal Counts.

### The Process

1.  **Extract Ground Truth**: Use the dataset's `target_signal_extractor` to get the actual observed signal for the predicted interval.
2.  **Transform Predictions**: Convert raw model outputs (logits) into the same unit as ground truth (e.g., read counts).
3.  **Visualization**: Plot the Predicted Profile vs. Observed Signal (Ground Truth) to visually assess performance.
4.  **Metrics**: Compute correlations (Pearson/Spearman) or MSE between the tracks.

**Example Reference**: The prediction notebooks demonstrate extracting ground truth, converting logits to counts, and plotting them side-by-side using `matplotlib`.

---

## Summary of Relationships

| Component | Training Role | Prediction Role |
| :--- | :--- | :--- |
| **GenomeConfig** | Defines genome & exclusions | Same (ensures consistent coordinates) |
| **DataConfig** | Defines I/O shapes & Augmentations | Defines I/O shapes (Augmentations auto-disabled) |
| **SamplerConfig** | Defines training batches (w/ jitter) | Not strictly used if providing explicit intervals |
| **ModelManager** | *Not used (created via Entrypoint)* | **Loads checkpoint & reinstantiates model** |
| **CerberusDataset** | Feeds DataLoader (`is_train=True`) | Feeds `predict_intervals` (`is_train=False`) |
| **predict_intervals** | *N/A* | Executes forward pass & aggregation |
