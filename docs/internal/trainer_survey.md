# Survey of Training Implementations in Sequence-to-Function Models

This document summarizes the analysis of training infrastructure in `gReLU`, `EUGENE`, and `chrombpnet-pytorch` to inform the design of the `cerberus` training module.

## Findings

### gReLU
*   **Approach**: Uses a generic `LightningModel` wrapper.
*   **Flexibility**: Handles various tasks (binary, regression, multiclass) dynamically based on configuration.
*   **Metrics**: Relies on `torchmetrics.MetricCollection` for standard evaluation.
*   **Loss**: Supports standard losses (Poisson, BCE, MSE).
*   **Takeaway**: Good for general purpose, but might lack specificity for complex dual-head architectures without custom extension.

### EUGENE
*   **Approach**: Uses a `SequenceModule` with a strict registry-based system.
*   **Structure**: Highly structured with registries for losses, optimizers, and schedulers.
*   **Configuration**: Relies on `settings` and strict typing.
*   **Takeaway**: Excellent for enforcing standards and reproducibility, but arguably over-engineered for a library that aims to be a flexible utility companion.

### chrombpnet-pytorch
*   **Approach**: Uses specific wrappers (`BPNetWrapper`, `ChromBPNetWrapper`) inheriting from a base `ModelWrapper`.
*   **Specificity**: Manually implements `multinomial_nll` and `pearson_corr`.
*   **Bias Handling**: Explicitly handles bias model subtraction in the forward pass/loss calculation.
*   **Loss Weighting**: Manually combines profile and count losses with `alpha` and `beta` weights.
*   **Takeaway**: This approach offers the control needed for the specific mathematical requirements of BPNet-style models (dual-head loss, bias correction).

### ASAP
*   **Approach**: Custom `Trainer` class (non-Lightning) with manual DDP handling.
*   **Unmap/Mappability**: Supports an auxiliary "unmap" loss (MSE) to penalize predictions in unmappable regions or learn mappability, triggered by `return_unmap=True` in the model.
*   **Metrics**: Computes Pearson, Spearman, Kendall, and MSE (log and non-log).
*   **Robustness**: Implements "robust batch" prediction using sliding windows and averaging.
*   **Takeaway**: Critical to support auxiliary losses (like mappability) alongside the primary objective.

### bpnet-lite
*   **Approach**: PyTorch-based, uses `MNLLLoss` (Multinomial NLL) for profiles and `log1pMSELoss` for counts.
*   **Loss Combination**: Combines losses within a `_mixture_loss` function: `profile_loss + count_loss_weight * count_loss`.
*   **Takeaway**: Confirms the standard BPNet loss formulation in PyTorch.

### bpAITAC
*   **Approach**: Custom training loop. Supports "Composite" losses (weighted sum of profile and count losses).
*   **Bias**: Explicitly handles bias as a model input.
*   **Evaluation**: Strong focus on JSD (Jensen-Shannon Divergence) and Pearson correlation.

### bpreveal
*   **Approach**: TensorFlow/Keras-based.
*   **Multi-Head**: Dynamically builds loss lists for multiple heads.
*   **Adaptive Weights**: Implements adaptive weighting for the count loss (learning the weight `λ` during training).

## Recommendation for Cerberus

We adopt a **hybrid approach** leaning heavily on the **Specialized Wrapper** pattern from `chrombpnet-pytorch`, but modernized with `torchmetrics`.

### Why a Specialized Wrapper?

1.  **Dual-Head Loss Management**: BPNet-style models require optimizing two distinct objectives simultaneously: the shape of the signal (Profile) and the total abundance (Count).
    *   *Profile Loss*: Multinomial Negative Log Likelihood (NLL). This is not a standard PyTorch loss function and requires careful implementation for numerical stability.
    *   *Count Loss*: Mean Squared Error (MSE) on log-counts.
    *   A specialized wrapper allows us to strictly define how these two losses are computed and weighted (`alpha` * count + `beta` * profile).
    *   *Note*: `bpnet-lite` and `bpreveal` also follow this pattern, with `bpreveal` even supporting adaptive weights.

2.  **Bias Correction**:
    *   In many S2F applications, a "bias model" (e.g., modeling enzyme bias) is used.
    *   The bias prediction is subtracted from the model's prediction *before* the loss is calculated, but effectively *after* the model's forward pass (in logit space).
    *   A generic wrapper would require hacking this into the model architecture or the loss function. A specialized `CerberusTask` can handle this gracefully as a first-class citizen of the training step.

3.  **Custom Metrics**:
    *   Standard accuracy or MSE is insufficient. We need to measure how well the predicted profile shape matches the ground truth (Pearson correlation of profiles).
    *   This requires accessing the raw logits/probabilities of the profile head, which a generic wrapper might abstract away.

### Implementation Strategy

*   **`CerberusModule`**: A `pl.LightningModule` that wraps a `model` and an optional `bias_model`.
    *   **Dual-Head Support**: Handles Profile (NLL) + Count (MSE) losses (BPNet style).
    *   **Bias Correction**: Handles explicit bias subtraction.
*   **`TrainConfig`**: A TypedDict to strictly validate training hyperparameters.
*   **`multinomial_nll`**: Custom implementation of the profile loss.
*   **`torchmetrics`**: Replace manual correlation calculations with robust, DDP-compatible metrics from `torchmetrics`.
