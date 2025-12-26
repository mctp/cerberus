# Models Implemented in Gopher

This document lists the models implemented in `../s2f-models/repos/gopher/gopher/modelzoo.py`.

## Basenji Variants

*   **`basenji_v2`**
    *   Description: Standard Basenji architecture.
    *   **Input**: `(Batch, Input_Length, 4)`
    *   **Output**: `(Batch, Output_Bins, Num_Tracks)`
    *   **Details**: 
        *   It calculates `window_size = Input_Length // Output_Bins // 2`.
        *   It applies an initial reduction (MaxPool size 2).
        *   It passes data through 11 dilated residual blocks.
        *   It applies a final Average Pooling of size `window_size` to reach the target number of output bins.
        *   It projects to `Num_Tracks` channels using a Dense layer.
    *   Arguments: `input_shape`, `output_shape`, `wandb_config`.

*   **`basenjimod`**
    *   Description: Basenji-based model that can change the output shape based on bin resolution. Defaults correspond to finetuned values. It adapts the number of max-pooling layers based on the output resolution.
    *   Arguments: `input_shape`, `output_shape`, `wandb_config`.

*   **`basenji_w1_b64`**
    *   Description: Base resolution Basenji-based model. It appears to use `w1=True` in `conv_block` which switches to `Conv2D` layers (likely treating the sequence as Height=L, Width=1 or similar).
    *   Arguments: `input_shape`, `output_shape`, `wandb_config`.

*   **`basenji_binary`**
    *   Description: Basenji architecture adapted for binary classification tasks (sigmoid output).
    *   Arguments: `input_shape`, `exp_num`, `wandb_config`.

## Basset

*   **`Basset`**
    *   Description: Implementation of the Basset architecture (Conv -> BatchNorm -> ReLU -> MaxPool x3 -> Flatten -> Dense -> Dropout -> Dense -> Dropout -> Sigmoid).
    *   Arguments: `inputs`, `exp_num`, `padding`, `wandb_config`.

## BPNet Variants

*   **`bpnet`**
    *   Description: BPNet implementation with dilated residual blocks and a profile head. Can optionally apply softplus activation.
    *   Arguments: `input_shape`, `output_shape`, `wandb_config`, `softplus`.

*   **`ori_bpnet`**
    *   Description: "Original" BPNet implementation that returns both profile outputs and total count outputs (multi-head).
    *   Arguments: `input_shape`, `output_shape`, `wandb_config`.

## Binary Classification Models

*   **`conv_binary`** (Baseline CNN - Binary)
    *   Description: A standard convolutional neural network serving as a **baseline**. Architecture: 3 Conv blocks (Conv-BN-Act-Pool) -> Flatten -> Dense -> Dense.
    *   Arguments: `input_shape`, `exp_num`, `wandb_config`.

*   **`residual_binary`**
    *   Description: A residual neural network (using `residual_block`) for binary classification.
    *   Arguments: `input_shape`, `exp_num`, `wandb_config`.

## Profile Prediction Models (Base Resolution)

*   **`conv_profile_task_base`** (Baseline CNN - Profile)
    *   Description: **Baseline CNN** matching the manuscript description.
    *   **Architecture**:
        1.  3 Convolutional Blocks (Conv -> BN -> Act -> MaxPool -> Dropout).
        2.  Flatten.
        3.  FC Block 1 (Bottleneck).
        4.  FC Block 2 (Rescale to `output_len * bottleneck`).
        5.  Reshape to `(output_len, bottleneck)`.
        6.  Convolutional Block.
        7.  **Task-Specific Heads**: Loop over `num_tasks`. Each head has its own `Conv1D` block followed by `Dense(1)`. This allows learning unique features per task.
    *   Arguments: `input_shape`, `output_shape`, `bottleneck`, `wandb_config`.

*   **`conv_profile_all_base`** (Baseline CNN - Profile Shared)
    *   Description: **Baseline CNN** variant with shared head. Same body as `conv_profile_task_base`.
    *   **Head Difference**: Instead of looping over tasks, it uses a single `Dense(num_tasks)` layer projecting from the shared body. This forces all tasks to share the same features up to the final linear combination, reducing parameters but potentially limiting task-specific feature learning.
    *   Arguments: `input_shape`, `output_shape`, `bottleneck`, `wandb_config`.

*   **`residual_profile_task_base`**
    *   Description: Residual model with task-specific heads at base resolution. Replaces the 3 initial Conv blocks with dilated residual blocks.
    *   Arguments: `input_shape`, `output_shape`, `bottleneck`, `wandb_config`.

*   **`residual_profile_all_base`**
    *   Description: Residual base resolution model without task-specific heads.
    *   Arguments: `input_shape`, `output_shape`, `bottleneck`, `wandb_config`.

## Profile Prediction Models (32bp Resolution)

*   **`residual_profile_all_dense_32`**
    *   Description: Residual model with no task-specific heads, designed for 32bp bin resolution.
    *   Arguments: `input_shape`, `output_shape`, `wandb_config`.

*   **`residual_profile_task_conv_32`**
    *   Description: Residual task-specific model designed for 32bp resolution.
    *   Arguments: `input_shape`, `output_shape`, `wandb_config`.

## Profile Prediction Model Analysis

### Comparison of Architectures
The profile prediction models listed above (`conv_profile_*`, `residual_profile_*`) share a common "Global -> Reshape" design pattern that distinguishes them from Fully Convolutional Networks (like BPNet or Basenji).

1.  **Architecture Body**:
    *   **`conv_*`**: Uses standard Convolutional blocks (Conv -> BN -> Act -> Pool) to extract features. Simpler, lower capacity.
    *   **`residual_*`**: Uses Dilated Residual blocks. Deeper, higher capacity, better at capturing long-range dependencies.

2.  **Head Structure (Task-Specific vs Shared)**:
    *   **`*_task_base`**: These models employ **Task-Specific Heads**. After a shared body and bottleneck, the network splits into separate branches (one per task/track), each with its own parameters. This allows the model to learn distinct features for each target in the final layers.
    *   **`*_all_base`**: These models employ a **Shared Head**. The shared body projects directly to all tasks via a single Dense layer. All tasks share the exact same features up to the final projection. This is more parameter-efficient but assumes tasks are highly correlated.

### Input/Output & Binning Capabilities

*   **Fixed Input Size**: Unlike BPNet or Basenji which are often fully convolutional (and thus length-agnostic), these models use a `Flatten()` layer followed by `Dense()` layers. This means they are **tied to a specific input length**. Changing the input length requires redefining the model structure (specifically the Dense layer input size).
*   **Input > Output**: Yes. These models typically use pooling layers to reduce the sequence length, followed by a Dense layer that projects to `output_len * bottleneck`. This allows the input sequence to be significantly longer than the output profile.
*   **Binning Support**: Yes, binning is supported and enforced by the `output_shape` parameter.
    *   The model reshapes the dense output to `[output_len, bottleneck]`.
    *   By setting `output_len` to be `input_len // bin_size`, the model effectively predicts binned profiles.
    *   This "Global Projection" approach differs from BPNet's "Local Pooling" approach. BPNet averages local activations to bin; these models learn a global projection matrix to map the entire input sequence to the binned output.

## Cerberus Implementation Analysis

### Recommendation
The most natural choice to implement first in Cerberus is **`ori_bpnet`**.

**Reasoning:**
1.  **Alignment with Existing Infrastructure**: Cerberus already contains `BPNetLoss` in `src/cerberus/loss.py`, which expects a dual-output model (Profile Logits + Log Counts). `ori_bpnet` is the only Gopher architecture that explicitly outputs this structure.
2.  **Capability Match**: `ori_bpnet` supports base-resolution prediction (standard BPNet) but also explicitly supports binning via its `AveragePooling1D` layer in the head. This aligns perfectly with Cerberus's `nr_bins` configuration.
3.  **Simplicity**: Unlike `Basenji` variants which are significantly deeper and more complex, `ori_bpnet` uses the same fundamental components (Dilated Residual Blocks) as other models but in a standard configuration that is well-understood and supported by the existing loss functions.

### Alternative: Simplified `bpnet` and Default Loss
The simplified **`bpnet`** model in Gopher outputs a single tensor (profiles) without a separate counts head.
*   **Loss Compatibility**: This architecture is directly compatible with Cerberus's default `PoissonNLLLoss`. If the model outputs log-counts (or counts via softplus) for each bin, `PoissonNLLLoss` can supervise both the shape and the magnitude of the signal simultaneously.
*   **Trade-off**: While simpler (single output), this approach entangles the shape and magnitude prediction tasks, which BPNet was specifically designed to decouple (via separate Profile and Count heads) to improve performance and stability. However, for a "quick start" using the default Cerberus loss, this is a viable candidate.

### Input/Output Shape Flexibility
Both `bpnet` and `ori_bpnet` in Gopher demonstrate excellent flexibility regarding input/output shapes, which is crucial for Cerberus:
*   **Dynamic Binning**: Both models calculate `window_size = input_len // output_len` dynamically. This matches Cerberus's `DataModule` logic where `nr_bins` is derived from configuration.
*   **Resolution Adaptation**: By using `AveragePooling1D(pool_size=window_size)` (or similar logic) in the head, these models can adapt to any target resolution (1bp, 32bp, 128bp) defined by the user without changing the core architecture. This implementation pattern should be preserved when porting to Cerberus to ensure the model respects the `output_bin_size` config.
