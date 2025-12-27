# bpAITAC Design Patterns and Architecture Review

This document summarizes key design patterns, model architectures, loss functions, and training strategies identified in the `bpAITAC` codebase (`../s2f-models/repos/bpAITAC/models/`). These patterns serve as a reference for potential integration into Cerberus.

## 1. Loss Architectures

The codebase implements a variety of loss functions, primarily focusing on "Composite" losses that combine profile shape prediction with total count prediction.

### Composite Loss Variants
The core pattern is `CompositeLoss = lambda * ProfileLoss + (1-lambda) * ScalarLoss` (or similar weightings).

- **`CompositeLoss`**: 
  - `lambda * CrossEntropy(profile) + MSELogs(scalar)`
  - Standard BPNet formulation.
  
- **`CompositeLossBalanced`**:
  - `lambda * CrossEntropy(profile) + (1-lambda) * MSELogs(scalar)`
  - Explicitly balances the two terms so `lambda` controls the trade-off between shape and count directly (where `lambda` is typically 0.5).

- **`CompositeLossMNLL`**:
  - `lambda * MNLL(profile) + (1-lambda) * MSELogs(scalar)`
  - Uses Multinomial Negative Log Likelihood for the profile instead of standard Cross Entropy.

- **`CompositeLossBalancedJSD`**:
  - Uses Jensen-Shannon Divergence (JSD) instead of Cross Entropy for the profile loss.

### Component Losses
- **`MSELogs`**: 
  - Used for scalar count prediction.
  - Formula: `(log(prediction + 1) - log(target + 1))^2`
  - Handles the large dynamic range of count data better than raw MSE.

- **`PoissonNLL` / `PoissonLoss`**:
  - Standard Poisson Negative Log Likelihood.
  - Can be applied to the profile prediction scaled by total counts.

## 2. Model Architectures

### Core Body (`Body`, `DialatedConvs`)
The backbone is a standard Dilated Convolutional Network (ResNet-style):
- **Structure**: Initial Conv1D -> Stack of `ResBlock` layers.
- **`ResBlock`**: Conv1D (`padding='same'`, `dilation=d`) -> BatchNorm -> ReLU -> Residual Add.
- **Dilation Schedule**: Typically doubles at every layer (1, 2, 4, 8, ...).
- **Default Config**: 9 dilated layers, 300 filters (in `bpAITAC`), filter width 3.

### Heads
The model splits into two heads after the body.

#### Profile Head (`ProfileHead`)
- **Purpose**: Predicts base-pair resolution probability distribution.
- **Structure**: 
  - Deconvolution (`Conv1d` with `kernel_size=25` typ.) to map features to output.
  - Bias Correction: Adds a bias track (unfrozen or fixed) to the logits.
  - **Binning**: Optional `Bin` layer to reduce resolution (e.g., sum every 2bp or 10bp).
  - Output: `Softmax` over sequence length.

#### Scalar Head Variants
This is where `bpAITAC` diverges from `BPnetRep`.

- **Standard BPNet Head (`ScalarHead`)**:
  - `AdaptiveAvgPool1d(1)` (Global Average Pooling) -> Linear -> ReLU.
  - Simple, aggregates global state.

- **bpAITAC Head (`ScalarHeadConvMaxpool`)**:
  - Used in `bpAITAC` and `BPcm` models.
  - **Structure**: Alternating layers of `MaxPool1d` and `Conv1d` + `BatchNorm` + `ReLU`.
  - **Purpose**: Preserves more spatial structure before flattening.
  - **Example Config**: 3 repetitions of [MaxPool(5), Conv1d], followed by flattened FC layers.
  - **Hypothesis**: This design suggests that hierarchical pooling is superior to Global Average Pooling (GAP) for total count prediction in this context. 
    - **Spatial Density**: GAP collapses all spatial information immediately, treating a dense cluster of peaks the same as dispersed peaks if their sum is identical. Hierarchical pooling preserves local density information through intermediate layers.
    - **Sparsity Handling**: ATAC-seq data is often sparse. GAP might "wash out" strong local signals against a large background of zeros. MaxPool can select the strongest signals in local windows, maintaining their magnitude before aggregation.
    - **Feature Interaction**: The interleaved Conv1D layers allow the network to learn interactions between features at increasing scales (e.g., "motif A" and "motif B" within 100bp) before making a final count prediction.

## 3. Training Parameters

Defaults extracted from `train.py`:
- **Epochs**: 200
- **Batch Size**: 20 (Note: Small batch size, likely due to memory or convergence preference).
- **Learning Rate**: 0.001
- **Lambda**: 0.5 (Weighting for profile loss in balanced composite loss).
- **Filters**: 300 (Higher than original BPNet's 64).
- **Patience**: 10 epochs (Early stopping).

## 4. Schedulers & Optimization

### Schedulers
The system supports multiple schedulers, with complex composition:
- **`WarmupCosineDecayLR`**: 
  - Linear Warmup: LR increases from `1e-6` to target LR over `warmup_steps` (default 1000).
  - Cosine Annealing: Decays LR following a cosine curve after warmup.
- **`StepLR`**: Standard step decay.
- **`Warmup`**: Linear warmup only.

### Custom "Variability Cut" Logic
A custom safeguard in the training loop:
- If `(current_loss - prev_loss) / initial_loss > 0.2` (Loss spikes significantly) OR
- If `loss_increased_count >= 3` (Loss increases for 3 consecutive epochs)
- **Action**: Multiply Learning Rate by **0.1**.
- This acts as an aggressive adaptive plateau reduction.

## 5. Learning Schemes

### Bias Correction
- Bias tracks (e.g., Tn5 bias) are explicitly handled.
- They are passed into the `forward` pass.
- In `ProfileHead`, bias is added to the *logits* (pre-softmax) of the prediction: `out = conv(x) + bias`.

### Early Stopping
- Monitors two metrics:
  1. **Validation Loss**: Standard.
  2. **Correlation**: Pearson correlation of total counts.
- Stops if neither improves for `patience` epochs.
- Saves two model checkpoints: `best_loss_model` and `best_corr_model`.

### Input/Output
- **MemmapDataset**: Heavily relies on memory-mapped numpy arrays for efficient data loading of large genomic datasets.
- **Off-by-two Correction**: Logic to handle specific shifting artifacts in ATAC-seq data (`trim_to_center`).

### Data Preprocessing
- **Signal Transformation**: 
  - **Profile Signal**: The input base-pair resolution signal (`bp_counts`) is **NOT** log-transformed. It undergoes **quantile normalization** (scaling) but remains in linear count space.
  - **Scalar Targets**: The scalar total count targets *are* log-transformed (`log(x+1)`) specifically within the `MSELogs` loss function during training, but not in the input data files.
  - **Bias**: Bias tracks are expected to be compatible with logits (often log-likelihoods) as they are added to the pre-softmax output.
