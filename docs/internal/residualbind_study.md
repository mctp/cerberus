# ResidualBind Architecture Study and Implementation Plan for Cerberus

## 1. Overview
This report analyzes the **ResidualBind** architecture from `s2f-models/repos/residualbind` and outlines its implementation within the **Cerberus** framework (`src/cerberus/models`).

ResidualBind is a convolutional neural network designed for predicting RNA-protein binding from sequence data. Its key feature is a dilated residual block that captures dependencies across varying receptive fields.

## 2. Architecture Analysis (Original Keras Model)

The original model is implemented in Keras (`residualbind.py`).

### 2.1. Input
- **Shape**: `(Batch, Length, 4)` (One-hot encoded DNA/RNA).
- **Standard Length**: ~41bp (used in RNAcompete 2013).

### 2.2. Layers
1.  **Stem / Layer 1**:
    -   `Conv1D`: 96 filters, kernel size 11, stride 1, padding 'same'.
    -   `BatchNormalization`.
    -   `ReLU` Activation.
    -   `Dropout`: 0.1.

2.  **Dilated Residual Block**:
    -   **Input**: `x` (Output of Stem).
    -   **Branch**:
        1.  `Conv1D`: 96 filters, kernel size 3, dilation 1, padding 'same'.
        2.  `BatchNormalization`.
        3.  **Loop** (dilations `[2, 4, 8]`):
            -   `ReLU` -> `Dropout(0.1)` -> `Conv1D` (96, k=3, d=d) -> `BatchNormalization`.
            -   This results in a chain of 4 convolutions (d=1, 2, 4, 8) in the branch.
    -   **Skip Connection**: `Add([x, branch_output])`.
    -   **Activation**: `ReLU`.

3.  **Pooling**:
    -   `AveragePooling1D`: pool size 10 (strides=10 implicitly in Keras unless specified, but `AveragePooling1D(pool_size=10)` usually implies stride 10? *Correction*: Keras `AveragePooling1D` default stride IS `pool_size`. So this reduces length by 10x).
    -   `Dropout`: 0.2.

4.  **Prediction Head**:
    -   `Flatten`.
    -   `Dense` (Linear): 256 units.
    -   `BatchNormalization`.
    -   `ReLU` Activation.
    -   `Dropout`: 0.5.
    -   `Dense` (Output): `num_class` units (Linear or Sigmoid).

## 3. Implementation in Cerberus

The implementation will reside in `src/cerberus/models/residualbind.py`. It should follow the pattern established in `src/cerberus/models/baseline_gopher.py` (`GlobalProfileCNN`).

### 3.1. Input/Output Shapes & Adapters

**Input Compatibility:**
- **Cerberus**: `(Batch, 4, Length)` (Channels First).
- **ResidualBind**: `(Batch, Length, 4)` (Channels Last).
- **Action**: The PyTorch implementation will naturally use `(N, C, L)`. No manual transposition is needed if we use `nn.Conv1d` which expects `(N, C, L)`.

**Output Compatibility:**
- **Cerberus**: Expects outputs in the format `(Batch, Output Channels, Bins)`. This is a unified format used for both profile prediction (where `Bins > 1`) and global prediction (where `Bins = 1`).
- **ResidualBind**: Originally designed for global classification/regression, outputting a flat vector of shape `(Batch, num_class)`.
- **The Mapping Strategy**:
    -   In the Cerberus context, `num_class` from ResidualBind corresponds to the total number of predictions we make per sequence.
    -   We define this total number as `num_output_channels * nr_bins`.
    -   **Scenario A (Global Prediction)**: If the task is to predict a single value (e.g. binding affinity) for each output channel (e.g. different proteins), then `nr_bins = 1`. The final layer size becomes `num_output_channels * 1`.
    -   **Scenario B (Coarse Profile)**: If the task is to predict values for regions (bins) of the sequence, `nr_bins > 1`. The final layer size scales accordingly.
    -   **Implementation**: We replace the original Keras `Dense(num_class)` with `nn.Linear(..., num_output_channels * nr_bins)`. We then immediately reshape (view) this output to `(Batch, num_output_channels, nr_bins)` to satisfy the Cerberus interface.

### 3.2. PyTorch Implementation Details

#### Padding 'Same' with Dilations
Keras `padding='same'` keeps the output length equal to the input length. In PyTorch `Conv1d`, we must calculate padding explicitly or use `padding='same'` (available in PyTorch 1.10+).
- Formula: `padding = (dilation * (kernel_size - 1)) // 2` (assuming odd kernel size and stride 1).
- For Kernel 11, Dilation 1: `padding = 5`.
- For Kernel 3, Dilation d: `padding = d`.

#### Dynamic Input Length
`GlobalProfileCNN` uses a dummy forward pass to determine the flattened size before the dense layer. `ResidualBind` should employ the same strategy to support varying `input_len`.

### 3.3. Proposed Code Structure

```python
import torch
import torch.nn as nn

class ResidualBind(nn.Module):
    def __init__(
        self, 
        input_len=2048, 
        output_len=1024, 
        output_bin_size=4, 
        num_input_channels=4, 
        num_output_channels=1
    ):
        super().__init__()
        
        # Output config
        self.nr_bins = output_len // output_bin_size
        self.num_output_channels = num_output_channels
        
        # 1. Stem
        # Keras: Conv1D(96, 11, same) -> BN -> ReLU -> Drop(0.1)
        self.stem = nn.Sequential(
            nn.Conv1d(num_input_channels, 96, kernel_size=11, padding='same'),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 2. Dilated Residual Block
        # We wrap this in a custom module or Sequential with functional logic
        self.residual_block = DilatedResidualBlock(96, kernel_size=3)

        # 3. Pooling
        # Keras: AvgPool(10) -> Drop(0.2)
        self.pooling = nn.Sequential(
            nn.AvgPool1d(kernel_size=10, stride=10),
            nn.Dropout(0.2)
        )

        # 4. Head (Dynamic Flattening)
        # Calculate flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, num_input_channels, input_len)
            x = self.stem(dummy)
            x = self.residual_block(x)
            x = self.pooling(x)
            flatten_size = x.view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256, bias=False), # Keras uses use_bias=False
            nn.BatchNorm1d(256), # BN on 1D vector (Batch, 256)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.nr_bins * self.num_output_channels)
        )

    def forward(self, x):
        # x: (Batch, 4, L)
        x = self.stem(x)
        x = self.residual_block(x)
        x = self.pooling(x)
        x = self.head(x)
        
        # Reshape to (Batch, OutCh, Bins)
        return x.view(x.shape[0], self.num_output_channels, self.nr_bins)

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv_d1 = nn.Conv1d(channels, channels, kernel_size, padding='same', dilation=1, bias=False)
        self.bn_d1 = nn.BatchNorm1d(channels)
        
        self.layers = nn.ModuleList()
        dilations = [2, 4, 8]
        for d in dilations:
            # Block structure inside loop: ReLU -> Drop -> Conv -> BN
            block = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding='same', dilation=d, bias=False),
                nn.BatchNorm1d(channels)
            )
            self.layers.append(block)
            
        self.final_activation = nn.ReLU()

    def forward(self, x):
        identity = x
        
        # Initial dilated conv (d=1)
        out = self.conv_d1(x)
        out = self.bn_d1(out)
        
        # Loop dilated convs
        for layer in self.layers:
            out = layer(out)
            
        # Residual connection
        out = identity + out
        return self.final_activation(out)
```

## 4. Key Differences & Implementation Notes

### 4.1. Batch Normalization
- Keras `BatchNormalization` works on the last axis (channels) by default for 3D inputs.
- PyTorch `BatchNorm1d` works on the second axis (channels) for 3D inputs `(N, C, L)`.
- This matches perfectly since we input `(N, C, L)`.

### 4.2. Bias in Convs
- The original code uses `use_bias=False` for all Convs and the first Dense layer in the head.
- The final output Dense layer uses `use_bias=True`.
- This should be preserved.

### 4.3. Pooling
- The Keras `AveragePooling1D(pool_size=10)` implies a stride of 10.
- If `input_len` is not divisible by 10, Keras/TF might ignore the last partial window or pad depending on settings (default `valid` drops). PyTorch `AvgPool1d` drops the last partial window by default (`ceil_mode=False`).
- **Constraint**: `input_len` should ideally be divisible by 10, or we accept slight data truncation at the end.

### 4.4. Input Dimensions
- `ResidualBind` was trained on short sequences (41bp).
- If applied to long sequences (e.g. 2048bp in Cerberus default), the `AveragePooling1D(10)` will result in ~204 length features.
- The `GlobalProfileCNN` has aggressive pooling.
- `ResidualBind` has only 10x reduction.
- **Memory Implication**: The flattened vector will be much larger for `ResidualBind` on long sequences compared to `GlobalProfileCNN`.
    - `GlobalProfileCNN` (2048 in): 2048 / 64 = 32. 512 channels * 32 = 16,384 features.
    - `ResidualBind` (2048 in): 2048 / 10 = 204. 96 channels * 204 = 19,584 features.
    - They are comparable in magnitude.

## 5. Conclusion
Porting ResidualBind to Cerberus is straightforward. The architecture fits well within the `GlobalProfileCNN` paradigm of "Sequence -> Features -> Flatten -> Projection". The main adaptation is ensuring channel-first ordering and correctly reconstructing the dilated convolution chain with residual addition.
