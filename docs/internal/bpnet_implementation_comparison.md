# BPNet Implementation Comparison

## Overview

This document provides a detailed technical comparison of four BPNet implementations:
1. `bpnet-lite` (PyTorch)
2. `chrombpnet-pytorch` (PyTorch)
3. `basepairmodels` (Keras/TF, inferred from `bpnet-lite` and partial dump)
4. `bpreveal` (Keras/TF)
5. `gopher` (Keras/TF)

The review focuses on neural architecture, mathematical form, and loss functions. The goal is to provide a specification sufficient for re-implementation.

## High-Level Summary

All implementations share the core BPNet architecture: a convolutional neural network with dilated residual convolutions predicting base-resolution profiles (multinomial) and total counts (MSE of log-counts).

**Biggest Differences:**
1.  **Residual Block Structure:**
    *   `bpnet-lite` (BPNet class), `chrombpnet-pytorch`, and `bpreveal` use a **Post-Activation** style (or standard activation): $X_{out} = X_{in} + \text{ReLU}(\text{Conv}(X_{in}))$.
    *   `basepairmodels` (as implemented in `bpnet-lite`'s `BasePairNet` class) uses a **Pre-Activation** style: $X_{out} = X_{in} + \text{Conv}(\text{ReLU}(X_{in}))$.
2.  **Padding & Cropping:**
    *   `bpnet-lite` uses `padding='same'` (or equivalent padding) to maintain sequence length throughout.
    *   `chrombpnet-pytorch` and `bpreveal` use `padding='valid'` and explicitly crop the residual path to match the valid convolution output. This results in "pyramidal" data flow where input length must be significantly larger than output length.
3.  **Counts Head Control Input:**
    *   `bpnet-lite` and `chrombpnet-pytorch` explicitly concatenate the log-sum of control tracks (if present) to the pooled embeddings before the final dense layer for count prediction.
    *   `bpreveal` (in the `soloModel` analyzed) does not appear to explicitly concatenate control counts in the architecture code provided, relying potentially on the model learning from the sequence or bias integration occurring at the `combinedModel` stage.

## Detailed Architecture Specification

### 1. Core BPNet Structure

The "Consensus" BPNet architecture (following `chrombpnet-pytorch` and `bpnet-lite.BPNet`) is defined as follows:

#### Inputs
*   **Sequence**: One-hot encoded DNA sequence tensor $S \in \{0,1\}^{B \times 4 \times L_{in}}$.
*   **Control Tracks** (Optional): Base-resolution control signal $C \in \mathbb{R}^{B \times N_{ctl} \times L_{in}}$.

#### Initial Convolution
*   **Operation**: 1D Convolution followed by ReLU.
*   **Kernel Size**: 21 (typical).
*   **Filters**: 64 (typical).
*   **Formula**: $X_0 = \text{ReLU}(\text{Conv1D}_{k=21}(S))$

#### Dilated Residual Layers (Iterated $i = 1 \dots N_{layers}$)
*   **Operation**: Dilated 1D Convolution with residual connection.
*   **Kernel Size**: 3.
*   **Dilation**: $2^i$ (1, 2, 4, 8, ...).
*   **Filters**: 64.
*   **Activation placement**:
    *   *Standard (BPNet-lite, ChromBPNet-pytorch, BPReveal)*:
        $$X_{conv} = \text{ReLU}(\text{Conv1D}_{k=3, d=2^i}(X_{i-1}))$$
        $$X_i = \text{Crop}(X_{i-1}) + X_{conv}$$
    *   *BasePairModels (Original)*:
        Based on `bpnet-lite`'s `BasePairNet` implementation:
        $$X_{act} = \text{ReLU}(X_{i-1})$$
        $$X_{conv} = \text{Conv1D}_{k=3, d=2^i}(X_{act})$$
        $$X_i = \text{Crop}(X_{i-1}) + X_{conv}$$
    *   *Note*: `Crop` ensures dimensions match if 'valid' padding is used. `BPNet-lite` uses padding to avoid cropping.
    *   *Note on BasePairModels*: The `basepairmodels` repository delegates architecture definition to an external dependency `genomics-DL-archsandlosses`. The behavior is inferred from the `BasePairNet` class in `bpnet-lite` which claims compatibility.

#### Profile Head
*   **Input**: Output of last residual layer $X_{final}$.
*   **Control Integration**: If control tracks exist, they are concatenated to $X_{final}$ along the channel dimension.
*   **Operation**: 1D Convolution (Width 75 typical).
*   **Output**: Logits $Y_{profile} \in \mathbb{R}^{B \times N_{out} \times L_{out}}$.
*   **Formula**: $Y_{profile} = \text{Conv1D}_{k=75}(\text{Concat}(X_{final}, C))$

#### Counts Head
*   **Input**: Output of last residual layer $X_{final}$.
*   **Pooling**: Global Average Pooling (over sequence length).
*   **Control Integration**:
    *   Calculate log-sum of control tracks: $c_{log} = \log(1 + \sum_{pos} C)$.
    *   Concatenate $c_{log}$ to pooled embeddings.
*   **Operation**: Dense (Linear) layer.
*   **Output**: Log-counts $Y_{counts} \in \mathbb{R}^{B \times N_{out}}$.
*   **Formula**:
    $$h = \text{AvgPool}(X_{final})$$
    $$h' = \text{Concat}(h, c_{log})$$
    $$Y_{counts} = \text{Dense}(h')$$

### 2. Gopher Implementation Specifics

The `gopher` repository provides two BPNet variants: `bpnet` and `ori_bpnet`.

*   **`ori_bpnet`**:
    *   Closest to the standard BPNet architecture (64 filters, 10 layers).
    *   **Dual Heads**: Predicts both profiles and total counts.
    *   **Binning Support**: Unique among surveyed implementations, it includes an `AveragePooling1D` layer in the profile head to support output binning (`window_size`), allowing the output resolution to be lower than the input resolution.
    *   **Head Architecture**: Uses `Conv2DTranspose` (kernel 25x1) followed by `AveragePooling1D` for the profile head, which is an unusual choice compared to the standard `Conv1D`.
    *   **Padding**: Uses `padding='same'`, matching `bpnet-lite` but differing from `chrombpnet-pytorch`.

*   **`bpnet`**:
    *   Appears to be a simplified or modified version.
    *   **Filters**: Defaults to 256 filters (vs 64 in `ori_bpnet`).
    *   **Single Head**: Only outputs profiles; lacks a dedicated counts head output in the provided code.
    *   **Activation**: Supports optional `softplus` activation on the profile output.

### 3. ChromBPNet (Bias Correction)

ChromBPNet combines a frozen **Bias Model** (learning enzyme bias) with a trainable **Accessibility Model** (learning sequence motifs).

*   **Models**: $M_{bias}$ and $M_{acc}$ are both BPNet architectures.
*   **Profile Combination**:
    $$P_{final} = P_{acc} + P_{bias}$$
    (Logits are added, equivalent to multiplying probabilities in probability space).
*   **Counts Combination**:
    $$C_{final} = \log(\exp(C_{acc}) + \exp(C_{bias}))$$
    (Log-Sum-Exp combination, representing additive contribution in count space).

### 3. Loss Functions

#### Profile Loss: Multinomial NLL
The loss treats the profile prediction as a multinomial distribution.
$$L_{profile} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{L} y_{true}^{(i,j)} \log(p_{pred}^{(i,j)})$$
Where $p_{pred} = \text{Softmax}(Y_{profile})$.
*Note*: `chrombpnet-pytorch` explicitly includes the log-factorial constant term of the Multinomial PDF, while others might omit it as it depends only on labels (constant during optimization).

#### Counts Loss: Mean Squared Error (Log Space)
$$L_{counts} = \frac{1}{N} \sum_{i=1}^{N} (\log(1 + y_{true\_total}^{(i)}) - Y_{counts}^{(i)})^2$$
*Note*: `BasePairNet` in `bpnet-lite` uses $\log(2 + y)$ for legacy compatibility, while others use $\log(1 + y)$.

#### Total Loss
$$L_{total} = L_{profile} + \lambda L_{counts}$$

## Implementation Comparisons

| Feature | `bpnet-lite` (BPNet) | `chrombpnet-pytorch` | `basepairmodels` (via `bpnet-lite` `BasePairNet`) | `bpreveal` | `gopher` (ori_bpnet) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Residual Block** | Post-Activation | Post-Activation | **Pre-Activation** | Post-Activation | Post-Activation |
| **Padding** | Same | Valid (Cropping) | Same | Valid (Cropping) | Same |
| **Conv Act** | ReLU after Conv | ReLU after Conv | ReLU before Conv (in loop) | ReLU after Conv | ReLU after Conv |
| **Count Head Ctl** | Concatenated | Concatenated | Concatenated | Not explicitly seen | Not explicitly seen |
| **Bias Integration** | ChromBPNet Class | ChromBPNet Class | N/A | `combinedModel` Class | N/A (Separate logic) |
| **Framework** | PyTorch | PyTorch | Keras/TF | Keras/TF | Keras/TF |

## Re-implementation Spec Checklist

To strictly re-implement **BPNet (standard)**:
1.  Use **Post-Activation** residual blocks: `x = x + relu(conv(x))`.
2.  Use **Valid Padding** if matching `chrombpnet`/`bpreveal` behavior (requires input > output). Use **Same Padding** for simpler dimension handling (`bpnet-lite`).
3.  **Profile Head**: Convolution kernel 75. Concatenate control tracks if used.
4.  **Counts Head**: Global Average Pool. Concatenate $\log(1+\sum \text{controls})$. Dense layer.
5.  **Loss**: Multinomial NLL (Profile) + MSE (Log Counts).

To re-implement **BasePairModels (Atlas)** legacy models:
1.  Use **Pre-Activation** residual blocks: `x = x + conv(relu(x))`.
2.  Use $\log(2 + \text{counts})$ for counts target transformation (check specific model config).
