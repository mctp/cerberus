# Pomeranian (Default) vs PomeranianK5: Evaluation and Recommendation

## Overview

Both `Pomeranian` (formerly K9) and `PomeranianK5` share the same Factorized Stem architecture (`[11, 11]`, Expansion 2) and Input/Output dimensions (`2112` -> `1024`). The key difference lies in the **Body** architecture: Kernel Size and Dilation Schedule.

| Feature | PomeranianK5 | Pomeranian (Default) |
| :--- | :--- | :--- |
| **Kernel Size** | 5 | **9** |
| **Dilations** | `[1, 2, 4, 8, 16, 32, 64, 128]` | `[1, 1, 2, 4, 8, 16, 32, 64]` |
| **Max Dilation** | 128 | **64** |
| **Parameters** | ~151,400 | ~153,000 |
| **Theoretical FLOPs** | Lower (Depthwise K=5) | Higher (Depthwise K=9) |
| **Memory Access** | Sparse (Dilation 128) | **Dense/Coherent** (Dilation 64) |

## Neural Network Best Practices Analysis

### 1. Modern Large Kernel Design
Contemporary CNN architectures (e.g., ConvNeXt, RepLKNet) have demonstrated that **larger kernels (K=7, K=9, or larger)** consistently outperform stacks of small kernels (K=3, K=5). Large kernels provide:
- **Larger Effective Receptive Field (ERF)**: Captures broader context immediately.
- **Shape Bias**: Encourages learning global shapes/motifs rather than just local textures.
- **Transformer Approximation**: Large kernels allow CNNs to mimic the global mixing capability of Vision Transformers.

**Winner: Pomeranian (K=9)**

### 2. Hardware Efficiency & Dilation
Dilated convolutions are powerful but can be inefficient on GPUs due to non-coalesced memory access.
- **K5** requires Dilation **128** to achieve the target receptive field. This wide stride often causes cache thrashing and lower GPU utilization.
- **Pomeranian** achieves the same coverage with Dilation **64** because its base kernel is larger. Dilation 64 is generally the upper limit for efficient execution on modern CUDA kernels before performance drops significantly.

**Winner: Pomeranian (K=9)**

### 3. Gradient Flow & Optimization
Larger kernels provide more "paths" for gradient flow and improved overlap between adjacent spatial positions. This spatial smoothing can lead to more stable training and smoother convergence, acting as a structural regularizer.

**Winner: Pomeranian (K=9)**

## Recommendation

**Default Model: Pomeranian**

While K5 is slightly lighter in FLOPs, **Pomeranian** (K9 configuration) aligns better with modern deep learning best practices. By trading a negligible amount of parameters (~1%) and depthwise FLOPs, it gains:
1.  **Better Hardware Utilization**: Avoids inefficient dilation=128.
2.  **Stronger Inductive Bias**: Large kernels (9) capture motif interactions more robustly than medium kernels (5).
3.  **Simpler Schedule**: Standard powers of 2 up to 64.

We recommend setting `Pomeranian` (K9) as the default configuration for future experiments.
