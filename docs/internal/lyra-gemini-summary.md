# Lyra Architecture Implementation Summary

Here is the implementation of the Lyra architecture, derived directly from the provided manuscript and supplementary code listings. This implementation reconstructs the model from the OCR-scanned code snippets, ensuring variable names and logic align with the paper's mathematical descriptions (specifically the "Projected Gated Convolution" and "S4D" sections).

## Implementation Notes & Generalization

The following code has been standardized for modern PyTorch (using `torch.fft` and `nn.functional.gelu`).

### Flexibility of Inputs

- **General Input Processing**: The implementation generalizes the input processing. While the paper focuses on one-hot encoded DNA/Protein sequences, the `Lyra` class accepts a `d_input` argument. This allows it to handle continuous features or pre-learned embeddings simply by changing the initialization arguments.
- **Encoder**: The encoder is a simple `Linear` layer. For tokenized inputs (like NLP), this could be easily swapped for `nn.Embedding`.

### Generalization of Architecture

- **Layer Depth & Width**: The `Lyra` class accepts `pgc_configs` (a list of tuples) allowing for arbitrary stacking of PGC layers with varying hidden dimensions.
- **Sequence Length**: The architecture utilizes fully convolutional and recurrent (SSM) principals. It handles variable sequence lengths `L` dynamically during the forward pass (see `S4DKernel` generating filters based on `L`).
- **S4D Parameterization**: The S4D block is implemented with the "kernel" approach (computing the impulse response and using FFT convolution), which is $O(N \log N)$ and highly efficient for long sequences.

## Mathematical Description of the Forward Pass

### 1. Notation Definitions
- $B$: Batch size
- $L$: Sequence length
- $D_{in}$: Input dimension
- $D_{model}$: Model hidden dimension
- $D_{out}$: Output dimension
- $\odot$: Element-wise multiplication (Hadamard product)
- $\ast$: Convolution operation
- $\mathcal{F}$: Fast Fourier Transform (FFT)
- $\mathcal{F}^{-1}$: Inverse Fast Fourier Transform (iFFT)

### 2. Component Layers

#### 2.1 RMSNorm
For an input vector $\mathbf{x} \in \mathbb{R}^d$:
$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \mathbf{\gamma}
$$
where $\mathbf{\gamma} \in \mathbb{R}^d$ is a learnable scale parameter and $\epsilon$ is a small constant for stability.

#### 2.2 Projected Gated Convolution (PGC)
Given an input sequence $\mathbf{U} \in \mathbb{R}^{L \times D_{model}}$ and expansion factor $E$:

**Projection & Normalization:**
Project input to an intermediate dimension $D' = E \cdot D_{model}$ and normalize.
$$
[\mathbf{X}, \mathbf{V}] = \text{RMSNorm}(\mathbf{U}\mathbf{W}_{in} + \mathbf{b}_{in})
$$
where $\mathbf{X}, \mathbf{V} \in \mathbb{R}^{L \times D'}$. The output is split into two chunks along the feature dimension.

**Depthwise Convolution Path:**
Apply a depthwise convolution (groups=$D'$) with kernel size $k$ to $\mathbf{X}$.
$$
\mathbf{X}_{conv} = \text{DepthwiseConv1d}(\mathbf{X})
$$

**Gating Mechanism:**
Element-wise multiplication of the convolved features and the value path.
$$
\mathbf{G} = \mathbf{X}_{conv} \odot \mathbf{V}
$$

**Output Projection:**
Project back to model dimension.
$$
\mathbf{Y} = \text{Dropout}(\text{RMSNorm}(\mathbf{G}\mathbf{W}_{out} + \mathbf{b}_{out}))
$$

#### 2.3 S4D Kernel & Layer

**Kernel Generation ($S4DKernel$):**
For hidden dimension $H$ and state dimension $N$, the kernel $\mathbf{K} \in \mathbb{R}^{H \times L}$ is computed using learnable parameters $\log \Delta, C, \log A_{real}, A_{imag}$.

**Parameter Materialization:**
For each channel $h \in \{1, \dots, H\}$ and state component $n \in \{1, \dots, N/2\}$:
$$
\Delta_h = \exp(\log \Delta_h)
$$
$$
A_{h,n} = -\exp(\log A_{real_{h,n}}) + i \cdot A_{imag_{h,n}}
$$
$$
\bar{C}_{h,n} = C_{h,n} \frac{\exp(\Delta_h A_{h,n}) - 1}{A_{h,n}}
$$

**Kernel Construction:**
For time steps $t = 0, \dots, L-1$:
$$
\mathbf{K}_{h, t} = 2 \cdot \text{Re}\left( \sum_{n=1}^{N/2} \bar{C}_{h,n} \exp(\Delta_h A_{h,n} t) \right)
$$

**Forward Pass ($S4D$):**
Given input sequence $\mathbf{U} \in \mathbb{R}^{H \times L}$ (transposed):

**FFT Convolution:**
Compute the output $\mathbf{Y}_{conv}$ via the Convolution Theorem.
$$
\mathbf{Y}_{conv} = \mathcal{F}^{-1}\Big(\mathcal{F}(\mathbf{U}, 2L) \odot \mathcal{F}(\mathbf{K}, 2L)\Big)_{[:L]}
$$
(Note: FFT is padded to $2L$ to perform linear convolution)

**Skip Connection:**
Add a learnable skip connection $\mathbf{D} \in \mathbb{R}^H$.
$$
\mathbf{Y}_{skip} = \mathbf{Y}_{conv} + \mathbf{D} \odot \mathbf{U}
$$

**Activation & Output:**
Apply GELU, Dropout, and a Gated Linear Unit (GLU) projection.
$$
\mathbf{Z} = \text{Dropout}(\text{GELU}(\mathbf{Y}_{skip}))
$$
$$
\mathbf{Y}_{out} = \text{GLU}(\text{Conv1d}_{1\times 1}(\mathbf{Z}))
$$
(where GLU splits the input into $\mathbf{A}, \mathbf{B}$ and outputs $\mathbf{A} \odot \sigma(\mathbf{B})$)

### 3. Lyra Model Architecture
The complete forward pass for an input $\mathbf{X}_{in} \in \mathbb{R}^{B \times L \times D_{in}}$:

**Input Encoder:**
$$
\mathbf{H}_0 = \mathbf{X}_{in}\mathbf{W}_{enc} + \mathbf{b}_{enc}
$$

**Local Context (PGC Layers):**
Apply $M$ stacked PGC layers.
$$
\mathbf{H}_{local} = \text{PGC}_M(\dots \text{PGC}_1(\mathbf{H}_0))
$$

**Global Context (S4D Blocks):**
Transpose $\mathbf{H}_{local}$ to shape $(B, D_{model}, L)$. For each layer $l$:
$$
\mathbf{Z} = \text{RMSNorm}(\mathbf{H}_{l-1})
$$
$$
\mathbf{Z} = \text{Dropout}(\text{S4D}(\mathbf{Z}))
$$
$$
\mathbf{H}_l = \mathbf{H}_{l-1} + \mathbf{Z} \quad (\text{Residual Connection})
$$
(Repeat for `num_s4` layers)

**Pooling:**
Average over the sequence dimension $L$.
$$
\mathbf{E} = \frac{1}{L} \sum_{t=1}^L \mathbf{H}_{final}[:, t]
$$

**Decoder:**
$$
\mathbf{Y}_{pred} = \text{Dropout}(\mathbf{E})\mathbf{W}_{dec} + \mathbf{b}_{dec}
$$
