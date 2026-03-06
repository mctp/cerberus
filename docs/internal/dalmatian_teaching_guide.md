# Dalmatian

## End-to-End Bias-Factorized Sequence-to-Function Model for ATAC-seq

*A Teaching Guide: From Biology to Architecture to Loss Design*

*Understanding ChromBPNet, its limitations, and the Dalmatian solution*

---

# Part I: The Problem

Before describing any model, we need to understand what ATAC-seq measures, what confounds it, and why decomposition matters.

## 1.1 What ATAC-seq Measures

ATAC-seq measures how accessible DNA is to enzymatic cutting. The Tn5 transposase is introduced into cells, where it preferentially inserts sequencing adapters into "open" regions of chromatin—places where transcription factors (TFs) have displaced nucleosomes, exposing the underlying DNA. After sequencing, you count how many Tn5 insertion events occurred at each base pair across the genome. Regions with many insertions are "peaks"—putative regulatory elements.

The raw data is a base-resolution signal track: at every position in the genome, you have a count of Tn5 insertions. This signal has two components that we want to separate.

## 1.2 The Two Components of the Signal

### Component 1: Enzymatic Bias (the "noise")

Tn5 does not cut DNA randomly. It has intrinsic sequence preferences—it prefers certain short DNA motifs (~10–19 bp) over others. This means that even in completely inaccessible DNA (background regions with no TF binding), Tn5 will cut more at some positions than others, purely because of the local DNA sequence. This is the Tn5 bias.

> **Key finding from ChromBPNet (Pampari et al., 2025):** Tn5 bias strongly affects the *shape* of the base-resolution profile (which positions get more or fewer cuts) but does NOT affect the *total number* of cuts in a region. Total counts in peaks are driven entirely by regulatory activity, not enzyme preference.

### Component 2: Regulatory Signal (the "signal")

When a TF binds DNA, it physically protects a ~10–40 bp region from Tn5 cutting, creating a "footprint"—a dip in the insertion profile at the binding site flanked by elevated signal where the displaced nucleosome boundaries are. The pattern, depth, and shape of these footprints encode information about which TFs are bound, how strongly they bind, and how they cooperate with other TFs.

This is the regulatory signal. It operates at multiple scales: individual TF motifs (4–25 bp), cooperative motif pairs (50–200 bp spacing), and nucleosome-scale patterns (150–200 bp). Extracting it cleanly requires removing the Tn5 bias.

## 1.3 Why Separation Matters

If you train a model on raw ATAC-seq profiles without separating these components, the model learns both Tn5 bias and regulatory signal entangled together. ChromBPNet showed that this entanglement corrupts model interpretation: when you ask "what sequence features drive this predicted profile?" using methods like DeepLIFT, the answer is contaminated by Tn5 sequence preferences that have nothing to do with biology. Separating the components gives you clean regulatory attributions that faithfully reflect TF binding.

## 1.4 The Mathematical Decomposition

We model the observed base-resolution profile as a composition of bias and regulatory signal. Let:

- **k = (k₁, k₂, ..., k_L)** be the observed read counts at L positions (L = 1,000 bp in ChromBPNet)
- **n = Σᵢ kᵢ** be the total read count in the window
- **qᵢ = kᵢ / n** be the observed profile shape (a probability distribution, Σqᵢ = 1)

We want to decompose the per-position cutting probability into:

```
P(cut at position i) = P_bias(i | local sequence) × P_signal(i | regulatory context)
                       \___________________________/   \_______________________________/
                       Tn5 enzyme preference             TF binding, nucleosome effects
```

And the total count into:

```
n_total = n_bias + n_signal
          \_____/   \______/
     background      regulatory
     cutting rate    accessibility boost
```

The profile shape decomposes **multiplicatively** (bias modulates the probability of cutting at each position), while total counts decompose **additively** (regulatory activity adds reads on top of the background rate). This asymmetry is biologically motivated and experimentally validated by ChromBPNet.

---

# Part II: How ChromBPNet Solves This

## 2.1 The Sequence-to-Profile Model

Both ChromBPNet and Dalmatian are sequence-to-profile models. ChromBPNet uses a 2,114 bp input window and L = 1,000 bp output; Dalmatian uses the Pomeranian backbone with a 2,112 bp input window and L = 1,024 bp output. The input is a one-hot encoded DNA sequence (4 channels: A, C, G, T). The output has two heads:

- **Profile head:** A vector of L logits, z = (z₁, ..., z_L). The predicted profile shape is obtained by applying the softmax function: pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ). This gives a probability distribution over positions.
- **Counts head:** A single scalar s representing the predicted log-total-counts: s = log(n̂). The predicted count at each position is then n̂ᵢ = exp(s) × pᵢ.

### What is softmax and why use it?

The softmax function converts any vector of real numbers (logits) into a probability distribution. Given logits z = (z₁, ..., z_L):

```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

Every output is positive, and they sum to 1. This is natural for modeling the profile shape: we want to predict the probability that a read falls at each position. The model learns logits (unconstrained real numbers) and the softmax converts them to valid probabilities. A logit of 0 means "average probability"; positive logits mean higher-than-average; negative means lower.

### What is the multinomial distribution?

If you have n total reads and each read independently falls at position i with probability pᵢ, the counts (k₁, ..., k_L) follow a multinomial distribution. The probability of observing a specific count vector is:

```
P(k₁, ..., k_L | n, p) = n! / (k₁! · k₂! · ... · k_L!) × p₁^k₁ × p₂^k₂ × ... × p_L^k_L
```

The negative log-likelihood of this distribution is the profile loss used by ChromBPNet. Taking the negative log and dropping the combinatorial constant (which doesn't depend on model parameters):

```
MNLL = -Σᵢ kᵢ · log(pᵢ)
```

This is equivalent to the cross-entropy between the observed count distribution qᵢ = kᵢ/n and the predicted distribution pᵢ, scaled by n. Minimizing MNLL forces the model's predicted profile shape to match the observed profile shape.

## 2.2 ChromBPNet's Two-Stage Training

### Stage 1: Train the Bias Model Alone

A small convolutional neural network (128 filters, 4 dilated layers, ~81 bp receptive field) is trained exclusively on background (non-peak) regions. In these regions, there is no regulatory signal—the observed profile shape is driven entirely by Tn5 sequence preferences. The bias model learns to predict this profile shape from DNA sequence.

The bias model is deliberately small and local. Its receptive field (~81 bp) means it can only "see" a small window of DNA around each output position. Tn5 bias is a local phenomenon (~19 bp motif), so this is sufficient. But the small receptive field physically prevents the model from learning long-range patterns like TF motif spacing.

> ⚠️ **Pitfall:** If background regions accidentally contain weakly accessible regulatory elements (false negatives from peak calling), the bias model will try to learn TF motifs alongside Tn5 bias. ChromBPNet addresses this with a manually tuned signal threshold: non-peak windows with suspiciously high total counts are excluded from bias model training. This requires iteratively decreasing the threshold and checking TF-MoDISco until no TF motifs appear in the bias model's learned features.

### Stage 2: Freeze the Bias Model, Train the Signal Model

The bias model's weights are frozen. A larger model (512 filters, 8 dilated layers, ~1,041 bp receptive field) is then trained on peak regions. Importantly, the signal model does not receive a pre-computed residual as its training target. Instead, it predicts its own independent profile and counts from the same DNA sequence. These predictions are combined with the frozen bias model's predictions using the combination rules below, and the combined prediction is compared to the observed data.

Because the bias model is frozen, the loss gradient flows only into the signal model. This gradient implicitly teaches the signal model to explain whatever the bias model is missing. At positions where the bias model already predicts the observed signal accurately, the combined prediction is correct and the gradient is near zero—the signal model learns to stay silent there. At positions where the bias model underpredicts (e.g., at a TF footprint that Tn5 bias alone cannot explain), the gradient pushes the signal model to increase its predicted probability. So the signal model learns the "residual"—but this residual learning emerges from the gradient of the combined loss, not from an explicit subtraction step.

The combination rule is:

```
pᵢ_combined = pᵢ_bias × pᵢ_signal     (multiplicative in probability space)
n_combined = n_signal + γ × n_bias      (additive in linear count space)
```

**The γ factor:** Because the bias model was trained on background regions (low total counts) but is now being applied to peak regions (high total counts), its count predictions need rescaling. ChromBPNet computes γ = exp(mean[log(n_bias + 1) − log(n_obs + 1)]) over training examples. This is a fixed scalar computed once between the two stages.

## 2.3 Limitations of Staged Training

- **Frozen errors:** Any mistakes the bias model makes (imperfect Tn5 representation, residual TF contamination) are permanently baked in. The signal model must compensate for these errors instead of the system correcting them jointly.
- **Manual threshold tuning:** The 'fraction' parameter for filtering background regions must be set by trial and error, with TF-MoDISco run at each setting to verify the bias model is clean. This is labor-intensive and error-prone.
- **No joint optimization:** The two models cannot negotiate the boundary between bias and signal. In ambiguous regions (e.g., weak peaks, TF motifs that overlap Tn5 preferences), the decomposition is determined by the staging order rather than a jointly optimal solution.
- **Fixed γ:** The count scaling factor is a single number computed between stages, not a learned parameter that adapts during training.

---

# Part III: The Dalmatian Architecture

Dalmatian replaces the two-stage procedure with end-to-end joint training. Both sub-networks train simultaneously, and a three-term loss function (the DalmatianLoss) forces them to specialize. The name follows the dog-breed convention (BPNet, Pomeranian) and alludes to a two-component spotted pattern.

## 3.1 Architecture Overview

```
              DNA Sequence (B, 4, input_len)
                         |
              +----------+----------+
              |                     |
         BiasNet               SignalNet
        ~35K params            ~2–3M params
        RF ~80 bp              RF ~1,000+ bp
              |                     |
         bias_logits          signal_logits
         bias_log_counts      signal_log_counts
              |                     |
              +----------+----------+
                         |
                  Combination Layer
                         |
                  DalmatianOutput
             (combined + decomposed fields)
```

Both sub-networks are instances of the Pomeranian architecture with different hyperparameters. They share the same input and produce the same output type (profile logits + log-counts). Their outputs are then combined.

## 3.2 The Bias Sub-Network: Design Rationale

The bias network is deliberately constrained along two axes: receptive field and capacity. Each constraint serves a specific purpose.

### Receptive Field Constraint

A convolutional neural network can only use information within its receptive field (RF)—the region of the input that influences a given output position. The RF is determined by the network's layer count, kernel sizes, and dilation rates.

**Why this matters:** Tn5 bias is a local phenomenon. The enzyme's sequence preference spans ~19 bp. A network with an ~80 bp RF can comfortably capture this and some flanking context, but it physically cannot represent relationships between sequence elements that are 200+ bp apart. Long-range regulatory grammar (TF motif spacing, enhancer-promoter syntax) spans hundreds of base pairs. By limiting the RF, we make it architecturally impossible for the bias network to learn regulatory signal.

This is the strongest form of constraint available in neural network design—it is not a soft penalty that can be overcome by strong gradients, but a hard structural limit. A network simply cannot compute a function that depends on inputs outside its receptive field, regardless of how many parameters it has or how long it trains.

> 💡 **Connection to neural network theory:** This exploits the locality property of convolutional networks. Unlike fully connected networks (which see all inputs), CNNs have bounded receptive fields determined by architecture. This is the same principle used in multi-scale feature extraction (e.g., U-Nets), where different network branches are designed to capture features at different spatial scales.

### Capacity Constraint

Even within its ~80 bp window, the bias network has only ~35K parameters (64 filters per layer), compared to ~2–3M for the signal network. This limits how complex a function it can represent.

**Why this matters:** Some TF motifs are short enough to fit within 80 bp (e.g., GATA = 6 bp, CTCF = 19 bp). A very powerful network with 80 bp RF could potentially learn to recognize these motifs. The capacity constraint makes this harder: with only 64 filters, the network can represent a limited number of patterns, and the L_sparse loss (described below) provides additional pressure against it.

> ⚠️ **Pitfall:** If the bias network is too small, it may not even capture the full complexity of Tn5 bias. ChromBPNet showed that a single position weight matrix (PWM) is insufficient—there are multiple variations of the Tn5 motif that require a CNN to represent. The ~35K parameter budget was chosen to be large enough for Tn5 bias but small enough to discourage TF motif learning.

### BiasNet Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| filters | 64 | Small capacity; ~35K total parameters |
| dilated layers | 4 | Shallow: limits depth and RF |
| dilations | [1, 1, 2, 4] | Max dilation 4 keeps RF ~80 bp |
| dil_kernel_size | 5 | Small spatial kernel |
| conv_kernel_size | 11 | Single stem layer (not factorized) |
| profile_kernel_size | 21 | Small smoothing for profile output |

## 3.3 The Signal Sub-Network: Design Rationale

The signal network is large and deep, with a receptive field exceeding 1,000 bp. This is necessary because regulatory grammar operates at multiple scales: individual TF motifs (4–25 bp), cooperative TF pairs (50–200 bp), and nucleosome positioning (~150–200 bp periodicity).

### SignalNet Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| filters | 256 | Large capacity; ~2–3M total parameters |
| dilated layers | 8 | Full depth for long-range features |
| dilations | [1,1,2,4,8,16,32,64] | Exponentially growing RF > 1,000 bp |
| dil_kernel_size | 9 | Standard Pomeranian kernel size |
| conv_kernel_size | [11, 11] | Factorized 2-layer stem |
| profile_kernel_size | 45 | Standard profile smoothing |

**Note on Pomeranian vs. BPNet:** ChromBPNet uses BPNet (dilated residual convolutions) as its backbone. Dalmatian uses Pomeranian (PGC blocks, factorized stems). These are structurally different, so the hyperparameters are not directly comparable. The design intent is preserved: a small/local bias network and a large/global signal network.

---

# Part IV: How the Sub-Networks' Outputs are Combined

This section derives the combination rules from first principles, showing why each choice is both biologically and mathematically motivated.

## 4.1 Profile Shape: Addition in Logit Space

### The biological model

We model the probability of a Tn5 cut at position i as a product of two factors: the enzyme's intrinsic preference for the local DNA sequence, and a modulation from regulatory activity (TF binding, nucleosome displacement):

```
P(cut at i) ∝ P_bias(i) × P_signal(i)
```

This multiplicative model is natural: Tn5 has a baseline cutting rate that varies by position (bias), and regulatory elements scale that rate up or down (signal).

### The mathematical equivalence

Each sub-network outputs logits: z_bias = (z₁_b, ..., z_L_b) and z_signal = (z₁_s, ..., z_L_s). The softmax of each gives the sub-network's probability distribution. The key insight is:

```
softmax(z_bias + z_signal)ᵢ
  = exp(zᵢ_b + zᵢ_s) / Σⱼ exp(zⱼ_b + zⱼ_s)
  = exp(zᵢ_b) · exp(zᵢ_s) / Σⱼ exp(zⱼ_b) · exp(zⱼ_s)
  = [softmax(z_bias)ᵢ × softmax(z_signal)ᵢ] / Z
```

where Z = Σⱼ softmax(z_bias)ⱼ × softmax(z_signal)ⱼ is a renormalization constant. So adding logits then applying softmax is equivalent to multiplying the two probability distributions and renormalizing. This is precisely the multiplicative decomposition we want.

> 💡 **Why use logit addition instead of probability multiplication?** Three reasons. (1) Numerical stability: logits are unconstrained real numbers, so addition never causes overflow or underflow. Multiplying small probabilities can underflow to zero. (2) Gradient flow: gradients flow cleanly through addition. With probability multiplication, you need to handle the renormalization constant's gradient. (3) Identity element: zero logits leave the distribution unchanged (adding zero is a no-op). This is critical for the DalmatianLoss, as we will see.

### The identity element property

If the signal network outputs all zeros (z_signal = 0), then:

```
combined_logits = z_bias + 0 = z_bias
softmax(combined_logits) = softmax(z_bias)
```

The combined profile reduces exactly to the bias-only profile. The signal model is perfectly invisible. This is mathematically exact—no approximation. This property is exploited by both the zero-initialization strategy and the L_sparse loss term.

> ⚠️ **Pitfall—this is NOT the same as probability-space multiplication:** softmax(a + b) ≠ softmax(a) × softmax(b) in general (the latter is not even normalized). They differ by the renormalization constant Z. This means the Dalmatian decomposition has slightly different mathematical properties from ChromBPNet's. In practice, both achieve the same goal (multiplicative modulation), but the decomposed probabilities are not identical between the two approaches.

## 4.2 Total Counts: Log-Sum-Exp

### The biological model

Total reads in a window are the sum of bias-driven cuts and signal-driven cuts:

```
n_total = n_bias + n_signal
```

This is straightforward: in a peak region, you get the background cutting rate from Tn5 bias, plus additional cuts from the regulatory accessibility boost.

### The implementation

Each sub-network predicts log-counts (s_bias, s_signal), meaning it predicts log(n). We need to compute log(n_bias + n_signal) = log(exp(s_bias) + exp(s_signal)). This is the log-sum-exp function:

```
combined_log_counts = logsumexp(s_bias, s_signal)
                    = log(exp(s_bias) + exp(s_signal))
```

> 💡 **Why not just add log-counts?** Adding in log space computes the *product* (log(a) + log(b) = log(a×b)), not the sum. We need the sum of counts in linear space, which requires logsumexp. This is a standard operation in numerical computing—PyTorch's `torch.logsumexp` is numerically stable (it subtracts the max before exponentiating to prevent overflow).

### The identity element for logsumexp

For logsumexp(a, b), the identity element is b = −∞ (since exp(−∞) = 0, and logsumexp(a, −∞) = a). This is important for initialization: if we want the signal model to contribute zero counts at the start of training, we need its log-count output to be a very large negative number, not zero.

```
If signal_log_counts = 0:   combined = log(exp(bias) + exp(0)) = log(exp(bias) + 1)
If signal_log_counts = -10: combined = log(exp(bias) + 0.000045) ≈ bias
```

With zero-initialized count output, the signal model contributes exactly 1 phantom read (exp(0) = 1) to every prediction. With bias initialized to −10, it contributes ~0.000045 reads—truly negligible.

## 4.3 Summary of Combination Rules

| Component | Operation | Math Space | Biology |
|-----------|-----------|-----------|---------|
| Profile shape | logit addition | Log-probability | Multiplicative: Tn5 preference × regulatory modulation |
| Total counts | logsumexp | Log-count | Additive: total = bias_reads + signal_reads |
| Profile identity | signal_logits = 0 | Exact | Signal model invisible |
| Counts identity | signal_log_cts = −∞ | Exact | Signal model invisible |

---

# Part V: The DalmatianLoss

The DalmatianLoss is the core innovation. It forces the bias and signal networks to specialize using three complementary terms, each with a clear biological motivation and well-defined gradient routing.

## 5.1 Notation

For a batch of B examples, each example has:

- **x:** one-hot encoded DNA sequence (4 × input_len)
- **k = (k₁, ..., k_L):** observed read counts at L output positions
- **n = Σᵢ kᵢ:** total observed read count in the window
- **peak_status ∈ {0, 1}:** whether this example is centered on a peak (1) or is a background region (0)

The model produces, for each example:

- **z_b, s_b:** bias model logits (L values) and log-counts (scalar)
- **z_s, s_s:** signal model logits (L values) and log-counts (scalar)
- **z_c = z_b + z_s:** combined logits
- **s_c = logsumexp(s_b, s_s):** combined log-counts

## 5.2 The Base Loss (Matching ChromBPNet)

Both L_recon and L_bias use the same base loss function, matching ChromBPNet's formulation. It has two parts:

### Profile loss: Multinomial Negative Log-Likelihood (MNLL)

```
pᵢ = softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)          (predicted profile shape)
L_profile = -Σᵢ kᵢ · log(pᵢ)                        (multinomial NLL)
```

This measures how well the predicted profile shape matches the observed profile. Minimizing it pushes p toward q = k/n (the observed distribution). The counts kᵢ act as importance weights: positions with more observed reads contribute more to the loss.

### Counts loss: Mean Squared Error on log-counts

```
L_counts = (log(1 + n) - log(1 + n̂))²
where n̂ = exp(s) is the predicted total count, s is the log-count output
```

**Why log(1 + n), not log(n)?** The +1 prevents log(0) in regions with zero reads (common in background). This is sometimes called "log1p" in numerical computing.

### Combined base loss with auto-scaled weighting

```
base_loss(z, s, k, n) = L_profile + λ · L_counts
where λ = median(n_obs over validation set) / 10
```

**Why this λ?** The profile loss (MNLL) scales linearly with n (more reads = larger loss magnitude). The counts loss (MSE on logs) does not scale with n. To keep the two terms balanced across different sequencing depths, λ is set proportional to the typical count magnitude. The /10 is an empirically chosen dampening factor from ChromBPNet.

## 5.3 Term 1: L_recon (Combined Reconstruction)

```
L_recon = base_loss(z_c, s_c, k, n)     // applied to ALL examples
```

This is the standard prediction loss. The combined model (bias + signal) should accurately predict the observed ATAC-seq profile everywhere—in peaks and in background. Both sub-networks receive gradients because z_c and s_c depend on both.

**Gradient flow:** z_c = z_b + z_s. By the chain rule, ∂L_recon/∂z_b = ∂L_recon/∂z_c and ∂L_recon/∂z_s = ∂L_recon/∂z_c. Both sub-networks receive identical profile gradients from L_recon because the combined logits z_c were computed from both z_b and z_s—both are ancestors of the loss in the computation graph. The gradient with respect to the combined logits is the difference between the predicted probability and the observed count proportion: ∂L_profile/∂zᵢ_c = pᵢ - kᵢ/n (a well-known result for cross-entropy with softmax).

## 5.4 Term 2: L_bias (Bias-Only Reconstruction on Background)

```python
# In the loss function:
bias_out = ProfileCountOutput(
    logits=output.bias_logits[non_peak_mask],
    log_counts=output.bias_log_counts[non_peak_mask])
L_bias = base_loss(bias_out, target[non_peak_mask])
```

In non-peak (background) regions, the observed signal is almost entirely Tn5 cutting bias—there is no regulatory activity. Therefore, the bias model alone should be able to explain the observed profile and counts in these regions. L_bias provides direct supervision for this.

### Clarification: How gradient routing works here

There are two completely independent mechanisms at work, and it is important not to confuse them:

**Mechanism 1 — Which examples contribute (Python indexing):** The line `output.bias_logits[non_peak_mask]` is ordinary tensor slicing. It selects which examples from the batch are used to compute L_bias. The `non_peak_mask` comes from the `peak_status` metadata in the batch. This is just Python selecting rows from a tensor—autograd is not involved in this decision.

**Mechanism 2 — Which parameters receive gradients (computation graph):** This is entirely determined by which network produced the tensors being differentiated. `output.bias_logits` was produced by `self.bias_model(x)` during the forward pass. The signal model's parameters are NOT ancestors of this tensor in the computation graph—they were used in a separate `self.signal_model(x)` call that produced `output.signal_logits` instead. When PyTorch calls `.backward()` on L_bias, it traces the computation graph backward from `bias_logits` and finds only bias network parameters. The signal network is excluded because its parameters simply aren't connected to the tensors being differentiated.

**These two mechanisms are independent:** Even if you computed L_bias on ALL examples (not just non-peak), only the bias network would receive gradients. The example selection (Mechanism 1) determines the biological meaning of the loss term. The computation graph (Mechanism 2) determines the gradient routing. No stop-gradient operations, no `detach()` calls, and no special autograd hooks are needed.

> 💡 **Why the decomposed output design matters:** The DalmatianOutput carries both combined tensors (logits, log_counts) and decomposed tensors (bias_logits, signal_logits, etc.) as separate fields. Because the decomposed tensors were produced by separate forward passes through separate networks, they naturally have separate computation graphs. If instead we tried to extract the bias contribution from the combined output (e.g., by subtracting signal from combined), both networks would be in the graph, and gradient routing would require explicit stop-gradient operations.

**Biological justification:** ChromBPNet demonstrated that a bias model alone achieves high profile concordance (median JSD = 0.74) in background regions but poorly predicts counts (r = −0.16). L_bias trains toward this same capability directly.

> ⚠️ **Pitfall:** If non-peak regions accidentally contain regulatory signal (false-negative peaks), L_bias will push the bias model to explain TF-driven signal, corrupting its Tn5 representation. Mitigation: filter background regions to exclude windows with suspiciously high total counts (see Section 7.2).

## 5.5 Term 3: L_sparse (Signal Sparsity on Background)

```
L_sparse = mean(|z_s|) + mean(|s_s|)    // applied to NON-PEAK examples only
```

This is an L1 penalty on the signal model's raw outputs in background regions. L1 regularization is the standard tool for inducing sparsity—it pushes values toward exactly zero rather than merely small values (unlike L2 regularization, which pushes toward small but non-zero values).

**Why L1 on logits specifically?** Because zero logits are the identity element for the profile combination. When signal_logits = 0, the combined profile equals the bias-only profile (see Section 4.1). L_sparse pushes the signal model toward this identity state in background regions, making it invisible there. If we used L2 instead, logits would shrink toward zero but never reach it, meaning the signal model would always contribute some small perturbation to the profile in background regions.

**Gradient flow:** Same principle as L_bias but in reverse. `output.signal_logits` and `output.signal_log_counts` were produced by `self.signal_model(x)`. The bias network's parameters are not ancestors of these tensors. When `.backward()` runs on L_sparse, only signal network parameters receive gradients. Again, this has nothing to do with peak status—it is purely a consequence of which network produced which tensors.

> 💡 **Connection to neural network practice:** L1 regularization (also called Lasso) is a foundational technique for feature selection and sparsity. In traditional machine learning, it drives irrelevant feature weights to exactly zero. Here, we apply it to the output activations rather than the weights, achieving the same effect at the functional level: the signal model's function output is driven to zero in regions where it shouldn't contribute.

## 5.6 Total Loss

```
L_total = L_recon + λ_bias · L_bias + λ_sparse · L_sparse
```

| Term | Default λ | Tune range | Applied to | Gradients to |
|------|-----------|-----------|-----------|-------------|
| L_recon | 1.0 | — | All examples | Both networks |
| L_bias | 1.0 | 0.5–2.0 | Non-peak only | Bias network only |
| L_sparse | 0.1 | 0.05–0.5 | Non-peak only | Signal network only |

## 5.7 Why This Forces Separation: A Walk-Through

Consider what each sub-network "experiences" during training:

### The bias network's perspective

It receives gradients from two sources: L_recon (be part of an accurate combined prediction everywhere) and L_bias (predict background regions alone). Both push it toward learning Tn5 bias, which is exactly what we want. In peak regions, the bias network contributes the Tn5 component of the combined prediction and gets gradients to refine that contribution.

### The signal network's perspective

It receives gradients from two sources: L_recon (be part of an accurate combined prediction everywhere) and L_sparse (be silent in background regions). In background regions, these conflict: L_recon wants it to help predict, but L_sparse suppresses it. However, because the bias model is already learning to explain background (via L_bias), the combined prediction is already accurate from bias alone—so L_recon's gradients in background regions are small. L_sparse wins, and the signal model learns to output zero in background.

In peak regions, L_sparse is not applied, and L_recon provides gradients for the signal component of the combined prediction. The bias model explains the Tn5 component; the remaining signal (TF footprints, regulatory grammar) must be captured by the signal model, because the bias model's ~80 bp receptive field physically cannot represent long-range patterns.

### What if both learn the same thing? (Mode collapse analysis)

Suppose the signal model started learning Tn5 bias (duplicating the bias model). Then:

- **L_sparse would penalize it:** Tn5 bias produces non-zero logits in background regions, triggering the L1 penalty.
- **L_recon would not reward it:** The bias model already explains the Tn5 component, so duplicating it in the signal model doesn't improve the combined prediction—it slightly worsens it (because the Tn5 contribution gets doubled).

So learning Tn5 bias is actively punished (via L_sparse) and not rewarded (via L_recon). The signal model's optimal strategy is to be silent in background and capture only the residual regulatory signal in peaks.

> ⚠️ **Pitfall—what if the bias model is too slow to converge?** If the bias model hasn't learned Tn5 bias yet, L_recon will push the signal model to explain background signal (because the combined prediction is inaccurate). The signal model could then "get stuck" learning Tn5 bias before L_sparse overwhelms it. The zero-initialization strategy mitigates this: the signal model starts at exactly zero output, giving the bias model a head start via L_bias before the signal model activates.

---

# Part VI: Zero-Initialization Strategy

A critical training detail that replaces explicit warm-up schedules with a simpler, more principled approach.

## 6.1 The Problem: Early Training Instability

At the start of training, both sub-networks have random weights. The signal model's random outputs would interfere with the bias model's learning: L_recon would see a combined prediction that is the sum of two random outputs, providing noisy gradients to both models.

## 6.2 The Solution: Start the Signal Model at the Identity

We initialize the signal model's final output layers so that it produces the identity element for both combination rules:

### Profile head: Initialize to zero

The final layers of the profile head (profile_pointwise and profile_spatial Conv1d layers) are initialized with zero weights and zero biases. This means signal_logits = 0 for all positions. Since adding zero logits is the exact identity for profile combination, the combined profile equals the bias-only profile at initialization.

### Count head: Initialize to −10

The final layer of the count MLP is initialized with zero weights and a bias of −10.0. This means signal_log_counts ≈ −10. Since exp(−10) ≈ 0.000045, the signal model contributes negligible counts via logsumexp.

**Why −10 and not 0?** The identity element for logsumexp is −∞, not 0. If we used bias = 0, then exp(0) = 1, and the signal model would contribute exactly 1 phantom read to every prediction: combined = log(exp(bias_counts) + 1). In high-coverage regions this is negligible, but it's mathematically incorrect as an identity. With −10, the contribution is truly negligible (0.000045 reads).

## 6.3 The Effect on Training Dynamics

At epoch 0:

- **Combined output = bias-only output.** L_recon trains only the bias model (signal model's contribution is zero, so its gradients from L_recon are also small).
- L_bias trains the bias model directly on background.
- **L_sparse is already satisfied** (signal outputs are zero).

As training progresses, the bias model converges on Tn5 bias. The signal model's intermediate representations develop (even though its outputs are zero, its internal features learn from the L_recon gradients that flow through the combination layer). Eventually, the signal model's features become informative enough that its output layers activate, and it begins contributing regulatory signal in peak regions.

> 💡 **Connection to neural network practice:** Zero-initializing the output of a residual branch is a well-established technique. It was introduced in ReZero (Bachlechner et al., 2020) and used in FixUp initialization (Zhang et al., 2019). The idea: in any architecture where a branch's output is added to a main path, initializing the branch to zero ensures stable training—the main path works immediately, and the branch gradually learns to contribute. This is exactly our setting: the signal model is a branch added to the bias model's logits.

---

# Part VII: Training Procedure

## 7.1 Data Pipeline

- **Input:** ATAC-seq bigWig (Tn5 insertion counts) + MACS3 narrowPeak calls.
- **Sampler:** PeakSampler provides peak-centered windows (peak_status = 1) interleaved with GC-matched background windows (peak_status = 0). Default ratio: 1 background per 1 peak.
- **Windows:** 2,112 bp input → 1,024 bp output. Peaks are jittered ±500 bp around summits each epoch; backgrounds are fixed.

## 7.2 Background Filtering

ChromBPNet's paper showed that non-peak regions can contain weak regulatory signal missed by peak callers. If L_bias supervises the bias model on such regions, it corrupts the Tn5 representation. Recommended filtering:

```
threshold = quantile(peak_total_counts, 0.01) × 0.8
exclude background windows where total_counts > threshold
```

This removes the top ~20% of background windows by signal intensity, keeping only regions clearly dominated by enzymatic preference.

## 7.3 Optimization

- **Optimizer:** Adam, lr = 0.001 for both networks.
- **Weight decay:** Standard weight decay (same value for both networks, e.g. 1e-4). Differential weight decay was considered but dropped as redundant given the large capacity asymmetry (35K vs 2–3M parameters); see Section 13.4.
- **Early stopping:** Monitor validation L_recon. Stop after 5 epochs without improvement. Restore best checkpoint.
- **Loss weights:** Constant from epoch 0 (λ_bias = 1.0, λ_sparse = 0.1). No warm-up needed because zero-init handles the cold-start problem.

## 7.4 Hyperparameter Sensitivity

### λ_sparse (signal sparsity weight)

This is the primary tunable knob. It controls the balance between the signal model's ability to contribute to peak predictions (L_recon pulls it active) and the pressure to stay silent in background (L_sparse pushes it to zero).

- **Too low (λ_sparse < 0.05):** Signal model learns Tn5 bias alongside TF signal. Detected by running TF-MoDISco on signal model contribution scores and finding Tn5 motifs.
- **Too high (λ_sparse > 0.5):** Signal model is suppressed even in weak peaks. Detected by low recall of known TF motifs in signal contribution scores.
- **Just right (0.1–0.3):** Signal model is silent in background, active in peaks. TF-MoDISco on signal model shows only TF motifs; TF-MoDISco on bias model shows only Tn5 motifs.

### Peak:background ratio

The default is 1:1, meaning half the batch contributes to L_bias and L_sparse. If you increase to 1:10 (more background per peak), the auxiliary losses fire on more examples per batch, effectively increasing their influence. The λ values should be re-tuned if the ratio changes.

---

# Part VIII: Monitoring and Quality Control

A trained Dalmatian model must be verified using the same interpretation-driven QC framework established by ChromBPNet. Training loss convergence alone is insufficient—you must inspect what each sub-network has learned.

## 8.1 TF-MoDISco Verification

TF-MoDISco is a motif discovery algorithm that clusters subsequences with high DeepLIFT contribution scores into non-redundant motifs (Contribution Weight Matrices, CWMs). Run it separately on each sub-network:

- **Bias model CWMs must show ONLY Tn5 bias motifs.** If TF motifs (CTCF, SP1, GATA, etc.) appear: increase λ_sparse, tighten background filtering, or verify RF is limited.
- **Signal model CWMs must show known TF motifs WITHOUT Tn5 contamination.** Count contribution scores should be naturally clean (Tn5 affects shape, not counts).

## 8.2 Marginal Footprinting

Insert a motif of interest into many random background sequences, predict with the model, and average the predicted profiles. This reveals the model's learned response to that motif in isolation.

- **Insert Tn5 motifs:** Bias model should show strong footprints. Signal model should show ZERO response.
- **Insert TF motifs:** Signal model should show cell-type-specific footprints (e.g., GATA1 in K562 but not GM12878). Bias model should show only the Tn5 component of the local sequence.
- **Negative control:** Insert motifs for TFs not expressed in the cell type. Signal model should show no footprint. Any footprint from the signal model indicates it has learned a spurious association.

## 8.3 Runtime Diagnostics

- **mean(|signal_logits|) in background:** Track per epoch. Should stabilize near zero.
- **Bias-only profile JSD in background:** How well does the bias model alone predict background profiles? Target: ~0.74 (ChromBPNet's reported value).
- **Combined counts Pearson r in peaks:** Target: ≥ 0.70.
- **Combined profile JSD in peaks:** Target: median ≤ 0.62.

---

# Part IX: Inference and Interpretation

## 9.1 Decomposed Predictions

At inference, three predictions are available for any genomic window:

- **Combined:** softmax(z_b + z_s), exp(logsumexp(s_b, s_s)). The full model prediction including bias and signal.
- **Bias-only:** softmax(z_b), exp(s_b). The predicted Tn5 cutting pattern from sequence alone.
- **Signal-only (bias-corrected):** softmax(z_s), exp(s_s). The debiased regulatory signal. Analogous to disconnecting the bias submodel in ChromBPNet.

## 9.2 Contribution Scores

DeepLIFT/DeepSHAP attribution is computed through each sub-network separately. For the profile head, per-position contribution scores c_{i,j} are weighted by predicted probabilities:

```
cᵢ = Σⱼ Pⱼ × c_{i,j}
```

where c_{i,j} is the contribution of input feature i to the logit at output position j, and Pⱼ is the predicted probability at position j. This produces profile contribution scores (sensitive to Tn5 bias in uncorrected models) and count contribution scores (naturally resistant to Tn5 bias even without correction—a key ChromBPNet finding).

## 9.3 Variant Effect Prediction

ChromBPNet introduced five complementary measures for predicting the impact of genetic variants. Dalmatian supports all of them. For a variant with alleles a₁ and a₂:

- **logFC = log(n̂(a₂)) − log(n̂(a₁)):** Log fold-change of predicted total coverage.
- **JSD = √[(D(p₁||m) + D(p₂||m)) / 2]:** Jensen-Shannon distance between bias-corrected allelic profiles, where m is the pointwise mean and D is KL divergence.
- **AAQ = max(percentile(n̂(a₁)), percentile(n̂(a₂))):** Active allele quantile relative to genome-wide peak coverage.
- **IES = logFC × JSD:** Integrative effect size (combines magnitude and shape change).
- **IPS = |logFC| × JSD × AAQ:** Integrative prioritization score (best for variant classification).

---

# Part X: Dalmatian vs. ChromBPNet

| | ChromBPNet | Dalmatian |
|---|---|---|
| Training | 2 stages: bias → freeze → signal on residual | Single stage, end-to-end |
| Disentanglement | Hard: staged freezing | Soft: DalmatianLoss + arch asymmetry |
| Profile combination | pᵢ^tf × pᵢ^bias (probability space) | logit addition (equivalent, better numerics) |
| Counts combination | n^tf + γ*n^bias (fixed γ) | logsumexp (no fixed γ needed) |
| Error correction | Bias frozen; errors permanent | Bias correctable via L_recon backprop |
| Threshold tuning | Manual iterative fraction search + TF-MoDISco | Background filter + λ_sparse tuning |
| Warm-up | N/A (staged) | Zero-init (implicit, no schedule) |
| Backbone | BPNet (dilated residual CNN) | Pomeranian (PGC blocks) |
| Guarantee | Hard separation (frozen weights) | Soft separation (requires QC verification) |

---

# Part XI: Variant — Continuous Signal-Intensity Weighting

The default DalmatianLoss uses a binary peak_status label to decide where L_bias and L_sparse apply: fully on in non-peak regions, fully off in peak regions. This section describes a variant that replaces the binary switch with a continuous weight derived from observed signal intensity, addressing several limitations of the binary approach.

## 11.1 The Problem with Binary Peak Labels

Peak calling (by MACS3 or any tool) imposes a hard decision boundary on a continuous phenomenon. Regulatory activity exists on a spectrum: some regions are strongly accessible (deep peaks with hundreds of reads), some are weakly accessible (shallow peaks near the significance threshold), and some are genuine background (sparse Tn5 cutting with single-digit reads). The binary label forces a region with 50 reads just below the peak threshold into the same category as a region with 3 reads, even though the 50-read region likely contains some regulatory signal.

This creates two problems. First, L_bias supervises the bias model on some regions that contain weak regulatory signal, potentially corrupting its Tn5 representation. This is exactly the contamination problem that ChromBPNet's manual threshold tuning was designed to address. Second, L_sparse applies full sparsity pressure on these same ambiguous regions, preventing the signal model from learning weak but real regulatory patterns.

## 11.2 Continuous Weighting by Signal Intensity

Instead of a binary switch, compute a per-example weight from the observed total counts:

```
wᵢ = σ((threshold - log(1 + nᵢ)) / temperature)
```

where:

- **σ is the sigmoid function:** σ(x) = 1 / (1 + exp(−x)). This smoothly maps any real number to the range (0, 1).
- **nᵢ is the observed total read count** in window i.
- **threshold is the log-count level** below which signal is believed to be dominated by Tn5 bias. A natural choice: the 0.01th quantile of peak total counts (matching ChromBPNet's background filter).
- **temperature controls the sharpness** of the transition. Low temperature → sharp (approaches binary). High temperature → soft (gradual transition). A reasonable starting point: temperature such that the sigmoid transitions over ~1 order of magnitude of counts.

The weight has intuitive behavior:

- **nᵢ ≪ threshold (low counts, clear background):** wᵢ ≈ 1. Full L_bias and L_sparse pressure.
- **nᵢ ≫ threshold (high counts, clear peak):** wᵢ ≈ 0. Only L_recon contributes.
- **nᵢ ≈ threshold (ambiguous region):** wᵢ ≈ 0.5. Partial auxiliary loss pressure.

## 11.3 Modified Loss

```
L_total = L_recon
        + λ_bias   · mean(wᵢ · L_bias_i)
        + λ_sparse · mean(wᵢ · L_sparse_i)
```

where L_bias_i and L_sparse_i are the per-example bias reconstruction and signal sparsity losses, and wᵢ is the continuous weight for example i. Note that wᵢ is computed from observed counts and detached from the computation graph (no gradients flow through the weight itself).

## 11.4 Advantages

- **No dependence on peak caller for training dynamics:** Peak calls are still used for the sampler (centering windows on summits), but the loss weighting is driven by actual signal intensity. This decouples the model's disentanglement from the peak caller's sensitivity.
- **Smooth treatment of ambiguous regions:** Weak regulatory regions get partial L_sparse pressure rather than all-or-nothing. The signal model can learn to contribute partial signal in these regions without being fully suppressed.
- **Absorbs the background filtering problem:** ChromBPNet's manual 'fraction' parameter and the background signal threshold are replaced by the threshold and temperature parameters, which are more interpretable and can be set directly from the data distribution.

## 11.5 Pitfalls and Considerations

- **Target-dependent loss weighting:** The weight wᵢ depends on the observed counts nᵢ, which are also the regression target for the counts head. This creates a subtle coupling: regions with higher counts (where the model needs to be most accurate) get less auxiliary loss pressure. This is intentional—high-count regions are peaks where we want the signal model active—but it means the model sees a biased view of the data in the auxiliary losses. The weight must be detached from the computation graph so no gradients flow through it.
- **Temperature sensitivity:** If temperature is too high, L_sparse applies non-trivially even in strong peaks, starving the signal model. If too low, you recover the binary case and lose the benefit. The temperature should be validated by checking that strong peaks have w ≈ 0 and clear background has w ≈ 1.
- **L_bias now partially supervises on weak peaks:** With continuous weights, the bias model receives some L_bias gradient from regions that contain weak regulatory signal. Whether this corrupts the Tn5 representation or actually helps it (by exposing it to the Tn5 component within partially accessible regions) is an empirical question. The bias model's limited receptive field should protect it from learning the regulatory component, but TF-MoDISco verification remains essential.
- **Two new hyperparameters:** threshold and temperature replace the binary peak label but introduce their own tuning. The threshold can be set from data (quantile of peak counts). The temperature is genuinely new and must be validated.

## 11.6 Implementation Sketch

```python
class ContinuousDalmatianLoss(nn.Module):
    def __init__(self, base_loss, bias_weight=1.0, sparse_weight=0.1,
                 threshold=None, temperature=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.bias_weight = bias_weight
        self.sparse_weight = sparse_weight
        self.threshold = threshold   # set from data
        self.temperature = temperature

    def _get_weights(self, obs_total):
        log_counts = torch.log1p(obs_total).detach()
        return torch.sigmoid(
            (self.threshold - log_counts) / self.temperature)

    def forward(self, output, target, obs_total):
        L_recon = self.base_loss(
            ProfileCountOutput(output.logits, output.log_counts),
            target)

        w = self._get_weights(obs_total)  # (B,) in [0, 1]

        # Per-example bias reconstruction (weighted)
        L_bias_per = self.base_loss.per_example(
            ProfileCountOutput(output.bias_logits, output.bias_log_counts),
            target)
        L_bias = (w * L_bias_per).mean()

        # Per-example signal sparsity (weighted)
        L_sparse_per = (output.signal_logits.abs().mean(dim=-1)
                      + output.signal_log_counts.abs())
        L_sparse = (w * L_sparse_per).mean()

        return (L_recon
                + self.bias_weight * L_bias
                + self.sparse_weight * L_sparse)
```

This variant can be used as a drop-in replacement for DalmatianLoss. The base DalmatianLoss with binary peak_status remains the default, recommended starting point. The continuous variant is recommended when peak calling quality is uncertain, when the dataset contains many weak regulatory regions, or when the binary threshold produces visible artifacts in TF-MoDISco QC.

---

# Part XII: Known Risks and Failure Modes

## 12.1 Background Contamination

If false-negative peaks exist in the background set, L_bias will push the bias model to explain TF-driven signal. With only ~35K parameters and ~80 bp RF, the bias model will distort its Tn5 representation trying to accommodate these signals. Detection: TF-MoDISco on bias model shows TF motifs. Mitigation: tighten the signal threshold for background filtering.

## 12.2 Signal Model Learning Tn5 Bias

If λ_sparse is too low, the signal model's L_recon gradients in background regions may overwhelm the sparsity penalty. It will learn Tn5 bias because doing so improves the combined prediction (even though the bias model is also learning it). Detection: TF-MoDISco on signal model shows Tn5 motifs. Mitigation: increase λ_sparse.

## 12.3 Signal Model Starved in Weak Peaks

If λ_sparse is too high, the sparsity pressure may suppress the signal model even in weak peaks (regions with low regulatory signal but real TF binding). Detection: known TF motifs missing from signal model contribution scores in weak peaks. Mitigation: decrease λ_sparse, or apply L_sparse only to windows well below the peak threshold.

## 12.4 No Hard Guarantee of Separation

ChromBPNet's staged freezing provides a mathematical guarantee: the frozen bias model cannot change, so the signal model can only learn residuals. Dalmatian provides no such guarantee—the DalmatianLoss is a soft constraint. In pathological cases (bad hyperparameters, corrupted background data), the networks could fail to separate. TF-MoDISco QC is mandatory, not optional.

## 12.5 The Logit Addition ≠ Probability Multiplication Issue

As discussed in Section 4.1, softmax(a + b) is not exactly equal to softmax(a) × softmax(b) renormalized. They differ by a normalization constant Z. This means the decomposed bias and signal profiles from Dalmatian are not identical to those from ChromBPNet, even if the combined prediction matches. Whether this matters empirically is an open question that should be tested by comparing TF-MoDISco motifs and variant effect scores between the two approaches on the same data.

---

# Part XIII: Alternative Disentanglement Mechanisms (Considered and Dropped)

An earlier version of this proposal described five layered mechanisms for forcing separation. The final Dalmatian design retains two and drops three. This section explains each decision.

## 13.1 Retained: Architectural Asymmetry

The bias network's limited receptive field (~80 bp) and low parameter count (~35K) versus the signal network's large RF (~1,000+ bp) and high capacity (~2–3M). This is the single strongest disentanglement mechanism—a hard structural constraint that no amount of training can overcome. Retained as Mechanism 1 in the final design without modification.

## 13.2 Retained (Modified): Background Region Regularization

The original proposal used an L2 penalty on signal outputs in non-peak regions. The final design uses L1 (the L_sparse term). L1 is strictly better here because zero is the identity element for logit addition, and L1 drives values to exact zero while L2 only drives them toward zero asymptotically. L2 would mean the signal model always contributes some small nonzero perturbation in background, never fully reducing to the bias-only profile. With L1, the signal model can achieve perfect silence. Additionally, the DalmatianLoss adds L_bias (direct bias model supervision on background), which the original proposal lacked—this is the more important change, because it gives the bias model a positive training signal rather than only relying on the signal model's suppression.

## 13.3 Dropped: Gradient Reversal Adversarial Training

The original proposal included a discriminator that predicts peak vs. non-peak from the bias network's features, with reversed gradients to prevent the bias model from encoding regulatory information.

### Why it was dropped

- **Unnecessary given L_bias + L_sparse:** The DalmatianLoss achieves the same goal (bias model doesn't learn TF motifs) through direct supervision. L_bias pulls the bias model toward Tn5 patterns. L_sparse prevents the signal model from learning Tn5 patterns. Together with the RF constraint, this is sufficient—there is no residual problem for adversarial training to solve.
- **Training instability:** Adversarial training (minimax optimization) is notoriously unstable. It requires careful balancing of the discriminator's learning rate, gradient penalty terms, and update schedules. GAN-style training failures (mode collapse, oscillation, discriminator domination) are well-documented and would add fragility to an otherwise stable training procedure.
- **Hyperparameter cost:** The gradient reversal strength (λ_adv), discriminator architecture, and discriminator learning rate are all additional hyperparameters. DalmatianLoss has only two (λ_bias, λ_sparse) and is much easier to tune.
- **Unclear biological necessity:** The concern was that short TF motifs (e.g., GATA = 6 bp) could be learned by the bias model's ~80 bp RF. But ChromBPNet demonstrated that a bias model with ~81 bp RF trained on background regions does not learn TF motifs when the background data is properly filtered. The architectural constraint is sufficient.

**Status:** Dropped from the default configuration. Could be added as a fourth loss term if TF-MoDISco QC reveals persistent bias model contamination that L_sparse + L_bias + RF constraint cannot resolve. We consider this unlikely.

## 13.4 Dropped: Differential Weight Decay

The original proposal applied stronger L2 weight decay to the bias network (1e-3) versus the signal network (1e-4) to further constrain the bias model's capacity.

### Why it was dropped as a named mechanism

It is not wrong—differential weight decay is harmless and may help marginally. But it is redundant given the already dramatic capacity asymmetry (35K vs. 2–3M parameters). Adding L2 weight decay to a 35K-parameter network that already has a limited RF provides minimal additional constraint beyond what the architecture enforces. It also risks impeding the bias model's ability to learn the full complexity of Tn5 bias (which ChromBPNet showed requires a CNN, not just a PWM).

**Status:** Not explicitly included as a disentanglement mechanism, but standard weight decay (same value for both networks) is applied as a normal regularization practice. If you want to experiment with differential decay, it is trivially added via separate optimizer parameter groups.

## 13.5 Dropped: Mutual Information Minimization

The original proposal included an optional MI penalty between the bias and signal networks' internal representations to penalize statistical dependence.

### Why it was dropped

- **Computational cost:** MI estimation between high-dimensional feature maps requires either variational bounds (MINE, Barber-Agakov) or kernel-based estimators, both of which are expensive and high-variance. This adds significant compute per batch for uncertain benefit.
- **Unclear target:** We want the networks' *outputs* to decompose cleanly, not necessarily their internal representations. Two networks can have correlated internal features (e.g., both detect GC content) while still producing functionally independent outputs. Penalizing representation MI could harm both networks' feature learning.
- **The problem it solves doesn't exist:** MI minimization addresses the case where both networks learn redundant representations. But the RF constraint already prevents this at the architectural level—the bias model's features are inherently local, and the signal model's features are inherently long-range. There is little redundancy to penalize.

**Status:** Dropped. No scenario has been identified where MI minimization would solve a problem that the other mechanisms cannot.

## 13.6 Summary: From Five Mechanisms to Three

| Mechanism | Status | Reason |
|-----------|--------|--------|
| Architectural asymmetry | Retained | Hard constraint; strongest mechanism |
| Background regularization | Retained (L1 + L_bias) | Core of DalmatianLoss; improved from L2-only |
| Gradient reversal | Dropped | Unstable; redundant given L_bias + L_sparse |
| Differential weight decay | Dropped | Redundant given capacity asymmetry |
| MI minimization | Dropped | Expensive; solves a non-existent problem |

---

# Part XIV: Training Curriculum

Should the model see a specific sequence of data during training, or is uniform mixing sufficient? This section analyzes the question and recommends an approach.

## 14.1 The Curriculum Question

In curriculum learning (Bengio et al., 2009), a model is trained on examples ordered from easy to hard. The intuition is that learning simple patterns first provides a scaffold for learning complex patterns later. In our setting, the question is: should the bias model be allowed to learn Tn5 bias from background regions before the signal model sees peak regions?

This is essentially what ChromBPNet does—its two-stage training is an extreme curriculum where stage 1 is 100% background (bias model only) and stage 2 is 100% peaks (signal model only, bias frozen). The Dalmatian approach mixes all data from epoch 0. Is there a middle ground?

## 14.2 The Case for Background-First Curriculum

### The argument

Start with epochs of mostly background data (e.g., 90% background, 10% peaks) then gradually shift toward the normal ratio. This gives the bias model a head start at learning Tn5 bias from clean background supervision (L_bias), before the signal model activates in peaks and L_recon gradients start flowing through both networks.

### What it would look like

```
Epoch 1–3:   90% background, 10% peaks
Epoch 4–6:   70% background, 30% peaks
Epoch 7–9:   50% background, 50% peaks
Epoch 10+:   normal ratio (50% background, 50% peaks)
```

### When it might help

- **Very large signal models** that converge quickly and could learn Tn5 bias before the bias model has a chance to claim it.
- **Low λ_sparse settings** where the sparsity pressure is insufficient to prevent early signal model overfitting.
- **Datasets where background and peaks are very different** (e.g., very deep sequencing where background regions have substantial counts), making the bias model's job harder.

## 14.3 The Case Against It

### Zero-init already provides an implicit curriculum

The signal model's output layers are initialized to zero (profile) and −10 (counts). At epoch 0, the signal model contributes nothing. The combined output IS the bias-only output. L_recon trains only the bias model. L_bias reinforces this. This is functionally equivalent to a background-first curriculum without actually changing the data ratio—the signal model is there but invisible.

As the signal model's internal representations develop (gradients flow through L_recon into its hidden layers even while its output is zero), its output layers gradually activate. This is a natural, self-paced curriculum: the signal model starts contributing when it has something useful to contribute, not on a fixed schedule.

### Changing the data ratio creates new problems

- **Bias model overfits to background early:** If the bias model sees 90% background for 3 epochs, it heavily optimizes for background regions. When peaks are introduced at epoch 4, L_recon asks it to contribute to peak predictions, and the distribution shift may destabilize its learned Tn5 representation.
- **Signal model undertrains on peaks:** Fewer peak examples in early epochs means the signal model's hidden representations develop more slowly. Since the signal model learns its internal features from L_recon gradients (even when its output is zero), reducing the peak fraction reduces the quality of the features it develops.
- **Adds hyperparameters:** The curriculum schedule (how many epochs at each ratio, what ratios) is a set of additional hyperparameters that interact with λ_bias, λ_sparse, and learning rate. The design philosophy of Dalmatian is to minimize tuning knobs.

## 14.4 Recommendation

**Do not use an explicit curriculum.** The zero-init strategy provides a better implicit curriculum that is self-paced, requires no additional hyperparameters, and avoids the distribution shift problems of changing data ratios mid-training. The bias model gets its head start not by seeing more background, but by being the only model that produces output in early training.

The one scenario where a curriculum might be justified is if you remove zero-init (e.g., because the Pomeranian architecture doesn't easily support it, or because you want both models active from the start for some reason). In that case, starting with more background would partially replace what zero-init provides. But zero-init is simpler and more principled.

> 💡 **Connection to ChromBPNet:** ChromBPNet's two-stage training is a degenerate curriculum with an infinitely sharp transition (100% background → 100% peaks, with freezing in between). Dalmatian's zero-init is a smooth curriculum with no transition at all—the data mix is constant, but the model's effective behavior shifts from bias-only to bias+signal as training progresses. The smooth version avoids the distribution shift and freezing artifacts of the sharp version.

---

# Part XV: Implementation Plan

## 15.1 Files to Create

- **src/cerberus/models/dalmatian.py:** Dalmatian model class, zero-init logic, constructor with BiasNet/SignalNet configs.
- **tests/test_dalmatian.py:** Forward pass shapes, ProfileCountOutput compatibility, zero-init verification, gradient flow, loss edge cases, parameter counts.

## 15.2 Files to Modify

- **src/cerberus/output.py:** Add DalmatianOutput dataclass extending ProfileCountOutput (with decomposed bias/signal fields and `detach()` method).
- **src/cerberus/loss.py:** Add DalmatianLoss with nested base loss instantiation (takes `base_loss_cls` string and `base_loss_args` dict, uses `import_class` internally to construct the base loss — compatible with existing `instantiate_metrics_and_loss` factory).
- **src/cerberus/module.py:** ~3-line isinstance check in `_shared_step` to pass `batch["peak_status"]` to DalmatianLoss.
- **src/cerberus/models/__init__.py:** Export Dalmatian.

See `dalmatian_architecture_plan.md` Section 10 for the detailed step-by-step implementation plan with implement-test cycles.

## 15.3 Verification Checklist

1. **Forward pass:** DalmatianOutput has correct shapes and is isinstance(ProfileCountOutput).
2. **Zero-init:** Signal profile logits = 0 everywhere; signal count output ≈ −10.
3. **Gradient routing:** L_recon → both; L_bias → bias only; L_sparse → signal only.
4. **Edge cases:** All-peak batch, all-non-peak batch, single-example batch.
5. **Post-training QC:** TF-MoDISco on both sub-networks; marginal footprinting; runtime diagnostics.
