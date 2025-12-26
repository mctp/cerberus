# ASAP Implementation Analysis

## Differences from ASAP vanillaCNN

### Architecture
- **Output Dimensions**: ASAP's `VanillaCNN` predicts the full input window (2048bp -> 512 bins). The Cerberus configuration in this notebook predicts the center 1024bp (256 bins). The Cerberus `VanillaCNN` class supports arbitrary output lengths via dynamic linear layer sizing, so this is a configuration choice.
- **Hard-coded Input/Output**: ASAP's `VanillaCNN` hard-codes the flattened feature size `512 * 13` in the first linear layer of the head (`nn.Linear(512 * 13, 1024)`), which implicitly hard-codes the expected input length to ~2048bp (resulting in 13 spatial bins after pooling). This rigidly couples the architecture to a specific input size.
- **Tensor Shape**: ASAP outputs `(Batch, Bins, Channels)`. Cerberus outputs `(Batch, Channels, Bins)`.

### Data Augmentation & Binning
- **Binning Phase**: ASAP bins signals *before* data loading/jittering. Jitter shifts are multiples of the bin size (4bp). This means the "phase" of the max-pooling bins is fixed relative to the genome.
- **Cerberus**: Jitter shifts are at 1bp resolution. Binning happens *after* jittering on the cropped window. This means the bin boundaries shift relative to the genome features, providing stronger data augmentation (1bp shift variance) compared to ASAP's fixed-phase binning.

### Training
- **Scheduler**: ASAP uses a custom Warmup + Cosine Decay to 0. Cerberus uses `timm`'s scheduler, which behaves similarly but defaults may vary.
- **Sampling**: ASAP's `WGDataset` loads a larger context (4096bp) and shifts the 2048bp window within it (up to 2048bp shift). The Cerberus config here loads a 2304bp context for a smaller jitter range (256bp). Matching ASAP's sampling coverage would require increasing `padded_size` and `max_jitter`.

### Analysis of Other ASAP Architectures
- **ConvNeXtCNN**: **Hard-coded Output Length.** The `forward` method explicitly reshapes the output to `(batch, 512, 1)`, fixing the output length to 512 bins. This imposes a hard constraint on the input window size (must produce 512 bins).
- **ConvNeXtLSTM**: **Flexible Output Length.** The final fully connected layer maps to `self.nr_bins`, allowing variable output lengths defined at initialization. However, it uses a global reduction (last LSTM state) which structurally separates it from dense prediction tasks.
- **ConvNextTransformer & ConvNeXtDCNN**: **Flexible Output Length.** These architectures use fully convolutional or sequence-preserving transformer blocks. The output length scales dynamically with the input length (typically `input_len // 4`), with no hard-coded shape constraints in the forward pass.
