# BPNet Training Troubleshooting Guide

Guide for diagnosing why `chip_ar_mdapca2b_bpnet.py` (or similar BPNet training runs) may fail to learn.

## Key Architectural Difference: BPNet Has No Normalization

BPNet is the only model in Cerberus without normalization layers.
Pomeranian uses RMSNorm throughout its architecture.
BPNet uses raw `Conv1d -> ReLU -> residual add` in its dilated tower.

This has concrete consequences that have been verified experimentally:

1. **Activations grow monotonically through residual layers** — each `x + ReLU(conv(x))` can
   only add non-negative values. After 8 layers, activation magnitudes increase ~3-4x
   (mean 0.12 -> 1.31 at init, growing further during training to mean ~3-4).

2. **Activation spikes under mixed precision** — without normalization to bound values,
   fp16 training produces instability spikes where latent activations can double
   (observed: max jumping from 18 to 37 at step 200 in fp16 overfitting test).
   Models with RMSNorm don't exhibit this.

3. **Profile loss converges much slower than count loss** — Count loss converges in ~30
   steps while profile loss takes 500+ steps. The multinomial NLL profile gradient is
   `softmax(logit_i) - target_i/sum(targets)`, which is ~O(1/output_len) = ~O(1/1000).
   Without normalization to amplify useful signal, these small gradients are easily
   drowned out by noise from background samples.

**Verified:** Single-batch overfit works perfectly in fp32 (Pearson -> 1.0 in ~100 steps)
and with fp16 autocast (with occasional spikes). The architecture and loss are correct.
The problem is specific to full-training dynamics.

---

## 1. Primary Suspects (Data Already Verified)

If other models train on the same data, skip data verification and focus here.

### 1.1 Mixed Precision Instability (Most Likely)

BPNet's unnormalized activations interact badly with fp16. During training, activation
magnitudes grow (to mean ~3-4, max ~15-20), and fp16's limited dynamic range causes
periodic instability spikes. On MPS, the script uses `16-mixed` (float16, less precise
than bfloat16).

**Diagnosis — check for spikes in training logs:**
```python
# If using TensorBoard or CSV logger, look for sudden loss spikes
# or NaN losses during training
import pandas as pd
metrics = pd.read_csv("path/to/lightning_logs/version_0/metrics.csv")
print(metrics[["step", "train_loss"]].describe())
# Check for NaN
print("NaN losses:", metrics["train_loss"].isna().sum())
```

**Fix — force full precision:**
```python
# In the example script, change precision_args:
precision_args = {
    "precision": "32-true",         # Full precision
    "accelerator": accelerator,
    "devices": devices,
    "strategy": "auto",
    "compile": False,
}
```

Or on GPU, use bfloat16 (more stable than float16):
```python
"precision": "bf16-mixed"  # Already the default on GPU, but verify
```

### 1.2 LR Schedule: Slow Start + Early Decay

The scheduler config is identical across all example scripts, but BPNet is more
sensitive to it due to small profile gradients without normalization.

**Actual LR schedule (verified via simulation):**

| Epoch | LR | Notes |
|-------|----|-------|
| 0 | 1.00e-05 | Warmup start — essentially no learning |
| 1 | 1.09e-04 | Still very low |
| 5 | 5.05e-04 | Starting to be meaningful |
| 10 | 9.05e-04 | Peak (never reaches 1e-3 because cosine already decays) |
| 20 | 6.58e-04 | Cosine decay |
| 30 | 3.52e-04 | |
| 40 | 1.05e-04 | Effectively done learning |
| 49 | 1.10e-05 | Back to min_lr |

The effective training window (LR > 2e-4) is roughly epochs 5-35. For BPNet's slow
profile convergence, this may not be enough.

**Profile gradient update at epoch 0:**
```
grad_magnitude * LR ≈ O(1/1000) * 1e-5 = O(1e-8)  # essentially zero
```

**Fix — try constant LR without scheduler:**
```python
train_config: TrainConfig = {
    ...
    "learning_rate": 1e-3,
    "scheduler_type": "default",   # No scheduler, constant LR
    "scheduler_args": {},
    ...
}
```

Or reduce warmup:
```python
"scheduler_args": {
    "num_epochs": 50,
    "warmup_epochs": 2,            # Reduce from 10 to 2
    "min_lr": 1e-5,
}
```

### 1.3 Profile Loss vs Count Loss Imbalance

At initialization, profile loss (~2800) is ~70x larger than count loss (~40).
But count loss gradients are larger per-parameter (~O(10)) while profile gradients
are tiny (~O(1/1000)). The count head converges in ~30 steps while the profile head
takes hundreds.

If you observe: "loss decreases initially then plateaus" — this is the count loss
converging while the profile loss stagnates.

**Diagnosis:**
```python
import torch
import torch.nn.functional as F

# During training, log both components separately:
with torch.no_grad():
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    profile_loss = -torch.sum(targets * log_probs, dim=-1).mean()
    target_gc = targets.sum(dim=(1, 2))
    count_loss = F.mse_loss(outputs.log_counts.flatten(), torch.log1p(target_gc))
    print(f"Profile: {profile_loss:.2f}  Count: {count_loss:.4f}")
```

**Fix — increase alpha to weight count loss higher relative to profile:**
```python
# Original alpha=1.0 means count_loss has equal weight as profile_loss
# But profile_loss is ~70x larger in absolute value
# Try reducing profile weight:
BPNetLoss(alpha=10.0)  # Makes count loss 10x profile loss
```

Or try different alpha values: `0.1, 1.0, 10.0, 100.0`

### 1.4 Background Sample Dilution

With `background_ratio=1.0`, 50% of training samples have zero signal:
- Profile loss = 0 for zero-signal samples (no gradient contribution)
- Count loss provides the only gradient from background samples
- Effective profile gradient is halved compared to peaks-only training

Models with normalization handle this better because their gradient flow is more robust.

**Fix — reduce or eliminate background for initial testing:**
```python
sampler_config: SamplerConfig = {
    "sampler_type": "peak",
    "padded_size": padded_size,
    "sampler_args": {
        "intervals_path": dataset_files["narrowPeak"],
        "background_ratio": 0.0,    # Peaks only
    }
}
```

---

## 2. Diagnostic Steps (Ordered by Priority)

### 2.1 Single-batch overfit (sanity check)

Confirms the model architecture and loss are fundamentally correct.

```python
from cerberus.models.bpnet import BPNet, BPNetLoss
from cerberus.output import ProfileCountOutput
import torch.nn.functional as F

model = BPNet(input_len=2114, output_len=1000,
    input_channels=['A', 'C', 'G', 'T'],
    output_channels=['signal'],
    filters=64, n_dilated_layers=8)

# Get a real batch from your datamodule
batch = next(iter(datamodule.train_dataloader()))
# Filter to nonzero-signal samples
mask = batch["targets"].sum(dim=(1,2)) > 0
inputs = batch["inputs"][mask]
targets = batch["targets"][mask]

criterion = BPNetLoss(alpha=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for step in range(500):
    optimizer.zero_grad()
    out = model(inputs)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        with torch.no_grad():
            probs = F.softmax(out.logits, dim=-1)
            pred = probs.flatten()
            targ = (targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)).flatten()
            corr = torch.corrcoef(torch.stack([pred, targ]))[0, 1]
        print(f"Step {step:3d}: loss={loss.item():.2f}  pearson={corr.item():.4f}")
```

**Expected:** Pearson should reach > 0.99 within 100-200 steps in fp32.
**If this fails:** There is a deeper issue (check shapes, gradient flow).

### 2.2 Check gradient flow

```python
model.train()
out = model(inputs)
loss = criterion(out, targets)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad.norm().item()
        print(f"  {name}: grad_norm={g:.4f}")
    else:
        print(f"  {name}: NO GRADIENT")
```

All parameters should have nonzero gradient norms. Expected ranges:
- `profile_conv.weight`: ~400-500
- `count_dense.weight`: ~100-110
- `res_layers.*.conv.weight`: ~50-110
- `iconv.weight`: ~80-90

### 2.3 Monitor activation magnitudes during training

Without normalization, BPNet activations can drift during training. Track the latent
representation after the residual tower:

```python
# Add to a callback or modify forward temporarily:
with torch.no_grad():
    h = F.relu(model.iconv(inputs))
    for layer in model.res_layers:
        h = layer(h)
    print(f"Latent: min={h.min():.2f} max={h.max():.2f} mean={h.mean():.2f}")
```

At init: mean ~1.3, max ~5. If during training max exceeds ~50 or mean exceeds ~10,
the model is becoming numerically unstable. This is a sign that fp32 or gradient
clipping is needed.

### 2.4 Verify LR schedule

```python
from timm.scheduler.scheduler_factory import create_scheduler_v2

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler, _ = create_scheduler_v2(optimizer, sched="cosine",
    num_epochs=50, warmup_epochs=10, min_lr=1e-5)

# Check: step_update is a NO-OP when t_in_epochs=True
# Only scheduler.step(epoch=N) changes the LR
for epoch in range(50):
    scheduler.step(epoch=epoch)
    print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.2e}")
```

Note: timm's `step_update(num_updates)` is a no-op when `t_in_epochs=True` (default).
The LR only changes at epoch boundaries via `scheduler.step(epoch)`. This is correct
but means there is NO per-step warmup — warmup is epoch-granularity only.

---

## 3. Recommended Fix Strategy

Try these in order. Each addresses one root cause.

### Step 1: Full precision, no scheduler, peaks only

Eliminate all secondary variables. If this trains, add them back one at a time.

```python
# Modify the example script:
precision_args = {
    "precision": "32-true",
    "accelerator": accelerator,
    "devices": devices,
    "strategy": "auto",
    "compile": False,
}

train_config: TrainConfig = {
    "batch_size": 64,
    "max_epochs": 50,
    "learning_rate": 5e-4,          # Lower than default 1e-3
    "weight_decay": 0.01,
    "patience": 10,
    "optimizer": "adamw",
    "scheduler_type": "default",    # Constant LR
    "scheduler_args": {},
    "filter_bias_and_bn": True,
    "reload_dataloaders_every_n_epochs": 0,
}

sampler_config: SamplerConfig = {
    "sampler_type": "peak",
    "padded_size": padded_size,
    "sampler_args": {
        "intervals_path": dataset_files["narrowPeak"],
        "background_ratio": 0.0,    # Peaks only
    }
}
```

### Step 2: If Step 1 works, add back scheduler

```python
"scheduler_type": "cosine",
"scheduler_args": {
    "num_epochs": 50,
    "warmup_epochs": 2,     # Reduced from 10
    "min_lr": 1e-5,
},
```

### Step 3: If Step 2 works, add back background

```python
"background_ratio": 0.5,   # Start lower than 1.0
```

### Step 4: If Step 3 works, try mixed precision

```python
"precision": "bf16-mixed",  # On GPU only; avoid 16-mixed on MPS for BPNet
```

---

## 4. Hyperparameter Reference

Compared to the original bpnet-lite / chrombpnet-pytorch implementations:

| Parameter | Cerberus Default | Original BPNet | Notes |
|-----------|-----------------|----------------|-------|
| LR | 1e-3 | ~4e-4 | Cerberus is 2.5x higher |
| Optimizer | AdamW | Adam | Weight decay differs |
| Scheduler | Cosine + 10-epoch warmup | None (flat LR) | Major difference |
| Precision | bf16-mixed / 16-mixed | fp32 | Major difference |
| Weight Decay | 0.01 | 0 | AdamW decoupled WD |
| Count Alpha | 1.0 | 1.0 | Same |

The original BPNet implementations train successfully with **flat LR ~4e-4, no warmup,
no scheduler, fp32 precision**. The Cerberus defaults were designed for the newer
normalized architectures (Pomeranian) and may not be optimal for BPNet.

---

## 5. Architecture Verification

### 5.1 Output length math

For default BPNet (input_len=2114, output_len=1000):

| Stage | Kernel | Dilation | Reduction | Running Length |
|-------|--------|----------|-----------|----------------|
| Input | - | - | 0 | 2114 |
| Initial Conv | 21 | 1 | 20 | 2094 |
| Dilated Block 1 | 3 | 2 | 4 | 2090 |
| Dilated Block 2 | 3 | 4 | 8 | 2082 |
| Dilated Block 3 | 3 | 8 | 16 | 2066 |
| Dilated Block 4 | 3 | 16 | 32 | 2034 |
| Dilated Block 5 | 3 | 32 | 64 | 1970 |
| Dilated Block 6 | 3 | 64 | 128 | 1842 |
| Dilated Block 7 | 3 | 128 | 256 | 1586 |
| Dilated Block 8 | 3 | 256 | 512 | 1074 |
| Profile Conv | 75 | 1 | 74 | 1000 |

Verified: model produces `(B, 1, 1000)` logits from `(B, 4, 2114)` input.

### 5.2 Loss equivalence with chrombpnet

```python
import torch
import torch.nn.functional as F
from cerberus.models.bpnet import BPNetLoss
from cerberus.output import ProfileCountOutput

logits = torch.randn(4, 1, 1000)
targets = torch.abs(torch.randn(4, 1, 1000)) * 10
log_counts = torch.randn(4, 1)

# Cerberus
criterion = BPNetLoss(alpha=1.0)
output = ProfileCountOutput(logits=logits, log_counts=log_counts)
cerberus_loss = criterion(output, targets)

# Manual chrombpnet-equivalent
log_probs = F.log_softmax(logits, dim=-1)
mnll = -torch.sum(targets * log_probs, dim=-1).mean()
target_total = targets.sum(dim=(1, 2))
target_log_total = torch.log1p(target_total)
mse = F.mse_loss(log_counts.flatten(), target_log_total)
manual_loss = mnll + mse

print(f"Cerberus: {cerberus_loss.item():.6f}")
print(f"Manual:   {manual_loss.item():.6f}")
# These should match
```

---

## 6. Checklist Summary

For the case where data is verified (other models train on same data):

| # | Check | Pass Condition |
|---|-------|----------------|
| 1 | Single-batch overfit (fp32) | Pearson > 0.99 in 200 steps |
| 2 | Gradients are nonzero | All params have grad_norm > 0 |
| 3 | Train with fp32, no scheduler, peaks only | Loss decreases over epochs |
| 4 | Add scheduler back | Still converges |
| 5 | Add background samples back | Still converges |
| 6 | Add mixed precision back | Still converges (try bf16 before fp16) |
| 7 | Profile loss decreases | Not just count loss decreasing |
| 8 | Activation magnitudes bounded | Latent max < 50 during training |

---

## 7. Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss barely changes for first 10 epochs | Warmup at LR=1e-5 | `scheduler_type: "default"` or `warmup_epochs: 2` |
| Loss decreases then plateaus early | Count loss converged, profile stuck | Lower alpha, reduce background_ratio |
| Sporadic loss spikes during training | fp16 instability + no normalization | Use `precision: "32-true"` or `"bf16-mixed"` |
| Loss is NaN | Activation overflow in fp16 | Use fp32; add gradient clipping |
| Total loss ~150 but Pearson ~0 | Profile shape not learned, only counts | Log profile/count loss separately; train longer |
| Works on GPU but not MPS | fp16 vs bf16 difference | Force `"32-true"` on MPS |
