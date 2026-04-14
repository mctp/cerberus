# Multitask Differential BPNet: Design Rationale

Design document for the two-phase differential chromatin accessibility
workflow implemented in cerberus via `MultitaskBPNet`, `MultitaskBPNetLoss`,
and `DifferentialCountLoss`.

## Motivation

Standard ChromBPNet / Pomeranian / BPNet trains one model per condition.
Comparing two separately-trained models confounds sequence grammar discovery
with model-training variance — the same sequence may receive different
attribution scores across runs due to random initialization and optimization
path differences.

The goal here is different: **learn what sequence features drive the change
in chromatin accessibility between two steady-state conditions**, rather than
what drives accessibility within each condition individually.  This requires
the differential signal to be a first-class training objective, not a
post-hoc subtraction of two independent models.

---

## Two-phase workflow

### Phase 1 — Multi-task absolute model (bpAI-TAC architecture)

**Model:** `MultitaskBPNet`

A single BPNet trunk is trained jointly on N steady-state conditions, one
profile head channel and one count head output per condition.

```
sequence (2114 bp)
    │
    ▼
[shared dilated tower]
    │
    ├─ profile_head → logits (B, N, L)   # N = n_conditions
    └─ count_head   → log_counts (B, N)
```

**Key architectural properties:**

- `predict_total_count=False` is enforced.  Each condition gets an
  independent log_counts scalar (shape `(B, N)`), which is the prerequisite
  for Phase 2 differential supervision.
- The trunk is shared across all conditions.  Cross-condition comparisons are
  therefore not confounded by separate training runs.
- No condition embeddings.  With a fixed set of steady-state conditions,
  embeddings add interpretive cost without benefit: the condition specificity
  lives in the output head weights, not in the shared latent `z`.
  Embeddings would be appropriate only if generalization to unseen conditions
  were required (e.g. scooby, which uses a hypernetwork cell decoder for
  single-cell resolution across arbitrary cells).

**Loss:** `MultitaskBPNetLoss`

`MSEMultinomialLoss` with `count_per_channel=True` and `average_channels=True`
enforced, matching the bpAI-TAC training objective.  Profile loss is averaged
across conditions; count loss is computed per condition independently.

**Reference:** Chandra et al. (2025). *Refining sequence-to-activity models by
increasing model resolution (bpAI-TAC).* bioRxiv 2025.01.24.634804.

---

### Phase 2 — Differential fine-tuning (Naqvi architecture)

**Loss:** `DifferentialCountLoss`

The Phase 1 model is loaded, profile loss weight is set to zero, and the
count heads for conditions A and B are retargeted to predict an external
differential statistic (DESeq2 / edgeR log2FC).

```
[Phase 1 weights loaded]
    │
    ├─ profile loss weight → 0   (profile heads receive no gradient)
    │
    └─ count head A, count head B → supervised by:
           MSE(log_counts_B − log_counts_A,  DESeq2 log2FC)
```

**Why count-only, no profile differential head:**

- There is no reliable ground truth for per-base differential insertion rates.
  DESeq2/edgeR operate on peak-level total counts and produce robust
  log2FC estimates; no equivalent exists at single-base resolution.
- The profile heads from Phase 1 already encode condition-specific TF
  footprint grammar.  They do not need to be re-trained for the differential
  task.
- This matches Naqvi et al. (2025) exactly: profile multinomial NLL weight
  is set to 0 during fine-tuning, and only the count head is retargeted.

**Why no separate delta head on `z_pooled`:**

Adding a new head that reads from the shared trunk `z` is architecturally
wrong for this design: `z` is condition-agnostic (both conditions pass through
the identical trunk and get the same `z`).  The condition specificity is
encoded in the count head weights, not in `z`.  A delta head on `z` would
need to learn differential from condition-blind features — harder and less
direct than fine-tuning the existing count heads whose weights already encode
each condition.

In Naqvi et al., no new head is added.  The existing count head IS the
differential head, repurposed by changing its supervision target.  The
equivalent in our multi-task model is supervising the difference of the two
existing count heads.

**Why no consistency loss `||Δ − (B − A)||`:**

If the model already outputs `log_counts_A` and `log_counts_B`, then
`B − A` is already a deterministic, differentiable function of the network.
Supervising a separate Δ head with `||Δ − (B − A)||` trains extra parameters
to approximate something the model can already compute exactly.  The gradient
is identical to differentiating through `B − A` directly.

A consistency penalty is only justified if the Δ head receives **independent
external supervision** (e.g. a statistic that is not trivially computable from
B − A, such as variance-stabilized fold change).  Without that, the penalty
is circular.

**Gradient flow:**

Following Naqvi et al., the entire model is updated end-to-end during Phase 2
(no frozen layers), but gradients arrive only from the count head difference
because profile loss weight is 0.  10 epochs at lr=1e-3 (Naqvi default)
is sufficient for the count heads to adapt without catastrophic forgetting of
Phase 1 features.

Optionally, individual parameters can be frozen (e.g. freeze the trunk and
only update `count_dense`) to reduce the risk of overfitting on small
differential datasets.

**Target resolution:**

`DifferentialCountLoss` accepts the log2FC target via:
1. `log2fc` batch-context kwarg — a `(B,)` float tensor (preferred for
   pipeline integration, analogous to `interval_source` in `DalmatianLoss`).
2. Fallback — `targets` averaged over all non-batch dimensions, supporting
   `(B, 1, 1)` scalar encoding or a constant-value track `(B, 1, L)`.

**Reference:** Naqvi S et al. (2025). *Transfer learning reveals sequence
determinants of the quantitative response to transcription factor dosage.*
Cell Genomics. PMC11160683.

---

---

## Differential target label: regression, not classification

### Why regression?

The most common framing of differential accessibility in published DL models is
binary classification: peaks are labelled as differentially accessible (DAR) or
not (non-DAR) at some FDR threshold (e.g. DeepAccess, Orca).  We reject this
framing for Phase 2 for three reasons:

1. **Attribution gradient requires a scalar with meaningful magnitude.**
   A binary {0, 1} label has no magnitude information; the gradient of the loss
   w.r.t. the input will point in the direction of "more differential" but
   cannot distinguish a 2× change from a 10× change.  A continuous log2FC label
   preserves this information: a peak with |log2FC| = 3 contributes proportionally
   larger gradients than one with |log2FC| = 0.5, so attribution maps reflect
   effect size.

2. **Regression is what Naqvi et al. (2025) do.**
   The Naqvi model is supervised with the quantitative differential statistic
   (ED50 scalar), not a binary label.  Their attribution analysis explicitly
   requires the continuous signal.

3. **Classification discards the null peaks.**
   When a classification head is added, non-DAR peaks are treated as background.
   With regression + shrunken LFC, non-DAR peaks get labels near 0 and serve
   as calibration examples — the model learns "these sequences do not change
   binding between conditions", which is equally informative for attribution.

### Why shrunken log2FC over raw log2FC?

Raw DESeq2 log2FC estimates for low-count peaks can be extreme (±10 or more)
even when the biological change is negligible.  These outliers dominate the MSE
loss and corrupt the count head representations.  Apeglm shrinkage pulls those
estimates toward zero proportionally to their uncertainty:

- A peak with baseMean = 5 and log2FC = 8 (high uncertainty) → shrunken to ~0.
- A peak with baseMean = 2000 and log2FC = 8 (low uncertainty) → shrunken
  to ~7.8 (minimal change).

The result is a label distribution that the MSE loss can actually fit without
the gradient being dominated by poorly-estimated outliers.

### S-value filtering (optional but recommended)

Apeglm computes an **s-value** per peak — the posterior probability that the
true fold change has the *opposite* sign from the point estimate (Stephens
2016).  It is the Bayesian analogue of the FDR for sign errors.  Filtering to
`svalue < 0.05` restricts fine-tuning to peaks where the model is ≥ 95%
confident in the *direction* of change, not just its magnitude.

This matters for attribution: a peak with shrunken LFC = 0.2 and svalue = 0.4
(uncertain direction) contributes noise.  Removing such peaks makes the
fine-tuning signal sharper.

### Existing literature using classification vs regression

| Paper | Mode | Target | Notes |
|---|---|---|---|
| DeepAccess (Cochran et al. 2022) | Classification | Binary DAR label | Cannot attribute magnitude |
| Orca (Zhou et al. 2022) | Classification | Binary Hi-C differential | Domain-level, not per-base |
| Naqvi et al. (2025) | **Regression** | ED50 scalar (LFC proxy) | Fine-tunes ChromBPNet |
| bpAI-TAC (Chandra et al. 2025) | **Regression** | Absolute counts per condition | Multi-task, no differential head |
| **This work** | **Regression** | log2 bigwig-sum ratio with pseudocount | Phase 1 bpAI-TAC → Phase 2 Naqvi |

---

## Label generation workflow

Labels are computed directly from per-peak bigwig signal using a log2 ratio
with a pseudocount.  No statistical framework (DESeq2, edgeR) is required,
making the workflow applicable even with a single replicate per condition.

### Step 1 — Sum bigwig signal over peaks

The recommended approach uses depth-normalised bigwig files (e.g. RPM) and
sums the signal over each peak interval directly, identical to how the count
head computes its own training target.  No BAM counting step is needed.

```python
from cerberus.interval import Interval, load_intervals_bed
from cerberus.differential import (
    compute_log2fc_from_bigwigs,
    DifferentialRecord,
    write_differential_targets,
)

# Load peak intervals (BED convention, 0-based)
intervals, _ = load_intervals_bed("peaks.bed")

log2fc = compute_log2fc_from_bigwigs(
    "condition_a.rpm.bw",
    "condition_b.rpm.bw",
    intervals,
    pseudocount=1.0,  # in bigwig signal units; larger → more shrinkage
)

records = [
    DifferentialRecord(chrom=iv.chrom, start=iv.start, end=iv.end,
                       log2fc=float(fc))
    for iv, fc in zip(intervals, log2fc)
]
write_differential_targets("targets.tsv", records)
```

The pseudocount acts as shrinkage: both-zero peaks get log2FC = 0; low-signal
peaks with uncertain fold changes are pulled toward zero proportionally.  The
effect is equivalent to apeglm shrinkage without requiring replicates.

**Relationship to the count head target:**  `compute_log2fc_from_bigwigs`
performs the same sum-over-bins operation as the count head loss
(`targets.sum(dim=2)` in `loss.py`), then takes the log2 ratio between
conditions.  Because the bigwigs are already depth-normalised, no additional
library-size correction is applied (`normalize=False` in
`compute_log2fc_cpm`).

### Legacy: Count reads from BAM files

If RPM-normalised bigwigs are not available, per-peak counts can be extracted
from BAM files and normalised to CPM before computing the ratio.

```bash
# deepTools multiBamSummary — per-peak counts from two BAMs
multiBamSummary BED-file \
    --BED peaks.bed \
    -b condition_a.bam condition_b.bam \
    --outRawCounts counts.tsv \
    -out counts.npz
```

```python
import numpy as np
import pandas as pd
from cerberus.differential import compute_log2fc_cpm, DifferentialRecord, write_differential_targets

counts = pd.read_csv("counts.tsv", sep="\t")
log2fc = compute_log2fc_cpm(
    counts["count_a"].values,
    counts["count_b"].values,
    pseudocount=1.0,  # in CPM units
    normalize=True,   # correct for library size differences between BAMs
)
records = [
    DifferentialRecord(chrom=row.chrom, start=row.start, end=row.end, log2fc=float(fc))
    for row, fc in zip(counts.itertuples(), log2fc)
]
write_differential_targets("targets.tsv", records)
```

Output (``targets.tsv``):
```
chrom   start   end     log2fc
chr1    1000    2000    1.23
chr1    5000    6000    -0.91
...
```

Coordinates are 0-based (BED convention), matching :class:`~cerberus.interval.Interval`.

### Step 3 — Build index and use in training

```python
from cerberus.differential import DifferentialTargetIndex

index = DifferentialTargetIndex.from_tsv("targets.tsv", default=0.0)

# Manually assemble the log2fc tensor for each batch from the index,
# then pass to DifferentialCountLoss via the log2fc kwarg:
from cerberus.loss import DifferentialCountLoss
import torch

loss_fn = DifferentialCountLoss(cond_a_idx=0, cond_b_idx=1)

# In your training loop:
log2fc_batch = torch.tensor(
    [index.get(interval) for interval in batch_intervals],
    dtype=torch.float32,
)
out = model(seq)
loss = loss_fn(out, targets_placeholder, log2fc=log2fc_batch)
```

---

## Design decisions rejected

| Proposal | Reason rejected |
|---|---|
| Condition embeddings (lookup → inject into latent) | Condition specificity lives in head weights, not trunk; embeddings add interpretive cost without gain for fixed condition sets |
| Profile differential head (Δ profile) | No reliable per-base differential ground truth; Naqvi explicitly sets profile loss to 0 during fine-tuning |
| Delta head on `z_pooled` | `z` is condition-agnostic; count head weights already encode condition; new head learns from condition-blind features |
| Consistency loss `‖Δ − (B−A)‖` | Mathematically redundant given existing absolute heads; only justified with independent external supervision |
| Frozen trunk during Phase 2 | Naqvi fine-tunes all weights; frozen trunk is optional but not the default; 10 epochs at 1e-3 does not destroy Phase 1 features |

---

## Usage

### Phase 1 training

```python
from cerberus.models.bpnet import MultitaskBPNet, MultitaskBPNetLoss

model = MultitaskBPNet(
    output_channels=["ctrl", "treat_4h", "treat_24h"],
    input_len=2114,
    output_len=1000,
    filters=64,
    n_dilated_layers=8,
)
loss_fn = MultitaskBPNetLoss(alpha=1.0, beta=1.0)

# targets: (B, N_conditions, L) absolute ATAC count tracks
out = model(seq)
loss = loss_fn(out, targets)
```

### Phase 2 fine-tuning

```python
import torch
from cerberus.differential import DifferentialTargetIndex
from cerberus.loss import DifferentialCountLoss

# 1. Load Phase 1 checkpoint
model = MultitaskBPNet(output_channels=["ctrl", "treat"], ...)
model.load_state_dict(torch.load("phase1_checkpoint.pt"))

# 2. Build the log2FC label index (see Label generation workflow)
index = DifferentialTargetIndex.from_tsv("targets.tsv", default=0.0)

# 3. Loss: profile weight = 0 (Naqvi), supervise log_counts diff with MSE
loss_fn = DifferentialCountLoss(
    cond_a_idx=0,   # ctrl
    cond_b_idx=1,   # treat
    abs_weight=0.0, # no absolute regularisation (Naqvi default)
)

# 4. In your training loop, assemble log2fc_batch from the index
log2fc_batch = torch.tensor(
    [index.get(interval) for interval in batch_intervals],
    dtype=torch.float32,
)
out = model(seq)
loss = loss_fn(out, targets_placeholder, log2fc=log2fc_batch)
loss.backward()
```

### Attribution

After Phase 2, DeepSHAP through the count head difference directly answers:
*"which base positions, as encoded by the shared steady-state grammar from
Phase 1, explain why this peak is differentially accessible between ctrl
and treat?"*

```python
from captum.attr import DeepLiftShap

def delta_target(output):
    # output is ProfileCountOutput; extract scalar delta
    return output.log_counts[:, 1] - output.log_counts[:, 0]

explainer = DeepLiftShap(model)
attributions = explainer.attribute(seq, target=delta_target)
```

---

## Summary

The combined workflow is novel: no published paper performs multi-task
absolute pretraining (bpAI-TAC Phase 1) specifically to improve the trunk
representation *before* the Naqvi-style differential fine-tune (Phase 2).
The improvement over Naqvi's original design is that the trunk has seen both
conditions jointly during Phase 1, so `z_pooled` encodes cross-condition
sequence grammar before a single differential label is ever seen.  The Phase 2
fine-tuning mechanism is otherwise identical to Naqvi et al.
