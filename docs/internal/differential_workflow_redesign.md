# Differential-Learning Workflow Redesign

**Date:** 2026-04-18
**Status:** Design selected — **Option B**. **Implemented** in commits
`b461a6e` (design doc) → `74e3999` (attribution unification) →
`e86ddaa` (loss simplification + Phase 2 tool rewrite), landed on
`marcin-feature` on 2026-04-18.
**Predecessors:**
- `docs/internal/multitask-differential-bpnet.md` (joanne-feature internal design, not ported)
- `docs/internal/variant_tool_design.md` (same proposal shape; adapted here)
- `docs/internal/correctness_audit_2026_04_16.md` (format of the audit sections)

---

> **TL;DR — Option B is the chosen design and shipped as described.**
> Sections §5 and §6 document the alternatives considered and the
> rationale for the choice. §7 and §7a are the implementation plan;
> see also §7.7 for notes on where the landed code deviates from the
> sketched plan. §8 covers the attribution-module unification (a
> part of Option B). §13 covers future extensions that are **not**
> required for the current work.

---

## Implementation notes (2026-04-18)

Added after implementation landed. The doc's concrete predictions
were mostly correct; the deviations worth flagging:

1. **`DifferentialCountLoss.loss_components` was kept, not removed.**
   `CerberusModule._shared_step` ([src/cerberus/module.py:169](src/cerberus/module.py#L169))
   calls `criterion.loss_components(...)` to log per-component
   metrics; every cerberus loss must provide it. Under Option B it
   returns a single entry: `{"delta_loss": <scalar>}`.

2. **`forward(outputs, targets, **kwargs)` keeps `**kwargs`.** The
   `CerberusLoss` protocol includes `**kwargs: object`, and
   `_shared_step` passes the full batch context (everything except
   `inputs` / `targets`) through. `DifferentialCountLoss.forward`
   accepts and ignores these. Interpreting the doc's
   `forward(outputs, targets)` as a *protocol change* would break
   every other caller; the contract stays the same, only the loss's
   use of the kwargs goes to zero.

3. **No explicit trunk-freeze hook was needed.** The design sketched
   a `_freeze_phase2_trunk(model)` helper (§7.2, §7a.7). In
   practice `DifferentialCountLoss` only touches the count heads, so
   the profile heads naturally receive no gradient; multi-GPU DDP
   handles this via a `ddp_find_unused_parameters_true` strategy
   override in [`_select_phase2_strategy`](tools/train_multitask_differential_bpnet.py).
   Freezing can be added later if a workflow demands it.

4. **`_data_config_two_channel` was not split out.** The Phase 1 and
   Phase 2 `DataConfig`s are inlined in `run_phase1` / `run_phase2`
   — cheaper than factoring out a 30-line helper called from two
   places.

5. **Line-count projections were optimistic.** Actual shrinks:
   `src/cerberus/loss.py` **−60** (doc projected −135),
   `src/cerberus/attribution.py` **−24** (projected −70),
   `tools/train_multitask_differential_bpnet.py` **−227** (projected
   −680). Net across tree ≈ **−379** lines (projected −940). The gap
   comes from (a) `loss_components` staying, (b) `run_interpretation`
   keeping its own DataModule setup, (c) `run_phase2` mirroring
   `run_phase1`'s scheduler / optimizer plumbing instead of a
   library-level helper.

6. **Test names drifted from the sketch.** Actual names on
   `marcin-feature`:
   - `test_differential_count_loss_derives_delta_from_targets`
   - `test_differential_count_loss_nonzero_when_prediction_off`
   - `test_differential_count_loss_pseudocount_affects_target`
   - `test_differential_count_loss_components_has_only_delta_loss`
   - `test_differential_count_loss_rejects_2d_targets`
   - `test_differential_count_loss_rejects_too_few_target_channels`
   - `test_attribution_target_delta_requires_tuple_channels`
   - `test_attribution_target_single_rejects_tuple_channels`
   - `test_attribution_target_delta_{log_counts,profile_window_sum_*,...}`
   Scope matches the plan (delta derivation, arity validation,
   channel-range validation); the breakdown is finer.

7. **Pyright config change.** The rewrite
   silences `reportPrivateImportUsage` project-wide in
   `pyrightconfig.json`; the rule was flagging 1769 legitimate
   `torch.*` references and drowning out real errors. Not part of
   the design but a prerequisite for treating pyright output as
   signal during the refactor.

---

## Executive Summary

The differential-accessibility workflow that landed from `joanne-feature`
is functionally correct but structurally at odds with the rest of
cerberus. The training tool carries four private helpers that duplicate
capabilities cerberus already provides; the Phase 2 training path goes
through a `_DiffWrapper` that precomputes per-peak `log2FC` offline and
re-injects it as a batch kwarg; and the attribution module now ships two
near-identical classes (`AttributionTarget`,
`DifferentialAttributionTarget`) whose fields only partially overlap.

**Design principle for this redesign:** keep `src/cerberus/` lean,
polished, and free of hacks. Every library symbol should be
justified by multiple consumers or a clean conceptual boundary.
`tools/train_multitask_differential_bpnet.py` is allowed — even
expected — to absorb workflow-specific quirks (BED union merging,
two-phase coordination, trunk freezing, pretrained-weight loading
logistics). The library gets the small, reusable pieces; the tool
gets the one-off workflow.

Under this principle the proposal is a tight triangle:

1. **Compute the per-peak log2FC inside the loss**, from the `(B, N, L)`
   `targets` tensor the count head is already supervised against.
   This makes `_compute_log2fc_from_bigwigs`, `_DiffWrapper`, and the
   provenance TSV unnecessary. `SignalExtractor` stays the single
   source of truth for bigwig reading.

2. **Collapse `DifferentialCountLoss` to a single-path, single-concept
   loss.** Inputs: `outputs`, `targets`. Output: one scalar. No
   kwargs, no shape overloading, no `abs_weight` regularizer, no
   `(B, 1, 1)` scalar fallback. Target: ~15 lines.

3. **Unify `AttributionTarget` and `DifferentialAttributionTarget`**
   into a single class dispatched by its `reduction` string. The two
   delta reductions become peers of the existing five. The
   `DifferentialAttributionTarget` name and the
   `DIFFERENTIAL_TARGET_REDUCTIONS` set are **removed** (breaking
   change); all in-tree callers (3 of them) update in the same
   commit. One class, one registry, one code path.

What stays out of `src/`:

- BED / peak union merging (lives in the tool).
- Phase 1 / Phase 2 config construction (lives in the tool).
- Trunk-freezing logic for Phase 2 fine-tuning (lives in the tool,
  via a direct `requires_grad = False` loop — no `TrainConfig`
  extension).
- Pretrained-weight loading glue (uses the existing
  `ModelConfig.pretrained` mechanism; no new library feature).
- Optional DeepLIFTSHAP + TF-MoDISco interpretation pipe (lives in
  the tool; uses the unified `AttributionTarget`).
- Optional absolute-count regularizer (removed; add back only when a
  real consumer surfaces).
- Externally-computed delta targets (not a requirement; §13
  describes how to add them later without touching the core).

Overall line count: the current tool is 1068 lines. A version built
on the design proposed here is expected to be ~350 lines, with the
saved ~700 lines distributed among (a) reusing `train_single` /
`train_multi`, (b) dropping the inlined helpers, and (c) dropping
the wrapper / precompute plumbing. Library side: `src/cerberus/loss.py`
**net shrinks** (~30 lines less than today — the kwarg + fallback +
`abs_weight` machinery is bigger than the single-path replacement).
`src/cerberus/attribution.py` shrinks by ~90 lines (single
`AttributionTarget` replaces two classes and the shim).

The design is **Option B** as described in §5.2. §6 documents the
rationale for the choice against the alternatives. §7 and §7a are
the implementation plan at the signature / config / migration
level. §8 spells out the attribution unification component in
isolation. §13 describes optional future-work extensions (external
delta labels, `abs_weight` regularizer restoration) and explicitly
commits that the planned code does not depend on any of them.

---

## 0. Design Principle: Library Polish, Tool Pragmatism

This redesign commits to a two-tier split between `src/cerberus/` and
`tools/`, more strictly than has been applied to previous feature
additions:

**`src/cerberus/` rules.** A symbol lives here only if:

- It has (or plausibly will have) multiple consumers, or
- It embodies a clean, self-contained concept that the rest of the
  library is built around (a model class, a loss, a data primitive), or
- It is the single source of truth for some I/O format or cross-cutting
  invariant (e.g. `SignalExtractor` for bigwig reading,
  `propagate_pseudocount` for count-space consistency).

Library code is written to a higher bar: no optional protocols,
no kwarg-driven shape changes, no "also works if you happen to pass
X" branches, no scaffolding for future features that don't have a
current consumer. Every library symbol should be explainable in one
sentence and callable from a notebook without reading the source.

**`tools/` rules.** A tool is allowed to:

- Absorb workflow-specific sequencing (Phase 1 → Phase 2 → optional
  interpretation).
- Know things the library does not (that this particular CLI takes
  two BED files and merges them; that Phase 2 wants trunk + profile
  heads frozen; that this specific call to `train_single` uses a
  pretrained checkpoint).
- Be verbose, imperative, and specific. A tool is a script, not a
  library.
- Contain logic that would be inappropriate in the library because
  it only serves one workflow.

A tool is not a second-class citizen — it is the correct home for
code that is genuinely one-off. The mistake the current
implementation makes is the inverse: putting workflow-specific
quirks (`log2fc` kwarg, `(B, 1, 1)` scalar fallback,
`DifferentialTargetIndex`) into the library, where every other
reader has to reason about them, while simultaneously inlining
generic helpers (`_compute_log2fc_from_bigwigs`) into the tool,
where they bypass existing library machinery.

Applying the principle:

| Concern | Today | Under this redesign |
|---|---|---|
| Compute log2FC from bigwigs | Tool (duplicates `SignalExtractor`) | **Library** (inside loss, uses the existing targets tensor) |
| Map per-condition names → channel indices | Tool | Tool |
| Merge two peak BEDs into union | Tool | Tool |
| Load Phase 1 checkpoint into Phase 2 model | Tool (manual `_find_phase1_model`) | Tool (via library's `ModelConfig.pretrained`) |
| Freeze trunk for Phase 2 | Tool (`_Phase2Module.__init__` loop) | Tool (same loop; no library change) |
| Supervise `log_counts[b] - log_counts[a]` | Library (with overloading) | **Library** (single path) |
| Absolute-track regularizer | Library (`abs_weight`) | **Not in library** (future work if needed) |
| External DESeq2 labels | Library (`log2fc` kwarg) | **Not in library** (future work if needed) |
| Provenance TSV of log2FCs | Tool (`_write_differential_targets`) | **Dropped** (never read back; derivable on demand) |
| Attribution target for delta | Library (`DifferentialAttributionTarget`) | **Library** (unified `AttributionTarget` with delta reductions) |

This split is the lens applied to each of the §5 options and the §7
blueprint.

---

## 1. Problem Statement

### 1.1 What the workflow is supposed to do

Train a model that distinguishes differential TF binding /
chromatin accessibility between two conditions (e.g. FOXA1 in LNCaP
vs 22Rv1). The published bpAI-TAC + Naqvi et al. 2025 recipe is:

- **Phase 1:** multi-task BPNet with one profile + count head per
  condition, trained jointly so the shared trunk sees both.
- **Phase 2:** keep the trunk, freeze it or the profile heads,
  fine-tune the count heads on an external scalar
  `log2FC_cond_B_vs_cond_A` per peak.
- **Optional interpretation:** attribute the fine-tuned model's
  `log_counts_B − log_counts_A` using DeepLIFTSHAP + TF-MoDISco to
  find motifs driving the differential.

### 1.2 What makes the current implementation hackish

Three observations from the post-merge audit:

**Observation 1: offline log2FC precompute duplicates the data
pipeline.** The tool opens the two bigwigs a second time (outside the
cerberus `SignalExtractor` path) purely to produce a per-peak scalar
`log2FC` array. It then wraps the `CerberusDataset` in `_DiffWrapper` to
inject that array back into each sample dict by integer index. But the
same bigwigs are *also* being read per-batch by `SignalExtractor` as
part of the normal `targets` tensor — they supply the same numbers,
just at per-base resolution.

**Observation 2: `DifferentialCountLoss` has three supported calling
patterns for `targets`.** Depending on `abs_weight` and whether a
`log2fc` kwarg is present, `targets` means one of three things:

- `(B, 1, 1)` or `(B, 1, L)` scalar delta target (fallback path),
- `(B, N, L)` absolute count tracks (when `abs_weight > 0`),
- ignored entirely (when `log2fc` kwarg is provided).

The caller has to know which combination applies. The validation fixes
landed in commit `0017795` close the most dangerous silent-wrong-math
cases, but the overloading itself is still the underlying smell.

**Observation 3: `DifferentialAttributionTarget` is a near-copy of
`AttributionTarget`.** They share the same `_resolve_window` logic,
the same `nn.Module` boilerplate, the same reduction-string dispatch
pattern. The only real difference is that the differential variant
carries `cond_a_idx`/`cond_b_idx` instead of `channel`. A user
looking for "what does attribution look like in cerberus" now has
two entry points with subtly different APIs.

### 1.3 The underlying question

Is differential accessibility:

- (i) **a different task** — warranting its own Loss / AttributionTarget
      hierarchies and its own batch-key plumbing, or
- (ii) **a different loss on the same task** — where the inputs,
      dataset, model I/O shapes, attribution targets, and batch flow all
      match Phase 1 and only the objective differs?

Joanne's design implicitly answered (i). This doc argues (ii) is
cleaner and enabled by what cerberus already has.

---

## 2. Audit of the Current Implementation

### 2.1 The training tool (`tools/train_multitask_differential_bpnet.py`, 1068 lines)

Top-of-file helpers, inlined during the port from
`origin/joanne-feature` commit `c1f5343`:

| Location | Symbol | Purpose |
|---|---|---|
| [lines 95–101](tools/train_multitask_differential_bpnet.py#L95-L101) | `_DifferentialRecord` | Dataclass (chrom, start, end, log2fc) for TSV provenance. |
| [lines 104–129](tools/train_multitask_differential_bpnet.py#L104-L129) | `_compute_log2fc_from_bigwigs` | Per-interval bigwig sum → `log2((sum_b + pc) / (sum_a + pc))`. Reads bigwigs a second time. |
| [lines 132–140](tools/train_multitask_differential_bpnet.py#L132-L140) | `_write_differential_targets` | TSV writer. Consumed only by human eyes (no reader exists). |
| [lines 293–309](tools/train_multitask_differential_bpnet.py#L293-L309) | `_read_bed_intervals` | 3-col BED/narrowPeak parser. |
| [lines 312–324](tools/train_multitask_differential_bpnet.py#L312-L324) | `_merge_and_write_peaks` | Union of the two condition peak BEDs. |
| [lines 327–334](tools/train_multitask_differential_bpnet.py#L327-L334) | `_find_phase1_model` | Glob for `model.pt` in fold subdir. |
| [lines 337–351](tools/train_multitask_differential_bpnet.py#L337-L351) | `_get_pos_intervals` | Materialize sampler's positives in iter order — must align with `log2fc[idx]`. |
| [lines 359–390](tools/train_multitask_differential_bpnet.py#L359-L390) | `_DiffWrapper` | Wraps base dataset; injects `log2fc[idx]` into each sample dict. |
| [lines 398–451](tools/train_multitask_differential_bpnet.py#L398-L451) | `_Phase2Module` | Minimal PL module; its `_step` calls `loss_fn(outputs, targets, log2fc=...)`. |
| [lines 850–870](tools/train_multitask_differential_bpnet.py#L850-L870) | `_dinuc_shuffle` | DLS baseline generator. |

The five workflow phases end-to-end:

1. **Phase 1 setup** ([run_phase1, lines 459-594](tools/train_multitask_differential_bpnet.py#L459-L594)): merge peaks, build `DataConfig.targets = {name_a: bw_a, name_b: bw_b}`, build `ModelConfig` pointing at `MultitaskBPNet` / `MultitaskBPNetLoss`, delegate to `train_single` / `train_multi`. Uses the standard cerberus pipeline end-to-end.
2. **Phase 2 log2FC precompute** ([lines 725–745](tools/train_multitask_differential_bpnet.py#L725-L745)): extract `train_intervals` / `val_intervals`, compute log2FC per interval by re-reading the bigwigs, write provenance TSV.
3. **`_DiffWrapper` injection** ([lines 748–766](tools/train_multitask_differential_bpnet.py#L748-L766)): each sample becomes `{inputs, targets, intervals, interval_source, log2fc}`.
4. **Phase 2 training loop** ([lines 768–824](tools/train_multitask_differential_bpnet.py#L768-L824)): bespoke PL module, trainer, DataLoader — all because `train_single` / `train_multi` expect a plain `CerberusDataset`, not a wrapped one.
5. **Optional interpretation** ([lines 873–1041](tools/train_multitask_differential_bpnet.py#L873-L1041)): reload Phase 2 model, `DifferentialAttributionTarget(reduction="delta_log_counts")`, DeepLIFTSHAP, TF-MoDISco.

Phases 1 and 5 already fit the cerberus shape. Phases 2–4 exist only
because the Phase 2 loss has a different idea of what `targets` means.

### 2.2 `DifferentialCountLoss` ([loss.py:547-714](src/cerberus/loss.py#L547-L714))

Constructor fields: `cond_a_idx`, `cond_b_idx`, `abs_weight`,
`count_pseudocount`.

**Target resolution** ([lines 613–638](src/cerberus/loss.py#L613-L638)):

```
if kwargs["log2fc"] is not None:
    delta_target = kwargs["log2fc"].flatten()
else:
    delta_target = targets.reshape(B, -1).mean(dim=-1)
```

The fallback only produces a sensible delta if `targets` is
`(B, 1, 1)` or `(B, 1, L)` *and the caller has pre-computed the delta*.
For `(B, N, L)` absolute-signal `targets`, the fallback silently
averages the whole thing, producing garbage.

**Absolute regularizer branch** ([lines 681–696](src/cerberus/loss.py#L681-L696)): requires `targets` to be `(B, N, L)` with `N >= max(cond_a_idx, cond_b_idx) + 1`. Raises on shape mismatch.

**Combined semantics:** the kwarg and shape together encode *which of
three protocols the caller intends*:

| `log2fc` kwarg | `abs_weight` | `targets` shape | Behaviour |
|---|---|---|---|
| present | 0.0 | `(B, 1, 1)` or `(B, 1, L)` | Delta from kwarg; targets ignored. |
| present | > 0 | `(B, N, L)` | Delta from kwarg; abs term reads targets. |
| absent | 0.0 | `(B, 1, 1)` | Fallback: delta = squeeze(targets). |
| absent | > 0 | `(B, N, L)` | Fallback returns garbage — averages the whole (B, N, L). Guarded against in `0017795` for the scalar-targets sub-case but still a footgun. |

The 4-row truth table is the smell.

### 2.3 `AttributionTarget` vs `DifferentialAttributionTarget` ([attribution.py](src/cerberus/attribution.py))

Side-by-side:

| Aspect | `AttributionTarget` | `DifferentialAttributionTarget` |
|---|---|---|
| Fields | `channel: int`, `bin_index: int \| None`, `window_start/_end: int \| None` | `cond_a_idx: int`, `cond_b_idx: int`, `window_start/_end: int \| None` |
| Reductions | 5 (`log_counts`, `profile_bin`, `profile_window_sum`, `pred_count_bin`, `pred_count_window_sum`) | 2 (`delta_log_counts`, `delta_profile_window_sum`) |
| `_resolve_window` | Private method | Identical copy of the above |
| Dispatch | `if self.reduction == ...` chain | Identical pattern, different reduction set |
| Entry points | `TARGET_REDUCTIONS` set | `DIFFERENTIAL_TARGET_REDUCTIONS` set |

The only structural difference is `channel: int` vs the pair
`(cond_a_idx, cond_b_idx)`. That can be unified trivially by making the
field `channels: list[int]` — `[ch]` for non-delta reductions,
`[a, b]` for delta reductions.

The differential reductions are arithmetically straightforward —
`log_counts[:, b] − log_counts[:, a]` is literally two selects and a
subtract. There is no new architectural concept in
`DifferentialAttributionTarget`. It is a second copy of the same
pattern with slightly different indexing.

**Why split at all?** The only justification is "reductions that use
two channels need two channel indices, and we wanted type-level
segregation." That's weak relative to the cost of maintaining two
classes with overlapping helpers.

### 2.4 Where cerberus is already correct

These pieces do not need to change:

- `MultitaskBPNet` ([bpnet.py:427-509](src/cerberus/models/bpnet.py#L427-L509)). Forces `predict_total_count=False`, emits `(B, N, L)` logits + `(B, N)` log_counts. Exactly the I/O the workflow needs.
- `MultitaskBPNetLoss` ([bpnet.py:512-556](src/cerberus/models/bpnet.py#L512-L556)). Pins `count_per_channel=True`, `average_channels=True`, `flatten_channels=False`, `log1p_targets=False`. Phase 1 training works end-to-end on the standard `train_single` / `train_multi` path today.
- `SignalExtractor` + `UniversalExtractor` ([signal.py](src/cerberus/signal.py)). Reads multi-channel bigwigs once, handles NaN / missing chrom / padding, is pickle-safe for DataLoader workers, has an in-memory variant. This is the canonical bigwig reader and the Phase 2 precompute duplicates its work.
- `CerberusDataset` sample dict ([dataset.py:326-330](src/cerberus/dataset.py#L326-L330)): `{inputs, targets, intervals, interval_source}`. Clean, already-established contract.
- `CerberusModule._shared_step` kwargs plumbing: all batch keys except `inputs` and `targets` flow to the loss as `**kwargs`. Already used by `DalmatianLoss` via `kwargs["interval_source"]` ([loss.py:785](src/cerberus/loss.py#L785)).

---

## 3. What the Library Already Provides (and What's Missing)

### 3.1 Per-condition signal is already in `targets`

The `(B, N, L)` absolute-signal track that Phase 2 wants to reduce to
log2FC is *already* being fed to the loss. Phase 1's
`MultitaskBPNetLoss` uses it as-is (MSE against per-channel
`log_counts`). The Phase 2 log2FC target can be derived from the same
tensor:

```
counts = targets.sum(dim=-1)                       # (B, N)
delta_target = log2((counts[:, b] + pc) / (counts[:, a] + pc))   # (B,)
```

This is arithmetically equivalent to what
`_compute_log2fc_from_bigwigs` returns — same bigwig signal, same
pseudocount, same log2. The difference is that it runs inside the
training step using the already-extracted tensor, not in a separate
offline pass.

### 3.2 External-labels support is not a current requirement

The `log2fc` kwarg exists in `DifferentialCountLoss` to let a caller
precompute a delta target externally (e.g. DESeq2/edgeR shrunk
estimates). Audit of the tree on marcin-feature:

- No test exercises the kwarg-override path with anything other than
  synthetic tensors for shape/type validation.
- The training tool populates the kwarg from its own bigwig
  precompute — which is exactly the work the redesign moves inside
  the loss.
- No CLI flag, external workflow, or user-facing documentation
  references an external log2FC source.

The bigwig-derived delta is the only delta any current code path
produces. Treating external-labels support as a requirement would
preserve machinery nobody is using. The recommended design drops
that machinery; §13 describes how to add it back in a targeted way
if and when a real external-label consumer appears.

### 3.3 What's missing (and cheap to add)

- **A derived-quantity transform.** The current transform API
  ([transform.py:10-20](src/cerberus/transform.py#L10-L20)) is
  `(inputs, targets, interval) -> (inputs, targets, interval)` — it
  cannot add new keys to the sample dict. A broader API that lets a
  transform return additional per-sample scalars would enable
  log2FC-in-dataset as one option, though the §5 recommendation does
  not require it (the computation fits naturally in the loss).
- **A `train_single`/`train_multi` entrypoint that can take an
  already-built dataset**, so a Phase 2 tool can reuse a Phase 1
  dataset without re-instantiating. Today `train_single` builds a
  `CerberusDataModule` from configs; a variant that accepts a
  pre-built datamodule would be useful for fine-tuning workflows.
  Not strictly required for this redesign — Phase 2 can build a
  fresh `CerberusDataModule` from the same config as Phase 1.

### 3.4 The joanne-feature `src/cerberus/differential.py` (not ported)

On `origin/joanne-feature` this module exposes seven symbols
(`compute_log2fc_cpm`, `compute_bigwig_counts`,
`compute_log2fc_from_bigwigs`, `DifferentialRecord`,
`load_differential_targets`, `write_differential_targets`,
`DifferentialTargetIndex`).

Usage audit (done during the port):
- Only `compute_log2fc_from_bigwigs`, `DifferentialRecord`, and
  `write_differential_targets` are called by the training tool.
- `DifferentialTargetIndex` is imported but never instantiated.
- `load_differential_targets` has no caller anywhere in the tree —
  the written TSV is never read back.
- `compute_log2fc_cpm` is only reached internally by the bigwig
  helper.
- `compute_bigwig_counts` is only reached internally by the same.

Conclusion: the module is scaffolding for a "bring your own DESeq2
labels" workflow that nothing in the tree exercises. The recommended
redesign does not need any of it; the external-labels case is served
by the existing `log2fc` kwarg.

---

## 4. The Native-Cerberus Solution in One Picture

```
                           ┌──────────────────────────────┐
                           │  CerberusDataModule          │
                           │  (same as Phase 1)           │
                           │  targets = {A: bw_a,         │
                           │             B: bw_b}         │
                           └──────────┬───────────────────┘
                                      │
                                      │ batch = {
                                      │   inputs: (B, 4, L),
                                      │   targets: (B, 2, L),
                                      │   intervals, interval_source
                                      │ }
                                      ▼
         ┌───────────────────────────────────────────────┐
         │  CerberusModule._shared_step                  │
         │     outputs = model(inputs)                   │
         │     loss = criterion(outputs, targets,        │
         │                      **batch_context)         │
         └──────────────┬────────────────────────────────┘
                        │
                        │ targets passes through as (B, 2, L);
                        │ no extra kwargs required.
                        ▼
         ┌───────────────────────────────────────────────┐
         │  DifferentialCountLoss   (single code path)   │
         │     counts = targets.sum(dim=-1)   # (B, 2)   │
         │     target_delta = log2((counts_B + pc) /     │
         │                         (counts_A + pc))      │
         │     delta_pred = log_counts[:,B] -            │
         │                  log_counts[:,A]              │
         │     return MSE(delta_pred, target_delta)      │
         └───────────────────────────────────────────────┘
```

Single protocol, one code path: `forward(outputs, targets)` with no
optional kwargs and no shape overloading. External-label support is
deferred to §13 as future work.

---

## 5. Design Options Considered

> **Option B (§5.2) is the selected design.** The other options are
> documented here as considered-and-rejected alternatives, for
> future maintainers who will want to understand why the API looks
> the way it does.

### 5.1 Option A — Minimal: inline the log2FC derivation; drop the kwarg + fallback + abs_weight *(not selected)*

**Scope.**

- `DifferentialCountLoss` gains a single-path target derivation:
  `targets` is always `(B, N, L)`, delta is computed inline from
  `targets.sum(dim=-1)` with `count_pseudocount`.
- Remove the `log2fc` kwarg, the `(B, 1, 1)` / `(B, 1, L)` scalar
  fallback, **and the `abs_weight` regularizer branch**. Resulting
  loss is ~15 lines: one init, one `forward`. No `loss_components`
  decomposition needed (only one component exists).
- Remove `_resolve_delta_target` and the `uses_count_pseudocount`
  machinery around the kwarg path; `count_pseudocount` stays as a
  constructor arg (used in the derivation).
- Rewrite `tools/train_multitask_differential_bpnet.py` to drop
  `_DiffWrapper`, `_Phase2Module`, `_compute_log2fc_from_bigwigs`,
  `_DifferentialRecord`, `_write_differential_targets`. Phase 2
  becomes a second `train_single` call with a different `ModelConfig`.
- `AttributionTarget` / `DifferentialAttributionTarget` unchanged.

**Pros.**

- Cleanest fix for the single biggest issue (the offline precompute +
  wrapper).
- `DifferentialCountLoss.forward(outputs, targets)` is the entire
  surface — no optional kwargs, no shape branches, no regularizer
  choice.
- Tool shrinks dramatically (est. 1068 → 350 lines).
- Library side shrinks: ~100 lines of `DifferentialCountLoss`
  machinery collapses to ~15.

**Cons.**

- Removes three `DifferentialCountLoss` features at once: `log2fc`
  kwarg, scalar fallback, `abs_weight` regularizer. None has an
  in-tree consumer (§3.2, §6.1); each is documented as future-work
  re-entry in §13.
- Attribution module keeps both classes; the near-duplication remains.

**Appropriate when:** you want the tool fixed this week and are OK
with the attribution-side duplication as a known debt.

### 5.2 Option B — **Selected**: inline derivation + unified `AttributionTarget` (no shim)

**Scope.** Option A plus:

- Collapse `DifferentialAttributionTarget` into `AttributionTarget`.
  Single class, single reductions set,
  `channels: int | tuple[int, int]` field (int for the 5
  single-channel reductions, 2-tuple for the 2 delta reductions).
- Fold `DIFFERENTIAL_TARGET_REDUCTIONS` into `TARGET_REDUCTIONS`.
  Remove the grouping set.
- **Remove** `DifferentialAttributionTarget` from the public API.
  In-tree callers update in the same commit:
  - `tools/train_multitask_differential_bpnet.py` (the interpret path)
  - `tools/run_tpcav.py` (unaffected — it uses `AttributionTarget` already)
  - `tests/test_attribution.py` (tests for the differential path)
- Remove `DIFFERENTIAL_TARGET_REDUCTIONS` and
  `DifferentialAttributionTarget` from `src/cerberus/__init__.py`
  exports.

**Pros.**

- Addresses both the tool-level mess and the attribution-module
  duplication.
- Library's attribution surface is closed-form: one class,
  seven reductions, one registry.
- No compatibility shim to maintain. Breaking change is contained
  to in-tree callers, all of which land in the same commit.

**Cons.**

- `AttributionTarget.channel` → `AttributionTarget.channels`
  (mechanical rename). Touches ~10 test + tool call sites.
- External callers of `DifferentialAttributionTarget` (none known)
  would need to update. Since it was added in the joanne-feature
  port and has been public for ~1 release cycle, this risk is low.

**Status:** **Selected.** This option accepts one intentional
breaking-change commit in exchange for a permanently simpler
attribution API and a clean library surface for the differential
workflow.

### 5.3 Option C — Radical: drop `DifferentialCountLoss` entirely; make it a composition *(not selected)*

**Scope.** Option B plus:

- Delete `DifferentialCountLoss`. Phase 2 becomes a generic
  `CoupledChannelDeltaLoss` that supervises `head_b − head_a` against
  a supplied target-delta function (default: the bigwig-derived
  log2FC from `targets.sum(dim=-1)`).
- Reduction selection (delta between which two channels) is provided
  via a config arg, not a subclass — matches the pattern of
  `BPNetLoss` being a pinned `MSEMultinomialLoss`.

**Pros.**

- Loss module becomes tidier: `DifferentialCountLoss` is literally
  "BPNet count-head delta, pinned". No standalone class.
- Symmetric with the `MultitaskBPNetLoss` / `BPNetLoss` pattern
  (subclass pins parameters of a generic base).
- Once the delta-target function is a first-class concept, adding a
  second delta objective later is cheap.

**Cons.**

- Premature generalization if only one delta objective ever
  materialises.
- Touches more files: `DifferentialCountLoss` callers, the
  tool's `ModelConfig`, and any doc/test that references the class
  by name.
- Slightly more surface than Option B for the same primary problem.

**Appropriate when:** delta-learning is a *growing* area (multiple
delta objectives, training-time attribution regularizers,
comparative multi-condition readouts) and you want the plumbing
ready for it.

### 5.4 Option D — Rejected: "log2FC transform on dataset"

Considered during research: make log2FC a transform that adds a
`log2fc` key to the sample dict; the loss reads that key.

Rejected because:
- The transform API cannot add new keys today (§3.3).
- The log2FC derivation is a one-liner inside the loss; pushing it
  up into the data pipeline adds machinery without solving a
  problem.
- The kwarg-based delivery path was already used by the offline
  precompute the redesign is removing. Reintroducing it via a
  transform repeats the same mistake through a different door.

---

## 6. Why Option B (Rationale Log)

Option B is selected. This section records the reasoning against
the alternatives, so future readers understand why the API looks
the way it does.

1. **Option A fixes only the worst problem** (offline precompute,
   wrapper, 1068-line tool). It leaves the attribution-module
   duplication in place — two classes, duplicated `_resolve_window`,
   two reduction sets. The duplication was the single most
   callable-out structural debt from the joanne-feature port;
   deferring it perpetuates a known hack.

2. **Option C's value depends on a use case that doesn't exist yet.**
   The generic `CoupledChannelDeltaLoss` base would pay off if a
   second delta objective appeared (e.g. a training-time attribution
   regularizer, a chromatin-state delta loss). None exists today.
   Designing the base class without a second consumer risks fitting
   it to the one we have; revisit when a second one appears.

3. **Option B's breaking-change cost is contained.** Three in-tree
   call sites need updating (the training tool, `tests/test_attribution.py`,
   `src/cerberus/__init__.py` exports). All land in one commit. No
   deprecation period, no shim, no external caller impact
   (`DifferentialAttributionTarget` has existed for ~one release
   cycle and has no known external users).

4. **The attribution unification in Option B is a win on its own.**
   Even if the differential-loss simplification stalled, pulling the
   two attribution targets into one class would be worth doing —
   the two classes diverge along a trivially-parameterizable axis
   (1 channel vs. 2). Under Option B we collect the wins together
   while the context is fresh.

5. **Option B's library delta is strictly negative** (~−200 lines
   across `loss.py`, `attribution.py`, `__init__.py`). Under Options
   A or C the library either stays the same size (A) or trades
   complexity between files without a clear net win (C). Option B is
   the option that leaves `src/cerberus/` smaller and more polished.

### 6.1 Constraints of the selected design

- **Do not keep the `log2fc` kwarg as a public API.** It exists
  today only to serve the offline precompute the redesign is
  removing. Keeping it "just in case" perpetuates the four-row
  truth table that made `DifferentialCountLoss` hackish. External
  labels, if ever needed, come back via the §13 re-entry plan.
- **Do not keep the `abs_weight` regularizer.** Per Naqvi et al.
  2025 the default is 0.0; no in-tree caller sets it; no test
  asserts that removing it is visible to users. Library code does
  not carry optional features "just in case". §13 documents how to
  restore it if a real need appears.
- **Do not keep a `DifferentialAttributionTarget` back-compat shim.**
  Three in-tree callers all update in the same commit. A shim
  preserves name continuity at the cost of two classes in
  `src/cerberus/attribution.py` forever. Not worth it.
- **Do not merge `MultitaskBPNet` back into `BPNet`.** The `≥ 2`
  output_channels invariant and `predict_total_count=False`
  enforcement are real guardrails, not cosmetic.
- **Do not add transform-API extensions for this redesign.** The
  single in-tree consumer (log2FC derivation) fits in the loss.
  Broader transform-API changes are a cerberus-wide concern with
  their own design process; do not couple them to this work.
- **Do not introduce a `TrainConfig.freeze_patterns` field for Phase
  2 trunk freezing.** The tool can do `for p in model.trunk.parameters():
  p.requires_grad = False` directly — a one-off workflow need that
  does not justify a library feature. When a second caller appears
  with the same need, promote then.

---

## 7. Implementation Plan

### 7.1 Library changes

**`src/cerberus/loss.py`** — `DifferentialCountLoss` collapses to
a single-path, single-concept loss. Landed body (schematic):

```
class DifferentialCountLoss(nn.Module):
    """MSE on log_counts[:, B] - log_counts[:, A].

    Delta target is derived from the (B, N, L) targets tensor:
        log2((sum_B + pc) / (sum_A + pc))
    where sums are over the length axis and pc is count_pseudocount
    (same value Phase 1 used so log_counts are in one space).
    """

    uses_count_pseudocount: bool = True

    def __init__(
        self,
        cond_a_idx: int = 0,
        cond_b_idx: int = 1,
        count_pseudocount: float = 1.0,
    ) -> None:
        super().__init__()
        if cond_a_idx == cond_b_idx:
            raise ValueError(...)
        if cond_a_idx < 0 or cond_b_idx < 0:
            raise ValueError(...)
        self.cond_a_idx = cond_a_idx
        self.cond_b_idx = cond_b_idx
        self.count_pseudocount = count_pseudocount

    def _delta_loss(
        self, outputs: object, targets: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(outputs, ProfileCountOutput):
            raise TypeError(...)

        a, b = self.cond_a_idx, self.cond_b_idx
        n = outputs.log_counts.shape[-1]
        if max(a, b) >= n:
            raise ValueError(...)
        if targets.ndim != 3:
            raise ValueError(...)
        if targets.shape[1] < max(a, b) + 1:
            raise ValueError(...)

        counts = targets.float().sum(dim=-1)              # (B, N)
        pc = self.count_pseudocount
        target_delta = torch.log2((counts[:, b] + pc) /
                                  (counts[:, a] + pc))    # (B,)
        delta_pred = (outputs.log_counts[:, b]
                      - outputs.log_counts[:, a])         # (B,)
        return F.mse_loss(delta_pred, target_delta)

    def loss_components(
        self, outputs: object, targets: torch.Tensor, **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        return {"delta_loss": self._delta_loss(outputs, targets)}

    def forward(
        self, outputs: object, targets: torch.Tensor, **kwargs: object,
    ) -> torch.Tensor:
        return self._delta_loss(outputs, targets)
```

No `abs_weight`, no `log2fc` kwarg, no `_resolve_delta_target`, no
shape overloading, no 4-row truth table. `loss_components` is kept
(required by `CerberusModule._shared_step` for per-component
logging) but returns a single entry. `**kwargs` is kept on the
public methods to match the `CerberusLoss` protocol; the loss
ignores all batch-context kwargs. Landed class body is ~120 lines
including docstring and validation (down from ~170).

Pseudocount semantics: use the loss's own `count_pseudocount`. In the
bpAI-TAC / Naqvi recipe this is the *same* pseudocount Phase 1 used
(propagated via `data_config.count_pseudocount` →
`propagate_pseudocount` → `MSEMultinomialLoss.count_pseudocount`), so
Phase 1 and Phase 2 operate in the same log-space.

**`src/cerberus/attribution.py`** — one class replaces two:

```
TARGET_REDUCTIONS = {
    # single-channel (channels: int)
    "log_counts",
    "profile_bin",
    "profile_window_sum",
    "pred_count_bin",
    "pred_count_window_sum",
    # multi-channel delta (channels: tuple[int, int], [a, b])
    "delta_log_counts",
    "delta_profile_window_sum",
}

class AttributionTarget(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        reduction: str,
        channels: int | tuple[int, int],
        bin_index: int | None = None,
        window_start: int | None = None,
        window_end: int | None = None,
    ):
        ...  # validates arity of `channels` against the reduction
```

`DifferentialAttributionTarget` is removed outright — no shim. The
three in-tree callers (§5.2) update in the same commit to use
`AttributionTarget(reduction="delta_log_counts", channels=(0, 1))`
etc.

`DIFFERENTIAL_TARGET_REDUCTIONS` is removed. Users who want to know
which reductions are "delta" reductions can inspect the arity of
`channels` the reduction implies, or we can expose a predicate like
`is_delta_reduction(name: str) -> bool` if a caller needs it (none
do today).

### 7.2 Tool rewrite

Everything workflow-specific stays in the tool. The library exposes
a model, a loss, and an attribution target; the tool composes them.

**`tools/train_multitask_differential_bpnet.py` landed structure:**

```
get_args()

# Workflow-specific one-offs (tool-only):
_read_bed_intervals(path)               # BED/narrowPeak parser
_merge_and_write_peaks(a, b, out)       # union BED merge
_find_phase1_checkpoint(path)           # glob **/model.pt
_plot_phase2_losses(phase2_dir)         # CSVLogger metrics.csv plot
_select_phase2_strategy(kwargs)         # override to ddp_find_unused_parameters_true
_dinuc_shuffle(...)                     # DLS baseline gen

# Workflow phases (compositions of library primitives):
run_phase1(args):
    # ModelConfig: MultitaskBPNet + MultitaskBPNetLoss.
    # train_single / train_multi on the merged peaks.

run_phase2(args, phase1_ckpt):
    # Same DataConfig shape (same two bigwigs).
    # ModelConfig: MultitaskBPNet + DifferentialCountLoss, with
    # pretrained=[PretrainedConfig(phase1_ckpt, ...)].
    # train_single / train_multi unchanged. DDP strategy overridden
    # to ddp_find_unused_parameters_true so the un-gradiented profile
    # heads don't trip DDP's bucket rebuild.

run_interpretation(args):
    # AttributionTarget(reduction="delta_log_counts", channels=(0, 1)).
    # DeepLIFTSHAP + TF-MoDISco pipe, as today.

main()
```

No `_freeze_phase2_trunk` (the profile heads naturally get no
gradient under `DifferentialCountLoss`, so DDP's
`find_unused_parameters=True` is sufficient). No
`_data_config_two_channel` (the Phase 1 and Phase 2 `DataConfig`s
are inlined).

Tool line counts (before → after, 2026-04-18):

| Section | Before | After |
|---|---|---|
| Argument parser + main | ~250 | ~230 |
| `_merge_and_write_peaks` + BED helpers | ~50 | ~40 |
| `_find_phase1_checkpoint` | ~10 | ~10 |
| `run_phase1` | ~140 | ~140 |
| `run_phase2` | ~200 | ~160 (drop wrapper+module, reuse `train_single`) |
| `_plot_phase2_losses` / `_select_phase2_strategy` | — | ~55 |
| `run_interpretation` | ~200 | ~175 |
| `_dinuc_shuffle` | ~20 | ~20 |
| Inlined differential helpers (`_compute_log2fc_from_bigwigs`, `_DifferentialRecord`, `_write_differential_targets`, `_get_pos_intervals`) | ~60 | 0 |
| `_DiffWrapper` / `_Phase2Module` | ~100 | 0 |
| **Total** | **~1134** | **~907** |

Phase 2 specifically became (schematically):

```
run_phase2(args, phase1_ckpt_path):
    data_config = DataConfig(
        targets={name_a: bw_a, name_b: bw_b}, max_jitter=0, ...)
    model_config = ModelConfig(
        name="MultitaskBPNet_Phase2",
        model_cls="cerberus.models.bpnet.MultitaskBPNet",
        loss_cls="cerberus.loss.DifferentialCountLoss",
        loss_args={"cond_a_idx": 0, "cond_b_idx": 1},
        pretrained=[PretrainedConfig(
            weights_path=str(phase1_ckpt_path),
            source=None, target=None, freeze=False)],
        model_args={...},
        count_pseudocount=args.count_pseudocount * args.target_scale,
    )
    precision_kwargs = get_precision_kwargs(args.precision, accel, devices)
    # Override the multi-GPU strategy: DifferentialCountLoss only
    # touches count heads, so profile heads receive no gradient.
    precision_kwargs = _select_phase2_strategy(precision_kwargs)
    train_fn = train_multi if args.multi else train_single
    train_fn(... model_config ... train_config ... **precision_kwargs)
```

That's it. No `_DiffWrapper`, no `_Phase2Module`, no offline log2FC
precompute, no `abs_weight` flag. The DataLoader emits `{inputs,
targets, intervals, interval_source}`; `DifferentialCountLoss`
derives the delta target from `targets` in the training step.

### 7.3 Pretrained-weight loading for Phase 2

Phase 2 initializes from Phase 1's weights via
`ModelConfig.pretrained: list[PretrainedConfig]`. On the landed
tool, `_find_phase1_checkpoint` globs `**/model.pt` under the Phase
1 output dir and feeds the path into
`PretrainedConfig(weights_path=..., source=None, target=None,
freeze=False)`. `load_pretrained_weights` ([src/cerberus/pretrained.py](src/cerberus/pretrained.py))
then calls `target.load_state_dict(state_dict, strict=True)` on the
freshly-built `MultitaskBPNet`, which succeeds because the Phase 1
and Phase 2 architectures are byte-identical (same
`output_channels`, `filters`, `n_dilated_layers`, ...).

**Optional fine-tuning shapes** (freeze trunk, freeze profile heads,
etc.) are not implemented in the landed tool: the
`DifferentialCountLoss` path naturally zeroes gradient on the
profile heads, and the DDP strategy override handles the multi-GPU
case. If a future workflow wants explicit freezing, express it
either via a new `PretrainedConfig(target="trunk", freeze=True)`
entry or a PL callback — both supported by the current cerberus
pipeline without library changes.

### 7.4 Back-compat and migration

The design is deliberately not back-compatible — the breaking
changes land in one commit, there is no deprecation shim, and the
three in-tree call sites migrate in the same commit. Full surface
diff is in §9. Summary:

- **`DifferentialAttributionTarget`**: removed. No shim. In-tree
  call sites rewrite to `AttributionTarget(reduction=...,
  channels=(a, b))`.
- **`DifferentialCountLoss(log2fc=...)` kwarg**: removed. Delta is
  derived from the `(B, N, L)` `targets` tensor inline. §13.1
  describes the future re-entry path for external labels.
- **`DifferentialCountLoss(abs_weight=...)`**: removed. §13.2
  describes the future re-entry path for the regularizer.
- **`DIFFERENTIAL_TARGET_REDUCTIONS`**: removed. Delta reductions
  are members of `TARGET_REDUCTIONS`.
- **CLI flags** on the tool (`--bigwig-a`, `--peaks-a`, etc.):
  mostly unchanged. `--abs-weight` and `--diff-pseudocount` drop
  (they serve library features that no longer exist).
- **`differential_targets.tsv` provenance output**: dropped under
  the redesign, since the log2FC is no longer precomputed. If the
  provenance is valued (e.g. for "what log2FC did my Phase 2 actually
  see on fold_3?"), it can be re-expressed as a post-hoc call:
  "load the val dataset, run it through the loss's target-derivation
  function once, write the TSV." Cheap to add; not on the critical
  path.

### 7.5 Tests

Tests that landed under Option B (see `tests/test_multitask_differential_bpnet.py`
and `tests/test_attribution.py`):

- **`test_differential_count_loss_derives_delta_from_targets`** —
  zero-MSE round-trip on a `(B, N, L)` target tensor with known
  per-channel length-sums.
- **`test_differential_count_loss_nonzero_when_prediction_off`** —
  MSE = mean(expected_delta ²) when the prediction is zero.
- **`test_differential_count_loss_pseudocount_affects_target`** —
  larger `count_pseudocount` → smaller |log2 ratio| → smaller MSE.
- **`test_differential_count_loss_components_has_only_delta_loss`**
  — the `loss_components` dict has a single `"delta_loss"` entry.
- **`test_differential_count_loss_rejects_2d_targets`** — targets
  must be `(B, N, L)`; `(B, N)` raises.
- **`test_differential_count_loss_rejects_too_few_target_channels`**
  — targets with fewer channels than `max(cond_a, cond_b) + 1` raise.
- **`test_differential_count_loss_same_idx_raises`,
  `test_differential_count_loss_out_of_range_idx`,
  `test_differential_count_loss_rejects_negative_idx`** — index
  validation (construction time for same-idx / negative, forward
  time for out-of-range).
- **`test_differential_count_loss_wrong_output_type`** — rejects
  non-`ProfileCountOutput`.
- **`test_attribution_target_delta_requires_tuple_channels`,
  `test_attribution_target_single_rejects_tuple_channels`** — arity
  validation on `channels`.
- **`test_attribution_target_delta_same_idx_raises`** — delta
  reduction with `channels=(k, k)` raises.
- **`test_attribution_target_delta_log_counts`,
  `test_attribution_target_delta_profile_window_sum_{full,partial}_window`,
  `test_attribution_target_delta_invalid_window_raises`,
  `test_attribution_target_delta_out_of_range_channel_raises`,
  `test_attribution_target_delta_gradient_flows`,
  `test_attribution_target_delta_ism_matches_manual_delta`** —
  delta-reduction math, window handling, gradient flow, and linear-
  model ISM parity (`attr(f_B - f_A) == attr(f_B) - attr(f_A)`).
- **`test_attribution_target_delta_reductions_in_target_reductions`**
  — `{delta_log_counts, delta_profile_window_sum} ⊂ TARGET_REDUCTIONS`.

Tests removed (they encoded protocols that no longer exist):

- `test_differential_count_loss_basic` — used `(B, 1, 1)` scalar
  fallback; rewritten against `(B, N, L)` primary.
- `test_differential_count_loss_log2fc_kwarg`,
  `test_differential_count_loss_log2fc_wrong_type`,
  `test_differential_count_loss_log2fc_batch_size_mismatch` — kwarg
  is gone.
- `test_differential_count_loss_abs_weight_components`,
  `test_differential_count_loss_abs_weight_scalar_targets_raises` —
  `abs_weight` is gone.
- `test_differential_count_loss_components_keys` — the components
  dict shape changed (now asserted by
  `…_components_has_only_delta_loss`).
- Tests that imported `DifferentialAttributionTarget` — rewritten
  against `AttributionTarget(reduction=…, channels=(a, b))`.
- `tests/test_train_multitask_differential_tool.py` was deleted
  (tested the removed `_DiffWrapper`).

Net on the landed state:
`tests/test_multitask_differential_bpnet.py` gained a handful of
single-path derivation tests; `tests/test_attribution.py` kept the
same coverage under a unified class; the `_DiffWrapper` test file
is gone.

### 7.6 Documentation

- `docs/usage.md` §Differential-Learning gains a section that
  cross-references the TPCAV / TF-MoDISco sections and makes the
  "Phase 1 and Phase 2 are the same CLI with different configs"
  story explicit.
- `CHANGELOG.md` entry under `### Changed` describing the `targets`
  contract change (now always `(B, N, L)`, scalar fallback reserved
  for explicit overrides via `log2fc` kwarg).

---

## 7a. Visual Reference

The following diagrams and tables are the ground-truth reference for
the selected design. If the prose in §7 and the visuals here
disagree, the visuals are authoritative.

### 7a.1 Phase 2 data flow: before vs. after

**Before (current):**

```
 ┌─────────────────┐     ┌──────────────────────────┐
 │ bigwig_a        │     │ bigwig_b                 │
 └────────┬────────┘     └──────────┬───────────────┘
          │                         │
          │   (offline pass #1: precompute per-peak log2FC)
          │                         │
          ▼                         ▼
 ┌────────────────────────────────────────────────────┐
 │  _compute_log2fc_from_bigwigs   (tool-private)     │
 │     sum_a, sum_b = sum(signal over each interval)  │
 │     log2fc = log2((sum_b + pc) / (sum_a + pc))     │
 └──────────────────────┬─────────────────────────────┘
                        │ np.ndarray (N,)
                        │ — written as provenance TSV
                        │   never read back
                        ▼
 ┌────────────────────────────────────────────────────┐
 │  _DiffWrapper  (tool-private Dataset wrapper)      │
 │     __getitem__(idx):                              │
 │         item = base[idx]                           │
 │         item["log2fc"] = self.log2fc[idx]  ← INJ. │
 │         return item                                │
 └──────────────────────┬─────────────────────────────┘
                        │
          (offline pass #2 via SignalExtractor, per-batch:
           CerberusDataset reads the SAME bigwigs again)
                        │
                        ▼
 ┌────────────────────────────────────────────────────┐
 │  batch = {inputs, targets, intervals,              │
 │           interval_source, log2fc}                 │
 └──────────────────────┬─────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────┐
 │  _Phase2Module._step  (tool-private PL wrapper)    │
 │     loss = loss_fn(out, targets, log2fc=log2fc)    │
 └──────────────────────┬─────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────┐
 │  DifferentialCountLoss   (4-row truth table)       │
 │     if "log2fc" in kwargs:                         │
 │         delta = kwargs["log2fc"]                   │
 │     elif targets is (B, 1, 1): fallback            │
 │     elif targets is (B, N, L) and abs_weight > 0:  │
 │         delta = fallback (BUG: silently garbage    │
 │                 if log2fc kwarg forgotten)         │
 │     ...                                            │
 └────────────────────────────────────────────────────┘
```

**After (Option B):**

```
 ┌─────────────────┐     ┌──────────────────────────┐
 │ bigwig_a        │     │ bigwig_b                 │
 └────────┬────────┘     └──────────┬───────────────┘
          │                         │
          │   (one pass via SignalExtractor, per-batch)
          ▼                         ▼
 ┌────────────────────────────────────────────────────┐
 │  CerberusDataset + SignalExtractor                 │
 │     targets = stack([bw_a, bw_b])    # (B, 2, L)   │
 └──────────────────────┬─────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────┐
 │  batch = {inputs, targets, intervals,              │
 │           interval_source}                         │
 │  (no log2fc key, no tool-private wrapper)          │
 └──────────────────────┬─────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────┐
 │  CerberusModule._shared_step  (library, unchanged) │
 │     loss = self.criterion(outputs, targets)        │
 └──────────────────────┬─────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────┐
 │  DifferentialCountLoss   (ONE code path)           │
 │     counts = targets.sum(-1)          # (B, 2)     │
 │     delta_t = log2((cB + pc)/(cA + pc))            │
 │     delta_p = log_counts[:,B] - log_counts[:,A]    │
 │     return MSE(delta_p, delta_t)                   │
 └────────────────────────────────────────────────────┘
```

Note: the bigwigs are read once per batch via `SignalExtractor`
(which is how Phase 1 already works). The before-state reads them
twice — once offline for precompute, once online for `targets`.

### 7a.2 Attribution-class unification

**Before:**

```
  ┌──────────────────────────┐        ┌──────────────────────────┐
  │ AttributionTarget        │        │ DifferentialAttribution  │
  │                          │        │ Target                   │
  ├──────────────────────────┤        ├──────────────────────────┤
  │ model                    │        │ model                    │
  │ reduction                │        │ reduction                │
  │ channel: int             │        │ cond_a_idx: int          │
  │ bin_index: int|None      │        │ cond_b_idx: int          │
  │ window_start: int|None   │        │ window_start: int|None   │
  │ window_end: int|None     │        │ window_end: int|None     │
  ├──────────────────────────┤        ├──────────────────────────┤
  │ TARGET_REDUCTIONS = {    │        │ DIFFERENTIAL_TARGET_     │
  │   log_counts,            │        │ REDUCTIONS = {           │
  │   profile_bin,           │        │   delta_log_counts,      │
  │   profile_window_sum,    │        │   delta_profile_window_  │
  │   pred_count_bin,        │        │     sum,                 │
  │   pred_count_window_sum, │        │ }                        │
  │ }                        │        │                          │
  ├──────────────────────────┤        ├──────────────────────────┤
  │ _resolve_window()        │   ←──  │ _resolve_window()        │
  │ (identical copies)       │        │                          │
  └──────────────────────────┘        └──────────────────────────┘
```

**After:**

```
  ┌───────────────────────────────────────────────────────┐
  │ AttributionTarget                                     │
  ├───────────────────────────────────────────────────────┤
  │ model                                                 │
  │ reduction                                             │
  │ channels: int | tuple[int, int]                       │
  │ bin_index: int | None                                 │
  │ window_start: int | None                              │
  │ window_end: int | None                                │
  ├───────────────────────────────────────────────────────┤
  │ TARGET_REDUCTIONS = {                                 │
  │   log_counts,                 # channels: int         │
  │   profile_bin,                # channels: int         │
  │   profile_window_sum,         # channels: int         │
  │   pred_count_bin,             # channels: int         │
  │   pred_count_window_sum,      # channels: int         │
  │   delta_log_counts,           # channels: (int, int)  │
  │   delta_profile_window_sum,   # channels: (int, int)  │
  │ }                                                     │
  ├───────────────────────────────────────────────────────┤
  │ _resolve_window()  (once)                             │
  │ _check_single_channel / _check_delta_channels         │
  │ (arity-specific validators, if/elif dispatch in       │
  │  forward — no separate registry)                      │
  └───────────────────────────────────────────────────────┘
```

As landed, the `forward` body stayed as an `if reduction == ...`
chain rather than a dict-of-callables registry — one file, seven
cases, same shape the old class already used. A registry can be
introduced later if the reduction set grows.

### 7a.3 Reduction × field compatibility matrix

Which `AttributionTarget` fields each reduction uses:

| Reduction | `channels` arity | `bin_index` | `window_start` / `window_end` |
|---|---|---|---|
| `log_counts` | `int` | — | — |
| `profile_bin` | `int` | optional *(defaults to center)* | — |
| `profile_window_sum` | `int` | — | optional *(defaults to full)* |
| `pred_count_bin` | `int` | optional *(defaults to center)* | — |
| `pred_count_window_sum` | `int` | — | optional *(defaults to full)* |
| `delta_log_counts` | `tuple[int, int]` | — | — |
| `delta_profile_window_sum` | `tuple[int, int]` | — | optional *(defaults to full)* |

As landed, unused fields are silently ignored — not raised. The
constructor validates `channels` arity against the reduction name;
it does not enforce that e.g. `bin_index` is `None` for
`log_counts`. If that stricter check is wanted, add it in a
follow-up; no in-tree caller relies on either behaviour today.

### 7a.4 `DifferentialCountLoss`: current vs. landed

| Property | Before | Landed (Option B) |
|---|---|---|
| Constructor args | `cond_a_idx`, `cond_b_idx`, `abs_weight`, `count_pseudocount` | `cond_a_idx`, `cond_b_idx`, `count_pseudocount` |
| Instance methods | `__init__`, `_resolve_delta_target`, `loss_components`, `forward` | `__init__`, `_delta_loss`, `loss_components`, `forward` |
| Supported `targets` shapes | `(B, 1, 1)`, `(B, 1, L)`, `(B, N, L)` | `(B, N, L)` only |
| Supported kwargs | `log2fc: Tensor \| None` | `**kwargs` accepted and ignored (protocol contract) |
| Return shape of `forward` | `Tensor` (sum of weighted components) | `Tensor` (MSE scalar) |
| Return of `loss_components` | `{"delta_loss", "abs_loss_a", "abs_loss_b"}` (conditionally) | `{"delta_loss"}` (always single-entry; required by `CerberusModule._shared_step`) |
| Optional behaviours | `abs_weight` regularizer | none |
| Branches inside `forward` | 4 (see §2.2 truth table) | 0 |
| Source line count *(incl. docstring + validation)* | ~170 | ~120 |
| `uses_count_pseudocount` class flag | `True` | `True` |

### 7a.5 File-by-file change summary *(landed)*

| File | Before | After | Δ lines | What changes |
|---|---|---|---|---|
| `src/cerberus/loss.py` | has `DifferentialCountLoss` with kwarg, fallback, `abs_weight` (~170 lines of the class) | single-path class (~120 lines incl. docstring + validation) | **−60** | Remove `_resolve_delta_target`, `log2fc` kwarg, scalar fallback, abs-weight branch. Add `(B, N, L)` shape validation. `loss_components` kept (returns a single `{"delta_loss": ...}` entry). |
| `src/cerberus/attribution.py` | two classes (`AttributionTarget`, `DifferentialAttributionTarget`) + duplicated `_resolve_window` + two reduction sets | single `AttributionTarget` + one `TARGET_REDUCTIONS` set | **−24** | Delete `DifferentialAttributionTarget` class. Rename `channel` → `channels`. Fold delta reductions into `TARGET_REDUCTIONS` (now 7 entries). Delete `DIFFERENTIAL_TARGET_REDUCTIONS`. Arity-validated in `__init__`. Dispatch stayed as an `if/elif` chain. |
| `src/cerberus/__init__.py` | exports `DifferentialAttributionTarget`, `DIFFERENTIAL_TARGET_REDUCTIONS` | doesn't | **−4** | Remove from `from .attribution import …` and from `__all__`. |
| `tools/train_multitask_differential_bpnet.py` | 1134 lines: `_DiffWrapper`, `_Phase2Module`, 4 inlined helpers, Phase 2 bespoke training loop | 907 lines: `_merge_and_write_peaks`, `_find_phase1_checkpoint`, `_plot_phase2_losses`, `_select_phase2_strategy`, `_dinuc_shuffle`, standard `train_single` / `train_multi` for both phases | **−227** | Drop wrapper, PL module, log2FC precompute, `_DifferentialRecord`, TSV writer, `_get_pos_intervals`. Drop `--abs-weight` / `--diff-pseudocount`. Phase 2 rewritten as a regular `ModelConfig`-driven `train_single` invocation with `pretrained=[...]`. |
| `tests/test_multitask_differential_bpnet.py` | 22 tests incl. log2fc kwarg path, abs-weight branch, scalar fallback | 25 tests against single path + channel/shape validation | **+12** | Remove: log2fc-kwarg tests, abs-weight tests, scalar-fallback test. Add: delta-derivation, pseudocount-effect, `(B, N, L)` shape validation, components-dict-shape. |
| `tests/test_attribution.py` | has `DifferentialAttributionTarget` tests | uses unified `AttributionTarget` with `channels=(a, b)` | **−23** | Mechanical: update imports, rename `channel` → `channels`, update call sites. Add arity-validation tests. |
| `tests/test_package_api.py` | — | — | **0** | No change (this file tests TPCAV helpers, not `DifferentialAttributionTarget` — prior claim in the draft was wrong). |
| `tests/test_train_multitask_differential_tool.py` | tests `_DiffWrapper` length invariant | **deleted** (wrapper no longer exists) | **−53** | The invariant being tested doesn't apply — no wrapper to wrap. |
| `CHANGELOG.md` | — | `### Added` / `### Changed` / `### Removed` / `### Internal` entries | +~90 | Document the breaking changes and new Phase 2 shape. |
| `docs/usage.md` | — | "Two-phase multitask-differential training" + "Scalar attribution targets" sections | +~90 | Added after the fact in a docs-only commit — see §7.6. |
| `docs/models.md` | — | "MultitaskBPNet" / "Loss: MultitaskBPNetLoss" / "DifferentialCountLoss" sections | +~100 | Same docs-only commit. |
| `pyrightconfig.json` | — | `"reportPrivateImportUsage": "none"` | +1 | Prerequisite for treating pyright output as signal; 1769 false positives on `torch.*` were drowning real errors. |

**Net across the tree (code + tests, excluding docs and design
doc):** **≈ −380 lines.** The design doc's original projection of
≈ −1000 was optimistic — see the Implementation notes at the top
of this doc for where the gap comes from.

### 7a.6 Call-site migrations

All in-tree call sites that break under Option B, with the before →
after rewrite.

**1. `tools/train_multitask_differential_bpnet.py` interpret path:**

```python
# Before
from cerberus.attribution import DifferentialAttributionTarget
...
diff_target = DifferentialAttributionTarget(
    model=model,
    reduction="delta_log_counts",
    cond_a_idx=0,
    cond_b_idx=1,
)

# After
from cerberus.attribution import AttributionTarget
...
diff_target = AttributionTarget(
    model=model,
    reduction="delta_log_counts",
    channels=(0, 1),
)
```

**2. Phase 2 loss construction (same tool, run_phase2):**

```python
# Before — tool instantiated the loss directly and called it from a
# bespoke _Phase2Module that passed log2fc via a dataset wrapper.
loss_fn = DifferentialCountLoss(
    cond_a_idx=0,
    cond_b_idx=1,
    abs_weight=args.abs_weight,
    count_pseudocount=args.count_pseudocount * args.target_scale,
)
# ...inside _Phase2Module._step:
loss = loss_fn(outputs, targets, log2fc=log2fc)

# After — tool no longer instantiates or calls the loss; it builds a
# ModelConfig that names it, and train_single / train_multi handle
# instantiation (with count_pseudocount injection) and invocation.
model_config = ModelConfig(
    ...,
    loss_cls="cerberus.loss.DifferentialCountLoss",
    loss_args={"cond_a_idx": 0, "cond_b_idx": 1},
    pretrained=[PretrainedConfig(
        weights_path=str(phase1_ckpt_path),
        source=None, target=None, freeze=False)],
    count_pseudocount=args.count_pseudocount * args.target_scale,
)
```

**3. `tests/test_attribution.py` (differential test block):**

```python
# Before
from cerberus.attribution import (
    DIFFERENTIAL_TARGET_REDUCTIONS,
    DifferentialAttributionTarget,
)
...
t = DifferentialAttributionTarget(model=m, reduction="delta_log_counts",
                                  cond_a_idx=0, cond_b_idx=1)

# After
from cerberus.attribution import AttributionTarget, TARGET_REDUCTIONS
...
t = AttributionTarget(model=m, reduction="delta_log_counts", channels=(0, 1))
```

**4. `src/cerberus/__init__.py`:**

```python
# Before
from .attribution import (
    DIFFERENTIAL_TARGET_REDUCTIONS,
    TARGET_REDUCTIONS,
    AttributionTarget,
    DifferentialAttributionTarget,
    ...
)
__all__ = [
    ...,
    "DIFFERENTIAL_TARGET_REDUCTIONS",
    "DifferentialAttributionTarget",
    ...,
]

# After
from .attribution import (
    TARGET_REDUCTIONS,
    AttributionTarget,
    ...
)
__all__ = [
    ...,
    "TARGET_REDUCTIONS",
    "AttributionTarget",
    ...,
]
```

**5. `channel=` → `channels=` rename (additional call sites):**

`AttributionTarget.channel` renamed to `.channels`. The kwarg
change touches three additional files, all updated in the same
commit:

- `src/cerberus/tpcav.py` — `build_tpcav_target_model`'s internal
  call to `AttributionTarget(... channel=channel ...)` becomes
  `AttributionTarget(... channels=channel ...)`. The helper's own
  public signature still takes `channel: int` (the TPCAV MVP
  supports only single-task BPNet, so exposing a tuple here would
  just be noise).
- `tools/export_tfmodisco_inputs.py` — `--target-channel` CLI flag
  preserved; internal `AttributionTarget(channel=...)` call
  updates.
- `notebooks/chip_ar_mdapca2b_attribution_ism_vs_taylor.py` — two
  `AttributionTarget(channel=0, bin_index=None, ...)` call sites
  updated to `AttributionTarget(channels=0)` (the `None`-valued
  positional arguments disappear because the new defaults match).

### 7a.7 Phase 1 / Phase 2 symmetry

Under Option B both phases reduce to the same top-level shape:

```
       ┌─ Phase 1 ─────────────────────────────────┐
       │                                           │
       │  data_config = _data_config_two_channel(  │
       │      {name_a: bw_a, name_b: bw_b})        │
       │                                           │
       │  model_config = ModelConfig(              │
       │      model_cls = MultitaskBPNet,          │
       │      loss_cls  = MultitaskBPNetLoss,      │
       │      model_args = {output_channels: 2},   │
       │      pretrained = [],                     │
       │  )                                        │
       │                                           │
       │  train_single(genome, data, sampler,      │
       │               model, train)               │
       │                                           │
       └───────────────┬───────────────────────────┘
                       │
                       │  Phase 1 checkpoint
                       ▼
       ┌─ Phase 2 ─────────────────────────────────┐
       │                                           │
       │  data_config = _data_config_two_channel(  │
       │      {name_a: bw_a, name_b: bw_b})        │
       │                                           │
       │  model_config = ModelConfig(              │
       │      model_cls = MultitaskBPNet,          │
       │      loss_cls  = DifferentialCountLoss,   │  ← only diff
       │      model_args = {output_channels: 2},   │
       │      pretrained = [phase1_checkpoint],    │  ← only diff
       │  )                                        │
       │                                           │
       │  _freeze_phase2_trunk(model)   (tool hook)│  ← tool-only
       │                                           │
       │  train_single(genome, data, sampler,      │
       │               model, train)               │
       │                                           │
       └───────────────────────────────────────────┘
```

Phase 2 is Phase 1 with a different `loss_cls`, the same
`DataConfig`, a populated `pretrained=[…]` list, and one tool-local
freeze hook. No `_Phase2Module`, no `_DiffWrapper`, no separate
training driver.

---

## 8. Attribution Module Unification

This section expands on the attribution-module part of the
implementation plan. The unification of `AttributionTarget` +
`DifferentialAttributionTarget` is a net simplification worth
understanding in isolation — its rationale stands on its own even
if the loss-side simplification ever needed to be rolled back.

### 8.1 Shape of the unified class

```
class AttributionTarget(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        reduction: str,
        channels: int | tuple[int, int],
        bin_index: int | None = None,
        window_start: int | None = None,
        window_end: int | None = None,
    ):
        ...
```

- `channels: int` — for the 5 single-channel reductions.
- `channels: tuple[int, int]` — for the 2 delta reductions.
- Validation in `__init__`: raise if the reduction's arity does not
  match the type of `channels`. Matrix of allowed combinations lives
  on a single class-level dict, not scattered.

### 8.2 Reduction dispatch

The design considered a dict-of-callables registry:

```
_REDUCTIONS: dict[str, Callable[[AttributionTarget, Output], Tensor]] = {
    "log_counts":              lambda self, out: out.log_counts[:, self.channels],
    "profile_bin":             lambda self, out: out.logits[:, self.channels, self._bin(out.logits.shape[-1])],
    ...
    "delta_log_counts":        lambda self, out: out.log_counts[:, self.channels[1]] - out.log_counts[:, self.channels[0]],
    ...
}
```

**Landed:** the `forward` body is still an `if reduction == ...` chain
— seven cases, one file, matches the shape the old class already had.
The `if/elif` is easy to read and keeps per-reduction validation
(channel arity, bin-index / window-end defaults, branch-specific
error messages) inline with the math. Swap to a registry if the
reduction set grows or if adding reductions starts requiring
cross-cutting changes.

### 8.3 The `channels` field change

Breaking change in the sense of "attribute type widens". Callers
that do `target.channel` break; callers that do `target.channels`
get `int | tuple[int, int]`.

In-tree call sites that need updating (Option B commit):

- `tools/train_multitask_differential_bpnet.py` — the interpret
  path's `DifferentialAttributionTarget(...)` call becomes
  `AttributionTarget(reduction="delta_log_counts", channels=(0, 1))`.
- `tools/run_tpcav.py` — already uses `AttributionTarget`;
  untouched.
- `tests/test_attribution.py` — the differential test block's
  imports and call sites update.
- `src/cerberus/__init__.py` — `DifferentialAttributionTarget` and
  `DIFFERENTIAL_TARGET_REDUCTIONS` removed from exports.

No back-compat shim. The whole change lands in one commit.

### 8.4 Gains vs. cost

Gains:
- One class instead of two. ~24 lines deleted
  (`src/cerberus/attribution.py` shrinks from 415 → 391 lines).
- Single `_resolve_window`, `_resolve_bin`, `_check_single_channel`
  / `_check_delta_channels` helpers (no duplication).
- Inline `if/elif` dispatch — one file, seven cases, same shape as
  before. A registry was considered (§8.2) but not adopted;
  per-reduction validation and error messages stay inline with the
  math.
- API discoverability: users find one `AttributionTarget` with
  seven reductions, not two classes with five + two.

Cost:
- `channels` is a union type at the field level. Enforcement lives
  in `__init__` validation rather than in the type system.

Worth it, on balance.

### 8.5 Why not go further (reductions as classes)?

Rejected: "each reduction becomes a subclass of
`AttributionTarget`". That pattern would:

- Multiply class count from 2 to 7.
- Force users to pick a class rather than a reduction string (harder
  to parameterize from config).
- Push per-reduction validation into a base-class abstract method,
  further away from the math it guards.

The reduction-string-dispatched single class is the sweet spot.

---

## 9. Migration & Compatibility

Under Option B:

| Path | Breaking? | Migration |
|---|---|---|
| `from cerberus.attribution import DifferentialAttributionTarget` | **Yes — removed** | Use `AttributionTarget(reduction=..., channels=(a, b))`. In-tree call sites updated in the same commit. |
| `from cerberus.attribution import DIFFERENTIAL_TARGET_REDUCTIONS` | **Yes — removed** | Check `reduction.startswith("delta_")` or maintain a local constant; no in-tree caller uses this export. |
| `cerberus.DifferentialAttributionTarget` top-level export | **Yes — removed** | Same as above. |
| `AttributionTarget.channel` attribute | **Yes — renamed** | `AttributionTarget.channels` (`int` for single-channel reductions, `tuple[int, int]` for delta reductions). |
| `DifferentialCountLoss(log2fc=tensor)` kwarg | **Yes — removed** | Pass `(B, N, L)` absolute-signal `targets` (the default Phase 2 dataset shape); delta is derived inline. See §13 for future re-entry if external labels become a requirement. |
| `DifferentialCountLoss(targets=(B, 1, 1))` scalar fallback | **Yes — removed** | Same as above. No in-tree caller uses this path. |
| `DifferentialCountLoss(abs_weight=...)` argument | **Yes — removed** | See §13 for future re-entry if an in-tree caller appears. No current caller sets this. |
| `DifferentialCountLoss(targets=(B, 2, L))` with no other args | **Semantic change**: previously silently averaged the whole tensor; now returns correct derived delta | Intentional fix. |
| `tools/train_multitask_differential_bpnet.py --flag-x` (CLI surface) | CLI mostly unchanged | `--abs-weight` and `--diff-pseudocount` flags dropped (no corresponding library feature). Other flags preserved. |

Within-repo call sites that need updates:

- [tools/train_multitask_differential_bpnet.py](tools/train_multitask_differential_bpnet.py) — rewritten per §7.2.
- [tools/run_tpcav.py](tools/run_tpcav.py) — uses `AttributionTarget`;
  its `choices=sorted(TARGET_REDUCTIONS)` for `--target-mode` now
  surfaces the new delta reductions. Might want an explicit filter
  if TPCAV doesn't support multi-channel deltas (it doesn't today).
- [tests/test_attribution.py](tests/test_attribution.py) — `AttributionTarget.channel` → `.channels` field rename.

Not affected:

- `export_tfmodisco_inputs.py`, `export_bigwig.py`, `export_predictions.py`,
  `score_variants.py` — none of these use `channels` as a tuple.

---

## 10. Open Questions

1. **Pseudocount consistency between Phase 1 and Phase 2.**
   Both should use the *same* `count_pseudocount` so `log_counts` in
   the loss are in the same space. This is already propagated via
   `propagate_pseudocount`, but the rewrite should add an explicit
   assertion at Phase 2 startup that the Phase 1 checkpoint's
   `count_pseudocount` matches what the Phase 2 loss will use.

2. **Frozen-trunk fine-tuning.** Naqvi et al. 2025's recipe keeps
   the trunk + profile heads frozen during Phase 2 — only the count
   heads update. The current tool does this via explicit
   `requires_grad = False` loops in `_Phase2Module`. The rewrite
   needs an equivalent in the standard training path (either via a
   `freeze_patterns` field on `TrainConfig`, a hook, or a
   module-level `setup` override). Scan `train.py` during
   implementation to pick the idiomatic option.

3. **Should the provenance TSV survive in any form?** See §7.4.
   Decision during implementation. Under the redesign there is
   nothing to precompute, so the TSV would be a post-hoc audit
   artifact (run the target-derivation over one epoch's worth of
   batches and dump). Nice-to-have, not blocking.

4. **Do we want Phase 2 to be callable on a Phase 1 *run*, not a
   *checkpoint*?** i.e., `train_multitask_differential_bpnet.py
   --resume-from-phase1 <path>` where `<path>` points at a full
   multi-fold training output and Phase 2 picks up each fold. The
   current tool does this implicitly for single-fold runs but is
   awkward for multi-fold. Matching the existing `--multi` flag
   logic is easy under the redesign since Phase 2 is now a regular
   `train_multi` call.

5. **Generalization to N ≥ 3 conditions.** MultitaskBPNet already
   supports it (`output_channels` list length ≥ 2). The current
   `DifferentialCountLoss` hardcodes a single pair. A generalization
   would be `list[tuple[int, int]]` — all-pairs or a user-specified
   pattern. Out of scope for this doc; flag for future work.

6. **Training-time differential attribution as a regularizer.**
   Some papers propose regularizing attribution consistency across
   conditions during training. This would need per-sample scalars
   derived from *gradients*, not from inputs, and is genuinely
   orthogonal to this redesign. Flag for future work; do not let
   it influence the Option A/B/C choice.

---

## 11. Relationship to Existing Docs

- `docs/internal/multitask-differential-bpnet.md` is joanne-feature's
  internal design doc. It was not ported to marcin-feature (excluded
  from Chunk C of the port). It captures the Naqvi / bpAI-TAC
  literature rationale, which this doc does not repeat. If the
  redesign lands, that doc should be ported (or its salient
  parts folded in) as the domain-science reference.
- `docs/internal/correctness_audit_2026_04_16.md` is the audit that
  flagged validation gaps; the §2.2 `targets`-overloading analysis
  extends from that audit.
- `docs/internal/variant_tool_design.md` is the template this doc
  follows (audit → library gaps → design options → recommendation →
  blueprint). Same shape; different domain.
- `docs/internal/attribution_methods_tangermeme_vs_captum.md` and
  `docs/internal/interpretability_performance_tradeoff.md` are the
  reference for the attribution-module design space.

---

## 12. What This Doc Does Not Propose

Explicitly out of scope (flagged to avoid scope creep during
implementation):

- Changes to `MultitaskBPNet` or `MultitaskBPNetLoss`. They work.
- A general "delta loss" base class. Premature — one consumer today.
- Transform-API extension to add per-sample scalars. Only needed
  under Option C; not recommended.
- Unification of `DifferentialCountLoss` with `MSEMultinomialLoss`
  into a generic `CoupledChannelLoss`. Possible later; out of scope.
- Removing `differential_targets.tsv` as part of the CLI behaviour.
  Decide during implementation.
- Changes to the TPCAV adapter (`src/cerberus/tpcav.py`). Orthogonal.
- **Support for externally-computed delta targets** (DESeq2, edgeR,
  etc.). Not a current requirement; §13 sketches how it could be
  re-added later, but the planned code does not depend on it.

---

## 13. Future Extensions (Not Required)

This section describes features **removed** under the current
redesign and how they could be restored in a later pass **if and
when** a real in-tree consumer appears. None of this is a
requirement. The code landed under Options A, B, or C does not need
to accommodate these extensions in advance.

### 13.1 External Delta Targets (DESeq2 / edgeR / custom)

**When would this be needed?**

- A user with biological replicates who wants DESeq2 / edgeR shrunk
  log2FC as the supervision target instead of the raw bigwig ratio.
- A user with *custom* delta scores (e.g. from a differential
  chromatin-state caller, peak-by-peak significance-weighted FC)
  that are not computable from `targets.sum(dim=-1)`.
- A benchmarking setup that wants to compare models trained on
  different delta definitions.

None of these exist in-tree today.

**Sketch: transform-based per-sample labels.**

The cleanest re-entry point is a label *source* tied to the dataset,
not a kwarg on the loss. Approximate shape:

```
# src/cerberus/differential.py  (new, minimal)
@dataclass(frozen=True)
class DeltaTargetRecord:
    chrom: str
    start: int
    end: int
    log2fc: float

def load_delta_targets(path: Path) -> dict[tuple[str, int, int], float]:
    """Parse a 4-col TSV (chrom start end log2fc) into a coord-keyed dict."""
    ...

# src/cerberus/transform.py  (extended)
class AttachDeltaTarget(DataTransform):
    """Looks up log2fc by (chrom, start, end); attaches to sample dict."""
    def __init__(self, table: dict[tuple[str, int, int], float],
                 default: float = 0.0): ...
    def __call__(self, inputs, targets, interval, *, extras: dict):
        key = (interval.chrom, interval.start, interval.end)
        extras["log2fc"] = self.table.get(key, self.default)
        return inputs, targets, interval, extras
```

The transform API extension (adding an `extras: dict` out-parameter
that flows into the sample dict) is the only core-cerberus change
required. Once that exists, a second loss class —
`ExternalDifferentialCountLoss` or similar — reads the kwarg
unconditionally (no fallback). Keeps `DifferentialCountLoss` pure.

**What can be resurrected from git history.**

Joanne-feature's `src/cerberus/differential.py` contained:

- `DifferentialRecord` dataclass (lines ~250-293).
- `load_differential_targets` TSV parser (lines ~301-376).
- `DifferentialTargetIndex` coord-keyed lookup (lines ~405-482).

These can be re-ported at feature-ship time rather than carried now.

### 13.2 Absolute-Count Regularizer (`abs_weight`)

**When would this be needed?**

- An in-tree workflow that observes count-head drift during Phase 2
  fine-tuning and wants the heads anchored to their absolute tracks.
  Naqvi et al. 2025 discuss this as an optional regularizer; their
  default is `abs_weight = 0.0`.

None of this is exercised today.

**Sketch: restoration path.**

The restored feature would be additive to `DifferentialCountLoss`,
but behind a cleaner interface than the current "abs_weight > 0
triggers a whole new computation path":

- **Option α**: a separate `AnchoredDifferentialCountLoss` subclass
  that adds the abs term. Composition > overloading; keeps
  `DifferentialCountLoss` a single-path function.
- **Option β**: a generic "composite loss" helper that weights
  arbitrary sub-losses, so Phase 2 becomes `compose(delta_loss,
  abs_loss_a, abs_loss_b, weights=...)`. Lines up with existing
  `MSEMultinomialLoss` + `count_weight` + `profile_weight`
  pattern.

Decision is deferred to whoever actually needs it. What the current
redesign removes is the *conjoined* "regularizer hidden inside
`DifferentialCountLoss`" interface.

### 13.3 Why not keep these features "just in case"

The argument for keeping `log2fc` kwarg / `abs_weight` ahead of real
demand:

> "Removing them means future re-addition will be a breaking change."

The counter-argument:

- **Adding a kwarg back to an `nn.Module.forward` is not
  breaking.** Existing callers pass `forward(outputs, targets)`;
  a future version that accepts an optional `log2fc=` kwarg accepts
  the same call shape.
- **Adding a new subclass is not breaking.** `AnchoredDifferentialCountLoss`
  can ship next quarter without touching anyone using
  `DifferentialCountLoss` today.
- **Keeping the features now preserves the four-row truth table**
  (§2.2) that made `DifferentialCountLoss` hackish in the first
  place.
- **Neither feature has ever had a non-internal caller.** Removing
  unused flexibility is cheap; keeping it costs every future reader
  who has to understand the overloading.

Net: drop now, re-add later only when a real consumer appears.

### 13.4 Architectural invariant worth preserving

Whatever the future-work path, the core invariant should be:

> **Each loss class in `src/cerberus/loss.py` has exactly one way to
> produce its target per call.** The way is determined at
> construction (by which class you picked), not at call time (via
> kwarg presence or shape overloading).

Concretely: `DifferentialCountLoss` *always* derives its delta from
`targets.sum(dim=-1)`. `ExternalDifferentialCountLoss` (hypothetical)
*always* reads an external per-sample kwarg.
`AnchoredDifferentialCountLoss` (hypothetical) *always* adds the
abs regularizer. The per-call "which protocol are we on today"
ambiguity stays gone.

---

## 14. Summary Diff (landed)

Measured against `b461a6e..e86ddaa` (2026-04-18):

**Library (`src/cerberus/`)** — net ~−90 lines, everything simpler:

- `loss.py`: `DifferentialCountLoss` collapses from ~170 lines to
  ~120 (validation + docstring included). Removes
  `_resolve_delta_target`, `log2fc` kwarg handling, `(B, 1, 1)`
  scalar fallback, `abs_weight` regularizer branch. Adds `(B, N, L)`
  shape validation. `loss_components` is kept (required by
  `CerberusModule._shared_step`) and returns a single
  `{"delta_loss": …}` entry. Net **−60**.
- `attribution.py`: `DifferentialAttributionTarget` class removed
  entirely. `AttributionTarget` gains delta reductions, a
  `channels: int | tuple[int, int]` field, and arity validation in
  `__init__`. `DIFFERENTIAL_TARGET_REDUCTIONS` removed; dispatch
  stayed as an `if/elif` chain. Net **−24**.
- `__init__.py`: `DifferentialAttributionTarget` and
  `DIFFERENTIAL_TARGET_REDUCTIONS` removed from exports. Net **−4**.
- Net library delta: **~−90 lines, zero optional protocols, zero
  overloading, zero `abs_weight`-style dormant features.**

**Tool (`tools/train_multitask_differential_bpnet.py`)** — net
**−227** lines:

- 1134 → 907. The tool still contains the BED merge, two-phase
  coordination, checkpoint glob, and DeepLIFTSHAP pipe — exactly
  the workflow-specific code that has no business in the library.
  No `_DiffWrapper`, no `_Phase2Module`, no offline log2FC
  precompute, no TSV writer, no `--abs-weight` / `--diff-pseudocount`
  flags. Phase 2 is a regular `train_single` / `train_multi` call
  with `pretrained=[PretrainedConfig(...)]`. The
  `run_interpretation` block is essentially unchanged (~175 lines)
  and was a larger fraction of the surviving tool than the design
  sketch anticipated.

**Tests** — net **−63** lines across differential + attribution + tool test files:

- +8 new tests on the single-path loss (delta derivation,
  pseudocount effect, shape / channel-count validation,
  components-dict shape).
- +~8 new tests on `AttributionTarget` delta reductions and
  `channels` arity validation.
- Removed: log2fc-kwarg tests, abs-weight tests, scalar-fallback
  test, the `_DiffWrapper` test file (53 lines).
- Existing attribution tests updated to use the unified
  `AttributionTarget` (mechanical; `channel=` → `channels=`).

**Docs** — net +~190 lines:

- `CHANGELOG.md`: consolidated `[Unreleased]` block with
  `### Added` / `### Changed` / `### Fixed` / `### Removed` /
  `### Internal` sections.
- `docs/models.md`: new "MultitaskBPNet" / "DifferentialCountLoss"
  sections (~100 lines).
- `docs/usage.md`: new "Two-phase multitask-differential training"
  and "Scalar attribution targets" sections (~90 lines).
- `docs/internal/taylor_ism_integration.md`: one-word fix for the
  `channel=` → `channels=` rename.

**Net across the tree (code + tests, excluding docs and this
design doc): ≈ −380 lines.** The design's original
≈ −940 estimate was optimistic — the gap is mostly (a)
`loss_components` staying, (b) `run_interpretation` not being a
library-level helper, (c) no library changes to avoid the
`run_phase2` scheduler/optimizer boilerplate.

**Intentional breaking changes** (all have zero known external
callers, all documented with §13 re-entry plans):

1. `DifferentialCountLoss(log2fc=...)` kwarg removed.
2. `DifferentialCountLoss(abs_weight=...)` argument removed.
3. `DifferentialCountLoss` scalar-fallback protocol removed.
4. `DifferentialAttributionTarget` class removed.
5. `DIFFERENTIAL_TARGET_REDUCTIONS` export removed.
6. `AttributionTarget.channel` → `AttributionTarget.channels`.
7. `DifferentialCountLoss.loss_components` returns a single
   `"delta_loss"` key (the `abs_loss_a` / `abs_loss_b` conditional
   entries are gone).
8. Tool CLI flags `--abs-weight` and `--diff-pseudocount` dropped.

Commits: `74e3999` (attribution unification + pyrightconfig) and
`e86ddaa` (loss simplification + Phase 2 tool rewrite). No
deprecation shim on either side.
