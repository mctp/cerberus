# Differential-Learning Workflow Redesign

**Date:** 2026-04-18
**Status:** Design proposal (no implementation)
**Predecessors:**
- `docs/internal/multitask-differential-bpnet.md` (joanne-feature internal design, not ported)
- `docs/internal/variant_tool_design.md` (same proposal shape; adapted here)
- `docs/internal/correctness_audit_2026_04_16.md` (format of the audit sections)

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

The proposal here is to do three things:

1. **Compute the per-peak log2FC inside the loss**, from the `(B, N, L)`
   `targets` tensor the count head is already supervised against. This
   makes `_compute_log2fc_from_bigwigs`, `_DiffWrapper`, and the
   provenance TSV unnecessary. `SignalExtractor` is already the single
   source of truth for bigwig reading; this plan keeps it that way.

2. **Unify `AttributionTarget` and `DifferentialAttributionTarget`** into
   a single class dispatched by its `reduction` string. The
   differential reductions (`delta_log_counts`,
   `delta_profile_window_sum`) become peers of the existing five. The
   field set widens slightly (`channels: list[int]` replaces
   `channel: int`), but the class count drops from two to one and
   `_resolve_window` is no longer duplicated.

3. **Simplify `DifferentialCountLoss` to a single protocol.** `targets`
   stays `(B, N, L)` absolute-signal always, matching what every other
   cerberus loss sees. The three fallback branches in
   `_resolve_delta_target` collapse to one path: derive the delta
   target from `targets.sum(dim=-1)` with a pseudocount. The
   `log2fc` kwarg and the `(B, 1, 1)` scalar fallback are both
   removed; `DifferentialCountLoss.forward` becomes a tight,
   self-contained function with no caller-facing protocol choice.

External-labels support (DESeq2/edgeR shrunk log2FC, etc.) is **not a
requirement** for this redesign. No current caller or test in cerberus
exercises that path. The bigwig-derived log2FC that `SignalExtractor`
yields is the only delta target the planned code needs to produce.
§13 sketches how external-label support could be added in a later
pass if demand appears; the recommended design is **not dependent on
it**.

After the rewrite the training tool loses `_DiffWrapper`, `_Phase2Module`
(both become unnecessary), and `_compute_log2fc_from_bigwigs` +
`_DifferentialRecord` + `_write_differential_targets` (the offline
precompute pipeline). Phase 2 can then be expressed in the same
`train_single` / `train_multi` call shape as Phase 1 — a second
ModelConfig with a different `loss_cls`. Phase 1 and Phase 2 stop being
"two training paradigms glued together in a bespoke tool" and become
"the same training CLI called twice with different config".

Overall line count: the current tool is 1068 lines. A version built on
the design proposed here is expected to be ~350 lines, with the saved
~700 lines distributed among (a) reusing `train_single` / `train_multi`,
(b) dropping the inlined helpers, and (c) dropping the wrapper /
precompute plumbing. Conversely, `src/cerberus/loss.py` **net shrinks**
(the inline log2FC derivation is smaller than the kwarg + fallback
machinery it replaces) and `src/cerberus/attribution.py` shrinks by
~70 lines (single `AttributionTarget` instead of two).

Three design options are presented (§5), with option **B** recommended.
§7 is a concrete blueprint for option B at the signature / config /
migration level. §8 spells out the attribution unification
independently — it is a win regardless of which of A/B/C is chosen.
§13 describes the optional future-work path for external delta
labels, with the explicit commitment that the planned code does not
depend on any of it.

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

## 5. Design Options

### 5.1 Option A — Minimal: inline the log2FC derivation; drop the kwarg path

**Scope.**

- `DifferentialCountLoss` gains a single-path target derivation:
  `targets` is always `(B, N, L)`, delta is computed inline from
  `targets.sum(dim=-1)` with `count_pseudocount`.
- Remove the `log2fc` kwarg and the `(B, 1, 1)` / `(B, 1, L)` scalar
  fallback. `_resolve_delta_target` goes away as a distinct method.
- Keep the `abs_weight > 0` abs-term branch as-is (already operates
  on `(B, N, L)` targets — no overloading).
- Rewrite `tools/train_multitask_differential_bpnet.py` to drop
  `_DiffWrapper`, `_Phase2Module`, `_compute_log2fc_from_bigwigs`,
  `_DifferentialRecord`, `_write_differential_targets`. Phase 2
  becomes a second `train_single` call with a different `ModelConfig`.
- `AttributionTarget` / `DifferentialAttributionTarget` unchanged.

**Pros.**

- Cleanest fix for the single biggest issue (the offline precompute +
  wrapper).
- `DifferentialCountLoss.forward(outputs, targets)` is the entire
  surface — no optional kwargs, no shape branches.
- Tool shrinks dramatically (est. 1068 → 350 lines).
- No core cerberus changes beyond the loss.

**Cons.**

- Removes the `log2fc` kwarg as a public API surface (currently
  unused outside the tool itself; see §3.2).
- Attribution module keeps both classes; the near-duplication remains.

**Appropriate when:** you want the tool fixed this week and are OK
with the attribution-side duplication as a known debt.

### 5.2 Option B — Recommended: inline derivation + unified `AttributionTarget`

**Scope.** Option A plus:

- Collapse `DifferentialAttributionTarget` into `AttributionTarget`.
  Single class, single reductions set, `channels: list[int]` field
  (length 1 or 2 depending on reduction).
- `DIFFERENTIAL_TARGET_REDUCTIONS` retained as a grouping helper but
  becomes a subset of `TARGET_REDUCTIONS`.
- Public API: `cerberus.DifferentialAttributionTarget` becomes an
  alias / thin constructor wrapper for `AttributionTarget` with
  a convenience signature (`cond_a_idx`, `cond_b_idx` kwargs that
  translate to `channels=[a, b]`). Source of truth is one class.

**Pros.**

- Addresses both the tool-level mess and the attribution-module
  duplication.
- The `AttributionTarget` API becomes closed-form for the set of
  "what scalar do I attribute": every supported question is one
  `reduction` string with a small number of typed fields.
- Zero breaking changes at the import site (`DifferentialAttributionTarget`
  remains importable).

**Cons.**

- `AttributionTarget.channel` becomes `channels: list[int]` — this
  is a mechanical change in existing callers (tools, tests). The
  `DifferentialAttributionTarget` shim can hide it from the
  multi-channel callers.
- Slightly more code in one place (one class with 7 reductions
  instead of two classes with 5+2).

**Appropriate when:** you want to pay down the attribution duplication
while you're already reshaping the differential workflow. This is the
recommended option.

### 5.3 Option C — Radical: drop `DifferentialCountLoss` entirely; make it a composition

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

## 6. Recommendation

**Ship Option B. Leave the door open for Option C.**

Rationale:

1. **Option A fixes the worst problem** (offline precompute, wrapper,
   1068-line tool). Option B additionally fixes the attribution-module
   duplication, which was the single most callable-out structural
   debt from the joanne-feature port. Both are small, localized
   changes — no cerberus-wide API changes.

2. **Option C's value depends on a use case that doesn't exist yet.**
   The transform-API extension is genuinely useful *if* there is a
   second derived-quantity consumer. Right now there is one
   (differential) and it fits inside the loss cleanly. If a second
   one (e.g. a training-time attribution regularizer that needs the
   gradient of `log_counts` as a per-sample scalar target) shows up,
   revisit the transform API at that point.

3. **Option B keeps the `DifferentialCountLoss` import path and the
   `log2fc` kwarg** — the DESeq2-style external-labels workflow (not
   currently exercised in-tree but part of the public API) keeps
   working without code changes.

4. **The attribution unification in Option B is a win on its own.**
   Even if the differential-loss redesign stalls, pulling the two
   attribution targets into one class is worth doing — the two
   classes diverge along a trivially-parameterizable axis
   (1 channel vs. 2).

### 6.1 What not to do even under Option B

- **Do not remove the `log2fc` kwarg.** It is the only supported
  escape hatch for DESeq2/edgeR-style shrunk estimators, which
  cerberus does not compute.
- **Do not remove `abs_weight`.** Per Naqvi et al. 2025 the default
  is 0.0 and the abs term is rarely used, but when the count heads
  drift away from absolute tracks during fine-tuning this is the
  only available anchor.
- **Do not merge `MultitaskBPNet` back into `BPNet`.** The `≥ 2`
  output_channels invariant and `predict_total_count=False`
  enforcement are real guardrails, not cosmetic.

---

## 7. Blueprint for Option B

### 7.1 Library changes

**`src/cerberus/loss.py`** — `DifferentialCountLoss` target resolution:

```
_resolve_delta_target(targets, kwargs):
    log2fc = kwargs.get("log2fc")
    if log2fc is not None:
        [existing kwarg path, unchanged]

    # New primary path: derive from per-channel targets.
    if targets.ndim == 3 and targets.shape[1] > max(cond_a_idx, cond_b_idx):
        counts = targets.float().sum(dim=-1)   # (B, N)
        pc = self.count_pseudocount
        return torch.log2((counts[:, cond_b_idx] + pc) /
                          (counts[:, cond_a_idx] + pc))

    # Legacy scalar-delta fallback: unchanged, kept for back-compat.
    return targets.float().reshape(targets.shape[0], -1).mean(dim=-1)
```

Pseudocount semantics: use the loss's own `count_pseudocount`. In the
bpAI-TAC / Naqvi recipe this is the *same* pseudocount Phase 1 used
(propagated via `data_config.count_pseudocount` →
`propagate_pseudocount` → `MSEMultinomialLoss.count_pseudocount`), so
Phase 1 and Phase 2 operate in the same log-space.

**`src/cerberus/attribution.py`** — unify the two attribution targets:

```
TARGET_REDUCTIONS = {
    # single-channel
    "log_counts",
    "profile_bin",
    "profile_window_sum",
    "pred_count_bin",
    "pred_count_window_sum",
    # multi-channel (delta)
    "delta_log_counts",
    "delta_profile_window_sum",
}

class AttributionTarget(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        reduction: str,
        channels: int | tuple[int, int],   # 1 for single, 2 for delta
        bin_index: int | None = None,
        window_start: int | None = None,
        window_end: int | None = None,
    ):
        ...

# Back-compat shim, no behaviour change for callers:
def DifferentialAttributionTarget(
    model, *, reduction, cond_a_idx, cond_b_idx, window_start=None, window_end=None
):
    return AttributionTarget(
        model=model,
        reduction=reduction,
        channels=(cond_a_idx, cond_b_idx),
        window_start=window_start,
        window_end=window_end,
    )
```

`DIFFERENTIAL_TARGET_REDUCTIONS` stays exported as a subset of
`TARGET_REDUCTIONS` for discoverability (lets `run_tpcav.py`'s
`--target-mode` flag keep its `choices=sorted(TARGET_REDUCTIONS)`
grouping intact).

### 7.2 Tool rewrite

**`tools/train_multitask_differential_bpnet.py` new structure:**

```
get_args()
run_phase1(args)      # unchanged — already uses train_single/train_multi
                      # with MultitaskBPNet + MultitaskBPNetLoss
run_phase2(args, phase1_model_path):
    # Same DataConfig.targets as Phase 1 (bw_a, bw_b)
    # Different ModelConfig: loss_cls=DifferentialCountLoss
    # Same CerberusDataModule, same train_single/train_multi
    # Pretrained weights loaded via ModelConfig.pretrained = [phase1_model_path]

run_interpretation(args)   # Uses unified AttributionTarget with
                           # reduction="delta_log_counts", channels=(0, 1)
main()
```

The tool collapses to:

| Section | Current lines | Projected lines |
|---|---|---|
| Argument parser + main | ~250 | ~230 |
| `_merge_and_write_peaks` + related BED helpers | ~50 | ~50 (kept — useful, not cerberus-generic) |
| `run_phase1` | ~140 | ~130 |
| `run_phase2` | ~200 | ~60 (was ~200: drop wrapper+module, reuse train_single) |
| `run_interpretation` | ~200 | ~200 |
| `_dinuc_shuffle` | ~20 | ~20 |
| Inlined differential helpers | ~50 | 0 (dropped) |
| `_DiffWrapper` / `_Phase2Module` | ~100 | 0 (dropped) |
| **Total** | **~1068** | **~350 + ~40 new = ~390** |

Phase 2 specifically becomes (schematically):

```
run_phase2(args, phase1_ckpt_dir):
    data_config = data_config_p2(args)              # same bigwigs as Phase 1
    model_config = ModelConfig(
        name="MultitaskBPNet_Phase2",
        model_cls="cerberus.models.bpnet.MultitaskBPNet",
        loss_cls="cerberus.loss.DifferentialCountLoss",
        loss_args={"cond_a_idx": 0, "cond_b_idx": 1,
                   "abs_weight": args.abs_weight},
        pretrained=[PretrainedConfig(path=phase1_ckpt_dir, strict=True)],
        model_args={...},
        count_pseudocount=args.count_pseudocount * args.target_scale,
    )
    train_single(... model_config ... train_config ... )
```

That's it. No `_DiffWrapper`, no `_Phase2Module`, no offline log2FC
precompute. The DataLoader emits `{inputs, targets, intervals,
interval_source}`; `DifferentialCountLoss` derives the delta target
from `targets` in the training step.

### 7.3 Pretrained-weight loading for Phase 2

Phase 2 needs to initialize from Phase 1's weights. Cerberus already
supports this via `ModelConfig.pretrained: list[PretrainedConfig]`;
the Phase 2 `ModelConfig` just lists the Phase 1 checkpoint in that
field. The current tool implements this manually in
[`_find_phase1_model`](tools/train_multitask_differential_bpnet.py#L327-L334);
the rewrite should verify that `ModelConfig.pretrained` + a matching
architecture (same `MultitaskBPNet`) reproduces the existing
behaviour. If there are subtle load-strict issues (head shape
retained, trunk frozen at specific layers), those can be expressed
via `PretrainedConfig` fields.

**Optional fine-tuning shapes:** the current tool supports
trunk-freeze variants via explicit `requires_grad = False` loops on
the PL module; under the redesign, freezing is typically expressed
via `TrainConfig.freeze_patterns` or an equivalent if present —
needs a small scan of `train.py` during implementation.

### 7.4 Back-compat and migration

- **`DifferentialAttributionTarget`**: stays importable
  (alias/shim under Option B). Zero breakage.
- **`DifferentialCountLoss(log2fc=...)` kwarg**: unchanged
  semantics — still overrides the derived target.
- **`DifferentialCountLoss(abs_weight=...)` branch**: unchanged.
- **CLI flags** on the tool (`--bigwig-a`, `--peaks-a`, etc.):
  unchanged.
- **`differential_targets.tsv` provenance output**: dropped under
  the redesign, since the log2FC is no longer precomputed. If the
  provenance is valued (e.g. for "what log2FC did my Phase 2 actually
  see on fold_3?"), it can be re-expressed as a post-hoc call:
  "load the val dataset, run it through the loss's target-derivation
  function once, write the TSV." Cheap to add; not on the critical
  path.

### 7.5 Tests

New tests required under Option B:

- `test_differential_count_loss_derives_from_targets` — parametrized
  over `(cond_a_idx, cond_b_idx)` combos. Constructs a known
  `(B, N, L)` target tensor whose per-channel sums are known; asserts
  `_resolve_delta_target(targets, {})` returns the expected
  `log2((sum_b + pc) / (sum_a + pc))`.
- `test_differential_count_loss_log2fc_kwarg_still_overrides` —
  regression test that the kwarg path wins when both inputs are
  present.
- `test_differential_count_loss_scalar_fallback_unchanged` —
  `(B, 1, 1)` target still returns squeezed mean.
- `test_attribution_target_multi_channel_reduction` —
  `AttributionTarget(reduction="delta_log_counts", channels=(0, 1))`
  on a toy model returns the expected delta.
- `test_differential_attribution_target_shim` — assert the shim
  constructs an equivalent `AttributionTarget`.

Existing tests
([test_multitask_differential_bpnet.py](tests/test_multitask_differential_bpnet.py),
[test_attribution.py](tests/test_attribution.py)) should continue to
pass unchanged under the shim.

### 7.6 Documentation

- `docs/usage.md` §Differential-Learning gains a section that
  cross-references the TPCAV / TF-MoDISco sections and makes the
  "Phase 1 and Phase 2 are the same CLI with different configs"
  story explicit.
- `CHANGELOG.md` entry under `### Changed` describing the `targets`
  contract change (now always `(B, N, L)`, scalar fallback reserved
  for explicit overrides via `log2fc` kwarg).

---

## 8. Attribution Module Unification (independent of §5–7)

The unification of `AttributionTarget` + `DifferentialAttributionTarget`
is a net simplification regardless of what happens on the loss side.
This section describes it independently so it can be pursued on its
own schedule.

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

### 8.2 Reduction registry

Internally, a dict mapping reduction-name → callable cleans up the
`if/elif` dispatch in `forward`:

```
_REDUCTIONS: dict[str, Callable[[AttributionTarget, Output], Tensor]] = {
    "log_counts":              lambda self, out: out.log_counts[:, self.channels],
    "profile_bin":             lambda self, out: out.logits[:, self.channels, self._bin(out.logits.shape[-1])],
    ...
    "delta_log_counts":        lambda self, out: out.log_counts[:, self.channels[1]] - out.log_counts[:, self.channels[0]],
    ...
}
```

The dispatch table is a one-screen diff — makes adding new reductions
tractable without adding `if self.reduction == "..."` branches.

### 8.3 The `channels` field change

Breaking change in the sense of "attribute type widens". Callers
that do `target.channel` break; callers that do
`target.channels[0]` work uniformly. Since most call sites are
internal (the three tools + tests), the blast radius is small.

The back-compat shim under `DifferentialAttributionTarget(
model, reduction, cond_a_idx, cond_b_idx, ...)` keeps every current
external call site working.

### 8.4 Gains vs. cost

Gains:
- One class instead of two. ~70 lines deleted.
- Single `_resolve_window`, `_check_channel(s)` helper.
- Registry-based dispatch; easier to extend.
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
- Lose the clean registry pattern.

The reduction-string-dispatched single class is the sweet spot.

---

## 9. Migration & Compatibility

Under Option B:

| Path | Breaking? | Migration |
|---|---|---|
| `from cerberus.attribution import DifferentialAttributionTarget` | No (shim) | None |
| `from cerberus.attribution import AttributionTarget` with `.channel` | Yes | `.channels` (union type); bare int still accepted via positional arg |
| `DifferentialCountLoss(log2fc=tensor)` kwarg | No | None |
| `DifferentialCountLoss(targets=(B, 1, 1))` scalar fallback | No | None (legacy path preserved) |
| `DifferentialCountLoss(targets=(B, 2, L))` with no kwarg | **Semantic change**: previously garbage; now returns correct derived delta | Intentional fix — no migration for well-behaved callers |
| `tools/train_multitask_differential_bpnet.py --flag-x` (CLI surface) | No | Same flags, internal refactor only |

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

3. **Should the `log2fc` kwarg stay in the library public API, or
   become a hidden / internal mechanism?** The only external caller
   in the joanne-feature design was the training tool itself, which
   is internal. If DESeq2-style external labels are a real use case,
   document them; otherwise consider privatizing the kwarg in a
   later pass.

4. **Should the provenance TSV survive in any form?** See §7.4.
   Decision during implementation.

5. **Do we want Phase 2 to be callable on a Phase 1 *run*, not a
   *checkpoint*?** i.e., `train_multitask_differential_bpnet.py
   --resume-from-phase1 <path>` where `<path>` points at a full
   multi-fold training output and Phase 2 picks up each fold. The
   current tool does this implicitly for single-fold runs but is
   awkward for multi-fold. Matching the existing `--multi` flag
   logic is easy under the redesign since Phase 2 is now a regular
   `train_multi` call.

6. **Generalization to N ≥ 3 conditions.** MultitaskBPNet already
   supports it (`output_channels` list length ≥ 2). The current
   `DifferentialCountLoss` hardcodes a single pair. A generalization
   would be `list[tuple[int, int]]` — all-pairs or a user-specified
   pattern. Out of scope for this doc; flag for future work.

7. **Training-time differential attribution as a regularizer.**
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

---

## 13. Summary Diff (projected, under Option B)

**Added:**

- `src/cerberus/loss.py`: ~20 lines of inline log2FC derivation in
  `_resolve_delta_target`.
- `src/cerberus/attribution.py`: ~15 lines for the reduction-dispatch
  registry.
- `tests/`: ~60 lines of new tests per §7.5.
- `docs/usage.md`: ~30 lines of unified differential-workflow docs.

**Changed:**

- `tools/train_multitask_differential_bpnet.py`: 1068 → ~390 lines
  (net −678).
- `src/cerberus/attribution.py`: two classes → one class + shim
  (net −70).
- `src/cerberus/loss.py`: `_resolve_delta_target` fallback rewired
  (net +10).

**Removed:**

- Inlined helpers in the tool: `_compute_log2fc_from_bigwigs`,
  `_DifferentialRecord`, `_write_differential_targets`,
  `_DiffWrapper`, `_Phase2Module` (net −180).

**Net ≈ −900 lines across the tree, zero public-API breakage**
(via the `DifferentialAttributionTarget` shim and the preservation
of the `log2fc` kwarg).
