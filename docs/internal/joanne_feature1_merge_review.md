# `joanne-feature1` Merge Review

Reviewer-facing companion to the merge of `joanne-feature1` into `marcin-feature`
(and onward to `main`). This document is the reference for the iterative
clean-up commits that follow the merge. It walks every merged commit, reviews
the semantics / structure / docs, and inventories the concerns to be fixed —
each tagged to one of the agreed focus areas:

- **(C)** consistency & correctness
- **(S)** semantics & logic
- **(A)** abstraction cleanliness — no overloaded metaphors, no leaky abstractions
- **(B)** remove all backwards-compatibility shims
- **(Y)** style consistency across the codebase

---

## 1. Branch topology

`marcin-feature` HEAD before the merge (`47242e3`, release `1.0.0a6`) **was the
merge-base** with `joanne-feature1`. The merge was therefore a **clean
fast-forward** — zero conflicts, no merge commit. `marcin-feature` now points at
`db21683`.

Five feature commits were brought in (the two `Merge branch 'marcin-feature'
into main` commits are release-merge bookkeeping already present in history):

| Commit    | Title                                                  |
|-----------|--------------------------------------------------------|
| `b2baa14` | ChromBPNet differential count-head-only mode           |
| `b491f6c` | Default single-task ChromBPNet precision to full       |
| `4df350f` | Parallel differential ChromBPNet training              |
| `cf4d40a` | Apply delta count pseudocount to differential predictions |
| `db21683` | Fixed-background negatives for ChromBPNet training     |

### Baseline health after merge

- **Tests:** `2189 passed, 26 skipped` — green.
- **Pyright:** `17 errors`. **16 introduced by `joanne-feature1`**, 1 pre-existing
  (`tests/test_run_tfmodisco.py:100`, unrelated `descriptive_report` import).
  This violates the CLAUDE.md "0 pyright errors" gate and is the single largest
  correctness item. Breakdown:
  - `src/cerberus/models/bpnet.py` — 9
  - `tests/test_multitask_differential_bpnet.py` — 5
  - `tests/test_export_tfmodisco_inputs.py` — 2

---

## 2. Theme 1 — Pseudocount-shrunk differential predictions (`cf4d40a`)

**The most semantically significant change in the branch.** It alters the
training signal of every differential model.

### 2.1 What changed

Files: `src/cerberus/loss.py`, `src/cerberus/metrics.py`.

A new helper computes a numerically stable `log(exp(x) + pc)`:

```python
def _log_count_plus_pseudocount(log_count_values, delta_count_pseudocount):
    log_count_values = log_count_values.float()
    if delta_count_pseudocount == 0:
        return log_count_values
    log_pc = log_count_values.new_tensor(float(delta_count_pseudocount)).log()
    return torch.logaddexp(log_count_values, log_pc)
```

The predicted delta in `DifferentialCountLoss._delta_loss` changed from:

```python
delta_pred = log_counts[:, b] - log_counts[:, a]                      # before
```
to:
```python
delta_pred = (_log_count_plus_pseudocount(log_counts[:, b], pc)
              - _log_count_plus_pseudocount(log_counts[:, a], pc))    # after
```

i.e. the prediction is now `log((exp(log_b) + pc) / (exp(log_a) + pc))`, which
**matches the form of the target** `log((sum_b + pc) / (sum_a + pc))`. The same
substitution is applied identically in
`metrics._extract_differential_log_count_pairs`.

### 2.2 Why this is a deliberate reversal — flag for sign-off (S)

The *prior* metrics code carried an explicit comment arguing the **opposite**:

> The prediction is read directly from `log_counts` regardless of whether the
> model's log-space includes the pseudocount or not — the loss optimises that
> same subtraction, so the metric must report on it. Inserting a pc-aware
> correction here would measure something the loss is not optimising and produce
> a biased training signal.

`cf4d40a` deletes that rationale and inverts the behavior on **both** sides
(loss *and* metric move together, so they stay mutually consistent — that part
is correct). The justification is sound: when `pc` is the empirical-Bayes
shrinkage prior, shrinking only the target while leaving the prediction
unshrunk creates a systematic bias for low-count regions — the model is
penalised for not matching a shrunk target with an unshrunk prediction. Applying
`pc` to both restores symmetry and gives the model a target it can actually hit
at the noise floor.

**This is a defensible and likely correct change**, but it changes the loss
surface for any already-trained or in-flight differential model. It deserves an
explicit decision and a one-line note in the changelog/docs that the
differential objective semantics changed in this release (not just "added
pseudocount").

### 2.3 Concerns

- **(A) Duplicated helper.** `_log_count_plus_pseudocount` is copy-pasted
  verbatim into both `loss.py:15` and `metrics.py:53`. The loss and the metric
  are *required* to compute the identical transform (that is the whole point of
  the change), so the duplication is a latent drift hazard: a future edit to one
  copy silently desynchronises the metric from the loss. It should live in one
  place. `cerberus/pseudocount.py` already owns pseudocount logic and is
  imported by both call sites' neighborhoods — the natural home.
- **(B) `count_pseudocount` alias.** Both `DifferentialCountLoss` and the two
  differential metrics now take `delta_count_pseudocount: float | None = None`
  **and** keep `count_pseudocount: float | None = None` as a
  "backwards-compatible alias", with a resolve-one-from-the-other dance in every
  `__init__`. Per focus area (B) this alias should be removed and call sites
  updated to pass `delta_count_pseudocount` only. Note `self.count_pseudocount =
  delta_count_pseudocount` is also set purely for compatibility — dead once the
  alias is gone.
- **(S) `pc == 0` short-circuit.** The helper returns the raw log-values when
  `pc == 0`, i.e. `delta_pred = log_b - log_a`, recovering the old behavior
  exactly. That is correct and worth keeping as documented behavior, but the
  default `pc` resolves to `1.0` (not `0.0`), so the *default* path is the new
  shrunk one — consistent with the intent.

---

## 3. Theme 2 — Fixed-background negatives (`db21683`)

Files: `src/cerberus/samplers.py`, `tools/train_chrombpnet.py`,
`tools/export_predictions.py`, `docs/samplers.md`,
`tests/test_fixed_background_sampler.py`.

### 3.1 What changed

- **`FixedBackgroundSampler(IntervalSampler)`** — a static negative set loaded
  once from a BED file and never regenerated on `resample()` (inherited no-op).
  It exists as a distinct class purely so `get_interval_source` reports
  `"FixedBackgroundSampler"` instead of the generic `"IntervalSampler"`.
- **`PeakFixedBackgroundSampler(MultiSampler)`** — mixes peaks (positives) with
  the fixed negative set, mirroring `PeakSampler`'s sub-sampler ordering
  (`samplers[0]` = peaks, `samplers[1]` = negatives).
- **`create_sampler` gains `"peak_fixed_background"`.**
- **`train_chrombpnet.py --negatives <bed>`** routes to the fixed sampler and
  estimates the bias log-count offset on the same negatives.
- **`train_chrombpnet.py --reload-dataloaders-every-n-epochs`** exposes
  Lightning's reload cadence (default `0` preserves prior frozen-negatives
  behavior).
- **`export_predictions.py --include-background`** now accepts the fixed sampler
  for evaluation on the identical negatives, and fixes a latent peak-counting
  bug (see 3.3).

Motivation is legitimate: it reproduces reference `chrombpnet-pytorch`'s
static-`negatives.bed` data setup so the two implementations can be compared on
an identical peak **and** negative set.

### 3.2 Review of the `split_folds` override (S/A)

`FixedBackgroundSampler` overrides `split_folds`, `_make_split`, and `_subset`
because the base `ListSampler.split_folds` returns plain `ListSampler` objects,
which would collapse the interval-source label to `"ListSampler"` and destroy
peak-vs-background separation in evaluation. The override re-wraps each fold
split as a `FixedBackgroundSampler` via `__new__` + `ListSampler.__init__` to
bypass the file-reading `__init__`.

This works and the docstring explains why, but it is a **leaky abstraction (A)**:
the class's identity is being used as a side-channel label, and preserving that
label across `split_folds` requires bypassing the constructor with `__new__`.
The `_make_split` / `_subset` / `split_folds` trio is non-trivial machinery in
service of a string tag. Worth discussing whether interval-source provenance
should be a first-class attribute on the sampler/interval rather than encoded in
the Python class name — that would remove the need for the override entirely.
(Not necessarily a merge blocker; a design-cleanliness candidate.)

### 3.3 The `export_predictions.py` peak-count fix (C) — good catch

```python
# Peaks are IntervalSampler before fold splitting, but split_folds()
# materializes them as ListSampler subsets.
peak_sources = {"IntervalSampler", "ListSampler"}
n_peaks = sum(1 for s in all_interval_source if s in peak_sources)
```

This is a genuine correctness fix: before, `n_peaks` counted only
`"IntervalSampler"`, but after `split_folds` peaks become `"ListSampler"`, so
the peak/background split reported in evaluation was wrong for any split sampler.
Note the **coupling** this creates with 3.2: peaks rely on collapsing to
`"ListSampler"` while negatives rely on *not* collapsing — the two behaviors are
load-bearing in opposite directions. This is exactly the fragility the
provenance-as-data refactor in 3.2 would dissolve.

### 3.4 Concerns

- **(C) Pyright:** no new errors in `samplers.py` itself — clean.
- **(A)** the class-name-as-label coupling (3.2/3.3).
- **(Y)** docstrings are thorough here (arguably verbose relative to the rest of
  the module) — fine, but note for tone consistency.

---

## 4. Theme 3 — Count-head-only fine-tuning (`b2baa14`)

File: `tools/train_chrombpnet_multitask_differential.py`.

Adds `--accessibility-count-head-only` and an `AccessibilityCountHeadOnly`
Lightning callback that freezes the entire `accessibility_model` branch except
`accessibility_model.count_dense`, re-applying on both `setup` and
`on_fit_start`.

### Review

- **(S)** The freeze is applied imperatively in a callback (`requires_grad_`),
  *parallel to* the declarative `FreezeSpec` mechanism (`ModelConfig.freeze`)
  that the rest of the codebase uses — and that this same tool already uses to
  freeze `bias_model`. This is an **overloaded approach to the same concept (A)**:
  two freezing mechanisms now coexist. Discussion point: can count-head-only be
  expressed as `FreezeSpec` patterns (freeze `accessibility_model`, then a
  narrower un-freeze, or freeze everything-except via pattern) so there is one
  freeze story? If `FreezeSpec` can't express "freeze all except X", that's a
  gap worth filling rather than routing around.
- **(C)** `_unwrap_compiled` and the `setup`+`on_fit_start` double-application
  are defensive and reasonable. The `_logged` guard avoids duplicate log spam.
- **(C)** `callbacks=callbacks` is forwarded through `train_single`/`train_multi`
  via `**trainer_kwargs` → `_train(callbacks=...)`. Verified the plumbing exists
  (`train.py` `_train` has `callbacks: list[pl.Callback] | None = None`). Works.

---

## 5. Theme 4 — Joint parallel differential loss + metrics (`4df350f`)

Files: `src/cerberus/models/bpnet.py` (new `MultitaskBPNetJointDifferentialLoss`,
`JointBPNetMetricCollection`), new tool
`tools/train_chrombpnet_multitask_differential_parallel.py`.

This is the source of **all 9 pyright errors in `bpnet.py`** and (transitively)
the 5 in `test_multitask_differential_bpnet.py`.

### 5.1 `MultitaskBPNetJointDifferentialLoss` (C/A)

```python
def loss_components(self, outputs: object, targets, **kwargs: object) -> dict[str, object]:
    ...
def forward(self, outputs: object, targets, **kwargs: object):
    components = self.loss_components(...)
    return (self.profile_weight * components["profile_loss"]
            + self.count_weight   * components["count_loss"]
            + self.delta_weight   * components["delta_loss"])
```

- **(C) Pyright errors 607/612/613/614:** the return type is annotated
  `dict[str, object]`, so `float * components["..."]` is `float * object` →
  `reportOperatorIssue`, and the dict return is incompatible with the parent's
  `dict[str, torch.Tensor]`. The parent `MSEMultinomialLoss.loss_components`
  returns `dict[str, torch.Tensor]` and `forward` returns `torch.Tensor`.
  **Fix:** match the parent's annotations (`dict[str, torch.Tensor]`,
  `-> torch.Tensor`) and drop the `object` typing of `outputs`/`targets` to match
  sibling losses. This also fixes the 3 test errors at
  `test_multitask_differential_bpnet.py:174-176`, which are the same
  `float * object` cascade.
- **(A) Reaching into a private method:** `forward`/`loss_components` call
  `self._differential_loss._delta_loss(...)` — a private method of a *composed*
  `DifferentialCountLoss` instance. Composing the differential loss and then
  reaching through its private API is a leaky boundary. Either `_delta_loss`
  should be public (it is effectively the reusable unit), or the delta
  computation should be a shared free function both losses call.
- **(B) `count_pseudocount` vs `delta_count_pseudocount`** dual params reappear
  here too (`count_pseudocount: float = 1.0` plus
  `delta_count_pseudocount: float | None = None`). Same removal applies.

### 5.2 `JointBPNetMetricCollection` (C/A)

```python
metrics = dict(BPNetMetricCollection(...))
metrics.update(dict(DifferentialBPNetMetricCollection(...)))
super().__init__(metrics)
```

- **(C) Pyright errors 715/716/723/724/734:** `dict(metric_collection)` — pyright
  types `MetricCollection` iteration as `Iterable[list[bytes]]` (it is a
  `ModuleDict`-like), so `dict(...)` and the resulting `dict[bytes, bytes]` are
  rejected. **Fix:** build the metric dict directly (instantiate the individual
  metrics in one dict literal, or use `.items()` explicitly with correct typing)
  instead of constructing two sub-collections only to immediately unpack them.
- **(A) Construct-to-destruct anti-pattern:** building two full
  `MetricCollection` objects purely to `dict(...)` them apart and rebuild a third
  is wasteful and obscures intent. A single dict assembled from the metric
  constructors is clearer and typed correctly.

### 5.3 Missing `__init__` exports (C)

`MultitaskBPNetJointDifferentialLoss` and `JointBPNetMetricCollection` are **not**
in `src/cerberus/models/__init__.py` `__all__`. They resolve at runtime via
fully-qualified `loss_cls`/`metrics_cls` strings through `import_class`, so
training works — but this violates CLAUDE.md commit-checklist item #1 (new public
API must be exported) and the sibling classes
(`DifferentialBPNetMetricCollection`, `MultitaskBPNetLoss`) *are* exported.
Inconsistent (Y) and a checklist miss (C).

### 5.4 The parallel training tool

`tools/train_chrombpnet_multitask_differential_parallel.py` (495 lines) is the
from-scratch counterpart to `train_chrombpnet_multitask.py`, reusing its private
helpers (`_export_accessibility_checkpoints`, `_merge_peaks`, etc.) via
`sys.path` injection — consistent with the sibling tool's existing pattern. Loss
wiring uses the two new classes by string. Tests cover only the arg-parser
surface (`test_train_chrombpnet_multitask_differential_parallel.py`), not an
end-to-end smoke train — acceptable given the other differential tools' coverage,
but note the gap.

---

## 6. Theme 5 — Precision default `bf16` → `full` (`b491f6c`)

File: `tools/train_chrombpnet.py`, `--precision` default changed `"bf16"` →
`"full"`.

- **(Y/S) Style/behavior flag.** This is a behavioral default change framed as a
  one-liner. Rationale (single-task ChromBPNet numerics) is plausible, but:
  - the *multitask* and *parallel* tools were **not** changed — so the default
    precision is now **inconsistent across the ChromBPNet tool family (Y)**.
    Either all should default to `full` or the divergence should be justified.
  - changing a training default silently affects throughput/memory for every
    existing invocation that relied on the default. Worth a changelog note (the
    current `[Unreleased]` does **not** mention it).
- **Decision needed:** unify the precision default across the tool family, or
  document why single-task differs.

---

## 7. Misc changes

### 7.1 `export_tfmodisco_inputs.py` accessibility-branch resolver (C) — good

New `_resolve_chrombpnet_accessibility_model` prefers legacy `chrombpnet_wo_bias`
then falls back to current `accessibility_model`. Correct generalisation.

- **(C) Pyright 66/74 (tests):** the helper is annotated
  `model: torch.nn.Module`, but the tests pass a duck-typed local `Model` class.
  Since the helper only does `getattr`, the annotation is too narrow. Fix the
  annotation (accept the structural shape it actually uses) — not the test.
- **(B)** This is itself a backwards-compat accommodation (`chrombpnet_wo_bias`
  is "older wrapper-style exports"). Under focus area (B), discuss whether legacy
  `chrombpnet_wo_bias` support should be dropped entirely, or whether old
  checkpoints are still in active use and must be read. **This one needs a
  product decision** — unlike the API aliases, dropping it can break loading real
  saved models.

### 7.2 `model_ensemble.parse_hparams_config` freeze-strip shim (B) — remove candidate

```python
if isinstance(pretrained, list):
    for entry in pretrained:
        if isinstance(entry, dict):
            entry.pop("freeze", None)
```

This strips a legacy `freeze` key from `pretrained` entries before validation,
because `PretrainedConfig` no longer has a `freeze` field (moved to
`ModelConfig.freeze` as `FreezeSpec` in the marcin refactor) and `extra="forbid"`
would otherwise reject old hparams.yaml files.

- **(B)** This is a pure backwards-compatibility shim for **old checkpoints'
  hparams**. Per focus area (B) it is a removal candidate — but, like 7.1, it
  trades off against re-loading already-trained models. **Needs the same
  product decision:** are there saved hparams.yaml files with the old
  `pretrained[].freeze` shape that must still load? If not, delete the shim.

### 7.3 `module.instantiate_metrics_and_loss` delta forwarding (C/A)

```python
metrics_params = inspect.signature(metrics_cls).parameters
if ("delta_count_pseudocount" in model_config.loss_args
        and "delta_count_pseudocount" in metrics_params):
    metrics_args.setdefault("delta_count_pseudocount",
                            model_config.loss_args["delta_count_pseudocount"])
```

Introspects the metric constructor to conditionally forward
`delta_count_pseudocount` from `loss_args`. Functional, but **(A)**: runtime
signature introspection to decide kwarg forwarding is a smell — it couples the
dispatcher to the metric's parameter names via reflection. If the
loss/metric pseudocount params are unified and named consistently (the (B)
clean-up), this special-case may be expressible more directly, or at least the
`count_pseudocount`/`delta_count_pseudocount` duality that motivates it
disappears.

---

## 8. Consolidated concern inventory

Ordered roughly by priority for the iterative fix commits (Section 9 of the
plan). Each is tagged with focus areas and whether it needs a **user decision**
vs. is a **mechanical** fix.

| # | Concern | Tags | Type |
|---|---------|------|------|
| 1 | 16 pyright errors (bpnet loss/metric typing; test cascades; tfmodisco resolver annotation) | C, Y | mechanical |
| 2 | Remove `count_pseudocount` alias everywhere; standardise on `delta_count_pseudocount` | B, C, Y | mechanical (+ confirm naming) |
| 3 | De-duplicate `_log_count_plus_pseudocount` into one home (`pseudocount.py`) | A, C | mechanical |
| 4 | Export `MultitaskBPNetJointDifferentialLoss`, `JointBPNetMetricCollection` in `models/__init__` | C, Y | mechanical |
| 5 | `JointBPNetMetricCollection` construct-to-destruct → single dict assembly | A, C | mechanical |
| 6 | `MultitaskBPNetJointDifferentialLoss` reaching into `_delta_loss` private API | A | small refactor |
| 7 | Sign-off + changelog/docs note on the pc-shrunk-prediction *semantics change* | S | decision |
| 8 | Precision default: unify across ChromBPNet tool family or document divergence | Y, S | decision |
| 9 | Count-head-only: imperative callback vs declarative `FreezeSpec` — unify? | A, S | decision |
| 10 | `FixedBackgroundSampler` class-name-as-label / `split_folds` override → provenance as data? | A, S | larger refactor / decision |
| 11 | Drop `chrombpnet_wo_bias` legacy resolver branch (7.1) | B | decision (checkpoint compat) |
| 12 | Drop `pretrained[].freeze` hparams strip shim (7.2) | B | decision (checkpoint compat) |
| 13 | `instantiate_metrics_and_loss` reflection-based kwarg forwarding (7.3) | A | follow-on to #2 |

**Decision items (7–12)** are where I will pause for your review before
implementing. **Mechanical items (1–6)** I can prepare as concrete diffs for your
approval. Items 11–12 specifically risk breaking the loading of already-trained
checkpoints — those need explicit confirmation that no live artifacts depend on
the legacy shapes.

---

## 9. Net assessment

The feature work is sound and well-motivated: the fixed-background sampler,
count-head-only fine-tuning, and parallel joint loss are real capabilities, and
the pc-shrunk differential prediction is a genuine correctness improvement to the
differential objective. The branch's weaknesses are uniformly in **polish**, not
direction:

- it ships with **16 pyright errors** (gate violation),
- it leans on **backwards-compat aliases/shims** the project now wants gone,
- it introduces **two duplicated helpers** and a **construct-to-destruct** metric
  pattern,
- and it adds a **second freezing mechanism** and a **class-name-as-label**
  coupling that overload existing abstractions.

None of these are blockers for the merge that already happened; all are
addressable as the additional commits in step 3.

---

## 10. Resolution log (step-3 clean-up commits)

Outcomes of the iterative fix pass on `marcin-feature` after the merge. Each
item references Section 8's inventory.

- **#1 pyright (14 errors)** — DONE. Joint-loss/metric type annotations matched
  to the parent (`dict[str, torch.Tensor]` / `-> Tensor`), `import torch` added,
  `JointBPNetMetricCollection` rebuilt by passing sub-collections as a list
  (also resolved #5's construct-to-destruct), tfmodisco resolver annotated
  `object`.
- **#2 `count_pseudocount` alias** — DONE. Removed from the differential
  loss/metrics; dispatch is now **capability-based** (`_constructor_accepts`
  walks the MRO), which also resolved **#13** (reflection special-case) and the
  dead `log_counts_include_pseudocount` params. Joint classes keep both
  pseudocounts (distinct there).
- **#3 duplicated `_log_count_plus_pseudocount`** — DONE. Single definition in
  `pseudocount.py`, imported by `loss.py` and `metrics.py`.
- **#4 exports** — DONE. `MultitaskBPNetJointDifferentialLoss` /
  `JointBPNetMetricCollection` exported from `cerberus.models`; the joint
  loss/metrics, the parallel trainer, and `--accessibility-count-head-only`
  (all undocumented by joanne) added to CHANGELOG + docs.
- **#6 private `_delta_loss` reach** — DONE. Joint loss calls the composed
  `DifferentialCountLoss`'s public `__call__` (its `forward` is exactly the
  delta scalar).
- **#7 pc-shrunk-prediction semantics (`cf4d40a`)** — KEPT + documented as a
  `Changed` entry (decision: the symmetric objective is more correct; `pc==0`
  recovers prior behavior; attribution unaffected).
- **#8 precision default** — RESOLVED by unifying: the **entire** ChromBPNet
  family (`chrombpnet`, `chrombpnet_bias`, `multitask`, `multitask_differential`,
  `multitask_differential_parallel`) now defaults to `--precision full`.
- **#9 count-head-only callback** — KEPT (decision A). Provenance: `FreezeSpec`
  was deliberately scoped to *static* freezing only (Marcin, 2026-04-18 design
  doc, §"out of scope v1: …unfreeze_epoch…"), explicitly anticipating "a
  dedicated callback" for selective unfreezing. Joanne's `AccessibilityCountHeadOnly`
  is exactly that. Documented the division of labor in `docs/configuration.md`
  and the callback docstring.
- **#10 class-name-as-label / `split_folds` override** — SCOPED FIX (decision A).
  The scheme (`get_interval_source` → `type(self).__name__`) is **pre-existing
  repo architecture**, and joanne's `FixedBackgroundSampler` override merely
  follows the existing `ComplexityMatchedSampler` convention — *not* a new leak.
  The full provenance-as-first-class-data refactor (which would change the
  `interval_source` strings written to BED manifests) was **deferred** as
  pre-existing debt beyond the joanne merge.
  - **Bug found and fixed.** Because `IntervalSampler` peaks collapse to
    `"ListSampler"` after `split_folds`, callers keying on a single label were
    wrong: `DalmatianLoss` used `!= "IntervalSampler"` for its bias-only term,
    so on the **split training sampler** peaks were misclassified as background
    and leaked into the bias reconstruction — an **active** training-objective
    bug (pre-existing, unrelated to joanne). `predict_misc.get_eval_intervals`
    had the same mistake (but is currently uncalled). Both, plus
    `export_predictions.py`, now route through the new
    `cerberus.samplers.PEAK_INTERVAL_SOURCES = {"IntervalSampler", "ListSampler"}`
    single source of truth. Regression tests added
    (`test_dalmatian_loss_split_peaks_labeled_listsampler`,
    `test_peak_interval_sources_separates_post_split`). The DalmatianLoss tests
    that hardcoded `"IntervalSampler"` still pass (it remains a peak label).
  - **Full fix scoped in a dedicated brief:**
    `docs/internal/interval_source_provenance_smell.md` (identifies the
    class-name-as-provenance smell and proposes role/`source_label` as
    first-class data; the manifest-string compatibility decision is why it's
    punted).

- **#11–#12 legacy shims** — pending (investigate-then-decide).
