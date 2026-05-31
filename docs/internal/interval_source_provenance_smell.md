# Code smell: interval-source provenance as a Python class name

**Status:** Issue identified; band-aid landed (`PEAK_INTERVAL_SOURCES`), proper
fix **punted**. This document scopes the smell and proposes the remediation.

**Related:**
- `docs/internal/joanne_feature1_merge_review.md` §10 (#10) — where this surfaced.
- `src/cerberus/samplers.py` — `get_interval_source`, `split_folds`, `PEAK_INTERVAL_SOURCES`.

---

## TL;DR

Cerberus decides whether a genomic interval is a **peak (positive)** or
**background (negative)** by reading the *Python class name of the sub-sampler
that produced it* — `get_interval_source(i)` returns `type(self).__name__`. That
string is then compared against hard-coded class names at several call sites to
drive **training-loss masks** and **evaluation interval selection**.

This couples a semantic property (peak vs. background, and which generator
produced an interval) to an implementation detail (the class name), and it is
fragile in a way that has already produced **one active training bug** and **one
latent bug**:

- A peak `IntervalSampler` **changes its label** from `"IntervalSampler"` to
  `"ListSampler"` after `split_folds`, because the base split returns plain
  `ListSampler` objects.
- To avoid that collapse, "must-stay-distinct" samplers
  (`ComplexityMatchedSampler`, `FixedBackgroundSampler`) carry **bespoke
  `split_folds`/`_subset`/`_make_split` overrides** whose *only* purpose is to
  preserve their class name as a label.
- Call sites disagreed on the predicate (`== "IntervalSampler"` vs.
  `!= "IntervalSampler"` vs. a two-element set), and at least one was wrong on
  the split sampler used in training.

The `PEAK_INTERVAL_SOURCES = {"IntervalSampler", "ListSampler"}` constant added
in the joanne-feature1 clean-up makes the *predicate* consistent, but it is a
**band-aid over the smell, not a fix** (see §4).

---

## 1. How provenance works today

```
src/cerberus/samplers.py
  261  BaseSampler.get_interval_source(idx)  -> return type(self).__name__
  427  MultiSampler.get_interval_source(idx) -> return type(self.samplers[k]).__name__
  507  ListSampler.split_folds(...)          -> returns plain ListSampler(...) subsets
  695  class IntervalSampler(ListSampler)    -> does NOT override split_folds
 1044  class ComplexityMatchedSampler        -> overrides split_folds (keeps its name)
  857  class FixedBackgroundSampler          -> overrides split_folds/_subset/_make_split
```

The per-interval source string flows into the dataset and the training batch:

```
src/cerberus/dataset.py:365   result["interval_source"] = self.sampler.get_interval_source(idx)
src/cerberus/datamodule.py    writes an interval_source column into the BED split manifests
```

and is consumed as a peak/background discriminator:

```
src/cerberus/loss.py:752          DalmatianLoss: non_peak = [s not in PEAK_INTERVAL_SOURCES]
src/cerberus/predict_misc.py:153  get_eval_intervals: peaks = [s in PEAK_INTERVAL_SOURCES]
tools/export_predictions.py:255   n_peaks = sum(s in PEAK_INTERVAL_SOURCES)
```

So the *meaning* "this interval is a peak" is carried by the *string equal to a
class name*, and that string is **not stable**: it depends on whether the
sampler has been fold-split, and on whether each class bothered to override
`split_folds` to preserve its identity.

---

## 2. Why this is a smell

1. **Semantic property encoded as an implementation detail.** "Peak vs.
   background" and "which generator produced this" are domain facts. Tying them
   to `type(self).__name__` means a rename, a refactor, or a new subclass can
   silently change behavior far away (a loss mask, an evaluation split). The
   compiler/type-checker cannot help — these are bare string literals.

2. **The label is not invariant under `split_folds`.** The base
   `ListSampler.split_folds` (samplers.py:507) returns plain `ListSampler`
   objects, so a peak `IntervalSampler` reports `"IntervalSampler"` *before* a
   split and `"ListSampler"` *after* one. The training path always uses the
   split sampler (`CerberusDataset.split_folds` → `sampler.split_folds`), so
   **training sees a different label than a freshly-built sampler does.**

3. **Bespoke overrides exist only to defend the label.**
   `ComplexityMatchedSampler` (samplers.py:1240) and `FixedBackgroundSampler`
   (samplers.py:882–908) each carry `split_folds`/`_subset`/`_make_split`
   overrides whose sole job is to keep `type(self).__name__` stable across a
   split. That is machinery in service of a string tag — and every *future*
   background-like sampler must remember to add the same overrides or it will
   silently collapse to `"ListSampler"` and be misread as a peak.

4. **Asymmetric, leaky contract.** Peaks are *allowed* to collapse to
   `"ListSampler"` (callers compensate), but backgrounds must *not* collapse
   (or they alias peaks). The same `split_folds` behavior is therefore
   load-bearing in opposite directions for two interval roles — exactly the
   coupling that produced the bugs in §3.

5. **No first-class notion of "background".** Callers express background as
   *"not a peak"* (`s not in PEAK_INTERVAL_SOURCES`). There is no positive,
   enumerable notion of background provenance, so anything unrecognised is
   silently treated as background.

---

## 3. Concrete failures this has already caused

- **Active training bug (fixed in `61a5726`).** `DalmatianLoss` masked its
  bias-only reconstruction term with `s != "IntervalSampler"`. On the split
  training sampler, peaks are `"ListSampler"`, so peaks were classified as
  background and leaked into the bias term — partially defeating the
  bias/peak factorization. Tests passed only because they **hard-coded**
  `interval_source=["IntervalSampler", ...]`, never exercising the split label.

- **Latent bug (fixed in `61a5726`).** `predict_misc.get_eval_intervals` used
  `== "IntervalSampler"` and would have returned **zero peaks** on any split
  `MultiSampler` (all intervals dumped into background). It is currently
  uncalled, so it never fired.

- **Predicate drift.** Three call sites independently chose
  `== "IntervalSampler"`, `!= "IntervalSampler"`, and
  `{"IntervalSampler","ListSampler"}` — three different answers to the same
  question.

---

## 4. Why `PEAK_INTERVAL_SOURCES` is a band-aid, not the fix

The constant (`samplers.py:31`) centralises the predicate so all callers agree,
and its docstring documents the split-collapse. That removes the *drift* and the
two bugs. But it does **not** remove the smell:

- It is still a **string whitelist of class names** that must be kept in sync by
  hand. Add a new peak-like sampler (say `SlidingWindowPeakSampler`) and you must
  remember to add its name here, or it is silently treated as background.
- It still encodes "peak" as *"happens to be one of these classes"* and
  "background" as *"anything else"* — there is no positive background identity.
- It does not address the **`split_folds` override tax** on background samplers,
  nor the asymmetry in §2.4.
- The `interval_source` strings are also written into **BED split manifests**
  (`datamodule.py:295`), so downstream tooling may already depend on the literal
  class-name strings — meaning the smell has leaked outside the process.

`PEAK_INTERVAL_SOURCES` is the right *interim* move (consistent, documented, test
-covered), but the underlying model should change.

---

## 5. Proposed remediation: provenance as first-class data

Replace "provenance = class name" with an explicit, enumerable label carried on
the interval/sub-sampler and preserved by construction across `split_folds`.

### 5.1 Shape

Introduce an explicit role/source value, e.g.:

```python
class IntervalRole(StrEnum):     # cerberus.samplers (or cerberus.interval)
    PEAK = "peak"
    BACKGROUND = "background"
```

and give every `ListSampler`/`ProxySampler` a `source_label: str` (free-form
generator id, e.g. `"complexity_matched"`, `"fixed_background"`, `"peaks"`) plus
a `role: IntervalRole`. `get_interval_source` returns the **stored label**, not
`type(self).__name__`. `MultiSampler.get_interval_source` returns the chosen
sub-sampler's stored label. Peak vs. background becomes `sampler.role_at(idx) ==
IntervalRole.PEAK` — a positive, type-checked test for *both* roles.

### 5.2 Why this dissolves the smell

- **Stable across split.** Because `_subset`/`split_folds` copy the stored
  `role`/`source_label` into the child sampler (one base-class change), the label
  no longer depends on the concrete class — so the bespoke
  `FixedBackgroundSampler`/`ComplexityMatchedSampler` `split_folds` overrides
  **can be deleted**. Item #10's override is removed, not merely tolerated.
- **No whitelist.** Callers test `role == PEAK` / `role == BACKGROUND`; there is
  no class-name set to keep in sync, and an unknown sampler can't masquerade as
  background by accident.
- **Type-checkable.** An enum (or a small `Literal`) lets pyright catch typos and
  unhandled roles.

### 5.3 Migration plan (incremental, low-risk)

1. Add `role`/`source_label` to the base sampler(s) with defaults that reproduce
   today's strings (`IntervalSampler` → label `"IntervalSampler"`, role `PEAK`;
   `ComplexityMatchedSampler` → `"ComplexityMatchedSampler"`, role `BACKGROUND`;
   etc.), and have `_subset`/`split_folds` propagate them. This is behavior-
   preserving — manifests still emit the same strings.
2. Switch the three call sites (`DalmatianLoss`, `predict_misc`,
   `export_predictions`) from `PEAK_INTERVAL_SOURCES` membership to
   `role == PEAK`. Delete `PEAK_INTERVAL_SOURCES`.
3. Delete the `split_folds`/`_subset`/`_make_split` overrides on
   `FixedBackgroundSampler` and `ComplexityMatchedSampler` that exist only to
   preserve the class-name label; verify the propagated `role`/`source_label`
   gives the same manifests.
4. **Manifest compatibility decision (the reason this is punted).** The
   `interval_source` column written to BED split manifests is downstream-visible.
   Either keep emitting the legacy class-name strings (store them as the default
   `source_label`) for backwards compatibility, or make a deliberate,
   documented breaking change to role-based labels. This choice needs an owner
   and a scan of downstream consumers (notebooks, analysis scripts) before it
   can land.

### 5.4 Tests to add with the fix

- `role`/`source_label` survive `split_folds` and `_subset` for every sampler
  (peaks **and** backgrounds) — the property tests that the class-name scheme
  silently failed.
- `DalmatianLoss` / `predict_misc` use the role, exercised on a **real split**
  `MultiSampler` (not stubbed labels).
- Manifest round-trip: the `interval_source` column matches the agreed
  (legacy or new) strings.

---

## 6. Decision

For the joanne-feature1 merge we deliberately **punt** the §5 refactor because it
(a) is pre-existing architecture unrelated to the merge, and (b) may change the
downstream-visible `interval_source` manifest strings, which needs its own owner
and consumer scan. The interim `PEAK_INTERVAL_SOURCES` band-aid (consistent,
documented, regression-tested) holds the line until then. This document is the
brief for the dedicated follow-up.
