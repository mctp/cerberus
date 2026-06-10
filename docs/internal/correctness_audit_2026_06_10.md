# Correctness Audit — `src/cerberus/` — 2026-06-10

**Scope**: All ~15.6k LoC under `src/cerberus/` (library + models). Focus is
*correctness* of the deep-learning pipeline — mathematical errors in losses and
metrics, false assumptions, code that silently produces wrong results under
realistic configs, multi-GPU/DDP correctness, worker/threading races, file
corruption, and coordinate/transform bugs. Style, formatting, and pure
micro-performance are out of scope.

**Method**: Parallel exploration of six module groups (training core,
data pipeline, model architectures, transforms/metrics/output,
prediction/attribution, utils/genome/config), followed by first-principles
re-verification of every CRITICAL/HIGH candidate against the actual source and
the installed dependency behavior (`pytorch_lightning==2.6.1`, `timm==1.0.26`).
Several initial candidates were demoted or rejected after direct reads; those
are listed at the end so the next audit does not re-raise them.

**Relationship to the 2026-04-16 audit**: the prior `extra="ignore"` finding
and the simple `predict_bigwig` floor-division truncation are *already on
record* and are not re-counted here as new. The previously-flagged DDP
`LOCAL_RANK` bug (its finding #2) **has been fixed** — see "Resolved /
verified correct". The headline issues below (#1 scheduler, #2 bigwig stride
misalignment, #3 stranded reverse-complement) are new.

**Note**: this document is an audit, not a patch set. Issues should be triaged
and fixed in follow-up PRs.

---

## Severity ranking (summary)

| # | Severity | File | One-line |
|---|----------|------|----------|
| 1 | **CRITICAL** | `module.py:158` | `lr_scheduler_step` uses the pre-2.0 signature → LR schedule runs on the wrong clock and `metric` is always `None` |
| 2 | **HIGH** | `predict_bigwig.py:277` | BigWig bins misaligned when `stride % output_bin_size != 0` (reachable with the *default* stride) |
| 3 | **HIGH** | `transform.py:172` | Reverse-complement reverses stranded targets along length but never swaps the strand channels |
| 4 | **MEDIUM** | `loss.py:178` | Profile loss summed over channels while count loss is mean-reduced → `profile_weight`/`count_weight` mis-scaled by a factor of `C` |
| 5 | **MEDIUM** | `metrics.py:202` | `CountProfile*` metrics lack the `log_counts_include_pseudocount` gate their siblings have → wrong reconstruction for Poisson/NB models with a non-zero pseudocount |
| 6 | **MEDIUM** | `sequence.py:153`, `signal.py:316` | In-memory & on-disk extractors silently truncate (or wrap) out-of-bounds intervals instead of raising |
| 7 | **LOW** | `predict_bigwig.py:79` | `regions=` export path can emit overlapping / out-of-order intervals → BigWig writer corruption/raise |
| 8 | **LOW** | `cache.py:81` | `prepare_data` cache write is non-atomic; concurrent same-config jobs can corrupt `metrics_cache.npz` |
| 9 | **LOW** | `samplers.py:613` | `RandomSampler._generate_intervals` samples closed fold regions with half-open arithmetic → loses 1 bp at each chromosome's 3′ end |

---

## 1. CRITICAL — `lr_scheduler_step` has the pre-2.0 Lightning signature; the LR schedule runs on the wrong clock

**File**: [module.py:158-168](src/cerberus/module.py#L158-L168), config at
[module.py:150-154](src/cerberus/module.py#L150-L154)
**Severity**: CRITICAL (silent mis-training of a core hyperparameter)

```python
def lr_scheduler_step(
    self, scheduler: Any, optimizer_idx: int, metric: float | None = None
) -> None:
    if hasattr(scheduler, "step_update"):
        scheduler.step(epoch=self.current_epoch, metric=metric)
        scheduler.step_update(num_updates=self.global_step)
    else:
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
```

PyTorch Lightning 2.x **removed** `optimizer_idx` from this hook. The installed
signature is `lr_scheduler_step(self, scheduler, metric)` and Lightning invokes
it **positionally**:

```text
# pytorch_lightning/loops/training_epoch_loop.py:511-516
call._call_lightning_module_hook(trainer, "lr_scheduler_step",
                                 config.scheduler, monitor_val)
```

Two compounding defects, both verified against the installed packages:

**a) `metric` is always `None`.** With the extra `optimizer_idx` positional
parameter, Lightning's `monitor_val` binds to `optimizer_idx`, and the named
`metric` keeps its default `None` forever. Any plateau-style scheduler routed
through the `else` branch calls `scheduler.step()` with no metric — it never
sees the value it is supposed to monitor.

**b) The timm schedule is stepped on the epoch clock every step.** The
optimizer config sets `"interval": "step"` ([module.py:152](src/cerberus/module.py#L152)),
so this hook fires on *every optimizer step*. `create_scheduler_v2` defaults
`step_on_epochs=True`, which builds the timm scheduler with `t_in_epochs=True`.
Under that flag (verified in `timm.scheduler.scheduler.Scheduler._get_values`):

* `scheduler.step(epoch=self.current_epoch)` is the **active** call — it sets
  the LR to the value for the *integer epoch index*, and it runs on every step.
* `scheduler.step_update(num_updates=self.global_step)` is a **no-op**
  (`_get_values(..., on_epoch=False)` returns `None` when `t_in_epochs=True`).

Net effect: the per-step warmup/cosine curve the `"interval": "step"` config
asks for never advances within an epoch. The LR collapses to an epoch
step-function (constant across all steps of an epoch, jumping at epoch
boundaries). Training proceeds without error, so the misbehavior is silent.

**Re-verification** (installed versions):
```text
PL 2.6.1  lr_scheduler_step(self, scheduler, metric)         # optimizer_idx gone
timm 1.0.26  create_scheduler_v2(step_on_epochs=True)  →  t_in_epochs=True
Scheduler._get_values: proceed = (on_epoch and t_in_epochs) or (not on_epoch and not t_in_epochs)
  step(epoch=...)            → on_epoch=True  → proceed=True  (LR set to epoch value)
  step_update(num_updates=…) → on_epoch=False → proceed=False (no-op)
```

**Reachability**: only when a non-`"default"` `scheduler_type` is configured
(by default no scheduler is attached, so the path is dormant). The moment a
user enables any timm scheduler, the configured schedule does not run as
intended.

**Fix**: match the PL 2.x signature `(self, scheduler, metric)` and issue a
single clock-consistent call — either build the scheduler with
`step_on_epochs=False` and call `scheduler.step_update(self.global_step)` for
true per-step scheduling, or keep epoch scheduling and step in
`on_train_epoch_end`. Forward the real `metric` to plateau schedulers.

---

## 2. HIGH — BigWig bins misaligned when stride is not a multiple of `output_bin_size`

**File**: [predict_bigwig.py:249-285](src/cerberus/predict_bigwig.py#L249-L285)
(`_process_island`)
**Severity**: HIGH (silent positional error in user-facing coverage tracks)

```python
min_start = min(iv.start + offset for iv in island_intervals)
...
n_bins = span_bp // output_bin_size            # L254
interval_bins = output_len // output_bin_size  # L255
...
out_start = interval.start + offset
rel_start_bin = (out_start - min_start) // output_bin_size   # L277
end_bin = rel_start_bin + interval_bins
accumulator[:, rel_start_bin:end_bin] += val                 # L284
```

`out_start - min_start` equals `k * stride` for the k-th window in an island.
When `stride % output_bin_size != 0`, the floor division at L277 snaps each
window to the wrong accumulator bin boundary — off by up to `bin_size - 1` bp.
Window k's bin-0 covers genomic `[out_start, out_start + bin_size)`, but it is
deposited at accumulator bin `floor(k·stride / bin_size)`, whose genomic origin
is `min_start + floor(k·stride / bin_size)·bin_size ≠ out_start`. Overlapping
windows are then averaged across **misaligned** bins, so the exported signal is
shifted relative to the genome.

This is reachable with the **default** stride:

```text
stride = output_len // 2                        # predict_bigwig.py:67
e.g. output_len = 1000, output_bin_size = 8     # 1000 % 8 == 0  (passes model asserts)
     stride = 500,  500 % 8 = 4  ≠ 0            # MISALIGNED
```

The model-side asserts only guarantee `output_len % output_bin_size == 0`; no
check anywhere requires `stride % output_bin_size == 0` (confirmed by grep).
The same root cause makes `n_bins = span_bp // output_bin_size` (L254) drop the
trailing partial bin, and lets `end_bin` overrun `n_bins` (silently truncated
by numpy slicing). Existing tests only exercise `output_bin_size == 1`, so the
path is untested.

**Fix**: validate `stride % output_bin_size == 0` (raise/normalize) or do the
accumulation in base-pair coordinates and bin once at the end.

---

## 3. HIGH — Reverse-complement augmentation does not swap stranded target channels

**File**: [transform.py:165-184](src/cerberus/transform.py#L165-L184)
**Severity**: HIGH (silent training-data corruption) — *conditional on stranded
multi-channel targets*

```python
inputs  = torch.flip(inputs,  dims=[-1])   # reverse along length
targets = torch.flip(targets, dims=[-1])   # reverse along length
if isinstance(self.dna_channels, slice):
    dna = inputs[self.dna_channels]
    inputs[self.dna_channels] = torch.flip(dna, dims=[-2])  # complement DNA
interval.strand = "-" if interval.strand == "+" else "+"
```

The transform reverses the target along the length axis but never swaps
strand-specific target **channels**. For a genuinely stranded target — e.g.
`targets = {"plus": plus.bw, "minus": minus.bw}` — the correct reverse-complement
must also swap channel `plus ↔ minus`, because forward-strand coverage becomes
reverse-strand coverage after RC. As written, a `+`-strand pileup stays in the
`+` channel (merely flipped in position), teaching the model an inconsistent
strand convention on half its augmented examples.

The codebase explicitly contemplates stranded targets: `DataConfig.targets` is
an arbitrary channel→bigWig mapping ([config.py:81](src/cerberus/config.py#L81)),
and `loss.py` documents `average_channels` for *"stranded profiles"*
([loss.py:118](src/cerberus/loss.py#L118)). The DNA-channel complement itself is
correct (channel-flip is valid for both `ACGT` and `AGCT` orderings).

**Reachability**: bites only when (a) two or more target channels are
strand-specific and (b) `reverse_complement` augmentation is enabled. The common
*unstranded* BPNet profile (single combined channel) is unaffected — which is
why this has gone unnoticed. Because the transform receives only a channel
*index* (no strand semantics), there is no general auto-swap; this needs an
explicit strand-pairing declaration in the config or an opt-out for stranded
runs.

**Fix**: add a strand-pair mapping to the augmentation (swap paired channels on
flip), or document/guard that RC augmentation is unsupported for stranded
targets.

---

## 4. MEDIUM — Profile loss summed over channels while count loss is mean-reduced

**File**: [loss.py:173-178](src/cerberus/loss.py#L173-L178) and
[loss.py:202-208](src/cerberus/loss.py#L202-L208)
**Severity**: MEDIUM (loss balance silently drifts with channel count)

In `MSEMultinomialLoss._compute_profile_loss` the default
(`average_channels=False`, [loss.py:105](src/cerberus/loss.py#L105)) is:

```python
return profile_loss_per_channel.sum(dim=-1).mean()   # sum over C channels, mean over batch
```

while the count loss uses `F.mse_loss(...)` with the default `mean` reduction
(L202 / L208). So the **profile** term scales ≈ linearly with the number of
channels `C`, but the **count** term does not. For multi-channel (e.g. stranded
or multi-task) targets, the effective `profile_weight : count_weight` balance is
off by a factor of `C` relative to what the weights suggest, and
`compute_counts_loss_weight`'s adaptive-alpha derivation (calibrated on the
single-channel ChromBPNet convention) is mis-scaled for `C > 1`.

Two related smells: (i) `MSEMultinomialLoss` defaults `average_channels=False`
while its sibling `PoissonMultinomialLoss` defaults `average_channels=True`
([loss.py:294](src/cerberus/loss.py#L294)) — inconsistent conventions for the
"same" knob; (ii) for `C == 1` there is no effect, so the bug only surfaces in
multi-channel runs.

This is documented behavior ("sums profile loss across channels"), so it may be
intentional — but the weight semantics are not channel-invariant. **Confirm the
intended convention** and either average both terms or document the
`C`-dependence of the weights.

---

## 5. MEDIUM — `CountProfile*` metrics always subtract the pseudocount (no `log_counts_include_pseudocount` gate)

**File**: [metrics.py:202-204](src/cerberus/metrics.py#L202-L204) and
[metrics.py:258-260](src/cerberus/metrics.py#L258-L260)
**Severity**: MEDIUM (narrow but silent wrong values)

```python
total_counts = (torch.exp(log_counts.float()) - self.count_pseudocount).clamp_min(0.0)
```

`CountProfilePearsonCorrCoef` and `CountProfileMeanSquaredError` declare
`count_pseudocount` but **not** `log_counts_include_pseudocount`, and they
subtract the pseudocount unconditionally. Their siblings
`LogCountsMeanSquaredError`/`LogCountsPearsonCorrCoef` *do* have the flag
([metrics.py:335](src/cerberus/metrics.py#L335),
[metrics.py:420](src/cerberus/metrics.py#L420)) and only subtract when the loss
trains in `log(count + pc)` space.

For a Poisson/NB model, `log_counts = log(count)` (no pseudocount), so the
correct inverse is `exp(log_counts)` with **no** subtraction. The wiring in
`instantiate_metrics_and_loss` injects `count_pseudocount =
model_config.count_pseudocount` *directly* (not via `get_log_count_params`,
[module.py:461-471](src/cerberus/module.py#L461-L471)); since these classes
accept `count_pseudocount` but ignore the flag, a Poisson/NB model with a
non-zero `count_pseudocount` gets `exp(log_counts) - pc`, systematically
under-counting (catastrophically for low-count bins, clamped to 0).

**Mitigating factors** (why MEDIUM, not HIGH): `ModelConfig.count_pseudocount`
defaults to `0.0` ([config.py:253](src/cerberus/config.py#L253)), so the
subtraction is a no-op unless explicitly set; and these two metrics are **not**
in `DefaultMetricCollection` — they require opting into a custom collection. The
bug is a latent inconsistency that bites only that specific combination.

**Fix**: add `log_counts_include_pseudocount: bool = False` to both classes and
gate the subtraction, mirroring the `LogCounts*` metrics.

---

## 6. MEDIUM — Extractors silently truncate (or wrap) out-of-bounds intervals

**Files**: [sequence.py:153-155](src/cerberus/sequence.py#L153-L155),
[sequence.py:210-218](src/cerberus/sequence.py#L210-L218),
[signal.py:312-318](src/cerberus/signal.py#L312-L318)
**Severity**: MEDIUM (silent wrong-coordinate / wrong-length output)

```python
# InMemorySequenceExtractor / InMemorySignalExtractor
# "Assume interval is valid (checked by Sampler)"
extracted = cached[:, interval.start : interval.end]   # no bounds check
vals      = chrom_data[interval.start : interval.end]
```

All extract paths trust an upstream "guaranteed by Sampler" invariant. But
several public entry points build `Interval`s *without* sampler validation —
`load_intervals_bed` / `parse_intervals` ([interval.py](src/cerberus/interval.py))
feed prediction and evaluation tools directly. When `end > chrom_length`, the
slice silently returns a **shorter** tensor (later crashing with an opaque shape
error in `torch.cat`/`torch.stack`, or worse, being used at wrong coordinates);
a negative `start` makes Python slicing **wrap from the chromosome end**,
yielding a sequence from an entirely different locus with no error at all. The
on-disk `SequenceExtractor` has the same stance (pyfaidx truncates silently,
L153-155). Note the signal in-memory path zero-pads only the *missing-chrom*
case, not the OOB case — an inconsistency.

**Fix**: assert `0 <= start < end <= chrom_length` in `extract()` for both
extractors (cheap, and converts a silent corruption into a loud error).

---

## 7. LOW — `regions=` BigWig export can emit overlapping / out-of-order intervals

**File**: [predict_bigwig.py:76-114](src/cerberus/predict_bigwig.py#L76-L114)
**Severity**: LOW (caller-dependent file corruption)

```python
sorted_regions = sorted(regions, key=lambda r: (r.chrom, r.start))
for region in sorted_regions:
    ...
    yield from _process_island(windows, ...)
```

User-supplied `regions` are only sorted, never merged/deduplicated. Each region
is tiled and emitted as an independent island. If two regions on the same
chromosome **overlap** (or a later region starts before the previous island's
emitted end), `_process_island` yields bins that overlap or precede
already-written bins. pybigtools (like all BigWig writers) requires strictly
increasing, non-overlapping intervals; out-of-order writes corrupt the file or
raise. The genome-wide path is safe (single sorted sliding sampler per chrom);
the `regions=` path is not.

**Fix**: merge overlapping regions before tiling, or assert disjointness.

---

## 8. LOW — `prepare_data` cache write is non-atomic

**File**: [cache.py:72-86](src/cerberus/cache.py#L72-L86)
**Severity**: LOW (rare multi-job file corruption)

```python
np.savez_compressed(cache_path, keys=keys, values=values)
(cache_dir / "ready").touch()
```

The `ready` sentinel correctly gates readers *within* one DDP job (rank 0
writes, others read after Lightning's post-`prepare_data` barrier). But two
independent training processes with the **same config hash** write to the same
`metrics_cache.npz` with no temp-file-plus-`os.replace()` and no lock — a
concurrent overwrite (or a reader hitting a stale `ready` while a second job
rewrites the npz) can yield a torn/corrupt file. Also `values =
np.array(list(cache.values()))` ([cache.py:82](src/cerberus/cache.py#L82))
silently becomes an `object` array if any metric row ever differs in length,
which then fails to round-trip through `np.load(..., allow_pickle=False)` — fine
today (uniform widths), flagged as a latent invariant.

**Fix**: write to a temp file in the same dir and `os.replace()` before touching
`ready`.

**Status (2026-06-10)**: *Layer 1 implemented* — `save_prepare_cache` now writes
to a unique temp file in the same directory and `os.replace()`s it into place
(with `fsync`), so the `.npz` can no longer be torn by concurrent writers; the
`ready` sentinel is touched only after the atomic publish, and ragged metric
arrays raise instead of writing a non-roundtrippable file. The remaining *Layer
2* (an advisory `flock` around the compute-and-write so only one process of a
sweep/CV recomputes) is not yet implemented — concurrent runs are now
corruption-safe but may still each recompute on a cold cache.

---

## 9. LOW — `RandomSampler._generate_intervals` consumes closed fold regions with half-open arithmetic

**Files**: [samplers.py:574-617](src/cerberus/samplers.py#L574-L617)
(consumer), region sources at
[samplers.py:566-572](src/cerberus/samplers.py#L566-L572) and
[genome.py:198-202](src/cerberus/genome.py#L198-L202)
**Severity**: LOW (1 bp at the 3′ end of every chromosome on the split path)
— *confirmed from first principles*

`_generate_intervals` iterates a region tree and samples with **half-open
arithmetic**: it allows the sampled window `end` to reach exactly `r_end`, which
is only valid if `r_end` is exclusive.

```python
start, end = interval
length = end - start                          # L590
...
max_start = r_end - self.padded_size          # L613
start = self.rng.randint(r_start, max_start)  # L616  randint() is INCLUSIVE on both ends
end = start + self.padded_size                # L617  → max sampled end == r_end
```

The same loop is fed by two region sources that use **different** end
conventions:

| Source | tuple | `length` | max sampled `end` | last base reachable |
|---|---|---|---|---|
| `_chrom_sizes_to_regions` ([samplers.py:570](src/cerberus/samplers.py#L570)) | `(0, size)` half-open | `size` | `size` | `size-1` ✓ |
| fold tree ([genome.py:201](src/cerberus/genome.py#L201)) | `(0, size-1)` **closed** | `size-1` | `size-1` | `size-2` ✗ |

So on the `split_folds` path (which sets `regions=` to the fold trees,
[samplers.py:653-685](src/cerberus/samplers.py#L653-L685)) the last base of every
chromosome (`size-1`) can never be sampled and the samplable span is 1 bp short;
the unsplit whole-genome path uses the half-open source and is correct.

**Root cause / why this is the *consumer's* bug, not the fold's**: the fold tree
is **correctly** stored closed — it is also overlap-queried as
`(start, end-1) in tree` ([samplers.py:153](src/cerberus/samplers.py#L153)),
and the code explicitly documents InterLap as closed
([genome.py:199](src/cerberus/genome.py#L199)). Changing fold storage to
half-open `(0, size)` would inject a phantom base `size` into those overlap
tests. The defect is that `_generate_intervals` applies half-open arithmetic to
a region that arrived in closed coordinates.

**Fix**: in the sampling consumer, treat a fold-derived region end as inclusive
(`max_start = r_end - padded_size + 1`, `length = r_end - r_start + 1`), or
normalize fold trees to half-open *only* where they are reused as
sampling-region bounds. Do **not** change the canonical closed fold storage.

---

## Resolved / verified correct (do not re-raise)

These were checked this round and are sound; several were prior-audit concerns
that have since been fixed.

* **DDP rank-zero gating is now correct** — [train.py:391](src/cerberus/train.py#L391)
  uses `rank_zero_only.rank == 0` (global rank, populated from
  RANK/LOCAL_RANK/SLURM_PROCID), with a comment explaining exactly the
  per-node race the prior `LOCAL_RANK` code caused. The 2026-04-16 audit's
  finding #2 is resolved.
* **ChromBPNet bias combination** — [chrombpnet.py:248-249](src/cerberus/models/chrombpnet.py#L248-L249):
  profile logits added in logit space before softmax; counts combined via
  `logaddexp` (log-space sum of totals). Correct. Dalmatian uses `.detach()` on
  the bias branch so the reconstruction loss cannot update it. Correct.
* **Multinomial / Negative-Binomial loss math** — [loss.py:155-178](src/cerberus/loss.py#L155-L178):
  multinomial NLL `-lgamma(N+1) + Σlgamma(xᵢ+1) - Σxᵢ·log_softmax` with stable
  `log_softmax`; NB parameterized so `μ = exp(log_counts)`. Coupled variants use
  `torch.logsumexp` over the correct axis. Correct.
* **Dilated-conv cropping / receptive field** — `layers.py` `_center_crop_to_length`
  and `bpnet.py` `compute_shrinkage` (`Σ dilation·(kernel-1)`) crop the valid-conv
  window symmetrically; equivalence tests pass for ASAP/Gopher/Pomeranian.
* **One-hot encoding & DNA reverse-complement** — `N`/ambiguous → all-zero
  column; channel-flip complement is valid for both `ACGT` and `AGCT`. (The
  *target-channel* swap gap is finding #3, a separate issue.)
* **Per-worker seeding** — `datamodule.py` uses `torch.initial_seed()` (unique
  per worker); per-rank/epoch resample seed `seed + epoch*world_size + rank` is
  distinct across ranks. No duplicate-data bug.
* **File-handle fork safety** — `SignalExtractor`/`SequenceExtractor` lazily open
  handles and null them in `__getstate__`; pre-fork helpers
  (`compute_median_counts`, etc.) use throwaway extractors. No cross-worker
  handle sharing / corruption.
* **Freeze + optimizer ordering** — `apply_freeze` runs before `trainer.fit()`,
  and `configure_optimizers` is invoked inside fit, so the optimizer/timm
  param-group filter sees `requires_grad=False`; `eval_mode` stops BatchNorm/
  dropout drift. Correct.
* **Variant coordinates / indel re-centering** — `record.POS - 1` (1-based→
  0-based), ref-allele FASTA verification, indel splice-and-trim to exact
  `input_len`. Correct. (Per-position profile metrics on indels are documented
  as not positionally meaningful — a known semantic caveat, not a bug.)
* **ISM / TISM attribution** — `repeat_interleave` + advanced-index alignment,
  ref delta 0, `g·x` Taylor formulation match the reference. Correct.
* **Batching order** — prediction paths use `itertools.batched` (order-preserving,
  includes final partial batch) with `strict=True` zips. No dropped/misaligned
  batches.
* **`aggregate_models` masked mean** ([output.py](src/cerberus/output.py)) divides
  by `clamp_min(1)` and zeros non-contributing positions — no divide-by-zero.
* **BigWig single-channel export** ([predict_bigwig.py:301-309](src/cerberus/predict_bigwig.py#L301-L309))
  collapses multi-channel to channel 0 — but it `logger.warning`s and is a
  documented single-track limitation, not a silent bug.

### Already on record (2026-04-16), still open

* `CerberusConfig` uses `extra="ignore"` ([config.py:269](src/cerberus/config.py#L269))
  while all seven sibling configs use `extra="forbid"` and the module docstring
  claims all do — typos in the top-level user-authored config are silently
  dropped. Tracked in the prior audit (its MEDIUM list); not re-counted here.
* `interval_bins`/`n_bins` floor-division truncation in `predict_bigwig` — the
  prior audit's finding #1; superseded in severity by the stride-misalignment
  root cause (#2 above).

---

## Triage recommendation

1. **Fix #1 first** — it silently breaks LR scheduling for every run that
   configures a scheduler, with no error and no log line. Highest blast radius.
2. **#2 and #3** produce silently-wrong scientific output (shifted coverage
   tracks; strand-confused training) in realistic configs; both need a guard or
   a fix plus a regression test covering the non-trivial case
   (`stride % bin != 0`; stranded targets with RC).
3. **#4–#6** are correctness inconsistencies that bite specific configurations;
   resolve the intended conventions and add the missing gates/asserts.
4. **#7–#9** are low-impact robustness/atomicity hardening.
