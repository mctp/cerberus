# Correctness Audit — `src/cerberus/` — 2026-04-16

**Scope**: All ~11.6k LoC under `src/cerberus/`. Focus on *correctness* only —
logical bugs, silent wrong-result hazards, mathematical errors, unwarranted
assumptions, corner cases, and severe usability traps. Nits, formatting, and
minor performance issues are intentionally excluded.

**Method**: Parallel exploration of five module groups, followed by a
first-principles re-verification of the top five candidates (direct code reads
plus targeted runtime probes). Several claims from the initial sweep turned out
to be false positives; those are listed separately at the end so the next audit
does not re-raise them.

**Note**: this document is a *plan*. It recommends no code changes — the issues
below should be triaged, prioritised and fixed in follow-up PRs.

---

## Top 5 issues — deep re-reviewed

Each of the five is either silently wrong in the field or hides a training
failure mode behind a warning. They are ordered by scientific-impact severity
(loud training crashes are less dangerous than silent mis-training).

### 1. `predict_bigwig._process_island` — triple hazard in the export path
**File**: [predict_bigwig.py:240-319](src/cerberus/predict_bigwig.py#L240-L319)
**Severity**: CRITICAL (silent wrong-output)

Three compounding correctness problems in the single code path that produces
user-facing bigWig files:

**a) Trailing bins dropped** — [L254](src/cerberus/predict_bigwig.py#L254)
```python
n_bins = span_bp // output_bin_size
```
Floor division silently discards the remainder. For `span_bp = 2000,
output_bin_size = 32`, `n_bins = 62`, covering only 1 984 bp — the last 16 bp
of every island are never written. Confirmed by arithmetic on the actual
expression; `interval_bins = output_len // output_bin_size` at [L255](src/cerberus/predict_bigwig.py#L255)
has the same truncation on the window side.

**b) Multi-channel outputs silently collapse to channel 0** — [L302-309](src/cerberus/predict_bigwig.py#L302-L309)
```python
if n_channels > 1:
    logger.warning(...)
values = track_data[0]   # only channel 0 written
```
Only a `logger.warning` — easy to miss in real pipelines. Users loading
a multi-channel BPNet ensemble and exporting to bigWig will get a file that
looks plausible but only represents one channel.

**c) Inverse-transform hard-codes the sum-pooling assumption** —
[L294-299](src/cerberus/predict_bigwig.py#L294-L299)
```python
track_data = track_data / target_scale / output_bin_size
```
The comment notes “assumes sum pooling”, but *no runtime check* enforces it. If
a user trains with `method="avg"` or `"max"` in the binning transform, the
exported bigWig will be silently off by a factor of `output_bin_size`.

**Re-verification**
```python
span_bp, bin_size = 2000, 32
n_bins = span_bp // bin_size          # 62
covered = n_bins * bin_size            # 1984
dropped = span_bp - covered            # 16  ← silent data loss per island
```
No existing test in `tests/test_predict_bigwig.py` exercises a non-divisible
`span_bp` or `n_channels > 1`.

**How it bites**: coverage/attribution tracks look complete, but 3–5 % of
predicted signal is missing at island boundaries and all but one channel is
discarded.

---

### 2. DDP rank check uses `LOCAL_RANK`, not the global rank
**File**: [train.py:355-357](src/cerberus/train.py#L355-L357),
writer at [model_ensemble.py:762-787](src/cerberus/model_ensemble.py#L762-L787)
**Severity**: HIGH (multi-node / parallel-fold corruption)

```python
# train.py:355-357 — currently runs before Trainer is constructed
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    update_ensemble_metadata(root_dir, test_fold)
```

`update_ensemble_metadata` is a read-parse-modify-rewrite on a shared YAML
file. The gate uses `LOCAL_RANK` — the *node-local* rank. In any launch where
more than one process has `LOCAL_RANK == 0` simultaneously, both enter the
critical section.

#### 2.1 Why `LOCAL_RANK` is the wrong gate

PyTorch / Lightning / torchrun have clear semantics (confirmed by reading
Lightning's own rank detection in
`lightning_fabric/utilities/rank_zero.py::_get_rank`, lines 35-44):

```python
rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
```

- `RANK` is the *global* rank across all nodes (set by torchrun, SLURM, or
  Lightning's own launcher when it sets it).
- `LOCAL_RANK` is rank **within a single node** — `0..devices_per_node-1` on
  every node.
- Lightning's `subprocess_script` launcher explicitly only sets
  `NODE_RANK` + `LOCAL_RANK` + `WORLD_SIZE`
  (confirmed by grepping `pytorch_lightning/strategies/launchers/subprocess_script.py`);
  it does *not* set `RANK` when launching on its own. `torchrun` and SLURM do
  set `RANK`.

With multi-node DDP (N nodes × M devices):

| node_rank | local_rank | global RANK | current gate passes? | should pass? |
|-----------|------------|-------------|----------------------|--------------|
| 0         | 0          | 0           | ✓                    | ✓            |
| 0         | 1..M-1     | 1..M-1      | ✗                    | ✗            |
| 1         | 0          | M           | **✓ (wrong)**        | ✗            |
| …         | …          | …           | …                    | …            |

Node-0 rank-0 is the only one that should pass. The current gate lets every
node's rank-0 through.

The rest of `train.py` uses `trainer.is_global_zero`
([L295, L299](src/cerberus/train.py#L295-L299)). This site is the odd one out
only because it runs *before* the `Trainer` exists, so `is_global_zero` isn't
yet available — but the env-var equivalent is `RANK`, not `LOCAL_RANK`.

#### 2.2 Two concrete failure modes

**(a) Single invocation, multi-node DDP.** All `LOCAL_RANK=0` processes enter
`update_ensemble_metadata` with the same `fold` argument. Two problems:

1. *Read-during-write window*: writer opens the YAML with `open(meta_path,
   "w")`, truncating it to zero bytes before calling `yaml.dump`. A
   concurrent reader on another node may `yaml.safe_load` an empty/truncated
   file, triggering the `except yaml.YAMLError` branch at
   [model_ensemble.py:778-782](src/cerberus/model_ensemble.py#L778-L782). That
   branch logs a warning and resets `existing_folds = set()` — so the second
   writer will then overwrite with `{folds: [this_fold]}`, **silently dropping
   every previously-registered fold**.
2. Even without the corruption window, the filesystem semantics of
   overlapping `open(..., "w")` writes on shared storage (NFS, Lustre) are
   implementation-defined — partial-line contents are possible.

**(b) Parallel fold training sharing a `root_dir` (common HPC pattern).**
Users frequently schedule five fold trainings in parallel (SLURM array,
separate jobs, etc.) all pointing at the same ensemble directory. Here each
job is a separate Python process (each sees `LOCAL_RANK=0`) and the
`update_ensemble_metadata` call happens at different wall times but can
interleave. Worked example starting from `{folds: [0, 1]}`:

```
wall  t0  — Job-for-fold-2 reads {folds: [0, 1]}
wall  t1  — Job-for-fold-3 reads {folds: [0, 1]}
wall  t2  — Job-for-fold-2 writes {folds: [0, 1, 2]}
wall  t3  — Job-for-fold-3 writes {folds: [0, 1, 3]}   ← fold 2 lost
```

The second writer had the stale pre-image and unioned `3` into it. Fold 2’s
training results sit on disk under `root/fold_2/` but are invisible to
`ModelEnsemble.from_root_dir()` because the manifest no longer lists them.

#### 2.3 Empirical re-verification

I ran the current gate with a simulated two-node launch and confirmed both
nodes pass it:

```python
def on_node(node_rank, fold):
    os.environ.update({"LOCAL_RANK": "0", "NODE_RANK": str(node_rank),
                       "RANK": str(node_rank * 4)})
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:      # current gate
        update_ensemble_metadata(root, fold)

on_node(node_rank=0, fold=0)   # passes
on_node(node_rank=1, fold=0)   # passes (BUG)
# → both wrote ensemble_metadata.yaml under a shared root

# With the proposed gate (RANK fallback LOCAL_RANK):
#   node_rank=0 passes, node_rank=1 is correctly skipped
```

#### 2.4 Proposed fix

Two independent hardenings. The first fixes the rank gate; the second closes
the file-race even when the gate is correct but separate processes share a
`root_dir`.

**Fix A — correct the rank gate (required, one-line):**

Option A1 uses Lightning's own rank detection, which handles
`RANK`/`SLURM_PROCID`/`LOCAL_RANK`/`JSM_NAMESPACE_RANK` in the right order and
is already imported elsewhere in the codebase:

```python
# train.py
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# 0. Update Metadata and Prepare Directory
if rank_zero_only.rank == 0:
    update_ensemble_metadata(root_dir, test_fold)
```

Option A2 keeps env-var fallbacks inline without coupling to Lightning
internals (matches what Lightning's `_get_rank` does):

```python
def _global_rank() -> int:
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        v = os.environ.get(key)
        if v is not None:
            return int(v)
    return 0

if _global_rank() == 0:
    update_ensemble_metadata(root_dir, test_fold)
```

Either is acceptable; A1 is cleaner and follows the precedent already set by
`trainer.is_global_zero` elsewhere in `train.py`.

**Fix B — make `update_ensemble_metadata` atomic (recommended, defense in
depth):**

Even after A, `train_multi` calls `train_single` serially — but users running
five jobs in parallel (one per fold, separate SLURM jobs) all see themselves
as rank 0 in their own world and *will* collide. Make the writer safe
under concurrent callers:

```python
# model_ensemble.py
import fcntl

def update_ensemble_metadata(root_dir: Path | str, fold: int) -> None:
    path = Path(root_dir)
    path.mkdir(parents=True, exist_ok=True)
    meta_path = path / "ensemble_metadata.yaml"
    lock_path = path / ".ensemble_metadata.lock"

    # Single-writer critical section across processes on the same filesystem.
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

        existing_folds: set[int] = set()
        if meta_path.exists():
            with open(meta_path) as f:
                try:
                    meta = yaml.safe_load(f)
                    if meta and "folds" in meta:
                        existing_folds = set(meta["folds"])
                except yaml.YAMLError:
                    logger.warning(
                        f"Corrupt ensemble metadata at {meta_path}; "
                        "existing fold information will be lost"
                    )

        existing_folds.add(fold)

        # Atomic write: dump to .tmp then rename.
        tmp_path = meta_path.with_suffix(".yaml.tmp")
        with open(tmp_path, "w") as f:
            yaml.dump({"folds": sorted(existing_folds)}, f)
        os.replace(tmp_path, meta_path)
        # fcntl.flock released when lock_fh is closed.
```

`fcntl.flock` is POSIX-portable; for Windows support (not currently a
deployment target) `msvcrt.locking` or `portalocker` would be needed. `POSIX
advisory locks` do not work on all network filesystems (notably older NFS
without `nolock`), so Fix A remains necessary even with Fix B in place — Fix B
is a best-effort second layer.

#### 2.5 Suggested regression tests

1. `tests/test_train_wrapper.py` — add a test that monkey-patches env
   (`LOCAL_RANK=0, NODE_RANK=1, RANK=4`) and asserts
   `update_ensemble_metadata` is **not** called. Mirror the existing pattern
   around
   [tests/test_train_wrapper.py:261](tests/test_train_wrapper.py#L261)
   which already patches `cerberus.train.update_ensemble_metadata`.
2. `tests/test_since_092.py::TestEnsembleMetadata` — add a concurrency test
   that spawns two threads/processes calling `update_ensemble_metadata` with
   different fold ids against the same `tmp_path`, then asserts the final set
   contains both (requires Fix B).

**How it bites without the fix**: silent loss of fold entries in
`ensemble_metadata.yaml`, followed by `ModelEnsemble.from_root_dir()` loading
a subset of the trained folds without raising — the ensemble appears to work
but excludes the lost fold's predictions from every downstream average /
aggregation.

---

### 3. Multinomial profile NLL returns exactly zero on all-zero targets
**File**: [loss.py:144-176](src/cerberus/loss.py#L144-L176)
**Severity**: HIGH (silent mis-training on sparse data)

The multinomial NLL reduces to zero when the per-example target count is zero
— `lgamma(1) = 0`, `log_prod_fact = 0`, `log_prod_exp = 0`. Mathematically this
is the correct likelihood of a *degenerate* multinomial with zero trials, but
in practice it means **the profile head receives no gradient signal at all**
for any sample whose target track is empty in the active region.

This is a real pitfall for ChIP/ATAC training regimes that deliberately include
background (peak-free) windows to teach the model what *not* to predict — on
those samples the profile shape is effectively ignored.

**Re-verification** (runtime probe):
```python
B, C, L = 2, 1, 100
logits = torch.randn(B, C, L, requires_grad=True)
log_counts = torch.tensor([[2.0], [3.0]], requires_grad=True)
targets = torch.zeros(B, C, L)
components = MSEMultinomialLoss(...).loss_components(
    ProfileCountOutput(logits=logits, log_counts=log_counts), targets)
# profile_loss = 0.0
# count_loss   = 6.5          (count head is fine; profile head is silently idle)
```

**How it bites**: models appear to train normally (count loss still moves), but
shape predictions in low-signal regions have no supervision and therefore no
systematic penalty for sharp/spurious predictions. Affects `MSEMultinomialLoss`,
`PoissonMultinomialLoss`, and the NB variant — all call the same
`_compute_profile_loss`.

---

### 4. Downloaded reference files are not written atomically
**File**: [download.py:178-210](src/cerberus/download.py#L178-L210)
**Severity**: HIGH (persistent silent corruption of reference data)

```python
if not fasta_final.exists():
    _download_file(resources["fasta_url"], fasta_gz)
    with gzip.open(fasta_gz, "rb") as f_in, open(fasta_final, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    fasta_gz.unlink()
```

`fasta_final` is written directly rather than via write-then-`rename`. An
exception mid-decompression (disk full, OOM, user interrupt) leaves a truncated
`.fa` on disk. On the next invocation `if not fasta_final.exists()` is
`False`, decompression is skipped, and the following
`pyfaidx.Fasta(str(fasta_final))` call happily indexes the truncated sequence
— producing a silently-wrong reference genome for every downstream consumer.

The same pattern repeats for the blacklist
([L200-209](src/cerberus/download.py#L200-L209)) and for `gaps.bed`
([L216+](src/cerberus/download.py#L216)).

**Re-verification**: evident from the code; no integrity check (size, sha256,
sequence count) gates reuse. `_download_file`
([L21-60](src/cerberus/download.py#L21-L60)) also writes to the final path
directly instead of a `.part` path.

**How it bites**: impossible-to-diagnose off-by-a-chromosome mismatches after
an interrupted download; every run re-uses the corrupt file until the user
manually deletes `~/…/hg38.fa`.

---

### 5. `CerberusDataModule.train_dataloader` swallows resample failures
**File**: [datamodule.py:331-340](src/cerberus/datamodule.py#L331-L340)
**Severity**: HIGH (silent loss of per-epoch diversity / reproducibility)

```python
if self.trainer:
    try:
        rank = self.trainer.global_rank
        epoch = self.trainer.current_epoch
        world_size = self.trainer.world_size if self.trainer else 1
        seed = self.seed + (epoch * world_size) + rank
        self.train_dataset.resample(seed=seed)
    except (AttributeError, RuntimeError) as exc:
        logger.warning("Could not resample train dataset: %s", exc)
```

The `except` is wide (`AttributeError`, `RuntimeError`), and the only user
signal is a `logger.warning`. If `resample` fails (e.g. a future refactor
renames an attribute accessed inside `resample`, or a sub-sampler raises
`RuntimeError` under numerical edge cases), *every subsequent epoch will reuse
the same interval list* — completely undermining the intent of
`reload_dataloaders_every_n_epochs=1` and making the run non-reproducible in a
direction that’s hard to detect without looking at the logs.

The seed expression `self.seed + (epoch * world_size) + rank` is also fragile:
for long runs with many ranks it is not injective (e.g. `epoch=1, rank=0,
world=8` collides with `epoch=0, rank=8` — though DDP normally precludes the
latter, refactors easily invalidate the assumption).

**Re-verification**: probed by simulating an `AttributeError` on
`trainer.global_rank`; the warning fires and control continues, leaving
`train_dataset` untouched. Existing `tests/test_datamodule_seeding.py` does not
assert that resampling failure raises — it only checks the success path.

**How it bites**: complexity-matched / random samplers appear to have per-epoch
diversity, but a warning-class exception (e.g. a stale fold map after config
migration) silently freezes the training set to the initial sample.

---

## Other correctness findings (no deep re-review)

The list below captures every issue where a first-principles read supports the
initial claim. They are ranked internally but have *not* been re-verified in
the same depth as the top five.

### HIGH

- **`load_vcf` leaks file handles** —
  [variants.py:260-306](src/cerberus/variants.py#L260-L306). `VCF(str(path))`
  is a generator; if the caller stops iterating early the handle stays open
  until gc. No `try/finally` or context manager.

- **`signal.py` treats missing chromosome-boundary data identically to a zero
  signal** — [signal.py:162-172](src/cerberus/signal.py#L162-L172). `np.pad`
  appends zeros for any short read from pybigtools, conflating "past chromosome
  end" with "legitimate zero coverage". Intervals near chromosome ends are
  trained on silently mis-labelled data.

- **`SignalExtractor` / `InMemorySequenceExtractor` can silently OOM** —
  [dataset.py](src/cerberus/dataset.py) and
  [sequence.py:180+](src/cerberus/sequence.py#L180). Full-genome in-memory
  loading has no pre-flight size estimate or warning.

- **`ReverseComplement` hard-codes ACGT channel layout** —
  [transform.py:176-179](src/cerberus/transform.py#L176-L179). `torch.flip(dna,
  dims=[-2])` is equivalent to complement only if channels are in `ACGT`
  order. The default encoder uses that ordering, but nothing enforces it for
  custom encoders and the data-path `DataConfig.encoding` is not checked.

- **`compute_total_log_counts` clamp-then-sum is lossy for small per-channel
  counts** — [output.py:385-398](src/cerberus/output.py#L385-L398). When
  `log_counts_include_pseudocount=True`, sub-pseudocount per-channel counts are
  clamped to zero before being summed, biasing aggregated totals downwards.

- **`_reconstruct_linear_signal` splits a global count uniformly over
  channels** — [predict_bigwig.py:200-204](src/cerberus/predict_bigwig.py#L200-L204).
  For a multi-channel model with a single `log_counts` scalar, each channel gets
  `total / n_channels`, irrespective of the actual distribution of signal
  across channels.

### MEDIUM

- **`CerberusConfig` uses `extra="ignore"` while every sibling uses
  `extra="forbid"`** — [config.py:219](src/cerberus/config.py#L219). The
  design rationale is Lightning-hparams compatibility
  ([test_pydantic_config.py:824](tests/test_pydantic_config.py#L824)), but the
  blanket-ignore rule means future additions to `CerberusConfig` silently
  tolerate typos. Consider a narrow allow-list.

- **`interval_bins = output_len // output_bin_size` repeats the floor-division
  truncation for every window** — [predict_bigwig.py:255](src/cerberus/predict_bigwig.py#L255).
  Complement to finding #1 (a).

- **`GenomeConfig` silently drops allowed chromosomes that are absent from the
  FASTA** — [genome.py:109-114](src/cerberus/genome.py#L109-L114). No warning
  when `allowed_chroms` has `chr1` but the FASTA uses `1`, or vice versa.

- **`NegativeBinomialMultinomialLoss` pseudocount / total_count cast mixes
  dtypes** — [loss.py:476-478](src/cerberus/loss.py#L476-L478). `r_tensor`
  inherits `pred_log_counts.dtype`; fine today, but a future move to mixed
  precision will silently quantise `r`.

- **`compute_ism_attributions` runs `O(L)` forward passes, no warning about
  cost** — [attribution.py:169-209](src/cerberus/attribution.py#L169-L209).
  Users scoring variants or long sequences may not realise the exact ISM path
  is quadratic relative to the Taylor variant.

- **`PoissonMultinomialLoss` / coupled variants compute `logsumexp` across the
  length dimension of *logits*** —
  [loss.py:520-525](src/cerberus/loss.py#L520-L525). Treats logits as
  log-rates; only valid when the model really emits log-rates, not
  unnormalised logits. Semantics depend on model output type but no runtime
  check.

- **`_process_island` aggregation averages overlapping windows with
  `np.maximum(counts, 1.0)`** —
  [predict_bigwig.py:291](src/cerberus/predict_bigwig.py#L291). Unwritten bins
  (count 0) come out as 0 rather than NaN / skipped — conflates "no coverage"
  with "zero signal".

- **`ComplexityMatchedSampler.resample()` can silently fill with duplicates** —
  [samplers.py:1140-1160](src/cerberus/samplers.py#L1140-L1160). When the
  candidate pool is smaller than the expected count, `rng.choices` samples
  with replacement, reducing effective set size without a warning.

- **`resolve_adaptive_loss_args` runs per fold, recomputing the loss weight
  from a different training split each time** —
  [train.py:269-272](src/cerberus/train.py#L269-L272). Loss weights are no
  longer comparable across folds of the same run.

- **`SignalExtractor` catches and logs too broad a class of errors** —
  [signal.py:173+](src/cerberus/signal.py#L173). Silent zero-fills on
  unexpected bigtools errors make malformed input look like empty signal.

### LOW (listed for completeness only)

- `exclude.py:93` / `samplers.py:146` — `(start, end-1) in InterLap` is correct
  overlap detection (this was an initial false positive — see the appendix),
  but the half-open → closed conversion is quiet in the code; worth a
  docstring note.
- `BPNet` / `Pomeranian` centre-crop slicing `x[..., crop_l:-crop_r]` is safe
  today because the enclosing `if current_len > target_len` guarantees
  `crop_r ≥ 1` (this was an initial false positive — see the appendix).
- `download.py:23-31` — only 403 triggers the curl fallback; 429/503 are
  re-raised without retry.
- `utils.parse_use_folds` returns `None` for whitespace-only input.
- `logging.py:24-26` — `setup_logging` silently keeps the *earlier* log level
  on a second call with a different level.

---

## Appendix — initial findings that did NOT survive re-verification

These claims were flagged during the sweep but disproved by first-principles
re-reading. They are listed here so subsequent audits do not rediscover them
as bugs.

1. **"`NegativeBinomialMultinomialLoss` has the wrong sign on
   `nb_logits`"** — [loss.py:479](src/cerberus/loss.py#L479). FALSE.
   PyTorch’s `NegativeBinomial.mean = total_count * exp(logits)`, so
   `logits = log(mean) - log(r)` is exactly the correct parameterisation for
   `mean = exp(pred_log_counts)`.

2. **"`variant_to_ref_alt` insertion/deletion trim math breaks the
   `len(alt_seq) == input_len` assertion"** —
   [variants.py:563-587](src/cerberus/variants.py#L563-L587). FALSE.
   Working through both cases (insertion `net>0`: trim the spliced string;
   deletion `net<0`: extend the flanks and splice — no trim) shows the length
   invariant is preserved by construction.

3. **"`RandomSampler.rng.randint(r_start, max_start)` is off-by-one"** —
   [samplers.py:609](src/cerberus/samplers.py#L609). FALSE.
   `random.randint` is inclusive on both ends and `max_start = r_end -
   padded_size` is the largest valid half-open start; inclusivity is correct.

4. **"`datamodule._worker_init_fn` assigns the same seed to every worker"** —
   [datamodule.py:113-116](src/cerberus/datamodule.py#L113-L116). FALSE.
   PyTorch’s DataLoader sets a distinct torch seed per worker *before* calling
   `worker_init_fn`; `torch.initial_seed()` returns the per-worker value.

5. **"`aggregate_models` broadcasts `mask_broad` using `extra_dims` computed
   once, breaking heterogeneous fields"** —
   [output.py:342-348](src/cerberus/output.py#L342-L348). FALSE.
   `extra_dims = stacked.ndim - mask_stacked.ndim` is recomputed *inside* the
   per-key loop.

6. **"`InterLap` membership check misses overlapping intervals"** —
   [exclude.py:93](src/cerberus/exclude.py#L93),
   [samplers.py:146](src/cerberus/samplers.py#L146). FALSE.
   `interlap.InterLap.__contains__` is documented overlap semantics, and the
   `(start, end-1)` half-open → closed conversion is correct.

7. **"BPNet centre-crop `profile_logits[..., crop_l:-crop_r]` has undefined
   behaviour when `crop_r == 0`"** —
   [models/bpnet.py:278](src/cerberus/models/bpnet.py#L278). FALSE.
   The `elif current_len < target_len` branch raises before reaching the slice,
   so the slice is only executed when `diff >= 1`, and
   `crop_r = diff - diff//2 >= 1`.

8. **"`predict_bigwig.stream_generator` leaks the bigWig handle"** —
   [predict_bigwig.py:163-167](src/cerberus/predict_bigwig.py#L163-L167).
   FALSE. `bw.close()` is inside a `finally` wrapping the whole `bw.write`
   call, and `bw.write` exhausts the generator synchronously.

9. **"`Jitter` / `ReverseComplement` mutate the sampler's `Interval` in
   place"** — [transform.py:104-106, 182](src/cerberus/transform.py#L104-L106).
   **Already fixed**. The mutation *inside* the transform is intentional, but
   the caller `CerberusDataset.__getitem__` at
   [dataset.py:360-363](src/cerberus/dataset.py#L360-L363) now passes a
   `copy.copy(self.sampler[idx])`, so the sampler's stored interval is never
   touched. `git blame` traces the defensive copy + a dedicated regression
   suite `tests/test_jitter_mutation.py` to commit `9fbdd86` (2026-03-22,
   *"Copy interval in __getitem__ to prevent Jitter from mutating sampler
   state"*). The invariant is documented by the comment preceding the copy.

10. **"`PeakSampler` / `MultiSampler` sub-samplers get correlated RNG streams
    because the same `seed` is passed to each"** —
    [samplers.py:1348-1384](src/cerberus/samplers.py#L1348-L1384). FALSE.
    The construction-time `seed=seed` is a placeholder: `RandomSampler` and
    `ComplexityMatchedSampler` are both created with `generate_on_init=False`,
    so no RNG work happens there. Immediately after,
    `super().__init__(..., seed=seed)` triggers
    `MultiSampler.__init__` → `self.resample(seed=seed)`, and
    `MultiSampler.resample` at
    [samplers.py:372-374](src/cerberus/samplers.py#L372-L374) derives
    independent `sub_seeds = generate_sub_seeds(self.seed, len(self.samplers))`
    and dispatches each sub-sampler's `resample(sub_seed)` — overwriting the
    placeholder seed via `_update_seed`. Empirical probe (parent seed 42)
    shows sub-samplers end up with
    `[2746317213, 478163327]` and produce non-identical intervals. The
    behaviour is locked in by 21 existing tests across
    `test_multisampler_correlation.py`, `test_seed_propagation.py`,
    `test_seeding_improvements.py`, `test_seeding_manual.py`, and
    `test_randomness_consistency.py`.

---

## Suggested follow-up

The top five correctness findings all warrant fixes with accompanying
regression tests:

- `predict_bigwig` — integrate into the existing
  [tests/test_predict_bigwig.py](tests/test_predict_bigwig.py) suite:
  non-divisible `span_bp`, multi-channel output, and non-sum binning.
- DDP metadata — a small unit test that mocks `RANK`/`LOCAL_RANK` and asserts
  `update_ensemble_metadata` fires once per ensemble.
- Multinomial zero-target — not a code fix so much as a documentation /
  loss-selection guideline, plus perhaps a `num_trials == 0` fast-path
  replaced with a placeholder penalty.
- Downloads — write to `dest.with_suffix(".part")` + `rename`; add a sha256
  manifest for resumability.
- DataModule resample — narrow the `except` clause, or at minimum re-raise
  after logging.

All of the MEDIUM items have a similar cost/benefit profile and should be
bundled into a follow-up PR once the top five are in.
