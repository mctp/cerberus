# Parameter Freezing Design

**Date:** 2026-04-18
**Status:** Proposal. Not implemented.
**Related:**
- `docs/internal/differential_workflow_redesign.md` — Phase 2 of the
  differential workflow is the primary in-tree consumer.
- `docs/internal/bias_factorized_models.md` — Dalmatian's
  `--pretrained-bias --freeze-bias` is the only shipped caller of
  the existing `PretrainedConfig.freeze` surface.

---

> **TL;DR.** Cerberus has exactly one freezing mechanism today —
> `PretrainedConfig.freeze` — and it is tied to weight loading,
> offers only subtree-wide granularity, does not pair
> `requires_grad=False` with `.eval()`, and does not coordinate
> with the DDP strategy selector. This doc maps the issues, then
> proposes a generic `ModelConfig.freeze: list[FreezeSpec]`
> surface with pattern-based selection, eval-mode pairing, and
> automatic DDP strategy promotion. Every external-library claim
> in §2–5 has been verified against the installed versions
> (PyTorch, `pytorch_lightning==2.6.0`, `timm==1.0.22`).

---

## 1. Today's freezing surface: `PretrainedConfig.freeze`

### 1.1 What it is

One `bool` field on `PretrainedConfig`
([src/cerberus/config.py:154-171](src/cerberus/config.py#L154-L171)):

```python
class PretrainedConfig(BaseModel):
    weights_path: str
    source:       str | None   # prefix to extract from the source state dict
    target:       str | None   # named child to load into; None = whole model
    freeze:       bool
```

The entire effect is three lines in `load_pretrained_weights`
([src/cerberus/pretrained.py:112-114](src/cerberus/pretrained.py#L112-L114)):

```python
if freeze:
    for p in target.parameters():
        p.requires_grad_(False)
```

`target` is either the whole model (`target_name is None`) or a
named child resolved via `getattr(root, target_name)`. So `freeze`
means "`requires_grad_(False)` on every parameter of the loaded
subtree, nothing else."

### 1.2 Where it's actually used

**Callers with `freeze=True` wired to a CLI flag:**

- [tools/train_dalmatian.py:424](tools/train_dalmatian.py#L424) and
  [tools/train_dalmatian_multitask.py:474](tools/train_dalmatian_multitask.py#L474)
  — both expose `--freeze-bias`, which loads `target="bias_model"`
  and optionally freezes it.

**Callers that hard-code `freeze=False`** (exposing `--pretrained`
for whole-model warm-starting only):

- `tools/train_bpnet.py`, `train_pomeranian.py`, `train_asap.py`,
  `train_gopher.py`, `train_biasnet.py`,
  `train_multitask_differential_bpnet.py`.

The only workflow on `main` that flips `freeze=True` is Dalmatian
with a pretrained BiasNet bias head.

### 1.3 Why the differential Phase 2 workflow didn't use it

The Option B redesign
([docs/internal/differential_workflow_redesign.md](docs/internal/differential_workflow_redesign.md))
didn't ship `PretrainedConfig(target="trunk", freeze=True)` because:

- `PretrainedConfig.freeze` cannot freeze something that isn't being
  loaded (Phase 2 wants to freeze trunk + profile heads *after*
  loading the full Phase 1 checkpoint into a fresh `MultitaskBPNet`).
- No granularity inside `target`: can't freeze `res_layers.0..4` but
  not `res_layers.5..7`.
- No interaction with DDP strategy (§4).

Phase 2 instead relies on `DifferentialCountLoss` only touching the
count heads (so the profile heads get zero gradient) plus
`ddp_find_unused_parameters_true` via `_select_phase2_strategy`.
That's **unfrozen-but-quiet**, not real freezing.

---

## 2. The `.eval()` / `.train()` issue

The core correctness gap in `PretrainedConfig.freeze`. Three
orthogonal controls that PyTorch exposes over autograd / training
behavior:

| Control | Scope | Set by | Effect |
|---|---|---|---|
| `Parameter.requires_grad` | per-parameter | `.requires_grad_(False)` | Autograd skips gradient accumulation for this tensor |
| `Module.training` | per-module (and descendants) | `.train(mode)` / `.eval()` | BatchNorm: update running stats vs. use them / Dropout: active vs. identity |
| `torch.no_grad()` | call-site context | context manager | Autograd graph not built inside the context at all |

`PretrainedConfig.freeze=True` flips only the first. Verified via
`torch.nn.Module.train` source (PyTorch 2.6+):

```python
def train(self, mode: bool = True) -> Self:
    self.training = mode
    for module in self.children():
        module.train(mode)
    return self
# .eval() is equivalent to .train(False).
```

`Module.training` is a plain Python `bool` attribute. `BatchNorm*`
consults it to decide whether to update `running_mean` /
`running_var`; `Dropout` consults it to decide whether to fire.
Verified empirically:

| Module | Behavior in `training=True` | Behavior in `training=False` |
|---|---|---|
| `nn.BatchNorm1d` | `running_mean` / `running_var` update every forward pass; forward uses batch stats | Running stats frozen; forward uses them |
| `nn.Dropout(p=0.5)` | Fires with probability `p`; scales active outputs by `1/(1-p)` | Identity |
| `nn.RMSNorm` | Stateless (no running buffers); identical in both modes | — |
| `nn.utils.parametrizations.weight_norm` | Stateless; identical in both modes | — |

Because `PretrainedConfig.freeze=True` never calls `.eval()`, any
BatchNorm inside the loaded subtree keeps drifting its running
statistics and any Dropout keeps firing — the "frozen" subtree is
still stateful / stochastic.

### 2.1 Per-architecture exposure on `main`

Enumerated by walking `model.modules()` on each default
architecture:

| Model | BN? | Dropout? | Stateless norms | `freeze` without `.eval()` safe? |
|---|---|---|---|---|
| `BPNet` (plain) | no | no | — | **yes — accidentally** |
| `BPNet` (`weight_norm=True`, stable mode) | no | no | `_WeightNorm` parametrization | **yes** |
| `MultitaskBPNet` | no | no | — | **yes — accidentally** |
| `BiasNet` | no | **yes** (5 × `Dropout(p=0.1)`) | — | **no — Dropout still fires** |
| `Pomeranian` | no | yes | `RMSNorm` | **no — Dropout still fires** |
| `ConvNeXtDCNN` (ASAP) | **yes (`BatchNorm1d`)** | yes | `RMSNorm` | **no — BN running stats drift** |

The shipped `--freeze-bias` caller (Dalmatian with pretrained
`BiasNet`) hits the Dropout row. Measured impact: two forward
passes through a "frozen" `bias_model` with different RNG produce
logits differing by up to 0.257 in absolute magnitude; only
explicit `bias_model.eval()` brings the difference to zero. The
signal branch (Pomeranian) is trained against a stochastic bias
signal and then, at inference time (`module.eval()`), sees a
deterministic bias — a silent train/inference distribution shift.

Severity is mild because BiasNet has no BatchNorm and dropout rates
are low. Any future BN-containing bias model would turn the same
class of bug into a correctness problem.

### 2.2 What "freezing" should mean

After freezing, the frozen submodule's output should depend only
on its inputs, not on the training step, batch contents, or other
RNG state. Concretely:

1. `requires_grad_(False)` on every `Parameter` in the subtree.
2. `.eval()` on the subtree root, so BatchNorm uses (and stops
   updating) its running stats and Dropout becomes the identity.

Wrapping the frozen subtree's forward in `torch.no_grad()` is a
performance-only optimization and complicates partial freezing
(the graph through the unfrozen branch still has to flow through
the frozen one). Leave it to explicit callers.

### 2.3 `.eval()` is per-module and strictly local

From the `Module.train` source quoted above, `parent.eval()`
recurses into `self.children()`, so the whole subtree is flipped.
There is **no upward propagation**: calling `.eval()` on a child
leaves the parent's `training` flag and sibling subtrees
unchanged.

On Dalmatian (the only shipped composite model, `bias_model` +
`signal_model` as top-level children):

```
>>> m = Dalmatian(...)
>>> m.train()                         # root + bias_model + signal_model all training
>>> m.bias_model.eval()               # eval only the bias branch
>>> m.training, m.bias_model.training, m.signal_model.training
(True, False, True)
>>> all(sub.training for sub in m.signal_model.modules())
True                                  # signal subtree untouched
>>> any(sub.training for sub in m.bias_model.modules())
False                                 # every descendant of bias_model in eval
```

So partitioning works — any subtree (whole branch, one residual
block, a single BatchNorm leaf) can be pinned without affecting
siblings.

### 2.4 PyTorch Lightning preserves submodule eval state

Relevant to the design: does the PL trainer flip `.eval()` back on
frozen subtrees? On `pytorch_lightning == 2.6.0` (the version
cerberus uses), **no.** This was a deliberate change in Lightning
2.2.0 (PR [#18951](https://github.com/Lightning-AI/pytorch-lightning/pull/18951),
released [2024-02](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.0)).
Release-note bullets verbatim:

- *"The `Trainer.fit()` loop no longer calls
  `LightningModule.train()` at the start; it now preserves the
  user's configuration of frozen layers."*
- *"The Trainer now restores the training mode set through
  `.train()` or `.eval()` on a submodule-level when switching from
  validation to training."*
- *"The `LightningModule.on_{validation,test,predict}_model_{eval,train}`
  now only get called if they are overridden by the user."*

Verified three ways:

1. **Fit loop source.** `pytorch_lightning/loops/fit_loop.py` in
   the installed 2.6.0 has **no** `lightning_module.train(...)`
   call anywhere (grepped over the full file — only
   `training_step` appears).
2. **Validation-boundary source.** `evaluation_loop.py:341-352`
   uses a new `_ModuleMode` helper
   ([`pytorch_lightning/utilities/model_helpers.py`](pytorch_lightning/utilities/model_helpers.py)):

   ```python
   class _ModuleMode:
       def capture(self, module: nn.Module) -> None:
           self.mode.clear()
           for name, mod in module.named_modules():
               self.mode[name] = mod.training

       def restore(self, module: nn.Module) -> None:
           for name, mod in module.named_modules():
               mod.training = self.mode[name]   # per-submodule restore
   ```

   The validation loop calls `capture(lightning_module)` before
   switching to eval and `restore(lightning_module)` after — a
   per-submodule restore, not a blanket `model.train()`.
3. **Live test.** A 2-epoch `trainer.fit` with
   `bias_model.eval()` set before `fit`: the recorded
   `bias_model.training` flag is `False` at every `training_step`
   in both epochs and at every `validation_step`. PL emits a
   `PossibleUserWarning` at fit start (added in PL 2.5.5 via
   [#21146](https://github.com/Lightning-AI/pytorch-lightning/issues/21146)):
   `"Found 3 module(s) in eval mode at the start of training. This
   may lead to unexpected behavior during training. If this is
   intentional, you can ignore this warning."`

**Implication for the design:** a one-shot `.eval()` at startup is
sufficient. No per-epoch callback is required.

### 2.5 Gotchas to document

- **User code that calls `self.train()` on the root.** Would
  cascade through children and undo `.eval()`. `CerberusModule`
  does not override or call `train()` / `eval()`, so this is
  safe today.
- **Overriding `on_validation_model_{eval,train}`.** Re-activates
  the pre-2.2 hook path (PL falls back to the user hook instead
  of `_module_mode.restore`). Cerberus doesn't override these;
  if it ever does, the override must re-apply `.eval()` on the
  frozen subtree explicitly.
- **PL version floor.** The no-callback design assumes PL ≥ 2.2.
  Add a version floor in `pyproject.toml` when the feature lands;
  the 2.6.0 already installed satisfies it.

---

## 3. Naming conventions across cerberus models

Pattern-based freezing depends on predictable `named_parameters` /
`named_modules` paths. Cerberus models do **not** share a naming
convention; each architecture is documented separately. Top-level
child names from `m.named_children()`:

| Model | Top-level children |
|---|---|
| `BPNet` / `MultitaskBPNet` | `iconv`, `iconv_act`, `res_layers`, `final_tower_relu`, `profile_conv`, `count_dense` |
| `Pomeranian` | `stem`, `layers`, `profile_pointwise`, `profile_act`, `profile_spatial`, `count_mlp` |
| `BiasNet` | `stem`, `layers`, `profile_spatial`, `count_mlp` |
| `Dalmatian` | `bias_model` (BiasNet), `signal_model` (Pomeranian) |
| `ConvNeXtDCNN` (ASAP) | `init_conv`, `init_pool`, `core` |

### 3.1 Implications for the proposal

- **Patterns are per-architecture, not per-library.** `"iconv*"`
  works for BPNet, `"stem.*"` works for Pomeranian, `"bias_model.*"`
  works for Dalmatian. The library's job is to match the pattern
  and report what matched, not to invent cross-architecture
  aliases.
- **Document the named children in `docs/models.md`** so users can
  write freeze patterns without reading the model source.
- **Reject unmatched patterns.** A typo like `"iconvv*"` must not
  silently match zero parameters — raise at `apply_freeze` time.
- **Don't codify a shared vocabulary in this proposal.** A
  `trunk` / `head` alias system is a separate refactor with its
  own design space.

The one cross-architecture convention in play is that **composite
models expose sub-models by name** (Dalmatian's `bias_model` /
`signal_model`). Freeze patterns on composites prefix naturally:
`"bias_model.*"`, `"signal_model.stem.*"`.

---

## 4. DDP strategy coupling

`get_precision_kwargs`
([src/cerberus/utils.py:77-137](src/cerberus/utils.py#L77-L137))
picks a Lightning trainer strategy from `cerberus.utils` at
tool-call time. For multi-GPU bf16 (`devices > 1`), it returns
`strategy="ddp_find_unused_parameters_false"` — the fast path, which
assumes every parameter receives a gradient on every backward pass.
Violating that assumption raises a DDP bucket-rebuild error mid-run.

`get_precision_kwargs` does not see `ModelConfig` and does not know
whether any parameter is frozen. Each tool currently reconciles the
two manually: BiasNet passes
`use_ddp_find_unused_parameters_false=False` (strategy becomes
`"auto"`), and the differential tool's `_select_phase2_strategy`
explicitly promotes `_false` → `_true`.

**The freezing design owns this reconciliation.** §6.3 introduces
`maybe_promote_ddp_strategy(trainer_kwargs, report)` — if any
parameter was frozen and the current strategy is
`ddp_find_unused_parameters_false`, it is promoted to
`ddp_find_unused_parameters_true`. `_select_phase2_strategy` is
deleted as a consequence.

Keeping `_false` as the default (instead of always using `_true`)
remains correct: `find_unused_parameters=True` adds an
`all_gather` over parameter usage on every backward pass, which is
wasted bandwidth on fully-used models.

---

## 5. Optimizer behavior

### 5.1 timm filters frozen params when `weight_decay > 0`

Verified by reading
`timm.optim._optim_factory.OptimizerRegistry.create_optimizer` in
the installed `timm==1.0.22`:

```python
if isinstance(model_or_params, nn.Module):
    if param_group_fn:                        # cerberus: None
        ...
    elif layer_decay is not None:             # cerberus: None
        params = param_groups_layer_decay(model, ...)
    elif weight_decay and weight_decay_exclude_1d:
        # hit when TrainConfig.weight_decay > 0 and filter_bias_and_bn=True
        params = param_groups_weight_decay(model, ...)
    else:
        params = model_or_params.parameters()
```

Inside `param_groups_weight_decay`
([`timm/optim/_param_groups.py`](timm/optim/_param_groups.py)):

```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue                              # ← frozen params skipped
    ...
```

So on the weight-decay-splitting path, frozen parameters are
dropped before the optimizer is constructed. They never appear in
any param group.

### 5.2 The fallthrough path (`weight_decay == 0`)

When `weight_decay == 0` (or `filter_bias_and_bn=False`), timm
falls through to `model.parameters()` unfiltered — frozen
parameters appear in the optimizer's param-group list. **This
does not waste meaningful memory**, because PyTorch optimizers
(Adam, AdamW, SGD, RMSprop) allocate per-parameter state
**lazily** on the first step where the parameter has a gradient.
Verified by checking `len(optimizer.state)` after a single step
on a model with 4 params (2 frozen, 2 live):

| Optimizer | Params in param group | Params with state after step |
|---|---|---|
| Adam | 4 | 2 |
| AdamW | 4 | 2 |
| SGD (momentum=0.9) | 4 | 2 |
| RMSprop | 4 | 2 |

Frozen parameters have `grad=None` after backward, so the
optimizer's per-step loop skips them and never allocates
`exp_avg` / `exp_avg_sq` / `momentum_buffer` tensors for them.
The only overhead is a few pointers in the Python param-group
list. **No action required in the freezing design.**

### 5.3 Parameter groups & scheduler interaction

`timm.scheduler.create_scheduler_v2` operates on the constructed
optimizer's existing groups. Frozen-param filtering (or lack
thereof) on the optimizer side does not change scheduler
behavior. `CerberusModule.configure_optimizers` needs no change.

---

## 6. Proposal: `ModelConfig.freeze: list[FreezeSpec]`

### 6.1 API

```python
class FreezeSpec(BaseModel):
    """Declarative freeze rule applied after model instantiation."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern:   str                # fnmatch glob on named_parameters + named_modules
    eval_mode: bool = True        # .eval() matched submodule roots
    # Out of scope v1: schedule, freeze_epoch, unfreeze_epoch, layer_lr_scale.

class ModelConfig(BaseModel):
    ...
    pretrained: list[PretrainedConfig] = Field(default_factory=list)
    freeze:     list[FreezeSpec]       = Field(default_factory=list)   # NEW
    count_pseudocount: float = ...
```

### 6.2 Pattern semantics

Matching uses `fnmatch.fnmatchcase` against (a) every name in
`model.named_modules()` and (b) every name in
`model.named_parameters()`. The semantics that matter:

- **`*` matches any characters including `.`.** Verified:
  `fnmatch("bias_model.layers.3.weight", "bias_model.*")` → `True`.
- **`.` in a pattern is a literal.** `"iconv.*"` matches
  `"iconv.weight"` but **not** `"iconv_act"` — the pattern
  requires a literal dot after `iconv`. This is the intended way
  to scope to a subtree.
- **`"iconv*"` (no dot) is footgun-prone.** It matches
  `"iconv.weight"` *and* `"iconv_act"`. Document in
  `docs/configuration.md` that users should prefer `"iconv.*"` for
  subtree patterns.
- **Module match → freeze all descendant parameters and call
  `.eval()` on that subtree root** (when `eval_mode=True`).
- **Parameter match → freeze that parameter only.** `.eval()` is
  not invoked for parameter-only matches.
- **Zero-match is an error.** Every `FreezeSpec` must match at
  least one module or parameter.
- **Overlapping matches are idempotent.** `requires_grad_(False)`
  and `.eval()` are both safe to call more than once on the same
  target.

### 6.3 Pipeline integration

`_train`
([src/cerberus/train.py:286-290](src/cerberus/train.py#L286-L290))
gains two steps:

```
 4. instantiate(model_config, ...)                       # unchanged
 5. load_pretrained_weights(module.model, ...)           # unchanged
+5a. report = apply_freeze(module.model, model_config.freeze)
+5b. trainer_kwargs = maybe_promote_ddp_strategy(trainer_kwargs, report)
 6. trainer.fit(module, datamodule=datamodule)           # unchanged
```

No Lightning callback is needed (§2.4).

New symbols in a new `src/cerberus/freeze.py` (sibling of
`pretrained.py`):

- `@dataclass FreezeReport` — tally of frozen parameters per
  pattern, frozen parameter total, unmatched patterns (empty if
  OK), subtree roots put in eval mode.
- `apply_freeze(model, specs) -> FreezeReport` — one-shot call at
  startup. Calls `_unwrap_compiled(model)` first (§7.2), walks
  `named_parameters()` / `named_modules()` of the unwrapped root
  to find matches, sets `requires_grad_(False)` on matched
  parameters, calls `.eval()` on the **minimal root set** of
  matched modules (if `bias_model` and `bias_model.layers.3` both
  match, call `.eval()` on `bias_model` only — the descendant
  call is redundant because `.eval()` recurses), and logs a
  summary line. Raises on any zero-match spec. Asserts
  `pl.__version__ >= (2, 2)` for the no-callback invariant.
- `maybe_promote_ddp_strategy(trainer_kwargs, report)` — if any
  parameter was frozen and the current strategy is
  `ddp_find_unused_parameters_false`, returns a copy of
  `trainer_kwargs` with strategy promoted to
  `ddp_find_unused_parameters_true`. Logs at `INFO`.

### 6.4 Interaction with `PretrainedConfig.freeze`

`PretrainedConfig.freeze` stays as-is. Mark it legacy in the
docstring. `apply_freeze` runs *after* `load_pretrained_weights`,
so both surfaces compose cleanly on Dalmatian: keep
`PretrainedConfig(target="bias_model", freeze=True)` for back-compat,
or switch to `PretrainedConfig(freeze=False)` +
`FreezeSpec(pattern="bias_model.*", eval_mode=True)` for the
correct-under-Dropout behavior.

### 6.5 `hparams.yaml` round-trip

`ModelConfig.freeze` is a Pydantic list, so `model_dump(mode="json")`
serializes it into `hparams.yaml` alongside the rest of
`ModelConfig`. Reloading a checkpoint via
`CerberusModule.load_from_checkpoint` or `ModelEnsemble` reapplies
the freeze specs, reproducing the same `requires_grad` / `training`
state the training run had. `PretrainedConfig.freeze` can't do this:
`requires_grad` is metadata on `Parameter`, not stored in the state
dict.

---

## 7. Applied to the two in-tree consumers

### 7.1 Phase 2 of the differential workflow

Current Phase 2
([tools/train_multitask_differential_bpnet.py::run_phase2](tools/train_multitask_differential_bpnet.py)):

```python
model_config = ModelConfig(
    ...,
    loss_cls="cerberus.loss.DifferentialCountLoss",
    loss_args={"cond_a_idx": 0, "cond_b_idx": 1},
    pretrained=[PretrainedConfig(
        weights_path=str(phase1_ckpt), source=None, target=None, freeze=False)],
)
precision_kwargs = _select_phase2_strategy(precision_kwargs)   # manual DDP override
```

With `ModelConfig.freeze`:

```python
model_config = ModelConfig(
    ...,
    loss_cls="cerberus.loss.DifferentialCountLoss",
    loss_args={"cond_a_idx": 0, "cond_b_idx": 1},
    pretrained=[PretrainedConfig(
        weights_path=str(phase1_ckpt), source=None, target=None, freeze=False)],
    freeze=[
        FreezeSpec(pattern="iconv.*"),
        FreezeSpec(pattern="res_layers.*"),
        FreezeSpec(pattern="profile_conv.*"),
    ],
)
# _select_phase2_strategy is deleted; maybe_promote_ddp_strategy does it generically.
```

The Naqvi et al. 2025 recipe ("keep trunk + profile heads frozen,
fine-tune count heads only") moves from tool-private
gradient-shape accident into a declarative, `hparams.yaml`-logged
config.

### 7.2 Dalmatian `--freeze-bias`

Current (with the Dropout-still-fires bug from §2.1):

```python
pretrained=[PretrainedConfig(
    weights_path=args.pretrained_bias,
    source=None, target="bias_model",
    freeze=args.freeze_bias,
)]
```

With `ModelConfig.freeze`:

```python
pretrained=[PretrainedConfig(
    weights_path=args.pretrained_bias,
    source=None, target="bias_model",
    freeze=False,
)]
freeze=[
    FreezeSpec(pattern="bias_model.*", eval_mode=True),
]
```

Behavior change: same parameter-level freeze, plus
`bias_model.eval()` is applied once at startup. PL 2.2+ preserves
it through training and validation (§2.4), so Dropout becomes the
identity in the frozen branch. Bias output is deterministic in
training, matching inference.

The CLI flag `--freeze-bias` stays; the tool translates it to the
new surface internally.

---

## 8. Edge cases

### 8.1 `torch.compile` interaction

`torch.compile(m)` returns an `OptimizedModule` wrapper. Verified
effects on naming:

| Surface | Uncompiled | Compiled |
|---|---|---|
| `named_children()` top | `bias_model`, `signal_model` | `_orig_mod` |
| `named_parameters()` | `bias_model.fc.weight`, … | `_orig_mod.bias_model.fc.weight`, … |
| `state_dict()` keys | `bias_model.fc.weight`, … | `_orig_mod.bias_model.fc.weight`, … |
| `mc.bias_model is m.bias_model` | — | `True` (wrapper's `__getattr__` delegates) |
| `mc.bias_model.eval()` flips underlying | — | Yes |

Cerberus already handles `torch.compile` on both loading sides:

- `load_pretrained_weights` unwraps via
  `_unwrap_compiled(model) → getattr(model, "_orig_mod", model)`
  ([src/cerberus/pretrained.py:13-15](src/cerberus/pretrained.py#L13-L15)).
- `_save_model_pt` strips the `_orig_mod.` prefix from checkpoint
  keys before writing `model.pt`
  ([src/cerberus/train.py:139-142](src/cerberus/train.py#L139-L142)).
  So on-disk `model.pt` files are uncompile-stable — consumers
  always see plain names.

**Design consequence.** `apply_freeze` calls `_unwrap_compiled`
first, then does all matching against the uncompiled root. User
patterns are written in the natural vocabulary
(`"bias_model.*"`, `"iconv.*"`) and never see the `_orig_mod.`
prefix. `apply_freeze` rejects patterns starting with
`_orig_mod.` with a user-error message directing the user to
drop that prefix.

### 8.2 `weight_norm` parametrization

`BPNet --stable` uses `nn.utils.parametrizations.weight_norm`,
which wraps a Conv1d's `weight` into a
`.parametrizations.weight[0]` submodule. The actual parameters it
exposes are `original0` and `original1`, not `weight_g` /
`weight_v`. Verified on a BPNet-stable instance:

```
iconv.bias
iconv.parametrizations.weight.original0
iconv.parametrizations.weight.original1
res_layers.0.conv.bias
res_layers.0.conv.parametrizations.weight.original0
res_layers.0.conv.parametrizations.weight.original1
...
```

A module-scoped pattern like `"iconv.*"` matches the `iconv`
module and all its descendants (including the parametrization
submodule), freezing both `original0` and `original1` together —
correct behavior. A parameter-name pattern like `"*.weight"` does
**not** match, because the actual parameter names are
`original0` / `original1`. Document this in
`docs/configuration.md` when the feature ships.

### 8.3 Other edge cases

- **`filter_bias_and_bn`.** About weight-decay exclusion for 1-D
  parameters, not freezing. No interaction.
- **Empty freeze list.** `apply_freeze(model, [])` returns a
  `FreezeReport` with `frozen_count=0` and is a no-op. DDP
  strategy is not promoted. Zero cost.
- **Frozen-but-still-forward compute.** Freezing doesn't skip the
  forward pass through the frozen subtree; the module is invoked
  normally but its output isn't consumed downstream (Phase 2
  case). Skipping the forward is an architecture-level concern
  and out of scope.
- **Partial unfreezing during training.** Static in v1. Gradual
  unfreezing (freeze trunk for N epochs, then unfreeze) would add
  `freeze_epoch` / `unfreeze_epoch` fields to `FreezeSpec` and a
  dedicated schedule callback. Optimizer state stays valid
  because PyTorch optimizers allocate lazily (§5.2).

### 8.4 Tests

New `tests/test_freeze.py` should cover:

- pattern matching (module path, parameter path, wildcards,
  `"*"`-matches-dots semantics, `iconv*` vs. `iconv.*` footgun);
- zero-match spec raises;
- a 2-epoch `trainer.fit` with a tiny LightningModule — assert
  `bias_model.training == False` at every `training_step` in both
  epochs and every `validation_step` (mirrors the live test that
  verified §2.4);
- DDP strategy promotion under `maybe_promote_ddp_strategy`;
- `apply_freeze` + `PretrainedConfig.freeze` composition (both
  apply cleanly);
- `torch.compile` wrapping — patterns like `"bias_model.*"` match
  correctly on `torch.compile(m)`; `"_orig_mod.*"` raises the
  user-error message;
- `weight_norm` parametrization — `"iconv.*"` freezes both
  `original0` and `original1` inside the parametrization wrapper.

---

## 9. Out of scope

- Removing or deprecating `PretrainedConfig.freeze`. Kept as a
  legacy shortcut.
- Cross-architecture aliases (`trunk`, `head`). Separate concern.
- Per-parameter-group LR overrides. Reuses similar pattern
  matching but addresses a different problem.
- Skipping the forward pass through frozen subtrees.
  Architecture-level concern.
- Complex freeze schedules (gradual unfreezing, layer-wise
  warmup). Future work.

---

## 10. Summary

| Concern | Today (`PretrainedConfig.freeze`) | Proposal (`ModelConfig.freeze`) |
|---|---|---|
| Scope | Loaded subtree, whole-or-nothing | Any subset via glob patterns |
| Ties to weight loading | Yes | No — independent |
| `.eval()` pairing | No — BN / Dropout drift | Yes — one-shot at startup, preserved by PL 2.2+ |
| Epoch re-`train()` handling | Not applicable | Not needed — PL 2.2+ preserves per-submodule state |
| Optimizer state for frozen | No issue (timm filters on `wd>0`; optimizers allocate lazily otherwise) | No issue |
| DDP strategy promotion | Manual per tool | Auto via `maybe_promote_ddp_strategy` |
| `torch.compile` interaction | Handled via `_orig_mod` unwrap | Same pattern; user patterns uncompile-stable |
| `hparams.yaml` round-trip | Freeze state lost on reload | Preserved as part of `ModelConfig` |
| Sanity on typos | Silent zero-match | Raises at `apply_freeze` time |
| Partial / patterned freezing | Not expressible | Native |

Rough implementation footprint:

- `src/cerberus/freeze.py`: ~90 lines (`FreezeSpec`, `apply_freeze`,
  `maybe_promote_ddp_strategy`, `FreezeReport`).
- `src/cerberus/config.py`: +1 field on `ModelConfig`.
- `src/cerberus/train.py::_train`: 2 new lines.
- `tools/train_multitask_differential_bpnet.py`: delete
  `_select_phase2_strategy`; Phase 2 gets a
  `freeze=[...]` list.
- `tests/test_freeze.py`: ~180 lines (§8.4).
- `docs/configuration.md`: `freeze` section.
- `docs/models.md`: per-architecture "Named submodules" subsection.
- `pyproject.toml`: pin `pytorch-lightning>=2.2`.

Net: small library addition, clear upgrade path for the two
in-tree freezing consumers (differential Phase 2 and Dalmatian),
no behavior change for any caller that doesn't opt in.

---

## Appendix: Library-version cross-check

Every external-library claim in the doc has been verified against
the installed versions. This appendix records the release-history
scan that confirms the behavior hasn't shifted in versions between
the semantic change and the installed pin.

### PyTorch Lightning

**Installed:** `pytorch_lightning == 2.6.0`. **Claims rely on the
2.2.0 `_ModuleMode` semantics** (§2.4).

Release-history scan (pages 1–4 of
[github.com/Lightning-AI/pytorch-lightning/releases](https://github.com/Lightning-AI/pytorch-lightning/releases),
covering 2.0.0 → 2.6.1):

- **v2.2.0** (2024-02): `_ModuleMode.capture/restore` landed via
  [PR #18951](https://github.com/Lightning-AI/pytorch-lightning/pull/18951).
  Release-note quote: *"the Trainer now captures the mode of every
  submodule before switching to validation, and restores the mode
  the modules were in when validation ends."*
- **v2.5.5** (2025-09): added `PossibleUserWarning` raised when
  modules are in eval mode at training start
  ([#21146](https://github.com/Lightning-AI/pytorch-lightning/issues/21146))
  — the exact INFO log observed in the live test (§2.4.3).
  Additive; does not change the preservation semantics.
- **v2.2.1 → v2.6.1**: no release notes reference
  `LightningModule.train()`, `on_validation_model_{eval,train}`,
  `_ModuleMode`, `fit_loop`, or frozen layers beyond the two above.

Conclusion: preservation semantics have been stable from 2.2.0
through 2.6.1. Pin `pytorch-lightning>=2.2` in `pyproject.toml`
when the feature lands.

### timm

**Installed:** `timm == 1.0.22`. **Claims rely on the
`OptimizerRegistry` + `param_groups_weight_decay` layout** (§5).

Release-history scan (pages 1–4 of
[github.com/huggingface/pytorch-image-models/releases](https://github.com/huggingface/pytorch-image-models/releases),
covering v0.6.12 → v1.0.26):

- **v1.0.12** (2024-12): optimizer factory refactor. Deprecated
  `optim.optim_factory`; moved to `optim/_optim_factory.py` and
  `optim/_param_groups.py`. Added `OptimizerRegistry`, `OptimInfo`,
  `list_optimizers`, `get_optimizer_class`, `get_optimizer_info`.
  This is the layout the design inspects.
- **v1.0.13 → v1.0.26**: additions only — new optimizers (Kron,
  Muon family, Big-Vision Adafactor, corrected weight decay option,
  Lion, etc.). No release note modifies
  `param_groups_weight_decay` behavior or the `requires_grad` skip
  inside it.

Conclusion: `param_groups_weight_decay` skipping
`requires_grad=False` params (§5.1) has been stable since the
v1.0.12 refactor. No version pin beyond `timm>=1.0.12` is needed,
and the installed 1.0.22 is well within the stable range.

### PyTorch

Core `nn.Module.train` / `nn.Module.eval` recursion and the
lazy state-allocation pattern of built-in optimizers (Adam, AdamW,
SGD, RMSprop) are long-standing behavior and not tracked here.
Verified empirically against the installed PyTorch in §2 and §5.2.
