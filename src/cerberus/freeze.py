"""Declarative parameter freezing for cerberus models.

Applies :class:`cerberus.config.FreezeSpec` rules to a model after
instantiation: ``requires_grad_(False)`` on matched parameters plus
``.eval()`` on matched module roots (to stop Dropout / BatchNorm from
firing or drifting inside a frozen subtree).

Pattern semantics are exact paths — a pattern either matches a
module name in ``named_modules()`` (freeze the subtree + optional
``.eval()``) or a parameter name in ``named_parameters()`` (freeze
one parameter).  Path matching runs on the uncompiled root so user
patterns are always written in the natural vocabulary (``"bias_model"``,
``"iconv"``) and never see ``torch.compile``'s ``_orig_mod.`` prefix.

Preservation of ``.eval()`` across the training loop relies on
PyTorch Lightning ≥ 2.2 (``_ModuleMode.capture/restore`` in
validation transitions; no root-level ``train()`` at epoch
boundaries) — pinned in ``pyproject.toml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from cerberus.config import FreezeSpec
from cerberus.pretrained import _unwrap_compiled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FreezeReport:
    """Summary of what :func:`apply_freeze` did."""

    frozen_param_count: int
    per_pattern: dict[str, int] = field(default_factory=dict)
    eval_roots: list[str] = field(default_factory=list)


def _minimal_root_set(matched_modules: list[str]) -> list[str]:
    """Drop any name whose ancestor is also in the list.

    ``Module.eval()`` recurses into children, so if ``bias_model`` and
    ``bias_model.layers.3`` both match, the descendant call is redundant.
    """
    ordered = sorted(matched_modules, key=len)
    roots: list[str] = []
    for name in ordered:
        if any(name == r or name.startswith(r + ".") for r in roots):
            continue
        roots.append(name)
    return roots


def apply_freeze(
    model: nn.Module,
    specs: list[FreezeSpec],
) -> FreezeReport:
    """Apply declarative freeze rules to ``model``.

    One-shot call at training startup. Mutations happen on the
    uncompiled root; the compile wrapper's ``__getattr__`` delegates
    so the underlying module state is what changes.

    Every :class:`FreezeSpec` must match at least one module or
    parameter — typos raise rather than silently freezing nothing.
    """
    if not specs:
        return FreezeReport(frozen_param_count=0)

    root = _unwrap_compiled(model)
    param_table = dict(root.named_parameters())
    module_table = dict(root.named_modules())
    # The "" entry in named_modules is the root itself. Including it
    # would turn pattern="" into a silent whole-model freeze; drop it.
    module_table.pop("", None)

    frozen_param_names: set[str] = set()
    eval_module_names: set[str] = set()
    per_pattern: dict[str, int] = {}

    for spec in specs:
        if spec.pattern.startswith("_orig_mod."):
            raise ValueError(
                f"FreezeSpec pattern {spec.pattern!r} starts with "
                "'_orig_mod.'. Patterns are matched against the "
                "uncompiled root; drop the '_orig_mod.' prefix."
            )

        if spec.pattern in module_table:
            # Module match: freeze the whole subtree.
            prefix = spec.pattern + "."
            matched_params = {
                p for p in param_table
                if p == spec.pattern or p.startswith(prefix)
            }
            if spec.eval_mode:
                eval_module_names.add(spec.pattern)
        elif spec.pattern in param_table:
            # Parameter match: freeze one parameter; eval_mode is moot.
            matched_params = {spec.pattern}
        else:
            raise ValueError(
                f"FreezeSpec(pattern={spec.pattern!r}) matched no "
                "module or parameter. Pattern must equal a path in "
                "named_modules() (e.g. 'bias_model', 'res_layers.0') "
                "or in named_parameters() (e.g. 'iconv.weight')."
            )

        for name in matched_params:
            param_table[name].requires_grad_(False)
        frozen_param_names |= matched_params
        per_pattern[spec.pattern] = len(matched_params)

    eval_roots = _minimal_root_set(sorted(eval_module_names))
    for name in eval_roots:
        module_table[name].eval()

    report = FreezeReport(
        frozen_param_count=len(frozen_param_names),
        per_pattern=per_pattern,
        eval_roots=eval_roots,
    )
    logger.info(
        "apply_freeze: froze %d param(s) across %d pattern(s); "
        "eval() on %d subtree root(s)",
        report.frozen_param_count,
        len(per_pattern),
        len(eval_roots),
    )
    return report


_DDP_FALSE = "ddp_find_unused_parameters_false"
_DDP_TRUE = "ddp_find_unused_parameters_true"


def maybe_promote_ddp_strategy(
    trainer_kwargs: dict[str, Any],
    report: FreezeReport,
) -> dict[str, Any]:
    """Promote DDP strategy to ``_true`` when frozen params are present.

    ``ddp_find_unused_parameters_false`` assumes every parameter
    receives a gradient on every backward pass; frozen parameters
    break that assumption and raise a bucket-rebuild error mid-run.
    Keeping ``_false`` as the default on fully-unfrozen models avoids
    the per-backward ``all_gather`` that ``_true`` adds.
    """
    if report.frozen_param_count == 0:
        return trainer_kwargs
    if trainer_kwargs.get("strategy") != _DDP_FALSE:
        return trainer_kwargs
    logger.info(
        "Promoting Trainer strategy %s -> %s because %d parameter(s) are frozen",
        _DDP_FALSE,
        _DDP_TRUE,
        report.frozen_param_count,
    )
    return {**trainer_kwargs, "strategy": _DDP_TRUE}
