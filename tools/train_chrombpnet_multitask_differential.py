#!/usr/bin/env python
"""Phase-2 differential fine-tuning for an existing multi-task ChromBPNet.

The ChromBPNet analogue of ``tools/train_multitask_differential_bpnet.py``:

1. Start from an already-trained :class:`cerberus.models.MultitaskChromBPNet`
   phase-1 model produced by ``tools/train_chrombpnet_multitask.py``.
2. Reload the accessibility branch as trainable.
3. Reload the shared bias branch as frozen.
4. Fine-tune with :class:`cerberus.loss.DifferentialCountLoss`, which
   supervises the pseudocount-shrunk predicted log fold-change between
   ``cond_b`` and ``cond_a`` against the inline-derived per-peak target
   log fold-change; the shrinkage pseudocount is derived from training
   data via :func:`cerberus.pseudocount.resolve_noise_floor_pseudocount`
   (override with ``--phase2-pseudocount-override``).

The resulting checkpoint keeps the same multi-condition output heads,
but its count representation is explicitly optimised for the requested
B-minus-A contrast.  For regulatory interpretation, run
``tools/export_tfmodisco_inputs.py --target-mode delta_log_counts
--target-cond-a/--target-cond-b`` against this output directory.

Pseudocount note: ``resolve_noise_floor_pseudocount`` samples the
training fold before the Lightning Trainer initialises, so under DDP
every rank repeats this work.  The result is deterministic across
ranks (fixed seed), so this is wasteful but not incorrect.  Pass
``--phase2-pseudocount-override`` to bypass when running multi-rank.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from pprint import pformat

import pytorch_lightning as pl
import torch

import cerberus
from cerberus import CerberusDataModule
from cerberus.config import (
    FreezeSpec,
    ModelConfig,
    PretrainedConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.model_ensemble import find_latest_hparams, parse_hparams_config
from cerberus.pretrained import extract_prefix
from cerberus.pseudocount import resolve_noise_floor_pseudocount
from cerberus.train import train_multi, train_single
from cerberus.utils import get_precision_kwargs

logger = logging.getLogger(__name__)


def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Return the original module if ``torch.compile`` wrapped the model."""
    return getattr(model, "_orig_mod", model)


def _freeze_accessibility_except_count_dense(model: torch.nn.Module) -> None:
    """Freeze the accessibility branch except its count head."""
    root = _unwrap_compiled(model)
    accessibility_model = getattr(root, "accessibility_model", None)
    if accessibility_model is None:
        raise AttributeError("Model does not expose an accessibility_model branch")

    for param in accessibility_model.parameters():
        param.requires_grad_(False)

    count_dense = getattr(accessibility_model, "count_dense", None)
    if count_dense is None:
        raise AttributeError("accessibility_model does not expose count_dense")
    for param in count_dense.parameters():
        param.requires_grad_(True)


class AccessibilityCountHeadOnly(pl.Callback):
    """Keep only ``accessibility_model.count_dense`` trainable in phase 2.

    This is a "freeze the whole branch **except** one head" operation, which
    ``FreezeSpec``/``ModelConfig.freeze`` deliberately does not express (it does
    static subtree/parameter freezing only — see ``docs/configuration.md`` and
    ``docs/internal/parameter_freezing_design.md`` §"out of scope v1"). Selective
    unfreezing is handled by a dedicated callback instead, re-applied at
    ``setup`` and ``on_fit_start`` so it survives module re-initialisation.
    """

    def __init__(self) -> None:
        self._logged = False

    def _apply(self, pl_module: pl.LightningModule) -> None:
        _freeze_accessibility_except_count_dense(pl_module.model)
        if not self._logged:
            trainable = [
                name
                for name, param in _unwrap_compiled(pl_module.model).named_parameters()
                if param.requires_grad
            ]
            logger.info(
                "Phase-2 count-head-only mode enabled; trainable parameters: %s",
                trainable,
            )
            self._logged = True

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str | None = None,
    ) -> None:
        if stage in (None, "fit"):
            self._apply(pl_module)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._apply(pl_module)


def _parse_devices(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        return value


def _find_phase1_fold_dir(checkpoint_dir: Path, fold: int) -> Path:
    """Locate the fold-specific subdirectory of a phase-1 training root."""
    candidate = checkpoint_dir / f"fold_{fold}"
    if candidate.is_dir():
        return candidate
    if (checkpoint_dir / "model.pt").exists():
        return checkpoint_dir
    raise FileNotFoundError(
        f"Could not find fold_{fold}/ or model.pt under phase-1 checkpoint dir: "
        f"{checkpoint_dir}"
    )


def _find_model_pt(fold_dir: Path) -> Path:
    """Resolve a fold directory to its ``model.pt`` checkpoint."""
    model_pt = fold_dir / "model.pt"
    if model_pt.exists():
        return model_pt
    candidates = sorted(fold_dir.glob("**/model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model.pt found under {fold_dir}")
    return candidates[0]


def _select_phase2_strategy(precision_kwargs: dict) -> dict:
    """Differential fine-tuning leaves profile-head params unused under DDP.

    :func:`get_precision_kwargs` returns ``ddp_find_unused_parameters_false`` by
    default; promote to ``_true`` so DDP doesn't trip the bucket-rebuild check
    on the profile heads that receive no gradient from :class:`DifferentialCountLoss`.
    Mirrors the BPNet phase-2 trainer's local helper.
    """
    strategy = precision_kwargs.get("strategy")
    if strategy == "ddp_find_unused_parameters_false":
        return {**precision_kwargs, "strategy": "ddp_find_unused_parameters_true"}
    return precision_kwargs


def _export_accessibility_checkpoints(root_dir: Path) -> None:
    """Write ``chrombpnet_wo_bias.pt`` next to every full phase-2 ``model.pt``."""
    for model_pt in sorted(root_dir.glob("**/model.pt")):
        state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)
        acc_sd = extract_prefix(state_dict, "accessibility_model")
        out_path = model_pt.with_name("chrombpnet_wo_bias.pt")
        torch.save(acc_sd, out_path)
        logger.info("Saved accessibility-only checkpoint to %s", out_path)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune an existing MultitaskChromBPNet with a differential "
            "log-count objective."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    phase1 = p.add_argument_group("Phase-1 model")
    phase1.add_argument(
        "--phase1-checkpoint-dir",
        type=Path,
        required=True,
        help="ModelEnsemble-compatible phase-1 root, e.g. models/run/single-fold.",
    )
    phase1.add_argument("--phase1-fold", type=int, default=0)
    phase1.add_argument(
        "--phase1-model",
        type=Path,
        default=None,
        help="Optional direct path to phase-1 model.pt. Defaults to fold_N/model.pt.",
    )

    data = p.add_argument_group("Data")
    data.add_argument(
        "--peaks",
        type=Path,
        default=None,
        help="Optional BED/narrowPeak for phase-2 sampling. Defaults to the "
        "phase-1 sampler_config.sampler_args['intervals_path'].",
    )
    data.add_argument(
        "--background-ratio",
        type=float,
        default=0.0,
        help="Phase-2 background ratio. Differential fine-tuning usually uses peaks only.",
    )
    data.add_argument("--cond-a-idx", type=int, default=0)
    data.add_argument("--cond-b-idx", type=int, default=1)

    out = p.add_argument_group("Output")
    out.add_argument("--output-dir", type=Path, required=True)
    out.add_argument("--multi", action="store_true")

    train = p.add_argument_group("Training")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--max-epochs", type=int, default=20)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--patience", type=int, default=7)
    train.add_argument("--optimizer", type=str, default="adam")
    train.add_argument("--scheduler-type", type=str, default="default")
    train.add_argument("--warmup-epochs", type=int, default=0)
    train.add_argument("--min-lr", type=float, default=1e-6)
    train.add_argument(
        "--phase2-pseudocount-quantile",
        type=float,
        default=0.10,
        help="Quantile of training-region per-channel counts for delta shrinkage.",
    )
    train.add_argument("--phase2-pseudocount-samples", type=int, default=2000)
    train.add_argument(
        "--phase2-pseudocount-override",
        type=float,
        default=None,
        help="Explicit pseudocount in scaled target units; bypasses the "
        "noise-floor estimate.",
    )
    train.add_argument(
        "--accessibility-count-head-only",
        action="store_true",
        help=(
            "Freeze the pretrained accessibility branch except "
            "accessibility_model.count_dense during phase-2 fine-tuning."
        ),
    )

    hw = p.add_argument_group("Hardware")
    hw.add_argument(
        "--accelerator", default="auto", choices=["auto", "gpu", "cpu", "mps"]
    )
    hw.add_argument(
        "--devices",
        default="1",
        help="Devices passed to Lightning. Default is single-GPU because the "
        "phase-2 pseudocount is sampled before Trainer setup; pass an explicit "
        "--phase2-pseudocount-override before multi-GPU DDP.",
    )
    hw.add_argument("--precision", default="full", choices=["bf16", "mps", "full"])
    hw.add_argument("--num-workers", type=int, default=8)
    hw.add_argument("--seed", type=int, default=42)
    hw.add_argument("--silent", action="store_true")

    return p.parse_args()


def main() -> None:
    cerberus.setup_logging()
    args = get_args()

    phase1_root = args.phase1_checkpoint_dir.resolve()
    phase1_fold_dir = _find_phase1_fold_dir(phase1_root, args.phase1_fold)
    phase1_model = (
        args.phase1_model.resolve()
        if args.phase1_model is not None
        else _find_model_pt(phase1_fold_dir)
    )
    if not phase1_model.exists():
        raise FileNotFoundError(f"Phase-1 model not found: {phase1_model}")

    hparams = find_latest_hparams(phase1_fold_dir)
    cfg = parse_hparams_config(hparams)
    if "ChromBPNet" not in cfg.model_config_.name:
        raise ValueError(
            "This tool expects a ChromBPNet phase-1 model; "
            f"got model_config.name={cfg.model_config_.name!r}"
        )

    fold_args = dict(cfg.genome_config.fold_args)
    test_fold = int(fold_args.get("test_fold", args.phase1_fold) or args.phase1_fold)
    val_fold = int(fold_args.get("val_fold", 1) or 1)

    peaks_path = args.peaks.resolve() if args.peaks is not None else None
    if peaks_path is None:
        raw_intervals = cfg.sampler_config.sampler_args.get("intervals_path")
        if raw_intervals is None:
            raise ValueError(
                "No --peaks provided and phase-1 sampler_config has no intervals_path."
            )
        peaks_path = Path(raw_intervals).resolve()
    if not peaks_path.exists():
        raise FileNotFoundError(f"Phase-2 peaks not found: {peaks_path}")

    output_root = args.output_dir.resolve() / (
        "multi-fold" if args.multi else "single-fold"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Phase-2 data: no jitter (stable per-region differential target),
    # everything else inherited from phase 1 so loss / metrics see consistent
    # log-spaces across the two phases.
    data_config = cfg.data_config.model_copy(update={"max_jitter": 0})
    sampler_config = SamplerConfig(
        sampler_type="peak",
        padded_size=data_config.input_len,
        sampler_args={
            "intervals_path": str(peaks_path),
            "background_ratio": args.background_ratio,
        },
    )

    if args.phase2_pseudocount_override is not None:
        phase2_pseudocount = args.phase2_pseudocount_override
        logger.info(
            "Using explicit phase-2 pseudocount %.4f (scaled units)",
            phase2_pseudocount,
        )
    else:
        # See module docstring: throwaway datamodule for the noise-floor
        # estimate, deliberately separate from the training datamodule so no
        # extra setup state leaks into ``train_single``.
        logger.info(
            "Deriving noise-floor pseudocount from training data "
            "(quantile=%.2f, n_samples=%d, seed=%s)",
            args.phase2_pseudocount_quantile,
            args.phase2_pseudocount_samples,
            args.seed,
        )
        quantile_dm = CerberusDataModule(
            genome_config=cfg.genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            test_fold=test_fold,
            val_fold=val_fold,
            seed=args.seed,
        )
        quantile_dm.prepare_data()
        quantile_dm.setup()
        phase2_pseudocount = resolve_noise_floor_pseudocount(
            quantile_dm,
            quantile=args.phase2_pseudocount_quantile,
            n_samples=args.phase2_pseudocount_samples,
            seed=args.seed,
        )
        del quantile_dm

    # Phase-2 ModelConfig: same architecture as phase 1, swap loss + metrics
    # to the differential pair, freeze bias_model via FreezeSpec (the
    # PretrainedConfig.freeze field was removed in the marcin freeze refactor).
    model_name = f"{cfg.model_config_.name}_Differential"
    if args.accessibility_count_head_only:
        model_name = f"{model_name}_AccessibilityCountHeadOnly"

    model_config = ModelConfig(
        name=model_name,
        model_cls=cfg.model_config_.model_cls,
        loss_cls="cerberus.loss.DifferentialCountLoss",
        loss_args={
            "cond_a_idx": args.cond_a_idx,
            "cond_b_idx": args.cond_b_idx,
            "delta_count_pseudocount": phase2_pseudocount,
        },
        metrics_cls="cerberus.models.bpnet.DifferentialBPNetMetricCollection",
        metrics_args={
            "cond_a_idx": args.cond_a_idx,
            "cond_b_idx": args.cond_b_idx,
        },
        model_args=dict(cfg.model_config_.model_args),
        pretrained=[
            PretrainedConfig(
                weights_path=str(phase1_model),
                source="accessibility_model",
                target="accessibility_model",
            ),
            PretrainedConfig(
                weights_path=str(phase1_model),
                source="bias_model",
                target="bias_model",
            ),
        ],
        freeze=[FreezeSpec(pattern="bias_model", eval_mode=True)],
        count_pseudocount=phase2_pseudocount,
    )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        optimizer=args.optimizer,
        filter_bias_and_bn=True,
        reload_dataloaders_every_n_epochs=0,
        scheduler_type=args.scheduler_type,
        scheduler_args={
            "num_epochs": args.max_epochs,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
        },
        adam_eps=1e-7,
        gradient_clip_val=None,
    )

    devices = _parse_devices(args.devices)
    accelerator = args.accelerator
    if accelerator == "auto" and torch.backends.mps.is_available():
        accelerator = "mps"
    precision_kwargs = get_precision_kwargs(args.precision, accelerator, devices)
    precision_kwargs = _select_phase2_strategy(precision_kwargs)

    logger.info("Phase-1 hparams: %s", hparams)
    logger.info("Phase-1 model: %s", phase1_model)
    logger.info("Phase-2 peaks: %s", peaks_path)
    logger.info("Phase-2 output: %s", output_root)
    logger.info("Data Config:\n%s", pformat(data_config))
    logger.info("Sampler Config:\n%s", pformat(sampler_config))
    logger.info("Train Config:\n%s", pformat(train_config))
    logger.info("Model Config:\n%s", pformat(model_config))
    logger.info("Precision and Hardware Args:\n%s", pformat(precision_kwargs))

    callbacks = []
    if args.accessibility_count_head_only:
        callbacks.append(AccessibilityCountHeadOnly())

    train_fn = train_multi if args.multi else train_single
    train_fn(
        genome_config=cfg.genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        model_config=model_config,
        train_config=train_config,
        test_fold=test_fold,
        val_fold=val_fold,
        num_workers=args.num_workers,
        in_memory=False,
        root_dir=str(output_root),
        enable_checkpointing=True,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_batch_size=args.batch_size * 4,
        enable_progress_bar=not args.silent,
        seed=args.seed,
        **precision_kwargs,
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        _export_accessibility_checkpoints(output_root)
        logger.info("Phase-2 differential ChromBPNet complete: %s", output_root)


if __name__ == "__main__":
    main()
