#!/usr/bin/env python
"""
ChromBPNet stage-2 training tool.

Loads a pre-trained small BPNet bias branch (trained in stage 1 via
``tools/train_chrombpnet_bias.py``), freezes it via
``ModelConfig.freeze``, optionally calibrates the bias-count scalar
offset on non-peak regions, and trains the full
:class:`cerberus.models.ChromBPNet` through the standard
``train_single`` / ``train_multi`` pipeline.

After training, exports an accessibility-only ``chrombpnet_wo_bias.pt``
checkpoint next to each ``model.pt`` (matches the reference
``chrombpnet-pytorch`` export naming), plots training curves, and
runs a held-out prediction evaluation on the test fold.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from pprint import pformat

import torch

import cerberus
from cerberus.config import (
    DataConfig,
    FreezeSpec,
    ModelConfig,
    PretrainedConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.datamodule import CerberusDataModule
from cerberus.download import download_human_reference
from cerberus.genome import create_genome_config
from cerberus.models.bpnet import BPNet
from cerberus.models.chrombpnet import estimate_bias_logcount_offset
from cerberus.pretrained import extract_prefix, load_pretrained_weights
from cerberus.train import train_multi, train_single
from cerberus.utils import get_precision_kwargs, resolve_device

from _pseudocount_cli import (
    add_pseudocount_cli_args,
    resolve_count_pseudocount_from_args,
)


def _parse_alpha(value: str) -> float | str:
    if value == "adaptive":
        return "adaptive"
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--alpha must be a float or 'adaptive', got {value!r}"
        ) from exc


def _export_accessibility_checkpoints(root_dir: Path) -> None:
    """Export accessibility-only state dicts next to each full model.pt."""
    for model_pt in sorted(root_dir.glob("**/model.pt")):
        state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)
        acc_sd = extract_prefix(state_dict, "accessibility_model")
        out_path = model_pt.with_name("chrombpnet_wo_bias.pt")
        torch.save(acc_sd, out_path)
        logging.info("Saved accessibility-only checkpoint to %s", out_path)


def _plot_training_curves(root_dir: Path) -> None:
    """Plot every Lightning CSV metrics file under a training root."""
    try:
        from plot_training_results import plot_metrics
    except Exception as exc:  # pragma: no cover - depends on optional plotting deps
        logging.warning("Skipping training plots: %s", exc)
        return

    metrics_files = sorted(root_dir.rglob("metrics.csv"))
    if not metrics_files:
        logging.warning("No metrics.csv files found under %s; skipping plots.", root_dir)
        return

    for metrics_path in metrics_files:
        logging.info("Plotting training metrics from %s", metrics_path)
        plot_metrics(metrics_path, metrics_path.parent)


def _run_prediction_evaluation(args: argparse.Namespace, model_root: Path) -> None:
    """Run held-out prediction/export evaluation for a trained model root."""
    eval_dir = model_root / "predict" / "test"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / "predictions.tsv.gz"

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("export_predictions.py")),
        str(model_root),
        args.peaks,
        args.bigwig,
        "--output",
        str(output_path),
        "--batch_size",
        str(args.batch_size * 4),
        "--eval-split",
        "test",
        "--seed",
        "1234",
        "--use_folds",
        "test",
        "--include-background",
    ]

    logging.info(
        "Running prediction-time evaluation; outputs will be written to %s", eval_dir
    )
    subprocess.run(cmd, check=True)


def _estimate_bias_offset(
    args: argparse.Namespace,
    genome_config,
    data_config: DataConfig,
    count_pseudocount_scaled: float,
) -> float:
    """Estimate the ChromBPNet bias-count scalar offset on non-peak training data.

    Constructs a standalone BPNet matching the stage-1 bias architecture,
    loads the pretrained bias weights into it, samples non-peak regions via
    a temporary CerberusDataModule, and returns the mean log-count residual
    suitable for ``ChromBPNet.bias_logcount_offset``.

    Two known limitations for future refactor:

    - Under DDP every rank repeats this estimation independently.  The
      result is deterministic across ranks (seeded sampler) so consistency
      is preserved -- it is wasteful but correct.  A future refactor could
      gate the estimation to rank 0 and broadcast the scalar, but
      torch.distributed is not initialised at this point in the tool.
    - The training dataloader applies jitter, so the offset is averaged
      over jittered windows rather than fixed centred windows.  Reference
      ``chrombpnet-pytorch`` behaves the same way (it iterates the same
      training loader).  Statistical efficiency only -- the mean still
      converges to the right scalar.
    """
    # TODO: ``residual_architecture`` is shared with the accessibility
    # branch (single CLI flag in stage 2).  If the user overrides it here
    # but stage-1 trained the bias with a different residual variant, the
    # subsequent ``load_pretrained_weights`` raises on shape mismatch.
    # A bias-prefixed flag (``--bias-residual-architecture``) would let
    # them diverge cleanly.
    bias_model = BPNet(
        input_len=data_config.input_len,
        output_len=data_config.output_len,
        output_bin_size=data_config.output_bin_size,
        input_channels=["A", "C", "G", "T"],
        output_channels=["signal"],
        filters=args.bias_filters,
        n_dilated_layers=args.bias_layers,
        conv_kernel_size=args.bias_conv_kernel_size,
        dil_kernel_size=args.bias_dil_kernel_size,
        profile_kernel_size=args.bias_profile_kernel_size,
        predict_total_count=True,
        activation="relu",
        weight_norm=False,
        residual_architecture=args.residual_architecture,
    )
    load_pretrained_weights(
        bias_model,
        [
            PretrainedConfig(
                weights_path=args.pretrained_bias,
                source=args.pretrained_bias_source,
                target=None,
            )
        ],
    )

    # TODO: ``padded_size`` reuses the main trainer's ``max_jitter``, which
    # forces background intervals to support the same ±jitter that peak
    # training needs.  For pure offset evaluation we don't need jitter; a
    # data_config copy with ``max_jitter=0`` would let the sampler use
    # narrower background regions and tighten the estimate.
    padded_size = data_config.input_len + 2 * data_config.max_jitter
    negative_sampler_config = SamplerConfig(
        sampler_type="negative_peak",
        padded_size=padded_size,
        sampler_args={
            "intervals_path": args.peaks,
            "background_ratio": 1.0,
        },
    )
    datamodule = CerberusDataModule(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=negative_sampler_config,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
        seed=args.seed,
    )
    datamodule.prepare_data()
    datamodule.setup(
        batch_size=args.batch_size,
        num_workers=0,
        in_memory=False,
    )

    device = resolve_device()
    logging.info(
        "Estimating bias log-count offset on non-peak training data with device=%s",
        device,
    )
    delta = estimate_bias_logcount_offset(
        bias_model,
        datamodule.train_dataloader(),
        count_pseudocount=count_pseudocount_scaled,
        device=device,
        max_batches=args.adjust_bias_max_batches,
    )
    logging.info("Estimated bias_logcount_offset=%.6f", delta)
    return delta


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ChromBPNet model with Cerberus"
    )

    # --- Input data ---
    parser.add_argument("--bigwig", type=str, required=True, help="Target BigWig")
    parser.add_argument(
        "--peaks",
        type=str,
        required=True,
        help="Peak BED/narrowPeak used for stage-2 ChromBPNet training",
    )

    # --- Genome / reference ---
    parser.add_argument("--genome", type=str, default="hg38")
    parser.add_argument("--species", type=str, default="human")
    parser.add_argument("--fasta", type=str)
    parser.add_argument("--blacklist", type=str)
    parser.add_argument("--gaps", type=str)

    # --- I/O & runtime ---
    parser.add_argument("--data-dir", type=str, default="tests/data")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--silent", action="store_true")

    # --- Cross-validation ---
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--test-fold", type=int, default=0)

    # --- Data config ---
    parser.add_argument("--input-len", type=int, default=2114)
    parser.add_argument("--output-len", type=int, default=1000)
    parser.add_argument("--jitter", type=int, default=256)
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=1.0,
        help="Non-peak to peak ratio during stage-2 training",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="Multiplicative scaling factor for targets (1.0 for raw-count pseudobulk BigWig values)",
    )

    # --- Pseudocount CLI family (shared with the other train_* tools) ---
    add_pseudocount_cli_args(parser, default_count_pseudocount=1.0)

    # --- Pretrained bias / accessibility branches ---
    parser.add_argument(
        "--pretrained-bias",
        type=str,
        required=True,
        help="Path to the stage-1 small-BPNet bias checkpoint (model.pt)",
    )
    parser.add_argument(
        "--pretrained-bias-source",
        type=str,
        default=None,
        help="Optional prefix to extract from the bias checkpoint "
        "(e.g. 'bias_model' when loading from a full ChromBPNet checkpoint)",
    )
    parser.add_argument(
        "--pretrained-accessibility",
        type=str,
        default=None,
        help="Optional BPNet checkpoint to warm-start the accessibility branch",
    )
    parser.add_argument(
        "--pretrained-accessibility-source",
        type=str,
        default=None,
        help="Optional prefix to extract from the accessibility checkpoint",
    )
    parser.add_argument(
        "--no-freeze-bias",
        action="store_true",
        help="Allow the loaded bias branch to keep training.  Default behaviour "
        "adds FreezeSpec(pattern='bias_model', eval_mode=True) to ModelConfig.freeze "
        "for ChromBPNet equivalence.",
    )
    parser.add_argument(
        "--adjust-bias-logcounts",
        action="store_true",
        help="Estimate and apply a scalar bias-branch log-count offset on non-peak "
        "regions before stage-2 training",
    )
    parser.add_argument(
        "--adjust-bias-max-batches",
        type=int,
        default=None,
        help="Optional cap on batches used during bias-count offset estimation",
    )

    # --- Accessibility branch architecture (chrombpnet-pytorch defaults) ---
    parser.add_argument("--filters", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--conv-kernel-size", type=int, default=21)
    parser.add_argument("--dil-kernel-size", type=int, default=3)
    parser.add_argument("--profile-kernel-size", type=int, default=75)

    # --- Bias branch architecture (must match stage-1 trainer) ---
    # TODO: the stage-1 trainer exposes its bias-model arch under the
    # unprefixed flags --filters / --n-layers / --conv-kernel-size / etc.
    # while stage-2 prefixes them as --bias-* (because --filters here
    # already controls the accessibility branch).  Users overriding
    # stage-1 architecture flags must remember to mirror them with the
    # --bias-* equivalents in stage 2 or the loaded bias state_dict will
    # mismatch shapes at load time.  A future refactor should unify the
    # CLI vocabulary.
    parser.add_argument("--bias-filters", type=int, default=128)
    parser.add_argument("--bias-layers", type=int, default=4)
    parser.add_argument("--bias-conv-kernel-size", type=int, default=21)
    parser.add_argument("--bias-dil-kernel-size", type=int, default=3)
    parser.add_argument("--bias-profile-kernel-size", type=int, default=75)

    parser.add_argument(
        "--residual-architecture",
        type=str,
        default="residual_post-activation_conv",
        choices=[
            "residual_post-activation_conv",
            "residual_pre-activation_conv",
            "activated_residual_pre-activation_conv",
        ],
    )

    # --- Loss ---
    parser.add_argument("--alpha", type=_parse_alpha, default="adaptive")
    parser.add_argument("--beta", type=float, default=1.0)

    # --- Optimisation ---
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler-type", type=str, default="default")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # --- Hardware ---
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu", "mps"],
    )
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument(
        "--precision",
        type=str,
        default="full",
        choices=["bf16", "mps", "full"],
    )

    return parser.parse_args()


def main() -> None:
    cerberus.setup_logging()
    logging.info("Starting ChromBPNet training tool...")
    args = get_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir).resolve()
    output_dir = output_dir / ("multi-fold" if args.multi else "single-fold")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.fasta:
        if args.genome != "hg38":
            raise ValueError(
                f"Fasta path must be provided for genome {args.genome} if not hg38"
            )
        logging.info("Downloading/Checking Human Reference (hg38)...")
        genome_files = download_human_reference(data_dir / "genome", name="hg38")
        fasta_path = genome_files["fasta"]
        blacklist_path = args.blacklist or genome_files["blacklist"]
        gaps_path = args.gaps or genome_files["gaps"]
    else:
        fasta_path = Path(args.fasta)
        blacklist_path = Path(args.blacklist) if args.blacklist else None
        gaps_path = Path(args.gaps) if args.gaps else None

    exclude_intervals: dict[str, Path] = {}
    if blacklist_path:
        exclude_intervals["blacklist"] = Path(blacklist_path)
    if gaps_path:
        exclude_intervals["gaps"] = Path(gaps_path)

    genome_config = create_genome_config(
        name=args.genome,
        fasta_path=fasta_path,
        species=args.species,
        fold_type="chrom_partition",
        fold_args={"k": 5, "val_fold": args.val_fold, "test_fold": args.test_fold},
        exclude_intervals=exclude_intervals,
    )

    input_len = args.input_len
    output_len = args.output_len
    output_bin_size = 1
    max_jitter = args.jitter
    target_scale = args.target_scale
    padded_size = input_len + 2 * max_jitter

    data_config = DataConfig(
        inputs={},
        targets={"signal": args.bigwig},
        input_len=input_len,
        output_len=output_len,
        max_jitter=max_jitter,
        output_bin_size=output_bin_size,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=True,
        use_sequence=True,
        target_scale=target_scale,
    )

    sampler_config = SamplerConfig(
        sampler_type="peak",
        padded_size=padded_size,
        sampler_args={
            "intervals_path": args.peaks,
            "background_ratio": args.background_ratio,
        },
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

    count_pseudocount_scaled = resolve_count_pseudocount_from_args(
        args, bin_size=output_bin_size, target_scale=target_scale,
    )

    bias_logcount_offset = 0.0
    if args.adjust_bias_logcounts:
        if args.multi:
            # TODO: estimate the offset per-fold inside the train_multi loop
            # instead of once at the top.  Each fold's training set differs
            # (different non-peak background) so a per-fold offset would be
            # marginally more accurate.  Acceptable for now -- the offset
            # captures bulk dataset-level bias scaling that is approximately
            # fold-invariant.
            logging.warning(
                "--adjust-bias-logcounts with --multi estimates a single shared "
                "offset using the configured val/test folds, then reuses it for all folds."
            )
        bias_logcount_offset = _estimate_bias_offset(
            args=args,
            genome_config=genome_config,
            data_config=data_config,
            count_pseudocount_scaled=count_pseudocount_scaled,
        )

    model_args = {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "bias_logcount_offset": bias_logcount_offset,
        "accessibility_args": {
            "filters": args.filters,
            "n_dilated_layers": args.n_layers,
            "conv_kernel_size": args.conv_kernel_size,
            "dil_kernel_size": args.dil_kernel_size,
            "profile_kernel_size": args.profile_kernel_size,
            "predict_total_count": True,
            "activation": "relu",
            "weight_norm": False,
            "residual_architecture": args.residual_architecture,
        },
        "bias_args": {
            "filters": args.bias_filters,
            "n_dilated_layers": args.bias_layers,
            "conv_kernel_size": args.bias_conv_kernel_size,
            "dil_kernel_size": args.bias_dil_kernel_size,
            "profile_kernel_size": args.bias_profile_kernel_size,
            "predict_total_count": True,
            "activation": "relu",
            "weight_norm": False,
            "residual_architecture": args.residual_architecture,
        },
    }

    pretrained: list[PretrainedConfig] = [
        PretrainedConfig(
            weights_path=args.pretrained_bias,
            source=args.pretrained_bias_source,
            target="bias_model",
        )
    ]
    if args.pretrained_accessibility:
        pretrained.append(
            PretrainedConfig(
                weights_path=args.pretrained_accessibility,
                source=args.pretrained_accessibility_source,
                target="accessibility_model",
            )
        )

    # Freeze the bias subtree by default.  ``eval_mode=True`` calls
    # ``.eval()`` on ``bias_model`` so any Dropout / BatchNorm inside the
    # frozen branch stops firing / drifting -- matches the reference
    # ChromBPNet behaviour where the bias branch is frozen pre-training.
    freeze: list[FreezeSpec] = []
    if not args.no_freeze_bias:
        freeze.append(FreezeSpec(pattern="bias_model", eval_mode=True))

    model_config = ModelConfig(
        name="ChromBPNet",
        model_cls="cerberus.models.chrombpnet.ChromBPNet",
        loss_cls="cerberus.models.bpnet.BPNetLoss",
        loss_args={"alpha": args.alpha, "beta": args.beta},
        metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
        metrics_args={},
        model_args=model_args,
        pretrained=pretrained,
        freeze=freeze,
        count_pseudocount=count_pseudocount_scaled,
    )

    devices: str | int = args.devices
    if devices != "auto":
        try:
            devices = int(devices)
        except ValueError:
            pass

    accelerator = args.accelerator
    if accelerator == "auto" and torch.backends.mps.is_available():
        accelerator = "mps"

    precision_args = get_precision_kwargs(args.precision, accelerator, devices)

    logging.info("Genome Config:\n%s", pformat(genome_config))
    logging.info("Data Config:\n%s", pformat(data_config))
    logging.info("Sampler Config:\n%s", pformat(sampler_config))
    logging.info("Train Config:\n%s", pformat(train_config))
    logging.info("Model Config:\n%s", pformat(model_config))
    logging.info("Precision and Hardware Args:\n%s", pformat(precision_args))

    train_fn = train_multi if args.multi else train_single
    train_fn(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        model_config=model_config,
        train_config=train_config,
        num_workers=args.num_workers,
        in_memory=False,
        root_dir=str(output_dir),
        enable_checkpointing=True,
        log_every_n_steps=10,
        val_batch_size=args.batch_size * 4,
        enable_progress_bar=not args.silent,
        seed=args.seed,
        **precision_args,
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        _export_accessibility_checkpoints(output_dir)
        _plot_training_curves(output_dir)
        _run_prediction_evaluation(args, output_dir)
        logging.info(
            "Training finished. Logs and checkpoints are in subdirectories of %s",
            output_dir,
        )


if __name__ == "__main__":
    main()
