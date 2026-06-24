#!/usr/bin/env python
"""
ChromBPNet bias-branch training tool.

Stage 1 of the reference ChromBPNet workflow: trains the small BPNet used as
the frozen bias model in :class:`cerberus.models.ChromBPNet`.  Designed for
ATAC-seq / DNase-seq background (non-peak) regions where the enzyme's
sequence preference dominates the signal.

The exported ``model.pt`` is then loaded into ``ChromBPNet.bias_model`` in
the stage-2 trainer (see ``docs/models.md#chrombpnet``).
"""

import argparse
import logging
import os
from pathlib import Path
from pprint import pformat

import torch
from _pseudocount_cli import (
    add_pseudocount_cli_args,
    resolve_count_pseudocount_from_args,
)

import cerberus
from cerberus.config import (
    DataConfig,
    ModelConfig,
    PretrainedConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.download import download_human_reference
from cerberus.genome import create_genome_config
from cerberus.train import train_multi, train_single
from cerberus.utils import get_precision_kwargs


def _parse_alpha(value: str) -> float | str:
    if value == "adaptive":
        return "adaptive"
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--alpha must be a float or 'adaptive', got {value!r}"
        ) from exc


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the small BPNet bias branch used by ChromBPNet"
    )

    # --- Input data ---
    parser.add_argument("--bigwig", type=str, required=True, help="Target BigWig")
    parser.add_argument(
        "--peaks",
        type=str,
        required=True,
        help="Peak BED/narrowPeak used to define non-peak background regions",
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
        "--sampler-type",
        type=str,
        default="negative_peak",
        choices=["negative_peak", "peak"],
        help="negative_peak matches the reference ChromBPNet bias stage",
    )
    parser.add_argument("--background-ratio", type=float, default=1.0)
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="Multiplicative scaling factor for targets (1.0 for raw-count pseudobulk BigWig values)",
    )

    # --- Pseudocount CLI family (shared with the other train_* tools) ---
    add_pseudocount_cli_args(parser, default_count_pseudocount=1.0)

    # --- Bias-branch architecture (chrombpnet-pytorch defaults: small BPNet) ---
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--conv-kernel-size", type=int, default=21)
    parser.add_argument("--dil-kernel-size", type=int, default=3)
    parser.add_argument("--profile-kernel-size", type=int, default=75)
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

    # --- Loss / pretrained ---
    parser.add_argument(
        "--loss",
        type=str,
        default="bpnet",
        choices=["bpnet", "profile-jsd"],
        help=(
            "Bias-model objective. 'bpnet' keeps the current Cerberus small "
            "BPNet profile MNLL + count MSE objective. 'profile-jsd' trains "
            "only the profile logits with Jensen-Shannon divergence, matching "
            "bpAI-TAC-style Tn5 bias training without scalar count-head loss."
        ),
    )
    parser.add_argument("--alpha", type=_parse_alpha, default="adaptive")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional pretrained BPNet ``.pt`` checkpoint to warm-start from.",
    )

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
    logging.info("Starting ChromBPNet bias-branch training tool...")
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
        sampler_type=args.sampler_type,
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

    model_args = {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "filters": args.filters,
        "n_dilated_layers": args.n_layers,
        "conv_kernel_size": args.conv_kernel_size,
        "dil_kernel_size": args.dil_kernel_size,
        "profile_kernel_size": args.profile_kernel_size,
        "predict_total_count": True,
        "activation": "relu",
        "weight_norm": False,
        "residual_architecture": args.residual_architecture,
    }

    pretrained: list[PretrainedConfig] = []
    if args.pretrained:
        pretrained.append(
            PretrainedConfig(
                weights_path=args.pretrained,
                source=None,
                target=None,
            )
        )

    if args.loss == "profile-jsd":
        loss_cls = "cerberus.loss.ProfileJSDLoss"
        loss_args = {"flatten_channels": False, "average_channels": True}
        model_name = "ChromBPNetBiasBPNet_ProfileJSD"
        count_pseudocount = 0.0
    else:
        loss_cls = "cerberus.models.bpnet.BPNetLoss"
        loss_args = {"alpha": args.alpha, "beta": args.beta}
        model_name = "ChromBPNetBiasBPNet"
        count_pseudocount = resolve_count_pseudocount_from_args(
            args,
            bin_size=output_bin_size,
            target_scale=target_scale,
        )

    model_config = ModelConfig(
        name=model_name,
        model_cls="cerberus.models.bpnet.BPNet",
        loss_cls=loss_cls,
        loss_args=loss_args,
        metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
        metrics_args={},
        model_args=model_args,
        pretrained=pretrained,
        count_pseudocount=count_pseudocount,
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
        logging.info(
            "Training finished. Logs and checkpoints are in subdirectories of %s",
            output_dir,
        )


if __name__ == "__main__":
    main()
