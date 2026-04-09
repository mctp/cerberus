#!/usr/bin/env python
"""Export attribution arrays for TF-MoDISco from a trained Cerberus model.

This script:
1. Loads a trained Cerberus model (prefer ``model.pt``, fallback ``.ckpt``).
2. Builds a datamodule from the training ``hparams.yaml``.
3. Collects one-hot DNA inputs (``ohe.npz``) and sequence attributions
   (``shap.npz``) using Captum Integrated Gradients (default), Captum
   DeepLiftShap, or mutation-based in-silico mutagenesis (ISM), with optional
   off-simplex gradient correction for Integrated Gradients attributions.

Notes
-----
- NPZ outputs are written using the default ``arr_0`` key, matching
  ``modisco motifs`` expectations.
- For ``peak`` samplers, this tool defaults to positive intervals only
  (``IntervalSampler`` source), which is generally preferred for motif discovery.
- This tool only exports attribution inputs. Run TF-MoDISco separately with
  ``tools/run_tfmodisco.py`` or the upstream ``modisco`` CLI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import cerberus
from cerberus.attribution import (
    AttributionTarget,
    apply_off_simplex_gradient_correction,
    compute_ism_attributions,
    resolve_ism_span,
)
from cerberus.datamodule import CerberusDataModule
from cerberus.model_ensemble import (
    find_latest_hparams,
    load_backbone_weights_from_fold_dir,
    parse_hparams_config,
)
from cerberus.module import instantiate_model

logger = logging.getLogger(__name__)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _resolve_fold_dir(checkpoint_dir: Path, fold: int) -> Path:
    # Direct fold directory
    if (checkpoint_dir / "model.pt").exists() or list(checkpoint_dir.glob("*.ckpt")):
        return checkpoint_dir

    # Common train root layout: root/fold_{i}
    direct = checkpoint_dir / f"fold_{fold}"
    if direct.is_dir():
        return direct

    # Recursive fallback
    recursive = sorted([p for p in checkpoint_dir.rglob(f"fold_{fold}") if p.is_dir()])
    if recursive:
        return recursive[0]

    raise FileNotFoundError(
        f"Could not locate fold directory for fold={fold} under {checkpoint_dir}."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export TF-MoDISco-compatible inputs (ohe.npz + shap.npz) from a trained "
            "Cerberus model."
        )
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help=(
            "Training output directory. Can be a fold directory (contains model.pt) "
            "or a parent directory with fold_* subdirectories."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to load if --checkpoint-dir is a parent directory.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to draw intervals from.",
    )
    parser.add_argument(
        "--intervals-path",
        type=Path,
        default=None,
        help=(
            "Optional peak/interval BED(narrowPeak) path override. "
            "Replaces sampler_config.sampler_args['intervals_path'] from hparams.yaml, "
            "allowing TF-MoDISco exports on custom interval sets (e.g. shared/unique peaks)."
        ),
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=2000,
        help="Number of examples to export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for attribution/export.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for datamodule sampling. "
            "Use the same value as training script --seed (default: 42)."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, cuda:0, mps, ...",
    )

    parser.add_argument(
        "--attribution-method",
        choices=["integrated_gradients", "deep_lift_shap", "ism"],
        default="integrated_gradients",
        help=(
            "Attribution method. 'integrated_gradients' uses Captum; "
            "'deep_lift_shap' uses Captum DeepLiftShap; "
            "'ism' computes mutation deltas via forward passes."
        ),
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=64,
        help="Integrated Gradients step count.",
    )
    parser.add_argument(
        "--ig-internal-batch-size",
        type=int,
        default=None,
        help="Optional internal batch size for Integrated Gradients.",
    )
    parser.add_argument(
        "--dls-n-baselines",
        type=int,
        default=20,
        help="Number of baselines per sequence for DeepLiftShap.",
    )
    parser.add_argument(
        "--dls-baseline-strategy",
        choices=["shuffle", "zero"],
        default="shuffle",
        help="DeepLiftShap baseline strategy: mononucleotide shuffle or all-zero baselines.",
    )
    parser.add_argument(
        "--dls-warning-threshold",
        type=float,
        default=1e-3,
        help="Warn when DeepLiftShap max |convergence delta| exceeds this threshold.",
    )
    parser.add_argument(
        "--ism-start",
        type=int,
        default=None,
        help=(
            "Inclusive input-base start index for ISM mutations. "
            "Default: 0 (full-length ISM)."
        ),
    )
    parser.add_argument(
        "--ism-end",
        type=int,
        default=None,
        help=(
            "Exclusive input-base end index for ISM mutations. "
            "Default: input length (full-length ISM)."
        ),
    )
    parser.add_argument(
        "--off-simplex-gradient-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply off-simplex correction to Integrated Gradients attributions: "
            "subtract, at each input position, the mean attribution across "
            "nucleotides (for arrays shaped (N, 4, L): "
            "attrs -= attrs.mean(axis=1, keepdims=True)). Ignored for ISM."
        ),
    )

    parser.add_argument(
        "--target-mode",
        choices=[
            "log_counts",
            "profile_bin",
            "profile_window_sum",
            "pred_count_bin",
            "pred_count_window_sum",
        ],
        default="log_counts",
        help="Scalar output target used for attribution.",
    )
    parser.add_argument(
        "--target-channel",
        type=int,
        default=0,
        help="Target channel index.",
    )
    parser.add_argument(
        "--bin-index",
        type=int,
        default=None,
        help="Output bin index for *_bin target modes (default: center bin).",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=None,
        help="Start bin (inclusive) for *_window_sum target modes.",
    )
    parser.add_argument(
        "--window-end",
        type=int,
        default=None,
        help="End bin (exclusive) for *_window_sum target modes.",
    )

    parser.add_argument(
        "--include-sources",
        type=str,
        default=None,
        help=(
            "Comma-delimited interval_source filters (from dataset['interval_source']). "
            "Example: IntervalSampler. If omitted and sampler_type=peak, defaults "
            "to the positive source inferred from the split sampler."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write ohe.npz and shap.npz.",
    )
    parser.add_argument(
        "--ohe-name",
        type=str,
        default="ohe.npz",
        help="Output filename for one-hot sequences.",
    )
    parser.add_argument(
        "--attr-name",
        type=str,
        default="shap.npz",
        help="Output filename for attributions.",
    )

    return parser


def _parse_include_sources(
    include_sources: str | None,
    sampler_type: str,
    peak_positive_source: str | None = None,
) -> set[str] | None:
    if include_sources is not None:
        return {x.strip() for x in include_sources.split(",") if x.strip()}

    # Peak sampler mixes positives + matched negatives. Default to positives.
    if sampler_type == "peak":
        if peak_positive_source is not None:
            return {peak_positive_source}
        # Backward-compatible fallback if sampler introspection is unavailable.
        return {"IntervalSampler", "ListSampler"}

    return None


def _infer_peak_positive_source(loader: object) -> str | None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    sampler = getattr(dataset, "sampler", None)
    if sampler is None:
        return None
    sub_samplers = getattr(sampler, "samplers", None)
    if not sub_samplers:
        return None
    return type(sub_samplers[0]).__name__


def _select_loader(datamodule: CerberusDataModule, split: str):
    if split == "train":
        return datamodule.train_dataloader()
    if split == "val":
        return datamodule.val_dataloader()
    if split == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"Unsupported split: {split}")


def _generate_dls_baselines(
    x: torch.Tensor, n_baselines: int, strategy: str, rng: np.random.Generator
) -> torch.Tensor:
    """Generate DeepLiftShap baselines for a single sample x (shape: 1 x C x L)."""
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(
            f"Expected x shape (1, C, L) for DeepLiftShap baselines, got {tuple(x.shape)}"
        )
    if n_baselines < 2:
        raise ValueError(
            f"DeepLiftShap requires >=2 baselines; got dls_n_baselines={n_baselines}"
        )

    _, n_channels, seq_len = x.shape
    if strategy == "zero":
        return torch.zeros(
            (n_baselines, n_channels, seq_len), device=x.device, dtype=x.dtype
        )

    if strategy == "shuffle":
        refs: list[torch.Tensor] = []
        for _ in range(n_baselines):
            perm = torch.as_tensor(rng.permutation(seq_len), device=x.device, dtype=torch.long)
            refs.append(x[0, :, perm])
        return torch.stack(refs, dim=0)

    raise ValueError(f"Unsupported DeepLiftShap baseline strategy: {strategy}")


def _export_arrays(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    IntegratedGradients = None
    DeepLiftShap = None
    if args.attribution_method == "integrated_gradients":
        try:
            from captum.attr import IntegratedGradients as _IntegratedGradients
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "captum is required for integrated_gradients. "
                "Install with: pip install captum or pip install -e .[attribution]"
            ) from exc
        IntegratedGradients = _IntegratedGradients
    elif args.attribution_method == "deep_lift_shap":
        try:
            from captum.attr import DeepLiftShap as _DeepLiftShap
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "captum is required for deep_lift_shap. "
                "Install with: pip install captum or pip install -e .[attribution]"
            ) from exc
        DeepLiftShap = _DeepLiftShap

    fold_dir = _resolve_fold_dir(args.checkpoint_dir.resolve(), args.fold)
    try:
        hparams_path = find_latest_hparams(fold_dir)
    except FileNotFoundError:
        # Fallback for custom checkpoint layouts where hparams live above fold_dir.
        hparams_path = find_latest_hparams(args.checkpoint_dir.resolve())
    logger.info("Using fold dir: %s", fold_dir)
    logger.info("Using hparams: %s", hparams_path)

    cfg = parse_hparams_config(hparams_path)
    if not cfg.data_config.use_sequence:
        raise ValueError("data_config.use_sequence is False; sequence attributions unavailable.")

    if args.intervals_path is not None:
        intervals_path = args.intervals_path.resolve()
        if not intervals_path.exists():
            raise FileNotFoundError(f"--intervals-path not found: {intervals_path}")
        sampler_args = dict(cfg.sampler_config.sampler_args)
        if "intervals_path" not in sampler_args:
            raise ValueError(
                "--intervals-path override is not supported for this sampler config: "
                "sampler_args does not define 'intervals_path'."
            )
        sampler_args["intervals_path"] = intervals_path
        cfg = cfg.model_copy(
            update={
                "sampler_config": cfg.sampler_config.model_copy(
                    update={"sampler_args": sampler_args}
                )
            }
        )
        logger.info("Overriding sampler intervals_path with: %s", intervals_path)

    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)

    model = instantiate_model(cfg.model_config_, cfg.data_config, compile=False)
    load_backbone_weights_from_fold_dir(model, fold_dir, device, strict=True)
    model.to(device)
    model.eval()

    target_model = AttributionTarget(
        model=model,
        mode=args.target_mode,
        channel=args.target_channel,
        bin_index=args.bin_index,
        window_start=args.window_start,
        window_end=args.window_end,
    ).to(device)
    target_model.eval()

    datamodule = CerberusDataModule(
        genome_config=cfg.genome_config,
        data_config=cfg.data_config,
        sampler_config=cfg.sampler_config,
        test_fold=cfg.genome_config.fold_args.get("test_fold"),
        val_fold=cfg.genome_config.fold_args.get("val_fold"),
        seed=args.seed,
    )
    datamodule.prepare_data()
    datamodule.setup(
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        in_memory=False,
    )

    loader = _select_loader(datamodule, args.split)
    peak_positive_source = None
    if args.include_sources is None and cfg.sampler_config.sampler_type == "peak":
        peak_positive_source = _infer_peak_positive_source(loader)
        if peak_positive_source is not None:
            logger.info(
                "Auto-detected positive interval_source for peak sampler: %s",
                peak_positive_source,
            )

    include_sources = _parse_include_sources(
        args.include_sources,
        cfg.sampler_config.sampler_type,
        peak_positive_source=peak_positive_source,
    )
    if include_sources is not None:
        logger.info("Filtering interval_source in: %s", sorted(include_sources))

    attribution = None
    if args.attribution_method == "integrated_gradients":
        assert IntegratedGradients is not None
        attribution = IntegratedGradients(target_model)
    elif args.attribution_method == "deep_lift_shap":
        assert DeepLiftShap is not None
        if args.target_mode in {"pred_count_bin", "pred_count_window_sum"}:
            raise ValueError(
                "target-mode pred_count_* is not supported with deep_lift_shap yet. "
                "Use log_counts/profile_* target modes or integrated_gradients."
            )
        if args.dls_n_baselines < 2:
            raise ValueError("--dls-n-baselines must be >= 2 for deep_lift_shap.")
        model_name = model.__class__.__name__.lower()
        if "bpnet" not in model_name:
            raise ValueError(
                f"deep_lift_shap is currently limited to BPNet models; got {model.__class__.__name__}."
            )
        if any(isinstance(m, torch.nn.GELU) for m in model.modules()):
            raise ValueError(
                "deep_lift_shap currently requires ReLU-only BPNet paths. "
                "Detected GELU modules; use integrated_gradients instead."
            )
        attribution = DeepLiftShap(target_model)
        logger.info(
            "DeepLiftShap enabled with baseline strategy '%s' and %d baselines per sequence.",
            args.dls_baseline_strategy,
            args.dls_n_baselines,
        )
    elif args.attribution_method != "ism":
        raise ValueError(f"Unsupported attribution method: {args.attribution_method}")

    ohe_batches: list[np.ndarray] = []
    attr_batches: list[np.ndarray] = []
    exported = 0
    logged_ism_span = False
    dls_rng = np.random.default_rng(args.seed)
    dls_delta_abs_sum = 0.0
    dls_delta_abs_max = 0.0
    dls_delta_count = 0

    logger.info("Beginning export: target %d examples", args.n_examples)
    apply_off_simplex_correction = (
        args.off_simplex_gradient_correction
        and args.attribution_method == "integrated_gradients"
    )
    if apply_off_simplex_correction:
        logger.info(
            "Enabled off-simplex gradient correction (subtract mean across "
            "nucleotide channels at each position)."
        )
    elif args.off_simplex_gradient_correction:
        logger.info(
            "Off-simplex gradient correction requested, but attribution method "
            "is '%s'; skipping correction.",
            args.attribution_method,
        )

    for batch_idx, batch in enumerate(loader):
        inputs = batch["inputs"].float()
        sources = batch.get("interval_source")

        if include_sources is not None and sources is not None:
            keep_mask = torch.tensor(
                [src in include_sources for src in sources], dtype=torch.bool
            )
            if not keep_mask.any():
                continue
            inputs = inputs[keep_mask]

        if inputs.numel() == 0:
            continue

        if exported >= args.n_examples:
            break

        remaining = args.n_examples - exported
        if inputs.shape[0] > remaining:
            inputs = inputs[:remaining]

        inputs = inputs.to(device)

        if args.attribution_method == "integrated_gradients":
            assert attribution is not None
            baseline = torch.zeros((1, inputs.shape[1], inputs.shape[2]), device=device)
            attributions = attribution.attribute(
                inputs,
                baselines=baseline,
                n_steps=args.ig_steps,
                internal_batch_size=args.ig_internal_batch_size,
            )
        elif args.attribution_method == "deep_lift_shap":
            assert attribution is not None
            attrs_per_example: list[torch.Tensor] = []
            for i in range(inputs.shape[0]):
                x_i = inputs[i : i + 1]
                baselines = _generate_dls_baselines(
                    x_i,
                    n_baselines=args.dls_n_baselines,
                    strategy=args.dls_baseline_strategy,
                    rng=dls_rng,
                )
                attr_i, delta_i = attribution.attribute(
                    x_i,
                    baselines=baselines,
                    return_convergence_delta=True,
                )
                attrs_per_example.append(attr_i)
                delta_abs = delta_i.detach().abs()
                dls_delta_abs_sum += float(delta_abs.sum().item())
                dls_delta_abs_max = max(dls_delta_abs_max, float(delta_abs.max().item()))
                dls_delta_count += int(delta_abs.numel())
            attributions = torch.cat(attrs_per_example, dim=0)
        else:
            if not logged_ism_span:
                span_start, span_end = resolve_ism_span(
                    inputs.shape[-1], args.ism_start, args.ism_end
                )
                span_len = span_end - span_start
                logger.info(
                    "ISM span [%d, %d) over input length %d (%d positions); "
                    "per batch this runs %d forward passes (%d mutants/position).",
                    span_start,
                    span_end,
                    inputs.shape[-1],
                    span_len,
                    span_len + 1,
                    inputs.shape[0] * 4,
                )
                logged_ism_span = True
            attributions = compute_ism_attributions(
                target_model=target_model,
                inputs=inputs,
                ism_start=args.ism_start,
                ism_end=args.ism_end,
            )

        # TF-MoDISco sequences and attribution files should both be (N, 4, L).
        ohe = inputs[:, :4, :].detach().cpu().numpy().astype(np.float32)
        attrs = attributions[:, :4, :].detach().cpu().numpy().astype(np.float32)
        if apply_off_simplex_correction:
            attrs = apply_off_simplex_gradient_correction(attrs)

        ohe_batches.append(ohe)
        attr_batches.append(attrs)
        exported += ohe.shape[0]

        if (batch_idx + 1) % 5 == 0 or exported >= args.n_examples:
            logger.info("Exported %d / %d examples", exported, args.n_examples)
            if args.attribution_method == "deep_lift_shap" and dls_delta_count > 0:
                logger.info(
                    "DeepLiftShap |delta| so far: mean=%.3e max=%.3e",
                    dls_delta_abs_sum / dls_delta_count,
                    dls_delta_abs_max,
                )

        if exported >= args.n_examples:
            break

    if exported == 0:
        raise RuntimeError(
            "No examples exported. Check split/source filters and dataset size."
        )

    ohe_all = np.concatenate(ohe_batches, axis=0)
    attr_all = np.concatenate(attr_batches, axis=0)

    if ohe_all.shape != attr_all.shape:
        raise RuntimeError(f"Shape mismatch: ohe {ohe_all.shape} vs attrs {attr_all.shape}")

    if args.attribution_method == "deep_lift_shap" and dls_delta_count > 0:
        dls_delta_abs_mean = dls_delta_abs_sum / dls_delta_count
        logger.info(
            "DeepLiftShap convergence |delta| summary: mean=%.3e max=%.3e n=%d",
            dls_delta_abs_mean,
            dls_delta_abs_max,
            dls_delta_count,
        )
        if dls_delta_abs_max > args.dls_warning_threshold:
            logger.warning(
                "DeepLiftShap max |delta| (%.3e) exceeded warning threshold (%.3e). "
                "Interpret attributions cautiously or prefer integrated_gradients.",
                dls_delta_abs_max,
                args.dls_warning_threshold,
            )

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ohe_path = output_dir / args.ohe_name
    attr_path = output_dir / args.attr_name

    # Save with default key arr_0 for modisco compatibility.
    np.savez_compressed(ohe_path, ohe_all)
    np.savez_compressed(attr_path, attr_all)

    logger.info("Saved sequences: %s  shape=%s", ohe_path, ohe_all.shape)
    logger.info("Saved attrs:     %s  shape=%s", attr_path, attr_all.shape)

    return output_dir, ohe_path, attr_path


def main() -> None:
    cerberus.setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()

    _, ohe_path, attr_path = _export_arrays(args)
    logger.info("Attribution export complete. Use tools/run_tfmodisco.py for aggregation.")
    logger.info("ohe:  %s", ohe_path)
    logger.info("attr: %s", attr_path)


if __name__ == "__main__":
    main()
