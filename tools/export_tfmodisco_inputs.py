#!/usr/bin/env python
"""Export TF-MoDISco inputs from a trained Cerberus model and run MoDISco.

This script:
1. Loads a trained Cerberus model (prefer ``model.pt``, fallback ``.ckpt``).
2. Builds a datamodule from the training ``hparams.yaml``. 
3. Collects one-hot DNA inputs (``ohe.npz``) and sequence attributions
   (``shap.npz``) using either Captum Integrated Gradients (default) or
   mutation-based in-silico mutagenesis (ISM), with optional off-simplex
   gradient correction for Integrated Gradients attributions.
4. Runs ``modisco motifs`` end-to-end, and optionally ``modisco report``.

Notes
-----
- NPZ outputs are written using the default ``arr_0`` key, matching
  ``modisco motifs`` expectations.
- For ``peak`` samplers, this tool defaults to positive intervals only
  (``IntervalSampler`` source), which is generally preferred for motif discovery.
- For ``modisco report`` motif matching, TF-MoDISco recommends using the
  MotifCompendium human motif database for human datasets.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import torch

import cerberus
from cerberus.datamodule import CerberusDataModule
from cerberus.model_ensemble import parse_hparams_config
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


def _find_latest_hparams(search_root: Path) -> Path:
    candidates = list(search_root.rglob("hparams.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No hparams.yaml found under: {search_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_fold_dir(checkpoint_dir: Path, fold: int) -> Path:
    # Direct fold directory
    if (checkpoint_dir / "model.pt").exists() or list(checkpoint_dir.glob("*.ckpt")):
        return checkpoint_dir

    # Common train root layout: root/fold_{i}
    direct = checkpoint_dir / f"fold_{fold}"
    if direct.is_dir():
        return direct

    # Recursive fallback
    recursive = sorted(
        [p for p in checkpoint_dir.rglob(f"fold_{fold}") if p.is_dir()]
    )
    if recursive:
        return recursive[0]

    raise FileNotFoundError(
        f"Could not locate fold directory for fold={fold} under {checkpoint_dir}."
    )


def _select_best_checkpoint(checkpoints: list[Path]) -> Path:
    def val_loss(path: Path) -> float:
        match = re.search(r"val_loss[=_](\d+\.?\d*)", path.name)
        if match is None:
            return float("inf")
        try:
            return float(match.group(1))
        except ValueError:
            return float("inf")

    return sorted(checkpoints, key=lambda p: (val_loss(p), p.name))[0]


def _load_model_weights(model: torch.nn.Module, fold_dir: Path, device: torch.device) -> None:
    pt_path = fold_dir / "model.pt"
    if pt_path.exists():
        logger.info("Loading clean state dict: %s", pt_path)
        state_dict = torch.load(pt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return

    checkpoints = list(fold_dir.glob("*.ckpt"))
    if not checkpoints:
        checkpoints = list(fold_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No model.pt or .ckpt files found in fold directory: {fold_dir}"
        )

    ckpt_path = _select_best_checkpoint(checkpoints)
    logger.info("Loading Lightning checkpoint: %s", ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_state = checkpoint["state_dict"]

    # Strip CerberusModule prefix and optional torch.compile prefix.
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not key.startswith("model."):
            continue
        new_key = key[6:]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[10:]
        state_dict[new_key] = value

    model.load_state_dict(state_dict, strict=True)


class AttributionTarget(torch.nn.Module):
    """Wrap Cerberus model output into a single tensor target for Captum."""

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str,
        channel: int,
        bin_index: int | None,
        window_start: int | None,
        window_end: int | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.mode = mode
        self.channel = channel
        self.bin_index = bin_index
        self.window_start = window_start
        self.window_end = window_end

    def _resolve_channel(self, tensor: torch.Tensor) -> int:
        n_channels = tensor.shape[1]
        if not (0 <= self.channel < n_channels):
            raise ValueError(
                f"Requested channel={self.channel}, but target has {n_channels} channels."
            )
        return self.channel

    def _resolve_window(self, length: int) -> tuple[int, int]:
        start = 0 if self.window_start is None else self.window_start
        end = length if self.window_end is None else self.window_end
        if not (0 <= start < end <= length):
            raise ValueError(
                f"Invalid window [{start}, {end}) for output length {length}."
            )
        return start, end

    def _resolve_bin(self, length: int) -> int:
        if self.bin_index is None:
            return length // 2
        if not (0 <= self.bin_index < length):
            raise ValueError(
                f"Invalid bin_index={self.bin_index} for output length {length}."
            )
        return self.bin_index

    def _predicted_counts_channel(
        self, logits: torch.Tensor, log_counts: torch.Tensor, channel: int
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        # If model predicts a single total count, broadcast to all profile channels.
        # log_counts is a 2D tensor with shape (batch_size, n_count_outputs)
        if log_counts.shape[1] == 1:
            count_scale = torch.exp(log_counts[:, 0]).unsqueeze(-1)
        else:
            count_scale = torch.exp(log_counts[:, channel]).unsqueeze(-1)
        return probs[:, channel, :] * count_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = out.logits
        log_counts = out.log_counts

        channel = self._resolve_channel(logits)

        if self.mode == "log_counts":
            if log_counts.shape[1] == 1:
                return log_counts[:, 0]
            if channel >= log_counts.shape[1]:
                raise ValueError(
                    f"Requested channel={channel}, but log_counts has "
                    f"{log_counts.shape[1]} channels."
                )
            return log_counts[:, channel]

        if self.mode == "profile_bin":
            bin_index = self._resolve_bin(logits.shape[-1])
            return logits[:, channel, bin_index]

        if self.mode == "profile_window_sum":
            start, end = self._resolve_window(logits.shape[-1])
            return logits[:, channel, start:end].sum(dim=-1)

        if self.mode == "pred_count_bin":
            pred_counts = self._predicted_counts_channel(logits, log_counts, channel)
            bin_index = self._resolve_bin(pred_counts.shape[-1])
            return pred_counts[:, bin_index]

        if self.mode == "pred_count_window_sum":
            pred_counts = self._predicted_counts_channel(logits, log_counts, channel)
            start, end = self._resolve_window(pred_counts.shape[-1])
            return pred_counts[:, start:end].sum(dim=-1)

        raise ValueError(f"Unsupported mode: {self.mode}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export TF-MoDISco inputs (ohe.npz + shap.npz) from a trained Cerberus "
            "model and optionally run modisco motifs/report."
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
        choices=["integrated_gradients", "ism"],
        default="integrated_gradients",
        help=(
            "Attribution method. 'integrated_gradients' uses Captum; "
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
        help="Directory to write ohe.npz, shap.npz, and MoDISco outputs.",
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

    parser.add_argument(
        "--run-modisco",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run `modisco motifs` after exporting NPZ files.",
    )
    parser.add_argument(
        "--max-seqlets",
        type=int,
        default=2000,
        help="`modisco motifs -n` max seqlets per metacluster.",
    )
    parser.add_argument(
        "--modisco-window",
        type=int,
        default=400,
        help="`modisco motifs -w` window size around center.",
    )
    parser.add_argument(
        "--modisco-output",
        type=str,
        default="modisco_results.h5",
        help="Output filename for `modisco motifs` HDF5 result.",
    )

    parser.add_argument(
        "--run-report",
        action="store_true",
        help="Run `modisco report` after motifs step.",
    )
    parser.add_argument(
        "--meme-db",
        type=Path,
        default=None,
        help=(
            "Motif DB (.meme) for `modisco report -m`. For human data, "
            "TF-MoDISco recommends MotifCompendium-Database-Human.meme.txt."
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="report",
        help="Output directory name for `modisco report`.",
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


def _resolve_ism_span(
    seq_len: int, start: int | None, end: int | None
) -> tuple[int, int]:
    span_start = 0 if start is None else start
    span_end = seq_len if end is None else end
    if not (0 <= span_start < span_end <= seq_len):
        raise ValueError(
            f"Invalid ISM span [{span_start}, {span_end}) for input length {seq_len}."
        )
    return span_start, span_end


def _compute_ism_attributions(
    target_model: torch.nn.Module,
    inputs: torch.Tensor,
    ism_start: int | None,
    ism_end: int | None,
) -> torch.Tensor:
    """Compute single-position ISM deltas as (B, 4, L) attribution scores."""
    if inputs.shape[1] < 4:
        raise ValueError(
            f"ISM requires >=4 DNA channels in inputs, got shape {tuple(inputs.shape)}"
        )

    batch_size = inputs.shape[0]
    seq_len = inputs.shape[-1]
    span_start, span_end = _resolve_ism_span(seq_len, ism_start, ism_end)

    attrs = torch.zeros(
        (batch_size, 4, seq_len), device=inputs.device, dtype=inputs.dtype
    )
    rows = torch.arange(batch_size * 4, device=inputs.device)
    nuc_idx = torch.arange(4, device=inputs.device).repeat(batch_size)
    sample_idx = torch.arange(batch_size, device=inputs.device)

    with torch.no_grad():
        ref_pred = target_model(inputs).reshape(batch_size)
        for pos in range(span_start, span_end):
            mutants = inputs.repeat_interleave(4, dim=0)
            mutants[:, :4, pos] = 0.0
            mutants[rows, nuc_idx, pos] = 1.0

            mut_pred = target_model(mutants).reshape(batch_size, 4)
            attrs[:, :, pos] = mut_pred - ref_pred[:, None]

            # For TF-MoDISco compatibility with one_hot * hypothetical_contribs,
            # fill the observed nucleotide channel with negative of the per-position mean
            # hypothetical contribution instead of forcing it to zero.
            ref_base = inputs[:, :4, pos].argmax(dim=1)
            mean_at_pos = attrs[:, :, pos].mean(dim=1)
            attrs[sample_idx, ref_base, pos] = -mean_at_pos

    return attrs


def _apply_off_simplex_gradient_correction(attrs: np.ndarray) -> np.ndarray:
    """Subtract per-position mean attribution across nucleotide channels."""
    if attrs.ndim != 3:
        raise ValueError(
            f"Expected 3D attribution array (N, 4, L), got shape {attrs.shape}"
        )
    return attrs - attrs.mean(axis=1, keepdims=True)


def _export_arrays(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    IntegratedGradients = None
    if args.attribution_method == "integrated_gradients":
        try:
            from captum.attr import IntegratedGradients as _IntegratedGradients
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "captum is required for integrated_gradients. "
                "Install with: pip install captum"
            ) from exc
        IntegratedGradients = _IntegratedGradients

    fold_dir = _resolve_fold_dir(args.checkpoint_dir.resolve(), args.fold)
    try:
        hparams_path = _find_latest_hparams(fold_dir)
    except FileNotFoundError:
        # Fallback for custom checkpoint layouts where hparams live above fold_dir.
        hparams_path = _find_latest_hparams(args.checkpoint_dir.resolve())
    logger.info("Using fold dir: %s", fold_dir)
    logger.info("Using hparams: %s", hparams_path)

    cfg = parse_hparams_config(hparams_path)
    if not cfg.data_config.use_sequence:
        raise ValueError("data_config.use_sequence is False; sequence attributions unavailable.")

    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)

    model = instantiate_model(cfg.model_config_, cfg.data_config, compile=False)
    _load_model_weights(model, fold_dir, device)
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
    elif args.attribution_method != "ism":
        raise ValueError(f"Unsupported attribution method: {args.attribution_method}")

    ohe_batches: list[np.ndarray] = []
    attr_batches: list[np.ndarray] = []
    exported = 0
    logged_ism_span = False

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
        else:
            if not logged_ism_span:
                span_start, span_end = _resolve_ism_span(
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
            attributions = _compute_ism_attributions(
                target_model=target_model,
                inputs=inputs,
                ism_start=args.ism_start,
                ism_end=args.ism_end,
            )

        # TF-MoDISco sequences and attribution files should both be (N, 4, L).
        ohe = inputs[:, :4, :].detach().cpu().numpy().astype(np.float32)
        attrs = attributions[:, :4, :].detach().cpu().numpy().astype(np.float32)
        if apply_off_simplex_correction:
            attrs = _apply_off_simplex_gradient_correction(attrs)

        ohe_batches.append(ohe)
        attr_batches.append(attrs)
        exported += ohe.shape[0]

        if (batch_idx + 1) % 5 == 0 or exported >= args.n_examples:
            logger.info("Exported %d / %d examples", exported, args.n_examples)

        if exported >= args.n_examples:
            break

    if exported == 0:
        raise RuntimeError(
            "No examples exported. Check split/source filters and dataset size."
        )

    ohe_all = np.concatenate(ohe_batches, axis=0)
    attr_all = np.concatenate(attr_batches, axis=0)

    if ohe_all.shape != attr_all.shape:
        raise RuntimeError(
            f"Shape mismatch: ohe {ohe_all.shape} vs attrs {attr_all.shape}"
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


def _run_modisco_motifs(
    ohe_path: Path,
    attr_path: Path,
    output_h5: Path,
    max_seqlets: int,
    window: int,
) -> None:
    cmd = [
        "modisco",
        "motifs",
        "-s",
        str(ohe_path),
        "-a",
        str(attr_path),
        "-n",
        str(max_seqlets),
        "-w",
        str(window),
        "-o",
        str(output_h5),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_modisco_report(
    modisco_h5: Path, report_dir: Path, meme_db: Path | None
) -> None:
    cmd = [
        "modisco",
        "report",
        "-i",
        str(modisco_h5),
        "-o",
        str(report_dir),
        "-s",
        str(report_dir),
    ]
    if meme_db is not None:
        cmd.extend(["-m", str(meme_db)])

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    cerberus.setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_dir, ohe_path, attr_path = _export_arrays(args)

    modisco_h5 = output_dir / args.modisco_output

    if args.run_modisco:
        try:
            _run_modisco_motifs(
                ohe_path=ohe_path,
                attr_path=attr_path,
                output_h5=modisco_h5,
                max_seqlets=args.max_seqlets,
                window=args.modisco_window,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`modisco` command not found. Install TF-MoDISco CLI first, e.g. `pip install modisco`."
            ) from exc

    if args.run_report:
        if not args.run_modisco and not modisco_h5.exists():
            raise FileNotFoundError(
                f"Cannot run report: MoDISco output not found at {modisco_h5}. "
                "Run with --run-modisco first or provide existing output path via --modisco-output."
            )

        report_dir = output_dir / args.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        _run_modisco_report(modisco_h5, report_dir, args.meme_db)


if __name__ == "__main__":
    main()
