"""Attribution utilities shared across interpretation tools."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


ATTRIBUTION_MODES = {
    "log_counts",
    "profile_bin",
    "profile_window_sum",
    "pred_count_bin",
    "pred_count_window_sum",
}

DIFFERENTIAL_ATTRIBUTION_MODES = {
    "delta_log_counts",
    "delta_profile_window_sum",
}


class AttributionTarget(torch.nn.Module):
    """Wrap Cerberus model output into a scalar target tensor."""

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
        if mode not in ATTRIBUTION_MODES:
            raise ValueError(
                f"Unsupported mode: {mode!r}. Must be one of {sorted(ATTRIBUTION_MODES)}."
            )
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
            raise ValueError(f"Invalid window [{start}, {end}) for output length {length}.")
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


class DifferentialAttributionTarget(nn.Module):
    """Wrap the differential output of a multi-task model into a scalar for attribution.

    The scalar target is the *difference* between two condition channels
    (``cond_b − cond_a``), enabling attribution of sequence features that
    drive differential TF binding or chromatin accessibility between two
    steady-state conditions.

    Two modes are supported, addressing complementary questions:

    ``"delta_log_counts"``
        Target = ``log_counts[:, cond_b_idx] − log_counts[:, cond_a_idx]``

        Attributes sequence features that change **total binding / accessibility**.
        Positive attribution → pushes binding higher in B (or lower in A).
        Use after Phase 2 fine-tuning when the model was explicitly trained to
        predict this scalar; attribution has direct semantic alignment with the
        training objective (Naqvi et al. 2025).

    ``"delta_profile_window_sum"``
        Target = ``(logits_B − logits_A)[:, window_start:window_end].sum(−1)``

        Attributes sequence features that change the **binding footprint shape**
        within the output window.  Gradients flow through ``profile_conv``, making
        the attribution sensitive to the exact nucleotide contacts encoded in the
        profile head.  Works on the Phase 1 model without any Phase 2 fine-tuning.
        Preferred for ChIP-seq where the footprint shape is interpretable.

    .. note::
        Do **not** compute attributions for each condition separately and then
        subtract the resulting maps.  The two maps share a common baseline but
        are not in the same reference frame — subtracting them conflates
        "features that push B up" with "features that pull A down" without the
        weighting the model learned.  Use this class instead.

    Args:
        model: A :class:`~cerberus.models.bpnet.MultitaskBPNet` or any model
            returning :class:`~cerberus.output.ProfileCountOutput` with
            ``log_counts`` of shape ``(B, N_conditions)``.
        mode: Attribution target mode.  One of
            ``"delta_log_counts"`` or ``"delta_profile_window_sum"``.
        cond_a_idx: Channel index for condition A (reference). Default: 0.
        cond_b_idx: Channel index for condition B (query). Default: 1.
        window_start: Inclusive start of the output-profile window used by
            ``"delta_profile_window_sum"``.  Defaults to 0 (full window).
        window_end: Exclusive end of the output-profile window.  Defaults to
            the full output length.

    Example::

        # Phase 2 fine-tuned model — count-level differential attribution
        target = DifferentialAttributionTarget(
            model, mode="delta_log_counts", cond_a_idx=0, cond_b_idx=1
        )
        attrs = compute_ism_attributions(target, seq, ism_start=None, ism_end=None)
        attrs = apply_off_simplex_gradient_correction(attrs.numpy())

        # Phase 1 model — profile-level differential attribution (center 200bp)
        target = DifferentialAttributionTarget(
            model, mode="delta_profile_window_sum",
            cond_a_idx=0, cond_b_idx=1,
            window_start=400, window_end=600,
        )

    References:
        - Naqvi et al. (2025). *Transfer learning reveals sequence determinants
          of the quantitative response to transcription factor dosage.*
          Cell Genomics. PMC11160683.
        - bpAI-TAC: Chandra et al. (2025). bioRxiv 2025.01.24.634804.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str,
        cond_a_idx: int = 0,
        cond_b_idx: int = 1,
        window_start: int | None = None,
        window_end: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in DIFFERENTIAL_ATTRIBUTION_MODES:
            raise ValueError(
                f"Unsupported mode: {mode!r}. "
                f"Must be one of {sorted(DIFFERENTIAL_ATTRIBUTION_MODES)}."
            )
        if cond_a_idx == cond_b_idx:
            raise ValueError(
                f"cond_a_idx and cond_b_idx must differ, got both={cond_a_idx}"
            )
        self.model = model
        self.mode = mode
        self.cond_a_idx = cond_a_idx
        self.cond_b_idx = cond_b_idx
        self.window_start = window_start
        self.window_end = window_end

    def _resolve_window(self, length: int) -> tuple[int, int]:
        start = 0 if self.window_start is None else self.window_start
        end = length if self.window_end is None else self.window_end
        if not (0 <= start < end <= length):
            raise ValueError(
                f"Invalid window [{start}, {end}) for output length {length}."
            )
        return start, end

    def _check_channel(self, n_channels: int, idx: int, name: str) -> None:
        if idx >= n_channels:
            raise ValueError(
                f"{name}={idx} is out of range for output with {n_channels} channels."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the differential scalar target for each sequence in the batch.

        Args:
            x: Input one-hot sequence tensor ``(B, 4, L)``.

        Returns:
            Scalar tensor of shape ``(B,)``.
        """
        out = self.model(x)
        log_counts = out.log_counts  # (B, N)
        logits = out.logits          # (B, N, L)

        n_channels = log_counts.shape[-1]
        self._check_channel(n_channels, self.cond_a_idx, "cond_a_idx")
        self._check_channel(n_channels, self.cond_b_idx, "cond_b_idx")

        if self.mode == "delta_log_counts":
            return log_counts[:, self.cond_b_idx] - log_counts[:, self.cond_a_idx]

        if self.mode == "delta_profile_window_sum":
            length = logits.shape[-1]
            start, end = self._resolve_window(length)
            delta = logits[:, self.cond_b_idx, start:end] - logits[:, self.cond_a_idx, start:end]
            return delta.sum(dim=-1)

        raise ValueError(f"Unsupported mode: {self.mode!r}")


def resolve_ism_span(seq_len: int, start: int | None, end: int | None) -> tuple[int, int]:
    """Resolve and validate the inclusive/exclusive ISM mutation window."""
    span_start = 0 if start is None else start
    span_end = seq_len if end is None else end
    if not (0 <= span_start < span_end <= seq_len):
        raise ValueError(
            f"Invalid ISM span [{span_start}, {span_end}) for input length {seq_len}."
        )
    return span_start, span_end


def compute_ism_attributions(
    target_model: torch.nn.Module,
    inputs: torch.Tensor,
    ism_start: int | None,
    ism_end: int | None,
) -> torch.Tensor:
    """Compute single-position ISM deltas as ``(B, 4, L)`` attribution scores.

    Positions outside ``[ism_start, ism_end)`` are left as zeros.  At each
    mutated position the observed-nucleotide channel is set to the negative
    per-position mean (TF-MoDISco ``one_hot * hypothetical_contribs``
    convention) rather than the raw zero delta.
    """
    if inputs.shape[1] < 4:
        raise ValueError(
            f"ISM requires >=4 DNA channels in inputs, got shape {tuple(inputs.shape)}"
        )

    batch_size = inputs.shape[0]
    seq_len = inputs.shape[-1]
    span_start, span_end = resolve_ism_span(seq_len, ism_start, ism_end)

    attrs = torch.zeros((batch_size, 4, seq_len), device=inputs.device, dtype=inputs.dtype)
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
            # fill the observed nucleotide channel with negative of the per-position
            # mean hypothetical contribution instead of forcing it to zero.
            ref_base = inputs[:, :4, pos].argmax(dim=1)
            mean_at_pos = attrs[:, :, pos].mean(dim=1)
            attrs[sample_idx, ref_base, pos] = -mean_at_pos

    return attrs


def apply_off_simplex_gradient_correction(attrs: np.ndarray) -> np.ndarray:
    """Subtract per-position mean attribution across nucleotide channels."""
    if attrs.ndim != 3:
        raise ValueError(
            f"Expected 3D attribution array (N, 4, L), got shape {attrs.shape}"
        )
    return attrs - attrs.mean(axis=1, keepdims=True)
