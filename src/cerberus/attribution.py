"""Attribution utilities shared across interpretation tools."""

from __future__ import annotations

import numpy as np
import torch


N_NUCLEOTIDES = 4
"""Number of DNA nucleotide channels (A, C, G, T) assumed by all attribution
methods. Inputs must have their first ``N_NUCLEOTIDES`` channels one-hot encode
DNA; trailing channels (if any) are treated as conditioning and ignored."""

IsmSpan = tuple[int | None, int | None]
"""Inclusive-start / exclusive-end ISM mutation window. ``None`` on either side
defaults to the sequence endpoint. ``(None, None)`` → full-length ISM."""


TARGET_REDUCTIONS = {
    "log_counts",
    "profile_bin",
    "profile_window_sum",
    "pred_count_bin",
    "pred_count_window_sum",
}

DIFFERENTIAL_TARGET_REDUCTIONS = {
    "delta_log_counts",
    "delta_profile_window_sum",
}


class AttributionTarget(torch.nn.Module):
    """Wrap Cerberus model output into a scalar target tensor.

    The ``reduction`` argument names the strategy used to reduce the model's
    multi-channel / multi-bin output to a single scalar per batch element,
    which is what attribution methods (ISM, TISM, IG, DLS) differentiate.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        reduction: str,
        channel: int,
        bin_index: int | None,
        window_start: int | None,
        window_end: int | None,
    ) -> None:
        super().__init__()
        if reduction not in TARGET_REDUCTIONS:
            raise ValueError(
                f"Unsupported reduction: {reduction!r}. "
                f"Must be one of {sorted(TARGET_REDUCTIONS)}."
            )
        self.model = model
        self.reduction = reduction
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

        if self.reduction == "log_counts":
            if log_counts.shape[1] == 1:
                return log_counts[:, 0]
            if channel >= log_counts.shape[1]:
                raise ValueError(
                    f"Requested channel={channel}, but log_counts has "
                    f"{log_counts.shape[1]} channels."
                )
            return log_counts[:, channel]

        if self.reduction == "profile_bin":
            bin_index = self._resolve_bin(logits.shape[-1])
            return logits[:, channel, bin_index]

        if self.reduction == "profile_window_sum":
            start, end = self._resolve_window(logits.shape[-1])
            return logits[:, channel, start:end].sum(dim=-1)

        if self.reduction == "pred_count_bin":
            pred_counts = self._predicted_counts_channel(logits, log_counts, channel)
            bin_index = self._resolve_bin(pred_counts.shape[-1])
            return pred_counts[:, bin_index]

        if self.reduction == "pred_count_window_sum":
            pred_counts = self._predicted_counts_channel(logits, log_counts, channel)
            start, end = self._resolve_window(pred_counts.shape[-1])
            return pred_counts[:, start:end].sum(dim=-1)

        raise ValueError(f"Unsupported reduction: {self.reduction}")


class DifferentialAttributionTarget(torch.nn.Module):
    """Reduce a multi-condition model output to ``f_B - f_A`` for attribution.

    The returned scalar is the *difference* between two condition channels, so
    attribution methods (ISM, TISM, IG, DLS) see a single delta target and
    produce sequence features that drive the differential. This wrapping is
    necessary for nonlinear attribution (DeepLIFT / DeepLIFTSHAP) where
    ``attr(f_B) - attr(f_A)`` does not equal ``attr(f_B - f_A)``; for linear
    methods (ISM, TISM, plain gradients) it is just a convenience.

    Two reductions:

    ``"delta_log_counts"``
        ``log_counts[:, cond_b_idx] - log_counts[:, cond_a_idx]``. Attributes
        features that change total binding / accessibility. Aligned with the
        :class:`~cerberus.loss.DifferentialCountLoss` training objective
        (Naqvi et al. 2025).

    ``"delta_profile_window_sum"``
        ``(logits_B - logits_A)[:, window_start:window_end].sum(-1)``.
        Attributes features that change the binding footprint shape. Works on
        the Phase 1 model without Phase 2 fine-tuning.

    Args:
        model: Model returning :class:`~cerberus.output.ProfileCountOutput`
            with ``log_counts`` of shape ``(B, N_conditions)`` — typically
            :class:`~cerberus.models.bpnet.MultitaskBPNet`.
        reduction: One of :data:`DIFFERENTIAL_TARGET_REDUCTIONS`.
        cond_a_idx: Channel index for condition A (reference). Default 0.
        cond_b_idx: Channel index for condition B (query). Default 1.
        window_start: Inclusive start of the output-profile window used by
            ``"delta_profile_window_sum"``. ``None`` → 0.
        window_end: Exclusive end. ``None`` → full output length.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        reduction: str,
        cond_a_idx: int = 0,
        cond_b_idx: int = 1,
        window_start: int | None = None,
        window_end: int | None = None,
    ) -> None:
        super().__init__()
        if reduction not in DIFFERENTIAL_TARGET_REDUCTIONS:
            raise ValueError(
                f"Unsupported reduction: {reduction!r}. "
                f"Must be one of {sorted(DIFFERENTIAL_TARGET_REDUCTIONS)}."
            )
        if cond_a_idx == cond_b_idx:
            raise ValueError(
                f"cond_a_idx and cond_b_idx must differ, got both={cond_a_idx}"
            )
        self.model = model
        self.reduction = reduction
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

    def _check_channels(self, n_channels: int) -> None:
        for name, idx in (("cond_a_idx", self.cond_a_idx), ("cond_b_idx", self.cond_b_idx)):
            if not (0 <= idx < n_channels):
                raise ValueError(
                    f"{name}={idx} is out of range for output with {n_channels} channels."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        log_counts = out.log_counts
        logits = out.logits

        self._check_channels(log_counts.shape[-1])

        if self.reduction == "delta_log_counts":
            return log_counts[:, self.cond_b_idx] - log_counts[:, self.cond_a_idx]

        if self.reduction == "delta_profile_window_sum":
            start, end = self._resolve_window(logits.shape[-1])
            delta = (
                logits[:, self.cond_b_idx, start:end]
                - logits[:, self.cond_a_idx, start:end]
            )
            return delta.sum(dim=-1)

        raise ValueError(f"Unsupported reduction: {self.reduction!r}")


def resolve_ism_span(seq_len: int, span: IsmSpan) -> tuple[int, int]:
    """Resolve and validate the inclusive/exclusive ISM mutation window.

    ``span`` is a ``(start, end)`` tuple where either side may be ``None`` to
    default to the sequence endpoint. ``(None, None)`` → full-length ISM.
    """
    start, end = span
    span_start = 0 if start is None else start
    span_end = seq_len if end is None else end
    if not (0 <= span_start < span_end <= seq_len):
        raise ValueError(
            f"Invalid ISM span [{span_start}, {span_end}) for input length {seq_len}."
        )
    return span_start, span_end


def _apply_tf_modisco_ref_override(
    attrs: torch.Tensor,
    inputs: torch.Tensor,
    span_start: int,
    span_end: int,
) -> None:
    """Overwrite the observed-nucleotide channel with ``-mean_j raw_delta_j``.

    Mutates ``attrs`` in place within ``[span_start, span_end)``. Implements
    the TF-MoDISco ``one_hot * hypothetical_contribs`` convention used by both
    exact ISM and first-order Taylor ISM: at each position in the span, the
    reference channel stores the negative mean of the four raw deltas, so that
    ``one_hot * attrs`` at the reference gives the importance score.

    Assumes ``attrs[:, :, span_start:span_end]`` holds raw deltas with the
    reference channel equal to zero (the natural output of both ISM and TISM).
    """
    ref_bases = inputs[:, :N_NUCLEOTIDES, span_start:span_end].argmax(dim=1)
    span_view = attrs[:, :, span_start:span_end]
    mean_in_span = span_view.mean(dim=1, keepdim=True)
    span_view.scatter_(dim=1, index=ref_bases.unsqueeze(1), src=-mean_in_span)


def compute_ism_attributions(
    target_model: torch.nn.Module,
    inputs: torch.Tensor,
    span: IsmSpan,
) -> torch.Tensor:
    """Compute single-position ISM deltas as ``(B, 4, L)`` attribution scores.

    ``span`` is a ``(start, end)`` tuple; positions outside the resolved span
    are left as zeros.  At each mutated position the observed-nucleotide
    channel is set to the negative per-position mean (TF-MoDISco
    ``one_hot * hypothetical_contribs`` convention) rather than the raw zero
    delta.
    """
    if inputs.shape[1] < N_NUCLEOTIDES:
        raise ValueError(
            f"ISM requires >={N_NUCLEOTIDES} DNA channels in inputs, "
            f"got shape {tuple(inputs.shape)}"
        )

    batch_size = inputs.shape[0]
    seq_len = inputs.shape[-1]
    span_start, span_end = resolve_ism_span(seq_len, span)

    attrs = torch.zeros(
        (batch_size, N_NUCLEOTIDES, seq_len), device=inputs.device, dtype=inputs.dtype
    )
    rows = torch.arange(batch_size * N_NUCLEOTIDES, device=inputs.device)
    nuc_idx = torch.arange(N_NUCLEOTIDES, device=inputs.device).repeat(batch_size)

    with torch.no_grad():
        ref_pred = target_model(inputs).reshape(batch_size)
        for pos in range(span_start, span_end):
            mutants = inputs.repeat_interleave(N_NUCLEOTIDES, dim=0)
            mutants[:, :N_NUCLEOTIDES, pos] = 0.0
            mutants[rows, nuc_idx, pos] = 1.0

            mut_pred = target_model(mutants).reshape(batch_size, N_NUCLEOTIDES)
            attrs[:, :, pos] = mut_pred - ref_pred[:, None]

    _apply_tf_modisco_ref_override(attrs, inputs, span_start, span_end)
    return attrs


def compute_taylor_ism_attributions(
    target_model: torch.nn.Module,
    inputs: torch.Tensor,
    span: IsmSpan,
    *,
    tf_modisco_format: bool = True,
) -> torch.Tensor:
    """First-order Taylor approximation of :func:`compute_ism_attributions`.

    Replaces the ``3 * L`` forward passes of exact ISM with one forward plus
    one backward pass. Per Sasse et al. 2024 (*iScience*, Eq. 7):

    .. math::
        \\mathrm{TISM}(l, b) = g[b, l] - g[\\mathrm{ref}(l), l]

    where :math:`g = \\partial f / \\partial x` is the model's input gradient
    and :math:`\\mathrm{ref}(l)` is the observed base at position :math:`l`.
    On models whose input-to-target map is linear (e.g. a fixed-weight linear
    scalar), the returned tensor is bit-identical to
    :func:`compute_ism_attributions` (verified by test).

    Reference-base gradient :math:`g[\\mathrm{ref}(l), l]` is computed as the
    dot product ``(g * x).sum(dim=1)``, matching
    [tism/torch_grad.py::correct_multipliers](../../../s2f-models/repos/TISM/tism/torch_grad.py).
    This form is bit-identical to ``g.gather(argmax)`` on one-hot inputs and
    is also well-defined on soft / PWM inputs.

    Input and output contracts are identical to :func:`compute_ism_attributions`:

    * ``inputs`` must have at least ``N_NUCLEOTIDES`` channels; the first
      ``N_NUCLEOTIDES`` one-hot encode DNA, trailing channels are ignored.
    * Returned tensor has shape ``(B, N_NUCLEOTIDES, L)`` and matches
      ``inputs.dtype``.
    * Positions outside the resolved ``span`` are left as zeros.
    * When ``tf_modisco_format=True`` (default), the observed-nucleotide
      channel holds ``-mean_j raw_delta_j`` — the TF-MoDISco
      ``one_hot * hypothetical_contribs`` convention exact ISM uses.
    * When ``tf_modisco_format=False``, raw TISM deltas are returned
      unchanged (reference channel == 0 in span), matching the TISM
      reference ``output='tism'``.

    The call is wrapped in :func:`torch.enable_grad` so callers nested inside
    ``@torch.no_grad()`` (common in eval loops) work correctly.
    """
    if inputs.shape[1] < N_NUCLEOTIDES:
        raise ValueError(
            f"TISM requires >={N_NUCLEOTIDES} DNA channels in inputs, "
            f"got shape {tuple(inputs.shape)}"
        )

    batch_size = inputs.shape[0]
    seq_len = inputs.shape[-1]
    span_start, span_end = resolve_ism_span(seq_len, span)

    x = inputs.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        out = target_model(x).reshape(batch_size)
        # grad_outputs=ones sums outputs across the batch; batch elements are
        # independent so per-sample gradients are recovered correctly.
        (grads,) = torch.autograd.grad(
            out,
            x,
            grad_outputs=torch.ones_like(out),
            create_graph=False,
            retain_graph=False,
        )

    dna_grads = grads[:, :N_NUCLEOTIDES, :]
    grad_ref = (dna_grads * inputs[:, :N_NUCLEOTIDES, :]).sum(dim=1, keepdim=True)
    raw = dna_grads - grad_ref

    attrs = torch.zeros(
        (batch_size, N_NUCLEOTIDES, seq_len),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    attrs[:, :, span_start:span_end] = raw[:, :, span_start:span_end]

    if tf_modisco_format:
        _apply_tf_modisco_ref_override(attrs, inputs, span_start, span_end)

    return attrs


def mean_center_attributions(attrs: np.ndarray) -> np.ndarray:
    """Subtract the per-position mean across nucleotide channels.

    Applies ``attrs - attrs.mean(axis=1, keepdims=True)`` to a ``(N, 4, L)``
    attribution tensor. This single operation is equivalent to three
    well-known formulations in the literature:

    * Majdandzic et al. 2023 off-simplex gradient correction.
    * Paper ATISM (Sasse et al. 2024, Eq. 8) when applied to raw TISM deltas.
    * TF-MoDISco hypothetical attributions with uniform 0.25 baseline.
    """
    if attrs.ndim != 3:
        raise ValueError(
            f"Expected 3D attribution array (N, 4, L), got shape {attrs.shape}"
        )
    return attrs - attrs.mean(axis=1, keepdims=True)
