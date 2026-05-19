"""Shared CLI helper for scale-aware count_pseudocount across train_* tools.

Lifts the argparse boilerplate and the args → scaled-pseudocount dispatch
out of every tool so the call sites stay uniform.  See
:func:`cerberus.pseudocount.resolve_read_coverage_pseudocount` for the math.
"""

from __future__ import annotations

import argparse

from cerberus.pseudocount import resolve_read_coverage_pseudocount


# Single shared default across every train_* tool.  ChIP-seq reads, DNase
# fragments, and the catch-all generic case all sit at ~150 bp.  ATAC-seq
# users should pass ``--read-length 100`` explicitly on the CLI -- preserving
# a per-tool override here would only mask a typo in one default vs another.
_DEFAULT_READ_LENGTH = 150


def add_pseudocount_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_count_pseudocount: float,
) -> None:
    """Register the standard scale-aware-pseudocount flag family on ``parser``.

    Five flags are added:

    - ``--count-pseudocount``: legacy linear-scale offset, multiplied by
      ``target_scale`` at resolve time.
    - ``--pseudocount-reads``: reads-equivalent override (recommended).
    - ``--read-length``: bp per read; only used with ``--pseudocount-reads``.
    - ``--input-scale {raw,cpm}``: bigWig normalisation.
    - ``--total-reads``: required when ``--input-scale=cpm``.

    Each adopting tool passes its canonical ``default_count_pseudocount``
    (e.g. 150.0 for BPNet, 1.0 for ASAP / Dalmatian / Gopher / Pomeranian /
    BiasNet); the other four flags share defaults across all tools.
    """
    parser.add_argument(
        "--count-pseudocount",
        type=float,
        default=default_count_pseudocount,
        help=(
            "Additive offset before log-transforming count targets "
            "(in raw coverage units).  Ignored when --pseudocount-reads is set."
        ),
    )
    parser.add_argument(
        "--pseudocount-reads",
        type=float,
        default=None,
        help=(
            "Scale-aware pseudocount in reads-equivalent units.  When set, "
            "overrides --count-pseudocount and computes the scaled value via "
            "cerberus.pseudocount.resolve_read_coverage_pseudocount."
        ),
    )
    parser.add_argument(
        "--read-length",
        type=int,
        default=_DEFAULT_READ_LENGTH,
        help=(
            "Read or fragment length in bp; used only when --pseudocount-reads "
            "is set.  Default 150 (ChIP-seq, DNase).  Use 100 for ATAC-seq."
        ),
    )
    parser.add_argument(
        "--input-scale",
        type=str,
        default="raw",
        choices=["raw", "cpm"],
        help=(
            "Input bigWig scale; used only when --pseudocount-reads is set.  "
            "Use 'cpm' for CPM-normalised bigWigs (requires --total-reads)."
        ),
    )
    parser.add_argument(
        "--total-reads",
        type=float,
        default=None,
        help=(
            "Library size (total mapped reads) for CPM normalisation; "
            "required when --input-scale=cpm and --pseudocount-reads is set."
        ),
    )


def resolve_count_pseudocount_from_args(
    args: argparse.Namespace,
    *,
    bin_size: int,
    target_scale: float,
) -> float:
    """Return the scaled ``count_pseudocount`` from a parsed CLI namespace.

    When ``args.pseudocount_reads`` is set, defers to
    :func:`cerberus.pseudocount.resolve_read_coverage_pseudocount`.
    Otherwise reproduces the pre-helper behaviour:
    ``args.count_pseudocount * target_scale``.
    """
    if args.pseudocount_reads is not None:
        return resolve_read_coverage_pseudocount(
            reads_equiv=args.pseudocount_reads,
            read_length=args.read_length,
            bin_size=bin_size,
            target_scale=target_scale,
            input_scale=args.input_scale,
            total_reads=args.total_reads,
        )
    return args.count_pseudocount * target_scale
