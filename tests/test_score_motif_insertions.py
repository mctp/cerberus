"""Tests for the bpAI-TAC-style motif marginalization scorer.

Pins the dinucleotide-shuffle determinism, motif substitution geometry,
and the MEME parsing surface.  The full prediction loop needs a real
ModelEnsemble fixture, so we don't exercise it here.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.score_motif_insertions import (  # noqa: E402
    _build_arg_parser,
    _deduplicate_motif_names,
    _dinucleotide_shuffle_indices,
    _insert_motif,
    _parse_inline_motifs,
    _wilcoxon_pvalue,
)


# ---------------------------------------------------------------------------
# Dinucleotide shuffle -- the no-motif background generator
# ---------------------------------------------------------------------------


def test_dinucleotide_shuffle_preserves_length():
    seed = 42
    indices = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    shuffled = _dinucleotide_shuffle_indices(indices, seed)
    assert shuffled.shape == indices.shape


def test_dinucleotide_shuffle_preserves_nucleotide_composition():
    """The shuffle preserves per-nucleotide counts (it's a permutation
    on the Eulerian path through the dinucleotide transition graph)."""
    indices = np.array(
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64,
    )
    shuffled = _dinucleotide_shuffle_indices(indices, seed=7)
    for nuc in range(4):
        assert (shuffled == nuc).sum() == (indices == nuc).sum()


def test_dinucleotide_shuffle_determinism():
    """Same seed -> same shuffle.  Different seeds usually differ, but the
    Eulerian-path construction over a 4-symbol alphabet can produce the
    same output for some seed pairs; the determinism guarantee is the
    important contract here."""
    indices = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
    a = _dinucleotide_shuffle_indices(indices, seed=1)
    b = _dinucleotide_shuffle_indices(indices, seed=1)
    assert np.array_equal(a, b)


def test_dinucleotide_shuffle_preserves_endpoints():
    """The Eulerian-path construction starts at the original first base
    and ends at the original last base."""
    indices = np.array([0, 2, 1, 3, 2, 0], dtype=np.int64)
    shuffled = _dinucleotide_shuffle_indices(indices, seed=11)
    assert shuffled[0] == indices[0]
    assert shuffled[-1] == indices[-1]


# ---------------------------------------------------------------------------
# Motif substitution geometry
# ---------------------------------------------------------------------------


def test_insert_motif_overwrites_window_with_one_hot_substitution():
    """``_insert_motif`` overwrites the centre region with a one-hot
    encoding of the motif string, leaving every other position untouched.

    Returns ``(mutated, start, end)`` where ``start``/``end`` describe
    where the motif was inserted (centre - len/2 + offset).
    """
    import torch
    # 1-hot input, all "A" (channel 0) over length 16, batch=1.
    inputs = torch.zeros(1, 4, 16, dtype=torch.float32)
    inputs[0, 0, :] = 1.0
    motif_seq = "CTCG"
    offset = 0  # centred -- start = 16//2 - 4//2 = 6
    mutated, start, end = _insert_motif(inputs, motif_seq, offset)

    assert mutated.shape == (1, 4, 16)
    assert (start, end) == (6, 10)
    # Outside the motif region, still all-A.
    assert torch.all(mutated[0, 0, :start] == 1.0)
    assert torch.all(mutated[0, 0, end:] == 1.0)
    # Inside the motif region, each column is one-hot at the motif index.
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for col, base in enumerate(motif_seq):
        nuc = base_to_idx[base]
        assert mutated[0, nuc, start + col] == 1.0
        for other in range(4):
            if other == nuc:
                continue
            assert mutated[0, other, start + col] == 0.0


def test_insert_motif_offset_shifts_window():
    """Positive offset shifts the insertion window downstream of centre."""
    import torch
    inputs = torch.zeros(1, 4, 16, dtype=torch.float32)
    _, start, end = _insert_motif(inputs, "AAA", offset=3)
    # centre - len/2 + offset = 8 - 1 + 3 = 10; end = 13
    assert (start, end) == (10, 13)


def test_insert_motif_does_not_mutate_input():
    """The helper must return a new tensor; in-place mutation would
    corrupt the background sequence cached by the caller."""
    import torch
    inputs = torch.zeros(1, 4, 16, dtype=torch.float32)
    inputs[0, 0, :] = 1.0
    original = inputs.clone()
    _ = _insert_motif(inputs, "CCC", offset=0)
    assert torch.equal(inputs, original)


def test_insert_motif_raises_when_window_outside_input():
    """A motif placed past the input boundary raises explicitly."""
    import torch
    inputs = torch.zeros(1, 4, 16, dtype=torch.float32)
    with pytest.raises(ValueError, match="outside input"):
        _insert_motif(inputs, "AAAAA", offset=20)


# ---------------------------------------------------------------------------
# Inline motif parser + deduplication
# ---------------------------------------------------------------------------


def test_parse_inline_motifs_accepts_name_equals_sequence():
    out = _parse_inline_motifs(["FOXA1=AAACAA", "GATA=GATAA"])
    assert out == [("FOXA1", "AAACAA"), ("GATA", "GATAA")]


def test_parse_inline_motifs_rejects_invalid_characters():
    with pytest.raises(ValueError, match="must contain only A/C/G/T"):
        _parse_inline_motifs(["bad=AANCC"])


def test_parse_inline_motifs_rejects_missing_equals():
    with pytest.raises(ValueError, match="NAME=ACGT"):
        _parse_inline_motifs(["no_equals_sign"])


def test_deduplicate_motif_names_adds_hash_suffix():
    """Duplicate names get ``#N`` suffix; the underlying motif sequences
    are preserved verbatim."""
    out = _deduplicate_motif_names([("a", "ACGT"), ("a", "TGCA"), ("a", "AAAA")])
    names = [n for n, _ in out]
    seqs = [s for _, s in out]
    assert names == ["a", "a#2", "a#3"]
    assert seqs == ["ACGT", "TGCA", "AAAA"]


# ---------------------------------------------------------------------------
# Wilcoxon helper
# ---------------------------------------------------------------------------


def test_wilcoxon_pvalue_small_sample_returns_none():
    """The helper short-circuits when too few non-zero deltas exist; this
    keeps the summary TSV from carrying spurious p-values."""
    assert _wilcoxon_pvalue([0.0, 0.0, 0.0]) is None
    assert _wilcoxon_pvalue([]) is None


def test_wilcoxon_pvalue_clear_signal():
    """A clear unsigned-positive delta signal produces a small p-value."""
    pytest.importorskip("scipy.stats")
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    p = _wilcoxon_pvalue(values)
    assert p is not None
    assert p < 0.05


# ---------------------------------------------------------------------------
# Parser surface smoke
# ---------------------------------------------------------------------------


def test_parser_has_chrombpnet_accessibility_only_flag():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--checkpoint-dir", "ckpt",
        "--intervals-path", "x.bed",
        "--motif", "FOXA1=AAACAA",
        "--output", "out.tsv",
    ])
    # Default: auto (None) -- the tool resolves at runtime.
    assert args.chrombpnet_accessibility_only is None


def test_parser_collects_multiple_motif_specs():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--checkpoint-dir", "ckpt",
        "--intervals-path", "x.bed",
        "--motif", "FOXA1=AAACAA",
        "--motif", "GATA=GATAA",
        "--output", "out.tsv",
    ])
    assert args.motif == ["FOXA1=AAACAA", "GATA=GATAA"]
