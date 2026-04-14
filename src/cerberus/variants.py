"""Variant representation and VCF loading for variant effect prediction.

Provides a frozen dataclass for genomic variants (SNPs, insertions, deletions)
using 0-based coordinates consistent with cerberus's :class:`Interval`, and
a :func:`load_vcf` generator for parsing VCF/BCF files via cyvcf2.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyfaidx
import torch

from cerberus.interval import Interval
from cerberus.output import (
    FactorizedProfileCountOutput,
    ModelOutput,
    ProfileCountOutput,
    compute_channel_log_counts,
    compute_profile_probs,
    compute_signal,
)
from cerberus.sequence import encode_dna

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=True, unsafe_hash=False)
class Variant:
    """A single genomic variant (SNP, insertion, or deletion).

    Uses 0-based coordinates consistent with :class:`Interval`.  VCF files
    use 1-based POS; convert on load by subtracting 1.

    The reference allele occupies ``[pos, pos + len(ref))`` in 0-based
    half-open coordinates — the same convention as
    ``Interval(chrom, pos, pos + len(ref))``.

    Attributes:
        chrom: Chromosome name (e.g., ``'chr1'``).
        pos: 0-based position of the first base of the reference allele.
        ref: Reference allele string (e.g., ``'A'``, ``'ACGT'``).
        alt: Alternative allele string (e.g., ``'G'``, ``'A'``).
        id: Variant identifier from the VCF ID column.  Defaults to ``'.'``.
        info: VCF INFO fields stored verbatim.  Defaults to empty dict.
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    id: str = "."
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ref:
            raise ValueError("ref allele must be non-empty")
        if not self.alt:
            raise ValueError("alt allele must be non-empty")
        if self.pos < 0:
            raise ValueError(f"pos must be non-negative, got {self.pos}")
        if self.ref == self.alt:
            raise ValueError(f"ref and alt alleles must differ, both are {self.ref!r}")

    # ------------------------------------------------------------------
    # Coordinate properties
    # ------------------------------------------------------------------

    @property
    def end(self) -> int:
        """0-based, half-open end of the reference allele footprint.

        ``end = pos + len(ref)``, so the ref allele spans ``[pos, end)``.
        """
        return self.pos + len(self.ref)

    @property
    def ref_len(self) -> int:
        """Length of the reference allele in base pairs."""
        return len(self.ref)

    @property
    def alt_len(self) -> int:
        """Length of the alternative allele in base pairs."""
        return len(self.alt)

    @property
    def ref_center(self) -> int:
        """Midpoint of the reference allele footprint (integer floor division).

        Equivalent to ``(pos + end) // 2``.  For SNPs this equals ``pos``.
        For multi-base variants this is the center of the affected region
        in reference coordinates, used to place the scoring window
        symmetrically around the variant.
        """
        return (self.pos + self.end) // 2

    # ------------------------------------------------------------------
    # Variant type classification
    # ------------------------------------------------------------------

    @property
    def is_snp(self) -> bool:
        """True if both ref and alt are single nucleotides."""
        return len(self.ref) == 1 and len(self.alt) == 1

    @property
    def is_insertion(self) -> bool:
        """True if alt is longer than ref (net bases inserted)."""
        return len(self.alt) > len(self.ref)

    @property
    def is_deletion(self) -> bool:
        """True if ref is longer than alt (net bases deleted)."""
        return len(self.ref) > len(self.alt)

    @property
    def size_change(self) -> int:
        """Net change in sequence length.

        Positive for insertions, negative for deletions, zero for SNPs
        and MNPs (multi-nucleotide polymorphisms where ``len(ref) == len(alt)``).
        """
        return len(self.alt) - len(self.ref)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_interval(self) -> Interval:
        """Return the reference allele footprint as an :class:`Interval`.

        The interval spans ``[pos, pos + len(ref))`` in 0-based half-open
        coordinates.  For SNPs this is a 1 bp interval.  For deletions it
        covers the full deleted span in the reference.  For insertions it
        covers only the anchor base(s) that appear in the reference.
        """
        return Interval(self.chrom, self.pos, self.end)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        """Hash on core variant identity, excluding the mutable ``info`` dict."""
        return hash((self.chrom, self.pos, self.ref, self.alt, self.id))

    def __str__(self) -> str:
        """Compact display: ``chrom:pos:ref>alt`` (0-based pos)."""
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"

    @classmethod
    def from_str(cls, s: str) -> Variant:
        """Parse a variant from ``chrom:pos:ref>alt`` format (0-based pos).

        Args:
            s: String like ``'chr1:100:A>G'`` or ``'chr1:100:ACGT>A'``.

        Raises:
            ValueError: If the string format is invalid or the resulting
                variant fails validation.
        """
        try:
            chrom, pos_str, alleles = s.split(":")
            ref, alt = alleles.split(">")
            pos = int(pos_str)
        except ValueError:
            raise ValueError(
                f"Invalid variant string: {s!r}. Expected 'chrom:pos:ref>alt'."
            ) from None
        return cls(chrom=chrom, pos=pos, ref=ref, alt=alt)


# ----------------------------------------------------------------------
# VCF loading
# ----------------------------------------------------------------------


def _interval_to_region(interval: Interval) -> str:
    """Convert a 0-based half-open :class:`Interval` to a tabix region string.

    cyvcf2 / htslib expect ``chrom:start-end`` in **1-based inclusive**
    coordinates.  ``Interval(chr1, 0, 100)`` → ``"chr1:1-100"``.
    """
    return f"{interval.chrom}:{interval.start + 1}-{interval.end}"


def load_vcf(
    path: str | Path,
    region: str | Interval | None = None,
    pass_only: bool = True,
    info_fields: list[str] | None = None,
) -> Iterator[Variant]:
    """Parse variants from a VCF/BCF file.

    Yields one :class:`Variant` per record.  Coordinates are converted from
    VCF 1-based to cerberus 0-based on the fly.

    The input VCF **must be biallelic and left-aligned** (i.e. normalized with
    ``bcftools norm -m- -f ref.fa``).  Multi-allelic records are skipped with
    a warning.  See :ref:`variant-vcf-prep` for preparation instructions.

    Requires the ``cyvcf2`` package (``pip install cyvcf2``).

    .. note:: Sample-aware loading (future)

       This function currently ignores genotype (GT) fields and yields every
       variant site in the VCF.  To support per-sample variant scoring, a
       ``sample`` parameter could be added that filters to records where the
       specified sample has at least one non-reference allele.  The key
       considerations are:

       - cyvcf2's ``VCF(path, samples=[name])`` restricts GT parsing to one
         sample for efficiency.
       - ``record.genotypes`` returns ``[[allele1, allele2, phased], ...]``.
         A non-ref call is any entry where allele1 > 0 or allele2 > 0.
       - For diploid dosage encoding (e.g. 0/1 → 0.5 in the one-hot matrix),
         the genotype would need to be stored on the Variant or returned
         alongside it.  GenVarLoader solves this by reconstructing full
         haplotype sequences instead of blending alleles.

    Args:
        path: Path to a VCF or BCF file.  If ``region`` is specified the
            file must be bgzipped and indexed (``.tbi`` or ``.csi``).
        region: Restrict to variants overlapping this region.  Accepts a
            cerberus :class:`Interval` (0-based half-open) or a tabix-style
            string (``"chr1:1000-2000"``, 1-based inclusive).  ``None``
            iterates the entire file.
        pass_only: If ``True`` (default), skip records whose FILTER is not
            ``PASS`` (or ``.`` / empty, which VCF treats as unfiltered).
        info_fields: List of INFO field names to capture into
            :attr:`Variant.info`.  ``None`` (default) captures nothing.
            Example: ``["AF", "DP"]``.

    Yields:
        :class:`Variant` objects in file order.

    Raises:
        ImportError: If ``cyvcf2`` is not installed.
    """
    try:
        from cyvcf2 import VCF
    except ImportError:
        raise ImportError(
            "cyvcf2 is required for VCF loading. Install it with: pip install cyvcf2"
        ) from None

    # Convert Interval to tabix region string
    region_str: str | None = None
    if isinstance(region, Interval):
        region_str = _interval_to_region(region)
    elif isinstance(region, str):
        region_str = region

    vcf = VCF(str(path))

    records = vcf(region_str) if region_str is not None else vcf

    want_info = info_fields or []
    n_skipped_multiallelic = 0
    n_skipped_filter = 0

    for record in records:
        # Skip multi-allelic records
        if len(record.ALT) != 1:
            n_skipped_multiallelic += 1
            continue

        # FILTER check: record.FILTER is None for PASS / ".", else a string
        if pass_only and record.FILTER is not None:
            n_skipped_filter += 1
            continue

        alt = record.ALT[0]
        if not alt:
            # Monomorphic or star allele — skip
            continue

        info: dict[str, Any] = {}
        for key in want_info:
            val = record.INFO.get(key)
            if val is not None:
                info[key] = val

        yield Variant(
            chrom=record.CHROM,
            pos=record.POS - 1,  # VCF 1-based → cerberus 0-based
            ref=record.REF,
            alt=alt,
            id=record.ID or ".",
            info=info,
        )

    if n_skipped_multiallelic > 0:
        logger.warning(
            "Skipped %d multi-allelic records. Normalize your VCF with: "
            "bcftools norm -m- -f ref.fa input.vcf.gz | bcftools view -v snps,indels",
            n_skipped_multiallelic,
        )
    if n_skipped_filter > 0:
        logger.debug("Skipped %d records failing FILTER (pass_only=True)", n_skipped_filter)


# ----------------------------------------------------------------------
# TSV loading (pos/ref/alt)
# ----------------------------------------------------------------------


def load_variants(
    path: str | Path,
    zero_based: bool = False,
) -> Iterator[Variant]:
    """Parse variants from a tab-separated file with ``chrom``, ``pos``, ``ref``, ``alt`` columns.

    This provides a lightweight alternative to :func:`load_vcf` for simple
    variant lists.  The file must be tab-delimited with a header row.
    Required columns (in any order): ``chrom``, ``pos``, ``ref``, ``alt``.
    An optional ``id`` column is used for :attr:`Variant.id`; if absent,
    each variant gets ``id='.'``.  Extra columns are silently ignored.
    Lines starting with ``#`` (other than the header) are skipped.

    By default, ``pos`` is interpreted as **1-based** (the convention used by
    VCF, dbSNP, ClinVar, HGVS, and most variant databases).  The loader
    subtracts 1 to convert to cerberus's 0-based coordinate system.
    Set ``zero_based=True`` if the file already uses 0-based positions
    (e.g. data derived from BED or 0-based interval pipelines).

    Args:
        path: Path to the TSV file.
        zero_based: If ``True``, ``pos`` values are already 0-based and
            no conversion is applied.  Default ``False`` (1-based input,
            subtract 1).

    Yields:
        :class:`Variant` objects in file order.

    Raises:
        ValueError: If required columns are missing from the header.
    """
    path = Path(path)
    offset = 0 if zero_based else 1
    n_variants = 0

    with open(path) as fh:
        header = next(fh).rstrip("\n")
        if header.startswith("#"):
            header = header[1:]
        cols = {name.strip().lower(): i for i, name in enumerate(header.split("\t"))}

        missing = {"chrom", "pos", "ref", "alt"} - cols.keys()
        if missing:
            raise ValueError(
                f"Missing required column(s) in {path}: {', '.join(sorted(missing))}"
            )

        ic, ip, ir, ia = cols["chrom"], cols["pos"], cols["ref"], cols["alt"]
        ii = cols.get("id")

        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")

            yield Variant(
                chrom=p[ic],
                pos=int(p[ip]) - offset,
                ref=p[ir],
                alt=p[ia],
                id=p[ii] if ii is not None and ii < len(p) else ".",
            )
            n_variants += 1

    logger.info("Loaded %d variants from %s", n_variants, path)


# ----------------------------------------------------------------------
# Saturation variant generation
# ----------------------------------------------------------------------


def generate_variants(
    interval: Interval,
    fasta: pyfaidx.Fasta,
    max_indel_size: int = 0,
    encoding: str = "ACGT",
) -> Iterator[Variant]:
    """Generate all possible variants within a genomic interval.

    Yields every single-nucleotide substitution at each position in
    *interval*.  When ``max_indel_size > 0``, also yields deletions
    up to that size and insertions of every possible sequence up to
    that length.

    Positions where the reference base is not in *encoding* (e.g. ``N``)
    are skipped.

    Variants are yielded in positional order.  At each position: SNVs
    first, then deletions from size 1 to ``max_indel_size``, then
    insertions from size 1 to ``max_indel_size`` (all ``4^size``
    possible inserted sequences per size).

    Args:
        interval: Genomic interval to scan.
        fasta: An open ``pyfaidx.Fasta`` for the reference genome.
        max_indel_size: Maximum indel length.  0 (default) means SNVs
            only.  For size *k*, deletions remove up to *k* reference
            bases and insertions add up to *k* bases after the anchor.
        encoding: DNA base alphabet (default ``"ACGT"``).

    Yields:
        :class:`Variant` objects in positional order.

    Raises:
        ValueError: If the interval's chromosome is not in the FASTA.
        ValueError: If ``max_indel_size`` is negative.
    """
    if max_indel_size < 0:
        raise ValueError(f"max_indel_size must be non-negative, got {max_indel_size}")

    chrom = interval.chrom
    if chrom not in fasta:
        raise ValueError(f"Chromosome {chrom!r} not found in FASTA")

    bases = tuple(encoding.upper())
    bases_set = set(bases)
    chrom_len = len(fasta[chrom])
    start = interval.start
    end = min(interval.end, chrom_len)

    # Fetch the full reference sequence once
    ref_seq = str(fasta[chrom][start:end]).upper()

    for i, ref_base in enumerate(ref_seq):
        if ref_base not in bases_set:
            continue

        pos = start + i

        # -- SNVs --
        for alt_base in bases:
            if alt_base != ref_base:
                yield Variant(chrom, pos, ref_base, alt_base)

        # -- Deletions (ref = anchor + deleted bases, alt = anchor) --
        for del_size in range(1, max_indel_size + 1):
            ref_end = i + 1 + del_size
            if ref_end > len(ref_seq):
                break
            del_ref = ref_seq[i:ref_end]
            if not bases_set.issuperset(del_ref):
                break
            yield Variant(chrom, pos, del_ref, ref_base)

        # -- Insertions (ref = anchor, alt = anchor + inserted bases) --
        for ins_size in range(1, max_indel_size + 1):
            for ins_seq in _product_bases(ins_size, bases):
                yield Variant(chrom, pos, ref_base, ref_base + ins_seq)


def _product_bases(size: int, bases: tuple[str, ...]) -> Iterator[str]:
    """Yield all DNA strings of a given length over *bases*."""
    if size == 0:
        yield ""
        return
    for prefix in bases:
        for suffix in _product_bases(size - 1, bases):
            yield prefix + suffix


# ----------------------------------------------------------------------
# Ref / alt sequence construction
# ----------------------------------------------------------------------


def variant_to_ref_alt(
    variant: Variant,
    fasta: pyfaidx.Fasta,
    input_len: int,
    encoding: str = "ACGT",
) -> tuple[torch.Tensor, torch.Tensor, Interval]:
    """Construct one-hot ref and alt sequences centered on a variant.

    The input window is centered on :attr:`Variant.ref_center` — the midpoint
    of the reference allele footprint.  Both ref and alt windows have length
    ``input_len``.

    For **SNPs**, the alt sequence is identical to the ref except at the
    variant position.  All tensor positions are aligned.

    For **indels**, the alt sequence is built by splicing in the alternative
    allele and trimming symmetrically from both flanks to restore
    ``input_len``.  No bases are positionally aligned between ref and alt —
    this is intentional (see ``docs/internal/variant_support.md``
    Section 5.2 for rationale).

    Args:
        variant: The variant to score.
        fasta: An open ``pyfaidx.Fasta`` for the reference genome.
        input_len: Model input length in bp (e.g. 2112).
        encoding: One-hot encoding channel order.  Defaults to ``'ACGT'``.

    Returns:
        ``(ref_tensor, alt_tensor, interval)`` where both tensors have shape
        ``(4, input_len)`` and ``interval`` is the ref-coordinate window
        (for :class:`ModelEnsemble` fold routing).

    Raises:
        ValueError: If the window extends beyond chromosome boundaries.
        ValueError: If the FASTA sequence at the variant position does not
            match ``variant.ref``.
    """
    chrom = variant.chrom
    center = variant.ref_center

    # -- Ref window (same for both ref and alt) --
    ref_start = center - input_len // 2
    ref_end = ref_start + input_len

    if chrom not in fasta:
        raise ValueError(f"Chromosome {chrom!r} not found in FASTA")
    chrom_len = len(fasta[chrom])
    if ref_start < 0 or ref_end > chrom_len:
        raise ValueError(
            f"Variant {variant} window [{ref_start}, {ref_end}) extends beyond "
            f"chromosome {chrom} boundaries [0, {chrom_len})"
        )

    # Extract reference sequence string
    ref_seq = str(fasta[chrom][ref_start:ref_end]).upper()

    # -- Verify ref allele matches FASTA --
    var_offset = variant.pos - ref_start  # offset of variant.pos within window
    fasta_ref = ref_seq[var_offset : var_offset + variant.ref_len]
    if fasta_ref != variant.ref.upper():
        raise ValueError(
            f"Ref allele mismatch for {variant}: FASTA has {fasta_ref!r} at "
            f"{chrom}:{variant.pos}-{variant.end}, expected {variant.ref!r}. "
            f"Check genome build and coordinate system."
        )

    # -- Build alt sequence --
    if variant.size_change == 0:
        # SNP or MNP: direct substitution, same length
        alt_seq = ref_seq[:var_offset] + variant.alt + ref_seq[var_offset + variant.ref_len :]
    else:
        # Indel: splice in alt allele into a wider FASTA context,
        # then trim symmetrically from both ends.
        net = variant.size_change  # positive=insertion, negative=deletion

        # Amount to extend/trim on each flank
        # For insertion (net>0): trim ceil(net/2) upstream, floor(net/2) downstream
        # For deletion (net<0): extend ceil(|net|/2) upstream, floor(|net|/2) downstream
        abs_net = abs(net)
        trim_upstream = (abs_net + 1) // 2  # ceil
        trim_downstream = abs_net // 2       # floor

        if net > 0:
            # Insertion: splice produces a longer string, trim from both ends
            raw_alt = ref_seq[:var_offset] + variant.alt + ref_seq[var_offset + variant.ref_len :]
            alt_seq = raw_alt[trim_upstream : len(raw_alt) - trim_downstream]
        else:
            # Deletion: splice produces a shorter string, need more context
            ext_start = ref_start - trim_upstream
            ext_end = ref_end + trim_downstream

            if ext_start < 0 or ext_end > chrom_len:
                raise ValueError(
                    f"Variant {variant} requires extended FASTA context "
                    f"[{ext_start}, {ext_end}) for indel trimming, which extends "
                    f"beyond chromosome {chrom} boundaries [0, {chrom_len})"
                )

            ext_seq = str(fasta[chrom][ext_start:ext_end]).upper()
            # Offset of variant within the extended sequence
            ext_var_offset = variant.pos - ext_start
            raw_alt = (
                ext_seq[:ext_var_offset]
                + variant.alt
                + ext_seq[ext_var_offset + variant.ref_len :]
            )
            alt_seq = raw_alt

    assert len(alt_seq) == input_len, (
        f"Alt sequence length {len(alt_seq)} != input_len {input_len} "
        f"for variant {variant}"
    )

    # -- One-hot encode --
    ref_tensor = encode_dna(ref_seq, encoding)
    alt_tensor = encode_dna(alt_seq, encoding)

    interval = Interval(chrom, ref_start, ref_end)

    return ref_tensor, alt_tensor, interval


# ----------------------------------------------------------------------
# Effect size computation
# ----------------------------------------------------------------------


def _jsd(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Jensen-Shannon divergence between two probability distributions.

    Args:
        p: First distribution (must sum to 1 along ``dim``).
        q: Second distribution (must sum to 1 along ``dim``).
        dim: Dimension over which the distributions are defined.

    Returns:
        JSD values with ``dim`` reduced.
    """
    m = 0.5 * (p + q)
    eps = 1e-10

    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * (torch.log(a + eps) - torch.log(b + eps))).sum(dim=dim)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _pearson(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Pearson correlation between two tensors along ``dim``."""
    x_c = x - x.mean(dim=dim, keepdim=True)
    y_c = y - y.mean(dim=dim, keepdim=True)
    cov = (x_c * y_c).sum(dim=dim)
    return cov / (x_c.pow(2).sum(dim=dim).sqrt() * y_c.pow(2).sum(dim=dim).sqrt() + 1e-10)


def compute_variant_effects(
    ref_output: ModelOutput,
    alt_output: ModelOutput,
    log_counts_include_pseudocount: bool = False,
    pseudocount: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute variant effect scores between ref and alt model outputs.

    Reconstructs linear predicted signal from both outputs via
    :func:`~cerberus.output.compute_signal`, then computes
    multiple complementary effect metrics per channel.

    For :class:`~cerberus.output.FactorizedProfileCountOutput` (Dalmatian),
    additionally computes signal-only effects using the decomposed signal
    sub-model outputs (the bias component should be unaffected by regulatory
    variants).

    Args:
        ref_output: Model output on the reference sequence.
        alt_output: Model output on the alternative sequence.
        log_counts_include_pseudocount: Whether ``log_counts`` include a
            pseudocount offset.  Obtain from
            ``get_log_count_params(model_config)``.
        pseudocount: The pseudocount value.  Default 1.0.

    Returns:
        Dict of metric name to tensor.  All tensors have shape ``(B, C)``
        for batched inputs or ``(C,)`` for unbatched.

        Metrics for all output types:

        - ``"sad"``: Sum of absolute differences across the profile.
        - ``"max_abs_diff"``: Maximum absolute difference at any position.
        - ``"pearson"``: Pearson correlation between ref and alt profiles.

        Additional metrics for :class:`~cerberus.output.ProfileCountOutput`:

        - ``"log_fc"``: Log fold change in total predicted counts
          (``alt_log_counts - ref_log_counts``).
        - ``"jsd"``: Jensen-Shannon divergence between ref and alt profile
          shapes (softmax of logits).

        Additional metrics for
        :class:`~cerberus.output.FactorizedProfileCountOutput` (Dalmatian):

        - ``"signal_sad"``, ``"signal_log_fc"``, ``"signal_jsd"``: Effect
          metrics computed on the signal sub-model only.
    """
    ps_kwargs = {
        "log_counts_include_pseudocount": log_counts_include_pseudocount,
        "pseudocount": pseudocount,
    }

    ref_signal = compute_signal(ref_output, **ps_kwargs)
    alt_signal = compute_signal(alt_output, **ps_kwargs)

    effects: dict[str, torch.Tensor] = {}

    # -- Signal-level metrics (all output types) --
    diff = alt_signal - ref_signal
    effects["sad"] = diff.abs().sum(dim=-1)
    effects["max_abs_diff"] = diff.abs().max(dim=-1).values
    effects["pearson"] = _pearson(ref_signal, alt_signal)

    # -- Count and shape metrics (ProfileCountOutput and subclasses) --
    if isinstance(ref_output, ProfileCountOutput) and isinstance(
        alt_output, ProfileCountOutput
    ):
        ref_log_c = compute_channel_log_counts(ref_output, **ps_kwargs)
        alt_log_c = compute_channel_log_counts(alt_output, **ps_kwargs)
        effects["log_fc"] = alt_log_c - ref_log_c

        effects["jsd"] = _jsd(
            compute_profile_probs(ref_output),
            compute_profile_probs(alt_output),
        )

    # -- Dalmatian: signal sub-model only --
    if isinstance(ref_output, FactorizedProfileCountOutput) and isinstance(
        alt_output, FactorizedProfileCountOutput
    ):
        ref_sig = ProfileCountOutput(
            logits=ref_output.signal_logits,
            log_counts=ref_output.signal_log_counts,
        )
        alt_sig = ProfileCountOutput(
            logits=alt_output.signal_logits,
            log_counts=alt_output.signal_log_counts,
        )

        ref_sig_signal = compute_signal(ref_sig, **ps_kwargs)
        alt_sig_signal = compute_signal(alt_sig, **ps_kwargs)
        sig_diff = alt_sig_signal - ref_sig_signal
        effects["signal_sad"] = sig_diff.abs().sum(dim=-1)

        effects["signal_log_fc"] = (
            compute_channel_log_counts(alt_sig, **ps_kwargs)
            - compute_channel_log_counts(ref_sig, **ps_kwargs)
        )
        effects["signal_jsd"] = _jsd(
            compute_profile_probs(ref_sig),
            compute_profile_probs(alt_sig),
        )

    return effects
